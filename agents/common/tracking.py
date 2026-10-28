"""Episode records, metric logging (local CSV + W&B) and W&B setup.

Local CSV files are the official numeric record. W&B receives the same numbers with the
environment-step counter `global_step` as x-axis. The code never reads, requires, prints or
stores a W&B API key: authentication is W&B's own (`wandb login` -> ~/.netrc).
"""
import hashlib
import json
import time

import gymnasium as gym
import numpy as np

from agents.common.run_io import utc_iso


# --------------------------------------------------------------------------- episode records
class EpisodeStats(gym.Wrapper):
    """Outermost wrapper: adds info["episode_record"] on the final step of every episode.

    return_raw is the env return (unaffected by any reward machine); return_rm is the RM
    return when an RM wrapper is present (equal on real trajectories). `progress` counts
    task milestones from the event stream: deliveries, correct walls, or services.
    """

    def __init__(self, env, env_key):
        super().__init__(env)
        self.env_key = env_key
        self._reset_counters()

    def _reset_counters(self):
        self.ret_raw, self.ret_rm, self.length, self.progress = 0.0, 0.0, 0, 0
        self.stage_exits, self.wall_touches = 0, 0
        self._held, self._ready = [], False
        self._has_rm = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._reset_counters()
        return obs, info

    def _count(self, events, info):
        for name, arg in events:
            if name == "stage_exit":
                self.stage_exits += 1
            elif name == "wall_touch":
                self.wall_touches += 1
            if self.env_key == "delivery" and name == "delivery":
                self.progress += 1
            elif self.env_key == "colour" and name == "room_reset":
                self.progress += 1
            elif self.env_key == "maintenance":
                if name == "code_collected":
                    self._held.append(arg)
                elif name == "readiness_on":
                    self._ready = True
                elif name == "service_attempt" and self._ready and arg in self._held:
                    self._held.remove(arg)
                    self.progress += 1
        if self.env_key == "colour" and info.get("is_success"):
            self.progress += 1                                   # final wall of the sequence

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        raw = info.get("raw_reward", reward)
        self.ret_raw += float(raw)
        if "rm" in info:
            self._has_rm = True
            self.ret_rm += float(info["rm"]["reward"])
        self.length += 1
        self._count(info.get("events", []), info)
        if terminated or truncated:
            info = dict(info)
            info["episode_record"] = {
                "return_raw": self.ret_raw, "return_rm": self.ret_rm if self._has_rm else None,
                "length": self.length, "success": bool(info.get("is_success", False)),
                "cause": info.get("termination_cause"), "progress": self.progress,
                "terminated": bool(terminated), "truncated": bool(truncated),
                # protocol v3 extra logging (None where the event does not exist in the task)
                "stage_exits": self.stage_exits if self.env_key == "maintenance" else None,
                "wall_touches": self.wall_touches if self.env_key == "colour" else None,
            }
        return obs, reward, terminated, truncated, info


def summarise(rows):
    r = np.array([x["return_raw"] for x in rows], dtype=float)
    causes = {}
    for x in rows:
        causes[x["cause"]] = causes.get(x["cause"], 0) + 1
    def _mean(key):
        vals = [x.get(key) for x in rows if x.get(key) is not None]
        return float(np.mean(vals)) if vals else None
    return {
        "n_episodes": len(rows),
        "stage_exits_mean": _mean("stage_exits"),
        "wall_touches_mean": _mean("wall_touches"),
        "return_mean": float(r.mean()), "return_std": float(r.std()),
        "return_min": float(r.min()), "return_max": float(r.max()),
        "success_rate": float(np.mean([x["success"] for x in rows])),
        "length_mean": float(np.mean([x["length"] for x in rows])),
        "progress_mean": float(np.mean([x["progress"] for x in rows])),
        "causes": causes,
    }


# --------------------------------------------------------------------------- W&B
def wandb_id_for(run_id):
    return hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:16]


def init_wandb(cfg, identity, cfg_hash, mode, run_path, resume=False):
    """Start (or resume) the W&B run. mode: 'online' | 'offline' | 'disabled'."""
    if mode == "disabled":
        return None
    import wandb
    w = cfg["wandb"]
    run = wandb.init(
        project=w["project"], entity=w["entity"], group=w["group"],
        job_type=f"{cfg['env']['key']}/{cfg['algo']['name']}", name=identity["run_id"], id=identity["wandb_id"],
        resume="allow" if resume else None, mode=mode, dir=run_path,
        config={"resolved": cfg, "config_hash": cfg_hash, "identity": identity},
        tags=[cfg["phase"], cfg["protocol_version"], cfg["mode"], cfg["algo"]["family"], cfg["env"]["key"],
              cfg["algo"]["name"], f"seed{cfg['seed']['train_seed']}"],
    )
    run.define_metric("global_step")
    run.define_metric("*", step_metric="global_step")
    return run


# --------------------------------------------------------------------------- metric logger
class MetricsLogger:
    def __init__(self, run_dir, wandb_run, cfg):
        self.rd = run_dir
        self.wb = wandb_run
        self.cfg = cfg
        names = ["train_episodes", "train_updates", "resources", "eval_periodic", "eval_points", "eval_final"]
        if cfg["eval"].get("secondary"):
            names += ["eval_periodic_noisy", "eval_points_noisy", "eval_final_noisy"]
        self.csv = {name: run_dir.csv_writer(name) for name in names}
        self.log_every = cfg["logging"]["train_log_every_steps"]
        self.next_log = self.log_every
        self.episodes = 0
        self._ep_buf = []
        self._upd_buf = []
        self._last_updates = {}
        self._t0 = time.time()
        self._last_res = (time.time(), 0)

    def restore(self, episodes, next_log):
        self.episodes, self.next_log = episodes, next_log

    def _wb(self, data, step):
        if self.wb is not None:
            data = dict(data)
            data["global_step"] = int(step)
            self.wb.log(data)

    # -- training
    def episode(self, step, rec):
        self.episodes += 1
        row = dict(rec, global_step=step, episode=self.episodes)
        self.csv["train_episodes"].write(row)
        self._ep_buf.append(rec)

    def update(self, step, n_updates, epsilon=None, metrics=None):
        """Called for every gradient update (custom agents); aggregated per log interval."""
        self._upd_buf.append(dict(metrics or {}, n_updates=n_updates, epsilon=epsilon))

    def flush_train(self, step, n_updates, epsilon=None, sb3_values=None, force=False):
        if not force and step < self.next_log:
            return
        while self.next_log <= step:
            self.next_log += self.log_every
        data = {"train/n_updates": n_updates, "train/episodes": self.episodes}
        if epsilon is not None:
            data["train/epsilon"] = epsilon
        if self._ep_buf:
            data["train/episode_return_mean"] = float(np.mean([e["return_raw"] for e in self._ep_buf]))
            data["train/episode_success_rate"] = float(np.mean([e["success"] for e in self._ep_buf]))
            data["train/episode_length_mean"] = float(np.mean([e["length"] for e in self._ep_buf]))
            data["train/episodes_in_interval"] = len(self._ep_buf)
        row = {"global_step": step, "n_updates": n_updates, "epsilon": epsilon}
        extra = {}
        if self._upd_buf:
            for key in ("loss", "q_mean", "grad_norm"):
                vals = [u[key] for u in self._upd_buf if u.get(key) is not None]
                if vals:
                    row[key] = float(np.mean(vals))
                    data[f"train/{key}"] = row[key]
            for key in self._upd_buf[-1]:
                if key not in ("loss", "q_mean", "grad_norm", "n_updates", "epsilon"):
                    vals = [u[key] for u in self._upd_buf if u.get(key) is not None]
                    if vals:
                        extra[key] = float(np.mean(vals))
                        data[f"train/{key}"] = extra[key]
        if sb3_values:
            for key, val in sb3_values.items():
                if isinstance(val, (int, float, np.floating, np.integer)):
                    data[f"sb3/{key}"] = float(val)
                    extra[key] = float(val)
            for key in ("loss", "q_mean", "grad_norm"):
                if f"train/{key}" in sb3_values:
                    row[key] = data[f"train/{key}"] = float(sb3_values[f"train/{key}"])
        row["extra_json"] = json.dumps(extra, sort_keys=True) if extra else None
        if self._upd_buf or sb3_values or n_updates:
            self.csv["train_updates"].write(row)
        self._wb(data, step)
        self._ep_buf, self._upd_buf = [], []

    def resources(self, step, replay_frames=None):
        now = time.time()
        t_last, s_last = self._last_res
        sps = (step - s_last) / (now - t_last) if now > t_last and step > s_last else None
        self._last_res = (now, step)
        row = {"global_step": step, "wall_time_s": now - self._t0, "env_steps_per_s": sps,
               "replay_frames": replay_frames}
        try:
            import psutil
            import os
            row["rss_gb"] = psutil.Process(os.getpid()).memory_info().rss / 1e9
        except ImportError:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                row["gpu_mem_alloc_gb"] = torch.cuda.memory_allocated() / 1e9
                row["gpu_mem_max_gb"] = torch.cuda.max_memory_allocated() / 1e9
        except ImportError:
            pass
        self.csv["resources"].write(row)
        self._wb({f"sys/{k}": v for k, v in row.items() if k != "global_step" and v is not None}, step)

    # -- evaluation
    def eval_point(self, idx, step, rows, n_updates, kind=""):
        """kind "" = headline greedy evaluation; "noisy" = v3 secondary evaluation."""
        sfx, pfx = (f"_{kind}", f"eval_{kind}") if kind else ("", "eval")
        for r in rows:
            self.csv[f"eval_periodic{sfx}"].write(dict(r, eval_index=idx, global_step=step))
        s = summarise(rows)
        self.csv[f"eval_points{sfx}"].write({**s, "eval_index": idx, "global_step": step, "n_updates": n_updates,
                                             "causes_json": json.dumps(s["causes"], sort_keys=True)})
        data = {f"{pfx}/{k}": v for k, v in s.items() if k != "causes" and v is not None}
        data.update({f"{pfx}/cause/{c}": n / s["n_episodes"] for c, n in s["causes"].items()})
        data[f"{pfx}/index"] = idx
        self._wb(data, step)
        return s

    def final(self, step, rows, kind=""):
        sfx, pfx = (f"_{kind}", f"final_eval_{kind}") if kind else ("", "final_eval")
        for r in rows:
            self.csv[f"eval_final{sfx}"].write(dict(r, global_step=step))
        s = summarise(rows)
        data = {f"{pfx}/{k}": v for k, v in s.items() if k != "causes" and v is not None}
        data.update({f"{pfx}/cause/{c}": n / s["n_episodes"] for c, n in s["causes"].items()})
        self._wb(data, step)
        if self.wb is not None:
            for k, v in data.items():
                self.wb.summary[k] = v
        return s

    def close(self):
        for w in self.csv.values():
            w.close()


def finish_wandb(run, exit_code):
    if run is None:
        return
    try:
        run.summary["run_status_utc"] = utc_iso()
        run.finish(exit_code=exit_code)
    except Exception as exc:                                     # never mask the run's own outcome
        print(f"W&B finish failed: {exc!r}", flush=True)
