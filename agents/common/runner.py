"""Unified baseline runner used by all six training entry points.

One run = one (environment, algorithm, seed, mode). The runner resolves the configuration
from agents/common/baseline_plan.py, creates an isolated run directory, records metadata,
starts W&B, trains for exactly the budget with periodic frozen-policy evaluation and
checkpoints, runs the final evaluation, verifies the run, and only then marks it complete.

Fresh training is the default; nothing is ever loaded from a shared directory. `--resume
RUN_DIR` continues a specific run after checking that its configuration, seed and identity
match. Resume is an EPISODE-BOUNDARY RESTART, not a bit-exact continuation:
  restored   network weights, target network, optimiser, update counters, exploration
             schedule (it is a function of the step), replay contents (off-policy), all RNG
             streams, evaluation/checkpoint progress, the training env's RNG
  not        the episode in progress at the checkpoint (a new episode starts), the
             recurrent actor state (reset with the new episode), and for on-policy methods the
             partially collected rollout (checkpoints are taken at rollout boundaries, so none
             is lost); local CSV rows logged after the checkpoint are moved to .bak files
Exit codes: 0 complete, 1 failed, 130 interrupted.
"""
import argparse
import json
import os
import random
import shutil
import signal
import sys
import time
import traceback

import numpy as np
import torch

from agents.common import baseline_plan as plan
from agents.common.env_factory import make_env_from_kwargs
from agents.common.evaluation import DRQNPolicy, QRMPolicy, SB3Policy, evaluate
from agents.common.run_io import RunDirectory, collect_metadata, read_json, utc_iso, write_json_atomic
from agents.common.seeding import RunRNG, eval_episode_seeds, seed_torch
from agents.common.tracking import EpisodeStats, MetricsLogger, finish_wandb, init_wandb, wandb_id_for
from agents.reward_machines import RewardMachineWrapper, make_reward_machine

# Device policy used by the launchers: the colour (image) runs need a GPU; delivery and
# maintenance agents are small MLPs dominated by environment simulation.
DEVICE_POLICY = {"delivery": "cpu", "colour": "cuda", "maintenance": "cpu"}


# --------------------------------------------------------------------------- environments
def build_env(cfg):
    """Same wrapper stack for training and evaluation (separate instances)."""
    key, algo = cfg["env"]["key"], cfg["algo"]["name"]
    base = make_env_from_kwargs(key, cfg["env"]["kwargs"])
    env = base
    if algo in ("qrm", "qrm_cue"):
        env = RewardMachineWrapper(base, make_reward_machine(key, base), obs_mode=None, reward_mode="rm")
    elif algo == "ppo_rm":
        env = RewardMachineWrapper(base, make_reward_machine(key, base), obs_mode="append", reward_mode="rm")
    elif algo == "dqn_ts":
        env = RewardMachineWrapper(base, make_reward_machine(key, base),
                                   obs_mode="dict" if key == "colour" else "append", reward_mode="env")
    return EpisodeStats(env, key)


# --------------------------------------------------------------------------- RNG state
def capture_rng(run_rng, env):
    return {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "run_numpy": run_rng.numpy.bit_generator.state,
        "run_replay": run_rng.replay.bit_generator.state,
        "run_exploration": run_rng.exploration.bit_generator.state,
        "run_python": run_rng.python.getstate(),
        "train_env": env.unwrapped.np_random.bit_generator.state,
    }


def restore_rng(state, run_rng, env):
    random.setstate(state["python"])
    np.random.set_state(state["numpy_global"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    run_rng.numpy.bit_generator.state = state["run_numpy"]
    run_rng.replay.bit_generator.state = state["run_replay"]
    run_rng.exploration.bit_generator.state = state["run_exploration"]
    run_rng.python.setstate(state["run_python"])
    gen = np.random.Generator(np.random.PCG64())
    gen.bit_generator.state = state["train_env"]
    env.unwrapped.np_random = gen


def _swap_restart_dir(rd, tmp):
    """Replace checkpoints/restart with tmp so a crash never leaves a half-written restart."""
    old = rd.restart_dir + ".old"
    if os.path.exists(old):
        shutil.rmtree(old)
    if os.path.exists(rd.restart_dir):
        os.replace(rd.restart_dir, old)
    os.replace(tmp, rd.restart_dir)
    if os.path.exists(old):
        shutil.rmtree(old)


# --------------------------------------------------------------------------- trainers
class Trainer:
    def __init__(self, cfg, device, rd, logger, run_rng, identity):
        self.cfg, self.device, self.rd, self.logger, self.run_rng, self.identity = cfg, device, rd, logger, run_rng, identity
        self.budget = cfg["budget"]["env_steps"]
        ev = cfg["eval"]
        self.eval_steps = [i * ev["every_steps"] for i in range(ev["n_points"] + 1)]
        self.periodic_seeds = eval_episode_seeds(ev["periodic_seed_block"], ev["periodic_episodes"])
        self.final_seeds = eval_episode_seeds(ev["final_seed_block"], ev["final_episodes"])
        self.ckpt_every = cfg["checkpoint"]["every_steps"]
        self.next_eval = 0
        self.next_ckpt = self.ckpt_every
        self.global_step = 0
        self.restored_steps = 0
        self.eval_env = build_env(cfg)
        self.last_eval = None
        self.fail_at_step = None                    # test aid: --debug-fail-at-step

    # -- hooks
    def hooks(self, step, allow_checkpoint=True):
        self.global_step = step
        if self.fail_at_step is not None and step >= self.fail_at_step:
            raise RuntimeError(f"injected failure at step {step} (--debug-fail-at-step)")
        if step >= self.logger.next_log:
            self.logger.flush_train(step, self.n_updates(), self.epsilon(), self.sb3_values())
        while self.next_eval < len(self.eval_steps) and step >= self.eval_steps[self.next_eval]:
            self.periodic_eval(self.next_eval, step)
            self.next_eval += 1
        if allow_checkpoint:
            self.maybe_checkpoint(step)

    def maybe_checkpoint(self, step):
        if step > 0 and step >= self.next_ckpt:
            self.save_checkpoint(step)
            while self.next_ckpt <= step:
                self.next_ckpt += self.ckpt_every

    def periodic_eval(self, idx, step):
        t = time.time()
        rows = evaluate(self.eval_env, self.eval_policy(), self.periodic_seeds)
        s = self.logger.eval_point(idx, step, rows, self.n_updates())
        self.logger.resources(step, self.replay_frames())
        self.last_eval = s
        print(f"[eval {idx}/{len(self.eval_steps) - 1}] step={step} success={s['success_rate']:.3f} "
              f"return={s['return_mean']:.2f} updates={self.n_updates()} ({time.time() - t:.0f}s)", flush=True)

    def final_eval(self):
        rows = evaluate(self.eval_env, self.eval_policy(), self.final_seeds)
        return rows, self.logger.final(self.global_step, rows)

    def progress_state(self):
        return {"global_step": self.global_step, "next_eval": self.next_eval, "next_ckpt": self.next_ckpt,
                "episodes": self.logger.episodes, "next_log": self.logger.next_log, "n_updates": self.n_updates()}

    def restore_progress(self, p):
        self.global_step = self.restored_steps = p["global_step"]
        self.next_eval, self.next_ckpt = p["next_eval"], p["next_ckpt"]
        self.logger.restore(p["episodes"], p["next_log"])

    def write_restart_meta(self, tmp, step):
        write_json_atomic(os.path.join(tmp, "state.json"), {
            "run_id": self.identity["run_id"], "config_hash": self.identity["config_hash"],
            "train_seed": self.cfg["seed"]["train_seed"], "global_step": step, "n_updates": self.n_updates(),
            "saved_utc": utc_iso(), "semantics": "episode-boundary restart",
        })

    # -- interface for subclasses
    def n_updates(self):
        raise NotImplementedError

    def epsilon(self):
        return None

    def sb3_values(self):
        return None

    def replay_frames(self):
        return None


class SB3Trainer(Trainer):
    def setup(self, resume=False):
        from sb3_contrib import RecurrentPPO
        from stable_baselines3 import DQN, PPO
        from stable_baselines3.common.logger import Logger
        from stable_baselines3.common.vec_env import DummyVecEnv
        classes = {"dqn": DQN, "dqn_ts": DQN, "ppo": PPO, "ppo_rm": PPO, "recurrent_ppo": RecurrentPPO}
        algo = self.cfg["algo"]["name"]
        self.cls = classes[algo]
        self.off_policy = self.cls is DQN
        self.recurrent = algo == "recurrent_ppo"
        cfg = self.cfg
        self.venv = DummyVecEnv([lambda: build_env(cfg)])
        if resume:
            self.model = self.cls.load(os.path.join(self.rd.restart_dir, "model.zip"), env=self.venv,
                                       device=self.device)
            if self.off_policy:
                self.model.load_replay_buffer(os.path.join(self.rd.restart_dir, "replay.pkl"))
        else:
            hp = dict(cfg["algo"]["hyperparams"])
            policy = hp.pop("policy")
            self.model = self.cls(policy, self.venv, seed=self.run_rng.torch_seed, device=self.device, verbose=0, **hp)
            # SB3 seeds the env with the model seed; use the protocol's training-env seed instead.
            self.venv.seed(cfg["seed"]["train_env_reset_seed"])
        self.model.set_logger(Logger(folder=None, output_formats=[]))   # values are read, never dumped

    def n_updates(self):
        return int(self.model._n_updates)

    def epsilon(self):
        return float(self.model.exploration_rate) if self.off_policy else None

    def sb3_values(self):
        return dict(self.model.logger.name_to_value)

    def replay_frames(self):
        return int(self.model.replay_buffer.size()) if self.off_policy else None

    def eval_policy(self):
        return SB3Policy(self.model, recurrent=self.recurrent)

    def save_checkpoint(self, step):
        self.model.save(os.path.join(self.rd.ckpt_dir, f"policy_step{step:09d}.zip"))
        tmp = self.rd.restart_dir + ".tmp"
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        os.makedirs(tmp)
        self.model.save(os.path.join(tmp, "model.zip"))
        if self.off_policy and self.cfg["checkpoint"]["save_replay"]:
            self.model.save_replay_buffer(os.path.join(tmp, "replay.pkl"))
        torch.save({"progress": self.progress_state(), "rng": capture_rng(self.run_rng, self.venv.envs[0])},
                   os.path.join(tmp, "state.pt"))
        self.write_restart_meta(tmp, step)
        _swap_restart_dir(self.rd, tmp)

    def load_restart(self):
        st = torch.load(os.path.join(self.rd.restart_dir, "state.pt"), weights_only=False)
        self.restore_progress(st["progress"])
        restore_rng(st["rng"], self.run_rng, self.venv.envs[0])
        if self.model.num_timesteps != self.global_step:
            raise RuntimeError("restart model step does not match the restart state")

    def train(self):
        from stable_baselines3.common.callbacks import BaseCallback
        trainer = self

        class Hooks(BaseCallback):
            def _on_step(self):
                for info in self.locals["infos"]:
                    if "episode_record" in info:
                        trainer.logger.episode(self.num_timesteps, info["episode_record"])
                # checkpoints only at rollout boundaries (after the previous update round)
                trainer.hooks(self.num_timesteps, allow_checkpoint=False)
                return True

            def _on_rollout_start(self):
                trainer.maybe_checkpoint(self.num_timesteps)

        start = self.global_step
        if start == 0:
            self.hooks(0)
        remaining = self.budget - start
        if remaining > 0:
            self.model.learn(total_timesteps=remaining, callback=Hooks(), log_interval=None,
                             reset_num_timesteps=(start == 0), progress_bar=False)
        if self.model.num_timesteps != self.budget:
            raise RuntimeError(f"SB3 executed {self.model.num_timesteps} env steps, budget is {self.budget}")
        self.global_step = self.model.num_timesteps
        self.logger.flush_train(self.global_step, self.n_updates(), self.epsilon(), self.sb3_values(), force=True)
        self.maybe_checkpoint(self.global_step)


class CustomTrainer(Trainer):
    def setup(self, resume=False):
        from agents.drqn.drqn_core import DRQNAgent, DRQNConfig
        from agents.qrm.qrm_core import QRMAgent, QRMConfig
        self.algo = self.cfg["algo"]["name"]
        self.env = build_env(self.cfg)
        hp = self.cfg["algo"]["hyperparams"]
        if self.algo == "drqn":
            self.acfg = DRQNConfig(**hp)
            self.agent = DRQNAgent(self.env.observation_space, self.env.action_space.n, self.acfg, self.device,
                                   self.run_rng)
        else:
            self.acfg = QRMConfig(**hp)
            rm = self.env.env.rm                                  # EpisodeStats -> RewardMachineWrapper
            self.agent = QRMAgent(self.env.observation_space, self.env.action_space.n, rm, self.acfg, self.device,
                                  self.run_rng)
        self._eps = self.acfg.eps_start

    def n_updates(self):
        return int(self.agent.n_updates)

    def epsilon(self):
        return float(self._eps)

    def replay_frames(self):
        return int(self.agent.replay.size)

    def eval_policy(self):
        return DRQNPolicy(self.agent) if self.algo == "drqn" else QRMPolicy(self.agent)

    def save_checkpoint(self, step):
        torch.save(self.agent.q.state_dict(), os.path.join(self.rd.ckpt_dir, f"policy_step{step:09d}.pt"))
        tmp = self.rd.restart_dir + ".tmp"
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        os.makedirs(tmp)
        torch.save({"q": self.agent.q.state_dict(), "target": self.agent.target.state_dict(),
                    "opt": self.agent.opt.state_dict(), "n_updates": self.agent.n_updates,
                    "progress": self.progress_state(), "rng": capture_rng(self.run_rng, self.env)},
                   os.path.join(tmp, "state.pt"))
        if self.cfg["checkpoint"]["save_replay"]:
            self.agent.replay.save(os.path.join(tmp, "replay.npz"))
        self.write_restart_meta(tmp, step)
        _swap_restart_dir(self.rd, tmp)

    def load_restart(self):
        st = torch.load(os.path.join(self.rd.restart_dir, "state.pt"), weights_only=False, map_location=self.device)
        self.agent.q.load_state_dict(st["q"])
        self.agent.target.load_state_dict(st["target"])
        self.agent.opt.load_state_dict(st["opt"])
        self.agent.n_updates = st["n_updates"]
        self.agent.replay.load(os.path.join(self.rd.restart_dir, "replay.npz"))
        self.restore_progress(st["progress"])
        restore_rng(st["rng"], self.run_rng, self.env)

    def train(self):
        from agents.drqn.drqn_core import linear_epsilon
        a, env, c = self.agent, self.env, self.acfg
        qrm = self.algo != "drqn"
        start = self.global_step
        if start == 0:
            obs, info = env.reset(seed=self.cfg["seed"]["train_env_reset_seed"])
            self.hooks(0)
        else:
            obs, info = env.reset()                       # episode-boundary restart
        head = info["rm"]["head"] if qrm else None
        a.begin_episode(obs)
        for step in range(start + 1, self.budget + 1):
            self._eps = eps = linear_epsilon(step - 1, self.budget, c)
            action = a.act(obs, head, eps) if qrm else a.act(obs, eps)
            next_obs, reward, term, trunc, info = env.step(action)
            if qrm:
                a.observe(action, reward, term, trunc, next_obs, info["rm"])
            else:
                a.observe(action, reward, term, trunc, next_obs)
            if term or trunc:
                self.logger.episode(step, info["episode_record"])
                obs, info = env.reset()
                head = info["rm"]["head"] if qrm else None
                a.begin_episode(obs)
            else:
                obs = next_obs
                head = info["rm"]["next_head"] if qrm else None
            if step >= c.learning_starts and step % c.train_freq == 0:
                m = a.update()
                self.logger.update(step, a.n_updates, eps, m)
            if step % c.target_update == 0:
                a.sync_target()
            self.hooks(step)
        self.logger.flush_train(self.global_step, self.n_updates(), self.epsilon(), force=True)


# --------------------------------------------------------------------------- device
def select_device(requested, require_gpu):
    cuda = torch.cuda.is_available()
    selected = ("cuda" if cuda else "cpu") if requested == "auto" else requested
    if selected.startswith("cuda") and not cuda:
        raise RuntimeError(f"device {requested!r} requested but CUDA is not available")
    if require_gpu and not selected.startswith("cuda"):
        raise RuntimeError("--require-gpu was given but the run would use the CPU")
    info = {"requested": requested, "selected": selected, "cuda_available": cuda, "is_gpu_run": selected.startswith("cuda"),
            "require_gpu": bool(require_gpu)}
    if info["is_gpu_run"]:
        idx = torch.device(selected).index or 0
        info["gpu_name"] = torch.cuda.get_device_name(idx)
        info["gpu_index"] = idx
    return info


# --------------------------------------------------------------------------- main flow
def _check_resume(rd, cfg, cfg_hash):
    identity = read_json(rd.file("identity.json"))
    stored_cfg = read_json(rd.file("config.resolved.json"))
    if identity["config_hash"] != cfg_hash or plan.config_hash(stored_cfg) != cfg_hash:
        raise RuntimeError("resume refused: the resolved configuration differs from the run being resumed "
                           f"({identity['config_hash'][:10]} on disk vs {cfg_hash[:10]} now)")
    if identity["seed"] != cfg["seed"]["train_seed"] or identity["algo"] != cfg["algo"]["name"] \
            or identity["env"] != cfg["env"]["key"]:
        raise RuntimeError("resume refused: env/algorithm/seed differ from the run being resumed")
    status = rd.status() or {}
    if status.get("state") == "complete":
        raise RuntimeError("resume refused: the run is already complete")
    state_path = os.path.join(rd.restart_dir, "state.json")
    if not os.path.exists(state_path):
        raise RuntimeError("resume refused: no restart checkpoint in this run directory")
    st = read_json(state_path)
    if st["run_id"] != identity["run_id"] or st["config_hash"] != cfg_hash or st["train_seed"] != identity["seed"]:
        raise RuntimeError("resume refused: restart checkpoint belongs to a different run")
    return identity, status, st


def run(cfg, args):
    cfg_hash = plan.config_hash(cfg)
    device = select_device(args.device, args.require_gpu)
    if args.threads:
        torch.set_num_threads(args.threads)
    threads = {"torch_num_threads": torch.get_num_threads(), "requested": args.threads}
    run_rng = RunRNG(cfg["seed"]["train_seed"])
    seed_torch(run_rng.torch_seed)
    random.seed(run_rng.torch_seed)
    np.random.seed(run_rng.torch_seed)
    launcher = {"stdout": args.launcher_stdout, "stderr": args.launcher_stderr}
    resume = args.resume is not None

    if resume:
        rd = RunDirectory(os.path.abspath(args.resume))
        identity, status, restart_meta = _check_resume(rd, cfg, cfg_hash)
        write_json_atomic(rd.file(f"metadata_resume_{time.strftime('%Y%m%dT%H%M%S')}.json"),
                          collect_metadata(device, threads, {"launcher_logs": launcher}))
        resumes = status.get("resumes", []) + [{"utc": utc_iso(), "restored_step": restart_meta["global_step"],
                                                 "launcher_logs": launcher, "device": device}]
        rd.update_status(state="running", resumes=resumes, error=None)
        rd.truncate_records(restart_meta["global_step"])
    else:
        rd = RunDirectory.create(cfg, cfg_hash, args.results_root)
        run_id = f"{cfg['phase']}-{cfg['protocol_version']}-{cfg['mode']}-{cfg['env']['key']}-{cfg['algo']['name']}" \
                 f"-s{cfg['seed']['train_seed']}-{cfg_hash[:10]}-{rd.run_uid}"
        identity = {
            "run_id": run_id, "run_uid": rd.run_uid, "wandb_id": wandb_id_for(run_id), "config_hash": cfg_hash,
            "protocol_version": cfg["protocol_version"], "phase": cfg["phase"], "mode": cfg["mode"],
            "env": cfg["env"]["key"], "algo": cfg["algo"]["name"], "family": cfg["algo"]["family"],
            "seed": cfg["seed"]["train_seed"], "machine_role": args.machine_role, "batch_id": args.batch_id,
            "results_root": os.path.abspath(args.results_root) if args.results_root else None,
            "run_dir": rd.path, "created_utc": utc_iso(),
        }
        write_json_atomic(rd.file("config.resolved.json"), cfg)
        write_json_atomic(rd.file("identity.json"), identity)
        write_json_atomic(rd.file("metadata.json"), collect_metadata(device, threads, {"launcher_logs": launcher}))
        rd.update_status(state="running", started_utc=utc_iso(), budget=cfg["budget"]["env_steps"],
                         env_steps_done=0, final_eval_done=False, resumes=[])
    print(f"run {identity['run_id']}\n  dir {rd.path}\n  device {device['selected']} (gpu run: {device['is_gpu_run']})",
          flush=True)

    wb = None
    logger = None
    exit_code = 1
    try:
        wb = init_wandb(cfg, identity, cfg_hash, args.wandb_mode, rd.path, resume=resume)
        if wb is not None:
            rd.update_status(wandb={"id": wb.id, "url": getattr(wb, "url", None), "mode": args.wandb_mode,
                                    "project": cfg["wandb"]["project"], "entity": cfg["wandb"]["entity"]})
        logger = MetricsLogger(rd, wb, cfg)
        TrainerCls = SB3Trainer if cfg["algo"]["impl"] == "sb3" else CustomTrainer
        trainer = TrainerCls(cfg, device["selected"], rd, logger, run_rng, identity)
        trainer.fail_at_step = args.debug_fail_at_step
        trainer.setup(resume=resume)
        if resume:
            trainer.load_restart()
            print(f"  resumed at step {trainer.restored_steps} (episode-boundary restart)", flush=True)
        session_start = trainer.global_step
        trainer.train()
        steps_done = trainer.global_step
        rows, summary = trainer.final_eval()
        n_points = trainer.next_eval
        problems = []
        if steps_done != cfg["budget"]["env_steps"]:
            problems.append(f"env steps {steps_done} != budget {cfg['budget']['env_steps']}")
        if n_points != cfg["eval"]["n_points"] + 1:
            problems.append(f"periodic evaluations {n_points} != {cfg['eval']['n_points'] + 1}")
        if len(rows) != cfg["eval"]["final_episodes"]:
            problems.append(f"final evaluation episodes {len(rows)} != {cfg['eval']['final_episodes']}")
        if problems:
            raise RuntimeError("verification failed: " + "; ".join(problems))
        final = {"global_step": steps_done, "n_updates": trainer.n_updates(), "train_episodes": logger.episodes,
                 "final_eval": summary, "last_periodic_eval": trainer.last_eval, "config_hash": cfg_hash,
                 "run_id": identity["run_id"], "completed_utc": utc_iso()}
        write_json_atomic(rd.file("final_summary.json"), final)
        rd.update_status(state="complete", env_steps_done=steps_done, final_eval_done=True,
                         n_updates=trainer.n_updates(), restored_steps=trainer.restored_steps,
                         new_steps_this_session=steps_done - session_start, finished_utc=utc_iso(), verified=True)
        print(f"complete: final success={summary['success_rate']:.3f} return={summary['return_mean']:.2f}", flush=True)
        exit_code = 0
    except KeyboardInterrupt:
        rd.update_status(state="interrupted", error="interrupted (SIGINT/SIGTERM)", finished_utc=utc_iso())
        print("interrupted", file=sys.stderr, flush=True)
        exit_code = 130
    except Exception as exc:
        rd.update_status(state="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(),
                         finished_utc=utc_iso())
        traceback.print_exc()
        exit_code = 1
    finally:
        if logger is not None:
            logger.close()
        finish_wandb(wb, exit_code)
    return exit_code


def _on_sigterm(signum, frame):
    raise KeyboardInterrupt


def build_parser(script_algo, description, task_state_flag=False):
    envs = sorted({e for a, spec in plan.ALGORITHMS.items() if plan.ALGORITHMS[a]["script"] ==
                   plan.ALGORITHMS[script_algo]["script"] for e in spec["envs"]})
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", choices=envs, required=True)
    p.add_argument("--seed", type=int, required=True, help="training seed (baseline: 1-5, expansion 6-10)")
    p.add_argument("--mode", choices=("full", "smoke"), default="full",
                   help="smoke = reduced pipeline check; outputs are labelled and kept apart")
    if task_state_flag:
        p.add_argument("--task-state", action="store_true", help="DQN-TS control: observation + RM task state")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY.PATH=VALUE",
                   help="override an existing resolved-config entry (recorded and included in the config hash)")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    p.add_argument("--require-gpu", action="store_true", help="fail instead of running on the CPU")
    p.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    p.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default=None,
                   help="default: online for full runs, offline for smoke runs")
    p.add_argument("--wandb-project", default=plan.DEFAULT_WANDB_PROJECT)
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--results-root", default=None, help="default: run_scripts/baseline/results")
    p.add_argument("--resume", default=None, metavar="RUN_DIR", help="continue this specific run directory")
    p.add_argument("--machine-role", default=None)
    p.add_argument("--batch-id", default=None)
    p.add_argument("--launcher-stdout", default=None)
    p.add_argument("--launcher-stderr", default=None)
    p.add_argument("--print-config", action="store_true", help="print the resolved config and hash, then exit")
    p.add_argument("--debug-fail-at-step", type=int, default=None,
                   help="pipeline tests only: raise an error once this env step is reached")
    return p


def cli(script_algo, description, argv=None, task_state_flag=False):
    args = build_parser(script_algo, description, task_state_flag).parse_args(argv)
    algo = plan.algo_for(script_algo, args.env)
    if task_state_flag and args.task_state:
        algo = "dqn_ts"
    overrides = dict(plan.parse_override(o) for o in args.overrides)
    cfg = plan.resolve_config(args.env, algo, args.seed, args.mode, overrides, args.wandb_project, args.wandb_entity)
    if args.wandb_mode is None:
        args.wandb_mode = "online" if args.mode == "full" else "offline"
    if args.print_config:
        print(json.dumps({"config_hash": plan.config_hash(cfg), "config": cfg}, indent=2))
        return 0
    signal.signal(signal.SIGTERM, _on_sigterm)
    return run(cfg, args)
