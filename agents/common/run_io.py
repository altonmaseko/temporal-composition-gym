"""Run directories, identity, metadata, status and local numeric records.

Layout (protocol v2 section 6.5):

  <results_root>/<protocol>/<mode>/<env>/<algo>/seed<seed>__<cfghash10>__<run_uid>/
      config.resolved.json  identity.json  metadata.json  status.json
      train_episodes.csv  train_updates.csv  resources.csv
      eval_periodic.csv  eval_points.csv  eval_final.csv  final_summary.json
      checkpoints/policy_step<N>.*   checkpoints/restart/   logs/

run_uid = UTC timestamp + 6 random hex characters, so two machines (or two launches on one
machine) never write to the same directory. Status files are written atomically. Missing
or failed runs keep state "failed"/"running"; nothing here ever writes a zero for a
result that was not measured.
"""
import csv
import datetime as _dt
import hashlib
import json
import os
import platform
import secrets
import shutil
import socket
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_RESULTS_ROOT = os.path.join(REPO_ROOT, "run_scripts", "baseline", "results")

# Environment variables that are safe and useful to record. Nothing else is read.
RECORDED_ENV_VARS = ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "WANDB_MODE",
                     "PYTHONHASHSEED", "BASELINE_MACHINE_ROLE", "BASELINE_BATCH_ID")

CSV_FIELDS = {
    "train_episodes": ["global_step", "episode", "return_raw", "return_rm", "length", "success", "cause", "progress",
                       "stage_exits", "wall_touches"],
    "train_updates": ["global_step", "n_updates", "epsilon", "loss", "q_mean", "grad_norm", "extra_json"],
    "resources": ["global_step", "wall_time_s", "env_steps_per_s", "rss_gb", "gpu_mem_alloc_gb",
                  "gpu_mem_max_gb", "replay_frames"],
    "eval_periodic": ["eval_index", "global_step", "episode_seed", "return_raw", "length", "success", "cause",
                      "progress", "stage_exits", "wall_touches"],
    "eval_points": ["eval_index", "global_step", "n_episodes", "return_mean", "return_std", "return_min",
                    "return_max", "success_rate", "length_mean", "progress_mean", "causes_json", "n_updates",
                    "stage_exits_mean", "wall_touches_mean"],
    "eval_final": ["global_step", "episode_seed", "return_raw", "length", "success", "cause", "progress",
                   "stage_exits", "wall_touches"],
}
# Protocol v3 secondary ("noisy") evaluation: same columns as the headline evaluation files.
for _name in ("eval_periodic", "eval_points", "eval_final"):
    CSV_FIELDS[f"{_name}_noisy"] = list(CSV_FIELDS[_name])

# Stress phase (decision D11): per-episode files gain the required count and the within-4,000-step
# metrics. Only stress run directories use this table; baseline and tuning files are unchanged.
STRESS_EPISODE_FIELDS = ["required", "progress_at_4000", "return_at_4000", "success_by_4000"]
STRESS_CSV_FIELDS = {name: list(fields) + (STRESS_EPISODE_FIELDS if name.split("_noisy")[0] in
                                           ("train_episodes", "eval_periodic", "eval_final") else [])
                     for name, fields in CSV_FIELDS.items()}


def utc_now():
    return _dt.datetime.now(_dt.timezone.utc)


def utc_iso():
    return utc_now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def new_run_uid():
    return utc_now().strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(3)


def write_json_atomic(path, data):
    tmp = f"{path}.tmp-{secrets.token_hex(4)}"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2, sort_keys=False, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- metadata
def _run(cmd, cwd=REPO_ROOT):
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def git_info():
    rev = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"])
    diff = _run(["git", "diff", "HEAD", "--", ".", ":(exclude)*.pyc"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    changed = [line[3:] for line in (status or "").splitlines() if line.strip()]
    return {
        "rev": rev.strip() if rev else None,
        "branch": branch.strip() if branch else None,
        "dirty": bool(changed) if status is not None else None,
        "changed_tracked_files": changed,                       # file names only, no contents
        "diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest() if diff else None,
    }


def package_versions():
    import importlib.metadata as md
    out = {}
    for name in ("torch", "stable_baselines3", "sb3_contrib", "gymnasium", "minigrid", "pymunk", "pygame",
                 "numpy", "wandb", "psutil"):
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = None
    return out


def hardware_info():
    info = {"hostname": socket.gethostname(), "platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor() or None, "cpu_count_logical": os.cpu_count()}
    try:
        import psutil
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["ram_total_gb"] = round(psutil.virtual_memory().total / 1e9, 2)
        info["ram_available_gb"] = round(psutil.virtual_memory().available / 1e9, 2)
        info["disk_free_gb"] = round(shutil.disk_usage(REPO_ROOT).free / 1e9, 2)
    except ImportError:
        pass
    try:
        import torch
        info["torch_cuda_available"] = torch.cuda.is_available()
        info["torch_cuda_version"] = torch.version.cuda
        info["gpus"] = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                info["gpus"].append({"index": i, "name": p.name, "vram_gb": round(p.total_memory / 1e9, 2),
                                     "capability": f"{p.major}.{p.minor}"})
    except ImportError:
        pass
    smi = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    info["nvidia_smi"] = smi.strip().splitlines() if smi else None
    return info


def collect_metadata(device_info, threads, extra=None):
    meta = {
        "created_utc": utc_iso(),
        "git": git_info(),
        "python": {"version": sys.version.split()[0], "executable": sys.executable},
        "packages": package_versions(),
        "hardware": hardware_info(),
        "device": device_info,
        "threads": threads,
        "env_vars": {k: os.environ.get(k) for k in RECORDED_ENV_VARS},
        "argv": sys.argv,
    }
    if extra:
        meta.update(extra)
    return meta


# --------------------------------------------------------------------------- run directory
class RunDirectory:
    csv_fields = CSV_FIELDS

    def __init__(self, path):
        self.path = path
        self.ckpt_dir = os.path.join(path, "checkpoints")
        self.restart_dir = os.path.join(self.ckpt_dir, "restart")
        self.log_dir = os.path.join(path, "logs")

    @classmethod
    def create(cls, cfg, cfg_hash, results_root=None):
        root = results_root or DEFAULT_RESULTS_ROOT
        uid = new_run_uid()
        rel = os.path.join(cfg["protocol_version"], cfg["mode"], cfg["env"]["key"], cfg["algo"]["name"],
                           f"seed{cfg['seed']['train_seed']}__{cfg_hash[:10]}__{uid}")
        path = os.path.join(root, rel)
        os.makedirs(path, exist_ok=False)                       # never reuse a directory
        rd = cls(path)
        os.makedirs(rd.ckpt_dir)
        os.makedirs(rd.log_dir)
        rd.run_uid = uid
        return rd

    def file(self, name):
        return os.path.join(self.path, name)

    # -- status
    def status(self):
        p = self.file("status.json")
        return read_json(p) if os.path.exists(p) else None

    def update_status(self, **fields):
        st = self.status() or {}
        st.update(fields)
        st["updated_utc"] = utc_iso()
        write_json_atomic(self.file("status.json"), st)
        return st

    # -- csv
    def csv_writer(self, name):
        return CSVLog(self.file(f"{name}.csv"), self.csv_fields[name])

    def truncate_records(self, max_step):
        """Resume support: drop rows logged after the restored step (originals kept as .bak)."""
        stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
        for name in self.csv_fields:
            path = self.file(f"{name}.csv")
            if not os.path.exists(path) or name == "eval_final":
                continue
            with open(path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            keep = [r for r in rows if int(float(r["global_step"])) <= max_step]
            if len(keep) == len(rows):
                continue
            shutil.copy2(path, f"{path}.pre-resume-{stamp}.bak")
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self.csv_fields[name])
                w.writeheader()
                w.writerows(keep)
        final = self.file("eval_final.csv")
        if os.path.exists(final):
            shutil.move(final, f"{final}.pre-resume-{stamp}.bak")


class CSVLog:
    """Append-only CSV with a fixed header; flushed after every write."""

    def __init__(self, path, fields):
        self.path = path
        self.fields = fields
        new = not os.path.exists(path)
        self._f = open(path, "a", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=fields, extrasaction="raise")
        if new:
            self._w.writeheader()
            self._f.flush()

    def write(self, row):
        self._w.writerow({k: _fmt(row.get(k)) for k in self.fields})
        self._f.flush()

    def close(self):
        self._f.close()


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        return repr(v)
    return v
