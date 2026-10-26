"""QRM entry point (protocol v2): QRM on Hazardous Delivery, QRM-Cue on Sequential Colour.

Prompt-2 state: runs the corrected QRM core on the frozen v2 environment settings and
prints per-episode training summaries (raw env return). Run directories, W&B logging,
periodic/final evaluation and checkpoints are added by the unified pipeline (Prompt 3).
It never loads or resumes from any existing checkpoint.
"""
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

from agents.common.env_factory import make_env  # noqa: E402
from agents.common.seeding import RunRNG, seed_torch, train_env_seed  # noqa: E402
from agents.qrm.qrm_core import QRM_DEFAULTS, QRMAgent, train_qrm  # noqa: E402
from agents.reward_machines import RewardMachineWrapper, make_reward_machine  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", choices=sorted(QRM_DEFAULTS), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--total-timesteps", type=int, required=True)
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    rng = RunRNG(args.seed)
    seed_torch(rng.torch_seed)
    base = make_env(args.env)
    rm = make_reward_machine(args.env, base)
    env = RewardMachineWrapper(base, rm, obs_mode=None, reward_mode="rm")
    agent = QRMAgent(env.observation_space, env.action_space.n, rm, QRM_DEFAULTS[args.env], device, rng)
    name = "QRM-Cue" if args.env == "colour" else "QRM"
    print(f"{name} {args.env} seed={args.seed} device={device} heads={rm.n_states} "
          f"replay={agent.replay.nbytes / 1e9:.2f} GB")

    def on_episode(step, ret, length, info):
        print(f"step={step} return={ret:.2f} length={length} cause={info.get('termination_cause')}")

    train_qrm(env, agent, args.total_timesteps, train_env_seed(args.seed), on_episode=on_episode)
    print(f"done: {args.total_timesteps} env steps, {agent.n_updates} updates")
    env.close()


if __name__ == "__main__":
    main()
