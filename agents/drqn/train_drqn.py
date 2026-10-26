"""DRQN entry point (protocol v2).

Prompt-2 state: runs the corrected DRQN core on the frozen v2 environment settings and
prints per-episode training summaries. Run directories, W&B logging, periodic/final
evaluation and checkpoints are added by the unified pipeline (Prompt 3). It never loads
or resumes from any existing checkpoint.
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
from agents.drqn.drqn_core import DRQN_DEFAULTS, DRQNAgent, train_drqn  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", choices=sorted(DRQN_DEFAULTS), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--total-timesteps", type=int, required=True)
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    rng = RunRNG(args.seed)
    seed_torch(rng.torch_seed)
    env = make_env(args.env)
    agent = DRQNAgent(env.observation_space, env.action_space.n, DRQN_DEFAULTS[args.env], device, rng)
    print(f"DRQN {args.env} seed={args.seed} device={device} replay={agent.replay.nbytes / 1e9:.2f} GB")

    def on_episode(step, ret, length, info):
        print(f"step={step} return={ret:.2f} length={length} cause={info.get('termination_cause')}")

    train_drqn(env, agent, args.total_timesteps, train_env_seed(args.seed), on_episode=on_episode)
    print(f"done: {args.total_timesteps} env steps, {agent.n_updates} updates")
    env.close()


if __name__ == "__main__":
    main()
