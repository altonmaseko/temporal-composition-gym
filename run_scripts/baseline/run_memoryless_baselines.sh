#!/usr/bin/env bash
# Protocol v2 -- MEMORY-LESS BASELINES (control group): DQN (delivery, colour), PPO (maintenance), then DQN-TS controls
# Runs only on the "memoryless" machine, one training process at a time, seed-major order.
#
#   wandb login                          # once per machine; the code never reads the key
#   bash run_scripts/baseline/run_memoryless_baselines.sh --python /path/to/env/bin/python
#
# Options: --help. Documentation: run_scripts/baseline/README.md
set -Eeuo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib_launcher.sh"
launch_role memoryless "$@"
