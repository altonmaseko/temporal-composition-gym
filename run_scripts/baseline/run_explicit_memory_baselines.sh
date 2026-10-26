#!/usr/bin/env bash
# Protocol v2 -- EXPLICIT-MEMORY BASELINES: QRM (delivery), QRM-Cue (colour), PPO-RM (maintenance)
# Runs only on the "explicit" machine, one training process at a time, seed-major order.
#
#   wandb login                          # once per machine; the code never reads the key
#   bash run_scripts/baseline/run_explicit_memory_baselines.sh --python /path/to/env/bin/python
#
# Options: --help. Documentation: run_scripts/baseline/README.md
set -Eeuo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib_launcher.sh"
launch_role explicit "$@"
