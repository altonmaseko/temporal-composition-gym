#!/usr/bin/env bash
# Protocol v2 -- IMPLICIT-MEMORY BASELINES: DRQN (delivery, colour), Recurrent PPO (maintenance)
# Runs only on the "implicit" machine, one training process at a time, seed-major order.
#
#   wandb login                          # once per machine; the code never reads the key
#   bash run_scripts/baseline/run_implicit_memory_baselines.sh --python /path/to/env/bin/python
#
# Options: --help. Documentation: run_scripts/baseline/README.md
set -Eeuo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib_launcher.sh"
launch_role implicit "$@"
