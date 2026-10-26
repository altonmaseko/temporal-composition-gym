#!/usr/bin/env bash
# DEPRECATED (protocol v2). This pilot "remediation" script launched six parallel jobs
# without budgets. Image DRQN and maintenance PPO-RM are now ordinary runs of the main
# launchers with explicit full budgets:
#   run_implicit_memory_baselines.sh   (includes DRQN on Sequential Colour)
#   run_explicit_memory_baselines.sh   (includes PPO-RM on Deferred Maintenance)
# It intentionally does nothing.
echo "rerun_drqn_image_and_ppo_rm_continuous.sh is deprecated and does nothing." >&2
echo "Use run_implicit_memory_baselines.sh / run_explicit_memory_baselines.sh (see run_scripts/baseline/README.md)." >&2
exit 2
