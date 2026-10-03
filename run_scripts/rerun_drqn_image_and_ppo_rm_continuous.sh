#!/bin/bash

# Ensure script fails if any command fails
set -e

# Make sure we're in the repository root directory
cd "$(dirname "$0")/.."

echo "======================================================"
echo " Starting 6 Temporal Composition Gym Runs in Parallel "
echo "======================================================"
echo "Hardware Profile: RTX 3090 (24GB VRAM), 40 vCPUs, 129GB RAM"
echo "Expected Resource Usage:"
echo "- VRAM: < 5GB total (very small networks)"
echo "- RAM: ~15-20GB total (due to image observation buffers)"
echo "- Compute: heavily CPU bound (6 parallel envs will easily fit on 40 vCPUs)"
echo ""

# Activate environment if needed (uncomment and modify if you use a specific conda/venv on Vast)
# source /path/to/venv/bin/activate
# or
# conda activate temporal_gym

# -------------------------------------------------------------
# 1. PPO-RM on Continuous Environment (Deferred Maintenance)
# -------------------------------------------------------------
echo "Starting PPO-RM (DeferredMaintenance-v0) - Seeds 1, 2, 3..."
python agents/ppo_rm/train_ppo_rm.py --env-id TemporalComp/DeferredMaintenance-v0 --seed 1 &
python agents/ppo_rm/train_ppo_rm.py --env-id TemporalComp/DeferredMaintenance-v0 --seed 2 &
python agents/ppo_rm/train_ppo_rm.py --env-id TemporalComp/DeferredMaintenance-v0 --seed 3 &

# -------------------------------------------------------------
# 2. DRQN on Image Environment (Sequential Colour)
# -------------------------------------------------------------
echo "Starting DRQN (SequentialColour-v0) - Seeds 1, 2, 3..."
python agents/drqn/train_drqn.py --env-id TemporalComp/SequentialColour-v0 --seed 1 &
python agents/drqn/train_drqn.py --env-id TemporalComp/SequentialColour-v0 --seed 2 &
python agents/drqn/train_drqn.py --env-id TemporalComp/SequentialColour-v0 --seed 3 &

echo "------------------------------------------------------"
echo "All 6 runs launched in the background!"
echo "Waiting for all background jobs to finish..."
echo "You can check W&B (if logged in) for live training curves."

# Wait for all background processes to finish
wait

echo "======================================================"
echo " All runs have successfully completed! "
echo "======================================================"
