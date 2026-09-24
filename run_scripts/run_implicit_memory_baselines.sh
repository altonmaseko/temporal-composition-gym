#!/bin/bash
# ==============================================================================
# SCRIPT 2: IMPLICIT MEMORY BASELINES (SEQUENCE MODELS)
# Algorithms: DRQN and Recurrent PPO
# ==============================================================================
echo "Starting Implicit Memory Baselines (Sequence Models)..."

# Array of seeds to ensure statistical significance
SEEDS=(1 2 3)

# ---------------------------------------------------------
# 1. DRQN (Handles Tabular and Image environments)
# ---------------------------------------------------------
for seed in "${SEEDS[@]}"; do
    echo "====================================================="
    echo "Running DRQN | Hazardous Delivery (Tabular) | Seed: $seed"
    echo "====================================================="
    python agents/drqn/train_drqn.py --env-id TemporalComp/HazardousDelivery-v0 --total-timesteps 1000000 --seed $seed

    echo "====================================================="
    echo "Running DRQN | Sequential Colour (Image) | Seed: $seed"
    echo "====================================================="
    python agents/drqn/train_drqn.py --env-id TemporalComp/SequentialColour-v0 --total-timesteps 3000000 --seed $seed
done

# ---------------------------------------------------------
# 2. Recurrent PPO (Handles Continuous Control environments)
# ---------------------------------------------------------
for seed in "${SEEDS[@]}"; do
    echo "====================================================="
    echo "Running Recurrent PPO | Deferred Maintenance (Continuous) | Seed: $seed"
    echo "====================================================="
    python agents/train_recurrent_ppo.py --env TemporalComp/DeferredMaintenance-v0 --timesteps 2000000 --seed $seed
done

echo "Script 2 (Implicit Memory Baselines) Complete!"
