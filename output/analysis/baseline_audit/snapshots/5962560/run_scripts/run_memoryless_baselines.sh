#!/bin/bash
# ==============================================================================
# SCRIPT 1: MEMORY-LESS BASELINES (CONTROL GROUP)
# Algorithms: DQN and PPO
# ==============================================================================
echo "Starting Memory-less Baselines (Control Group)..."

# Array of seeds to ensure statistical significance
SEEDS=(1 2 3)

# ---------------------------------------------------------
# 1. DQN (Handles Tabular and Image environments)
# ---------------------------------------------------------
for seed in "${SEEDS[@]}"; do
    echo "====================================================="
    echo "Running DQN | Hazardous Delivery (Tabular) | Seed: $seed"
    echo "====================================================="
    python agents/train_dqn.py --env TemporalComp/HazardousDelivery-v0 --timesteps 1000000 --seed $seed

    echo "====================================================="
    echo "Running DQN | Sequential Colour (Image) | Seed: $seed"
    echo "====================================================="
    python agents/train_dqn.py --env TemporalComp/SequentialColour-v0 --timesteps 3000000 --seed $seed
done

# ---------------------------------------------------------
# 2. PPO (Handles Continuous Control environments)
# ---------------------------------------------------------
for seed in "${SEEDS[@]}"; do
    echo "====================================================="
    echo "Running PPO | Deferred Maintenance (Continuous) | Seed: $seed"
    echo "====================================================="
    python agents/train_ppo.py --env TemporalComp/DeferredMaintenance-v0 --timesteps 2000000 --seed $seed
done

echo "Script 1 (Memory-less Baselines) Complete!"
