#!/bin/bash
# ==============================================================================
# SCRIPT 3: EXPLICIT MEMORY BASELINES (AUTOMATA MODELS)
# Algorithms: QRM and PPO-RM
# ==============================================================================
echo "Starting Explicit Memory Baselines (Automata Models)..."

# Array of seeds to ensure statistical significance
SEEDS=(1 2 3)

# ---------------------------------------------------------
# 1. QRM (Handles Tabular environments)
# ---------------------------------------------------------
for seed in "${SEEDS[@]}"; do
    echo "====================================================="
    echo "Running QRM | Hazardous Delivery (Tabular) | Seed: $seed"
    echo "====================================================="
    python agents/qrm/train_qrm.py --env-id TemporalComp/HazardousDelivery-v0 --total-timesteps 1000000 --seed $seed
done

# ---------------------------------------------------------
# 2. PPO-RM (Handles Continuous Control environments)
# ---------------------------------------------------------
for seed in "${SEEDS[@]}"; do
    echo "====================================================="
    echo "Running PPO-RM | Deferred Maintenance (Continuous) | Seed: $seed"
    echo "====================================================="
    python agents/ppo_rm/train_ppo_rm.py --env-id TemporalComp/DeferredMaintenance-v0 --timesteps 2000000 --seed $seed
done

echo "Script 3 (Explicit Memory Baselines) Complete!"
