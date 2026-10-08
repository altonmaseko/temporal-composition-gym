#!/bin/bash
# ==============================================================================
# HYPERPARAMETER TUNING: PPO-RM (Continuous Control)
# ==============================================================================
echo "Starting Optuna Hyperparameter Tuning for PPO-RM (DeferredMaintenance-v0)..."
echo "Budget: 20 Trials"
echo "Timesteps: 500,000 per trial"
echo "Concurrency: 1 sequential job (due to WandB/Optuna deadlock)"
echo "Estimated Time: ~2 - 3 hours"
echo "=============================================================================="

# Use the correct python interpreter as per environment rules
python run_scripts/hyperparameter_tuning/tune_ppo_rm.py

echo "PPO-RM Tuning Complete!"
