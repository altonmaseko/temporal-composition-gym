#!/bin/bash
# ==============================================================================
# HYPERPARAMETER TUNING: DRQN (Image-Based)
# ==============================================================================
echo "Starting Optuna Hyperparameter Tuning for DRQN (SequentialColour-v0)..."
echo "Budget: 15 Trials"
echo "Timesteps: 300,000 per trial"
echo "Concurrency: 2 parallel jobs (Limited to avoid CUDA OOM on RTX 3090 Ti)"
echo "Estimated Time: ~1.5 - 2.5 hours"
echo "=============================================================================="

# Use the correct python interpreter as per environment rules
python run_scripts/hyperparameter_tuning/tune_drqn.py

echo "DRQN Tuning Complete!"
