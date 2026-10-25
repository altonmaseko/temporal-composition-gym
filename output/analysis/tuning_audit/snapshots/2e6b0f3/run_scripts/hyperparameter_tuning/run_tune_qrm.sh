#!/bin/bash
# ==============================================================================
# HYPERPARAMETER TUNING: QRM (Tabular)
# ==============================================================================
echo "Starting Optuna Hyperparameter Tuning for QRM (HazardousDelivery-v0)..."
echo "Budget: 15 Trials"
echo "Timesteps: 300,000 per trial"
echo "Concurrency: 8 parallel jobs (CPU heavy)"
echo "Estimated Time: ~45 mins - 1.5 hours (depends on single-core performance)"
echo "=============================================================================="

# Use the correct python interpreter as per environment rules
python run_scripts/hyperparameter_tuning/tune_qrm.py

echo "QRM Tuning Complete!"
