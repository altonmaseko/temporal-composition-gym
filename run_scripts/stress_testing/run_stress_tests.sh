#!/bin/bash

echo "Starting Phase 4 Stress Tests in parallel..."

python run_scripts/stress_testing/stress_test_hazardous.py &
python run_scripts/stress_testing/stress_test_maintenance.py &
python run_scripts/stress_testing/stress_test_sequential.py &

# Wait for all background processes to finish
wait

echo "All stress tests completed successfully!"
