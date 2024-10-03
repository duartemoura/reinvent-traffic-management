#!/bin/bash
set -e

# Ensure the conda environment is properly initialized
source /opt/conda/etc/profile.d/conda.sh

if [ "$PROCESS_TYPE" == "TRAIN" ]; then
    echo "Starting training process..."
    conda activate tf_gpu
    python sagemaker_train.py
elif [ "$PROCESS_TYPE" == "TEST" ]; then
    echo "Starting testing process..."
    conda activate tf_gpu
    python testing_main.py
else
    echo "Invalid PROCESS_TYPE. Please set to TRAIN or TEST."
    exit 1
fi
