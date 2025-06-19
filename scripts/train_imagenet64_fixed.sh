#!/bin/bash

# Fixed ImageNet64 DSNO Training Script for single GPU
CONFIG_FILE="configs/custom/imagenet64-dsno.yaml"
LOG_DIR="exp/ImageNet64-DSNO-Custom"

echo "Starting ImageNet64 DSNO training (Single GPU)..."
echo "Configuration: $CONFIG_FILE"
echo "Log directory: $LOG_DIR"

# Create log directory
mkdir -p $LOG_DIR

# Always use single GPU to avoid multi-GPU issues
echo "Running single GPU training..."
python train_imagenet.py \
    --config $CONFIG_FILE \
    --log \
    --seed 42 \
    --local_rank 0 \
    --num_proc_node 1 \
    --num_gpus_per_node 1 \
    --port 9041 \
    --master_addr localhost \
    --amp

echo "Training started successfully!"
