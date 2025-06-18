#!/bin/bash

# ImageNet64 DSNO Training Script
# Usage: ./scripts/train_imagenet64.sh

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Adjust based on your available GPUs

# Training configuration
CONFIG_FILE="configs/custom/imagenet64-dsno.yaml"
NUM_GPUS=1  # Adjust based on your setup
BATCH_SIZE=8  # Per GPU batch size
LOG_DIR="exp/ImageNet64-DSNO-Custom"

# Create log directory
mkdir -p $LOG_DIR

echo "Starting ImageNet64 DSNO training..."
echo "Configuration: $CONFIG_FILE"
echo "Number of GPUs: $NUM_GPUS"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Log directory: $LOG_DIR"

# Single GPU training
if [ $NUM_GPUS -eq 1 ]; then
    echo "Running single GPU training..."
    python train_imagenet.py \
        --config $CONFIG_FILE \
        --log \
        --seed 42 \
        --local_rank 0 \
        --num_proc_node 1 \
        --num_gpus_per_node 1 \
        --amp

# Multi-GPU training
else
    echo "Running multi-GPU training on $NUM_GPUS GPUs..."
    python train_imagenet.py \
        --config $CONFIG_FILE \
        --log \
        --seed 42 \
        --local_rank 0 \
        --num_proc_node 1 \
        --num_gpus_per_node $NUM_GPUS \
        --port 9040 \
        --master_addr localhost \
        --amp
fi

echo "Training started. Monitor progress in the log directory: $LOG_DIR"
echo "You can also monitor with wandb if enabled in the config."