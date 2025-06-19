#!/bin/bash

echo "=== Starting ImageNet64 DSNO Training ==="
echo "Configuration: Single GPU optimized setup"

# Check if trajectory data exists
if [ ! -d "data/imagenet64_trajectories/lmdb" ]; then
    echo "❌ Trajectory data not found. Please run data processing first:"
    echo "   ./scripts/simple_process.sh"
    exit 1
fi

# Start training with optimized config
python train_imagenet.py \
    --config configs/custom/imagenet64-dsno-single.yaml \
    --log \
    --seed 42 \
    --local_rank 0 \
    --num_proc_node 1 \
    --num_gpus_per_node 1 \
    --port 9042 \
    --master_addr localhost \
    --amp

echo "Training completed or stopped."
