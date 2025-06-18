#!/bin/bash

# Simple ImageNet64 trajectory processing script
# Run this after your data verification shows files are ready

echo "=== Processing ImageNet64 Data to Trajectories ==="
echo ""

# Check if data exists
if [ ! -f "./data/train_data_batch_1" ]; then
    echo "❌ No training data found in ./data/"
    echo "Please ensure ImageNet64 files are in ./data/ directory"
    exit 1
fi

# Check if already processed
if [ -d "./data/imagenet64_trajectories/lmdb" ] && [ -f "./data/imagenet64_trajectories/labels.npy" ]; then
    echo "✅ Trajectory data already exists!"
    echo "Delete './data/imagenet64_trajectories/' if you want to reprocess"
    exit 0
fi

echo "🔄 Starting trajectory processing..."
echo "This will take several hours depending on your hardware..."
echo ""

# Create output directory
mkdir -p data/imagenet64_trajectories

# Run the processing script
python scripts/process_imagenet64.py \
    --data_dir ./data \
    --output_dir ./data/imagenet64_trajectories \
    --num_timesteps 9 \
    --batch_size 50

# Check if processing succeeded
if [ -d "./data/imagenet64_trajectories/lmdb" ] && [ -f "./data/imagenet64_trajectories/labels.npy" ]; then
    echo ""
    echo "✅ Trajectory processing completed successfully!"
    echo ""
    echo "Ready to train! Run:"
    echo "  ./scripts/train_imagenet64.sh"
else
    echo ""
    echo "❌ Trajectory processing failed!"
    echo "Check the error messages above"
    exit 1
fi