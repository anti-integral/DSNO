#!/bin/bash

# ImageNet64 DSNO Setup Script
# This script sets up the environment and processes ImageNet64 data for DSNO training

set -e

echo "Setting up ImageNet64 DSNO training environment..."

# Create directory structure
echo "Creating directory structure..."
mkdir -p data/imagenet64_trajectories
mkdir -p configs/custom
mkdir -p scripts
mkdir -p exp

# Check if ImageNet64 data already exists
DATA_DIR="./data"
echo "Checking for existing ImageNet64 data in $DATA_DIR..."

# Check for training batch files
TRAIN_FILES_FOUND=0
for i in {1..10}; do
    if [ -f "$DATA_DIR/train_data_batch_$i" ]; then
        TRAIN_FILES_FOUND=$((TRAIN_FILES_FOUND + 1))
    fi
done

# Check for validation data
VAL_FILE_EXISTS=false
if [ -f "$DATA_DIR/val_data" ]; then
    VAL_FILE_EXISTS=true
fi

echo "Found $TRAIN_FILES_FOUND training batch files"
if [ "$VAL_FILE_EXISTS" = true ]; then
    echo "Found validation data file"
else
    echo "Validation data file not found"
fi

# If no data files found, download them
if [ $TRAIN_FILES_FOUND -eq 0 ]; then
    echo "No existing ImageNet64 data found. Downloading..."
    
    # Create raw data directory
    mkdir -p data/imagenet64_raw
    cd data/imagenet64_raw

    # Download ImageNet64 data
    echo "Downloading ImageNet64 training part 1..."
    wget https://image-net.org/data/downsample/Imagenet64_train_part1.zip

    echo "Downloading ImageNet64 training part 2..."
    wget https://image-net.org/data/downsample/Imagenet64_train_part2.zip

    echo "Downloading ImageNet64 validation data..."
    wget https://image-net.org/data/downsample/Imagenet64_val.zip

    # Unzip data files
    echo "Extracting data files..."
    unzip -q Imagenet64_train_part1.zip
    unzip -q Imagenet64_train_part2.zip
    unzip -q Imagenet64_val.zip

    # Move files to main data directory
    echo "Moving files to data directory..."
    mv train_data_batch_* ../
    mv val_data ../
    
    cd ../..
    
    # Clean up
    rm -rf data/imagenet64_raw
    
    echo "Data download and extraction complete!"
else
    echo "Using existing ImageNet64 data in $DATA_DIR"
fi

# Install required Python packages
echo "Installing required packages..."
echo "First, fixing NumPy installation..."
pip uninstall -y numpy
pip install numpy==1.24.3

echo "Installing other required packages..."
pip install lmdb pillow tqdm omegaconf lpips wandb clean-fid

echo "Data check completed. Found $TRAIN_FILES_FOUND training files."

# Check if trajectory data already exists
if [ -d "data/imagenet64_trajectories/lmdb" ] && [ -f "data/imagenet64_trajectories/labels.npy" ]; then
    echo "Processed trajectory data already exists!"
    echo "Delete 'data/imagenet64_trajectories/' if you want to reprocess."
else
    # Process ImageNet64 data to create trajectories
    echo "Processing ImageNet64 data to create trajectory dataset..."
    echo "This may take several hours depending on your hardware..."

    python scripts/process_imagenet64.py \
        --data_dir data \
        --output_dir data/imagenet64_trajectories \
        --num_timesteps 9 \
        --batch_size 100

    echo "Data processing complete!"
fi

# Verify data structure
echo "Verifying data structure..."
echo ""
echo "Current data directory contents:"
ls -la data/ | head -15

if [ -d "data/imagenet64_trajectories/lmdb" ]; then
    echo "✓ Training trajectory data exists"
else
    echo "✗ Training trajectory data missing"
    exit 1
fi

if [ -f "data/imagenet64_trajectories/labels.npy" ]; then
    echo "✓ Training labels exist"
else
    echo "✗ Training labels missing"
    exit 1
fi

# Count available training files
FINAL_TRAIN_COUNT=$(ls data/train_data_batch_* 2>/dev/null | wc -l)
echo "✓ $FINAL_TRAIN_COUNT training batch files available"

echo ""
echo "Setup complete! You can now start training with:"
echo "python train_imagenet.py --config configs/custom/imagenet64-dsno.yaml --log"
echo ""
echo "Final data structure:"
echo "├── data/"
echo "│   ├── train_data_batch_1        # Original ImageNet64 training files"
echo "│   ├── train_data_batch_2"
echo "│   ├── ..."
echo "│   ├── train_data_batch_10"
echo "│   ├── val_data                  # Original validation file"
echo "│   └── imagenet64_trajectories/  # Processed trajectory data"
echo "│       ├── lmdb/                 # Training trajectories"
echo "│       ├── labels.npy            # Training labels"
echo "│       └── val/                  # Validation data (if processed)"
echo "├── configs/custom/"
echo "│   └── imagenet64-dsno.yaml      # Configuration file"
echo "└── scripts/"
echo "    ├── process_imagenet64.py     # Data processing script"
echo "    ├── setup_imagenet64.sh       # This setup script"
echo "    └── train_imagenet64.sh       # Training script"