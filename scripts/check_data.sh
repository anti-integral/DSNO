#!/bin/bash

# Check existing ImageNet64 data script
# This script verifies the existing data structure and content

echo "=== ImageNet64 Data Verification ==="
echo ""

DATA_DIR="./data"

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Data directory '$DATA_DIR' does not exist!"
    exit 1
fi

echo "📁 Data directory: $DATA_DIR"
echo ""

# Check training batch files
echo "🔍 Checking training batch files..."
TRAIN_COUNT=0
TRAIN_SIZES=""

for i in {1..10}; do
    FILE="$DATA_DIR/train_data_batch_$i"
    if [ -f "$FILE" ]; then
        SIZE=$(ls -lh "$FILE" | awk '{print $5}')
        echo "  ✅ train_data_batch_$i (size: $SIZE)"
        TRAIN_COUNT=$((TRAIN_COUNT + 1))
        TRAIN_SIZES="$TRAIN_SIZES $SIZE"
    else
        echo "  ❌ train_data_batch_$i (missing)"
    fi
done

echo ""
echo "📊 Summary: Found $TRAIN_COUNT/10 training batch files"

# Check validation file
echo ""
echo "🔍 Checking validation file..."
VAL_FILE="$DATA_DIR/val_data"
if [ -f "$VAL_FILE" ]; then
    VAL_SIZE=$(ls -lh "$VAL_FILE" | awk '{print $5}')
    echo "  ✅ val_data (size: $VAL_SIZE)"
else
    echo "  ❌ val_data (missing)"
fi

# Check if files are readable and seem to be pickle files
echo ""
echo "🔍 Testing file readability..."

# Test first training file if it exists
FIRST_TRAIN="$DATA_DIR/train_data_batch_1"
if [ -f "$FIRST_TRAIN" ]; then
    if python3 -c "
import pickle
import sys
try:
    with open('$FIRST_TRAIN', 'rb') as f:
        data = pickle.load(f)
    print('  ✅ train_data_batch_1 is readable as pickle file')
    if 'data' in data:
        print(f'  📈 Contains {len(data[\"data\"])} images')
    if 'labels' in data:
        print(f'  🏷️  Contains {len(data[\"labels\"])} labels')
except Exception as e:
    print(f'  ❌ Error reading train_data_batch_1: {e}')
    sys.exit(1)
" 2>/dev/null; then
        echo "  ✅ File format verification passed"
    else
        echo "  ❌ File format verification failed"
    fi
else
    echo "  ⚠️  Cannot test - no training files found"
fi

# Check processed trajectory data
echo ""
echo "🔍 Checking processed trajectory data..."
TRAJ_DIR="$DATA_DIR/imagenet64_trajectories"
if [ -d "$TRAJ_DIR" ]; then
    echo "  📁 Trajectory directory exists"
    
    if [ -d "$TRAJ_DIR/lmdb" ]; then
        echo "  ✅ LMDB database exists"
    else
        echo "  ❌ LMDB database missing"
    fi
    
    if [ -f "$TRAJ_DIR/labels.npy" ]; then
        echo "  ✅ Labels file exists"
    else
        echo "  ❌ Labels file missing"
    fi
else
    echo "  ❌ Trajectory directory does not exist"
    echo "  💡 Run './scripts/setup_imagenet64.sh' to process the data"
fi

echo ""
echo "=== Verification Complete ==="

# Provide recommendations
echo ""
echo "📋 Next Steps:"

if [ $TRAIN_COUNT -eq 0 ]; then
    echo "  1. ❌ No training data found. Please ensure ImageNet64 files are in ./data/"
    echo "     Expected files: train_data_batch_1, train_data_batch_2, ..., train_data_batch_10"
elif [ $TRAIN_COUNT -lt 10 ]; then
    echo "  1. ⚠️  Only $TRAIN_COUNT/10 training files found. Training will work but with less data."
else
    echo "  1. ✅ All training files present"
fi

if [ -f "$VAL_FILE" ]; then
    echo "  2. ✅ Validation data present"
else
    echo "  2. ⚠️  Validation data missing (optional for training)"
fi

if [ -d "$TRAJ_DIR/lmdb" ] && [ -f "$TRAJ_DIR/labels.npy" ]; then
    echo "  3. ✅ Processed trajectory data ready - you can start training!"
    echo "     Run: ./scripts/train_imagenet64.sh"
else
    echo "  3. 🔄 Need to process data into trajectories"
    echo "     Run: ./scripts/setup_imagenet64.sh"
fi

echo ""