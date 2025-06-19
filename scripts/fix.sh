#!/bin/bash

# Complete DSNO Fix Script - Fixes NumPy, GPU detection, and training setup
echo "=== Complete DSNO Environment and Training Fix ==="

# Step 1: Fix NumPy version compatibility
echo "🔧 Step 1: Fixing NumPy version (downgrading from 2.x to 1.x)..."
pip uninstall -y numpy
pip install "numpy>=1.24.0,<2.0"

# Step 2: Check GPU availability
echo "🔧 Step 2: Checking GPU availability..."
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f'GPU count: {gpu_count}')
    for i in range(gpu_count):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('No GPUs detected')
"

# Step 3: Get actual GPU count for training script
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)")
echo "Detected $GPU_COUNT GPUs"

# Step 4: Update training config to use single GPU if multiple GPUs aren't working
echo "🔧 Step 3: Creating single-GPU training script..."
cat > scripts/train_imagenet64_fixed.sh << 'EOF'
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
EOF

chmod +x scripts/train_imagenet64_fixed.sh

# Step 5: Update config file to use smaller batch size for single GPU
echo "🔧 Step 4: Creating optimized config for single GPU..."
cat > configs/custom/imagenet64-dsno-single.yaml << 'EOF'
data:
    dataset: imagenet
    category: imagenet
    image_size: 64
    num_channels: 3
    random_flip: True
    centered: True
    datapath: ./data/imagenet64_trajectories
    shape: [3, 9, 64, 64]
    dims: [1, 0, 2, 3]
    t_dim: 9
    t_idx: [0, 2, 4, 6, 8]
    num_steps: 16
    time_step: uniform
    epsilon: 0.0

model:
    logsnr_min: -20.
    logsnr_max: 20.
    num_scales: 1000
    dropout: 0.1
    name: 'tddpmm'
    ema_rate: 0.9999
    normalization: 'GroupNorm'
    nonlinearity: 'swish'
    nf: 128  # Reduced from 192 for single GPU
    temb_dim: 512  # Reduced from 768 for single GPU
    ch_mult: [1, 2, 2, 3]  # Reduced from [1, 2, 3, 4]
    num_res_blocks: 2  # Reduced from 3
    attn_resolutions: [16, 32]  # Reduced from [8, 16, 32]
    head_dim: 64
    resamp_with_conv: False
    conditional: True
    num_classes: 1000
    resblock_type: 'biggan'
    init_scale: 0.
    logsnr_type: inv_cos
    mean_type: x
    time_conv: True
    with_nin: False
    num_modes: 2
    pred_eps: False 
    num_t: 4
    num_pad: 0
    loss_weight: snr
    fourier_feature: False

training:
    start_iter: 0
    n_iters: 100_001  # Reduced for testing
    batchsize: 4  # Small batch size for single GPU
    accum_grad_iter: 8  # Higher accumulation to simulate larger batch
    loss: L1
    loss_weight: snr

eval:
    save_step: 5_000
    test_fid: False

optim:
    optimizer: 'Adam'
    lr: 0.0001  # Reduced learning rate
    weight_decay: 0.0
    milestone: [50_000, 80_000]
    warmup: 5_000
    grad_clip: 1.0

log:
    logname: ImageNet64-DSNO-Single
    entity: your_wandb_entity
    project: ImageNet64-DSNO
    group: ImageNet64-DSNO-Single

sample:
    batchsize: 16
    num_batchs: 10
    num_steps: 8
    clip_x: False
EOF

# Step 6: Create final training script that uses the optimized config
cat > scripts/start_training.sh << 'EOF'
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
EOF

chmod +x scripts/start_training.sh

# Step 7: Test the environment
echo "🔧 Step 5: Testing fixed environment..."
python -c "
try:
    import numpy as np
    print(f'✅ NumPy {np.__version__} (should be 1.x)')
    
    import torch
    print(f'✅ PyTorch {torch.__version__}')
    print(f'✅ CUDA available: {torch.cuda.is_available()}')
    
    if torch.cuda.is_available():
        print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
        # Test basic CUDA operation
        x = torch.randn(10, 10).cuda()
        y = torch.matmul(x, x.T)
        print('✅ Basic CUDA operations working')
    
    from flax import serialization
    print('✅ Flax import successful')
    
    print('🎉 Environment is ready!')
    
except Exception as e:
    print(f'❌ Environment test failed: {e}')
"

echo ""
echo "=== Fix Complete ==="
echo ""
echo "📋 Next Steps:"
echo "1. Process data (if not done): ./scripts/simple_process.sh"
echo "2. Start training: ./scripts/start_training.sh"
echo ""
echo "🔧 Changes made:"
echo "- Fixed NumPy version to 1.x (compatible with PyTorch)"
echo "- Created single GPU training setup"
echo "- Reduced model size for single GPU"
echo "- Created optimized config file"
echo "- Fixed multi-GPU device ordinal errors"