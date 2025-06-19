#!/usr/bin/env python3
"""
Process ImageNet64 data to create trajectory data for DSNO training.
This script generates denoising trajectories from clean ImageNet64 images.
"""

import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import lmdb
from PIL import Image
import argparse

def unpickle(file):
    """Unpickle ImageNet64 data files."""
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def load_imagenet64_batch(batch_file):
    """Load a single ImageNet64 batch file."""
    data_dict = unpickle(batch_file)
    
    # Extract images and labels
    images = data_dict['data']  # Shape: (N, 3072) for 64x64x3 images
    labels = np.array(data_dict['labels']) - 1  # Convert to 0-based indexing
    
    # Reshape images from (N, 3072) to (N, 3, 64, 64)
    images = images.reshape(-1, 3, 64, 64)
    
    # Convert to float32 and normalize to [-1, 1]
    images = images.astype(np.float32) / 255.0
    images = images * 2.0 - 1.0
    
    return images, labels

def get_logsnr_schedule(logsnr_max=20.0, logsnr_min=-20.0):
    """Create log SNR schedule for trajectory generation."""
    b = np.arctan(np.exp(-0.5 * logsnr_max))
    a = np.arctan(np.exp(-0.5 * logsnr_min)) - b
    
    def get_logsnr(t):
        return -2.0 * torch.log(torch.tan(a * t + b))
    
    return get_logsnr

def generate_trajectory(clean_image, timesteps, logsnr_fn, device):
    """Generate a denoising trajectory for a single image."""
    
    # Convert numpy array to tensor with proper error handling
    try:
        if isinstance(clean_image, np.ndarray):
            clean_image = torch.from_numpy(clean_image.copy())
        clean_image = clean_image.to(device)
    except Exception as e:
        print(f"Error converting image to tensor: {e}")
        # Fallback: use CPU if GPU conversion fails
        if isinstance(clean_image, np.ndarray):
            clean_image = torch.from_numpy(clean_image.copy())
        clean_image = clean_image.to('cpu')
        device = torch.device('cpu')
        print("Falling back to CPU processing...")
    
    trajectory = []
    
    for t in timesteps:
        try:
            t_tensor = torch.tensor([t], device=device)
            logsnr = logsnr_fn(t_tensor)
            
            # Convert log SNR to alpha and sigma
            snr = torch.exp(logsnr)
            alpha = torch.sqrt(snr / (1 + snr))
            sigma = torch.sqrt(1 / (1 + snr))
            
            # Generate noise
            noise = torch.randn_like(clean_image)
            
            # Create noisy image: x_t = alpha * x_0 + sigma * noise
            noisy_image = alpha * clean_image + sigma * noise
            
            trajectory.append(noisy_image.cpu().numpy())
        except Exception as e:
            print(f"Error generating trajectory at timestep {t}: {e}")
            # Return what we have so far or create a dummy trajectory
            if len(trajectory) == 0:
                # Create a dummy trajectory if nothing worked
                dummy_trajectory = np.tile(clean_image.cpu().numpy()[None, ...], (len(timesteps), 1, 1, 1))
                return dummy_trajectory
            break
    
    return np.stack(trajectory, axis=0)  # Shape: (T, C, H, W)

def process_imagenet64_data(data_dir, output_dir, num_timesteps=9, batch_size=100):
    """Process ImageNet64 data to create trajectory dataset."""
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if processing already completed
    lmdb_path = os.path.join(output_dir, 'lmdb')
    labels_path = os.path.join(output_dir, 'labels.npy')
    
    if os.path.exists(lmdb_path) and os.path.exists(labels_path):
        print(f"Processed data already exists at {output_dir}")
        print("Delete the output directory if you want to reprocess the data.")
        return
    
    # Setup device - with fallback to CPU for RTX 5090 compatibility issues
    if torch.cuda.is_available():
        try:
            # Test if CUDA operations work
            test_tensor = torch.randn(10, 10).cuda()
            _ = test_tensor * 2  # Simple operation test
            device = torch.device('cuda')
            print(f"Using device: cuda")
        except Exception as e:
            print(f"CUDA available but operations failing: {e}")
            print("Falling back to CPU...")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
        print(f"Using device: cpu")
    
    # Create log SNR schedule
    logsnr_fn = get_logsnr_schedule(logsnr_max=20.0, logsnr_min=-20.0)
    
    # Create timesteps (uniform spacing)
    t0, t1 = 1.0, 0.0
    timesteps = np.linspace(t0, t1, num_timesteps)
    
    print(f"Timesteps: {timesteps}")
    print(f"Number of timesteps: {num_timesteps}")
    
    # Process training data - check for existing batch files
    train_files = []
    missing_files = []
    
    for i in range(1, 11):  # train_data_batch_1 to train_data_batch_10
        batch_file = os.path.join(data_dir, f'train_data_batch_{i}')
        if os.path.exists(batch_file):
            train_files.append(batch_file)
        else:
            missing_files.append(f'train_data_batch_{i}')
    
    print(f"Found {len(train_files)} training batch files")
    if missing_files:
        print(f"Missing files: {missing_files}")
        print("Continuing with available files...")
    
    # Create LMDB database for training data
    train_lmdb_path = os.path.join(output_dir, 'lmdb')
    train_env = lmdb.open(train_lmdb_path, map_size=200 * 1024**3*5)  # 1000GB
    
    all_labels = []
    total_samples = 0
    
    with train_env.begin(write=True) as txn:
        for batch_file in tqdm(train_files, desc="Processing training batches"):
            images, labels = load_imagenet64_batch(batch_file)
            all_labels.extend(labels)
            
            print(f"Processing {len(images)} images from {batch_file}")
            
            # Process images in smaller batches to save memory
            for i in tqdm(range(0, len(images), batch_size), desc="Generating trajectories", leave=False):
                batch_images = images[i:i+batch_size]
                
                for j, img in enumerate(batch_images):
                    trajectory = generate_trajectory(img, timesteps, logsnr_fn, device)
                    
                    # Store trajectory in LMDB
                    key = f'{total_samples}'.encode()
                    value = trajectory.astype(np.float32).tobytes()
                    txn.put(key, value)
                    
                    total_samples += 1
        
        # Store dataset length
        txn.put('length'.encode(), str(total_samples).encode())
    
    train_env.close()
    
    # Save labels
    labels_path = os.path.join(output_dir, 'labels.npy')
    np.save(labels_path, np.array(all_labels))
    
    print(f"Processed {total_samples} training samples")
    print(f"Saved to {output_dir}")
    print(f"Labels saved to {labels_path}")
    
    # Process validation data if it exists
    val_file = os.path.join(data_dir, 'val_data')
    if os.path.exists(val_file):
        print("Processing validation data...")
        val_images, val_labels = load_imagenet64_batch(val_file)
        
        val_output_dir = os.path.join(output_dir, 'val')
        os.makedirs(val_output_dir, exist_ok=True)
        
        val_lmdb_path = os.path.join(val_output_dir, 'lmdb')
        val_env = lmdb.open(val_lmdb_path, map_size=50 * 1024**3*5)  # 250GB
        
        with val_env.begin(write=True) as txn:
            for i, img in enumerate(tqdm(val_images, desc="Processing validation images")):
                trajectory = generate_trajectory(img, timesteps, logsnr_fn, device)
                
                key = f'{i}'.encode()
                value = trajectory.astype(np.float32).tobytes()
                txn.put(key, value)
            
            txn.put('length'.encode(), str(len(val_images)).encode())
        
        val_env.close()
        
        # Save validation labels
        val_labels_path = os.path.join(val_output_dir, 'labels.npy')
        np.save(val_labels_path, val_labels)
        
        print(f"Processed {len(val_images)} validation samples")

def main():
    parser = argparse.ArgumentParser(description='Process ImageNet64 data for DSNO')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing ImageNet64 data files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for processed trajectory data')
    parser.add_argument('--num_timesteps', type=int, default=9,
                        help='Number of timesteps in trajectory')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size for processing')
    
    args = parser.parse_args()
    
    process_imagenet64_data(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_timesteps=args.num_timesteps,
        batch_size=args.batch_size
    )

if __name__ == '__main__':
    main()