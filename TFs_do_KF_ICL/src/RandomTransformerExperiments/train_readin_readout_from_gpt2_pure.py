"""
Train BOTH read-in and read-out layers from GPT2 Pure checkpoint.
Backbone is frozen from GPT2 Pure checkpoint (step%3D99000.ckpt).

Read-in layer initialization (same as train_readin_from_gpt2_pure.py):
- W_scaled = W_random * (||W_pure||_F / ||W_random||_F)  [scaled to match pure norm]
- b_scaled = b_random * (||b_pure||_2 / ||b_random||_2)
- Random init loaded from Experiment1/initial_random_model.ckpt

Read-out layer initialization:
- Initialize each entry as N(0, sigma) where sigma = (1/sqrt(128)) * sqrt(5/128) * sqrt(2) / 2
- Then scale: W_readout_scaled = W_readout_random * (||W_pure_readout||_F / ||W_random_readout||_F)
- Same for bias

Key features:
- Loads GPT2 Pure checkpoint (backbone only)
- Read-in layer: Scaled random init from Experiment1
- Read-out layer: Scaled random init with custom std
- Freezes backbone, trains read-in + read-out
- Adam lr = 10^-4 * sqrt(128/57)
- Uses same batch structure: 20 batches, 10000 traces, 128 gradient accumulation
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import math
from core import Config
from models import GPT2
from tqdm import tqdm
import time
import random
import re

# Import functions from GPT2 Modified training script
from inputOutputLayerGPT2ModifiedAdamBothPretraining import (
    generate_pretraining_traces,
    load_pretraining_batches,
)

def compute_single_trace_mse_with_dtype(model, trace, device, model_dtype):
    """
    Compute MSE for a single trace with proper dtype handling.
    trace: numpy array [seq_len, 57]
    """
    # Convert to appropriate dtype based on device
    if model_dtype == torch.float64:
        trace_tensor = torch.from_numpy(trace).double().to(device).unsqueeze(0)  # [1, seq_len, 57]
    else:
        trace_tensor = torch.from_numpy(trace).float().to(device).unsqueeze(0)  # [1, seq_len, 57]
    
    # Forward pass
    embeds = model._read_in(trace_tensor)
    hidden = model._backbone(inputs_embeds=embeds).last_hidden_state
    preds = model._read_out(hidden)  # [1, seq_len, 5]
    
    # Compute loss at valid positions
    targets = trace_tensor[:, 1:, -5:]  # [1, seq_len-1, 5]
    preds_shifted = preds[:, :-1, :]    # [1, seq_len-1, 5]
    mask = trace_tensor[:, 1:, 51] != 0  # [1, seq_len-1]
    
    if mask.sum() == 0:
        return torch.tensor(0.0, device=device, dtype=model_dtype, requires_grad=True)
    
    # MSE over valid positions
    diff = (preds_shifted - targets) ** 2  # [1, seq_len-1, 5]
    diff_sum = diff.sum(dim=-1)  # [1, seq_len-1]
    mask_dtype = mask.type_as(diff_sum)
    mse = (diff_sum * mask_dtype).sum() / mask_dtype.sum()
    
    return mse

def train_epoch(model, optimizer, samples, batch_size, device, model_dtype):
    """
    Train read-in and read-out layers for one epoch (20 batches).
    samples: list of traces (different lengths allowed)
    Processes traces individually, accumulates gradients over 128 traces before stepping.
    Returns single epoch MSE calculated on all training data at the end.
    """
    model.train()
    
    # Ensure model is on correct device and dtype
    model.to(device)
    if model_dtype == torch.float64:
        model.double()
    else:
        model.float()
    
    num_samples = len(samples)
    
    # Shuffle samples
    shuffled_samples = samples.copy()
    np.random.shuffle(shuffled_samples)
    
    # Split into 20 batches
    num_batches = 20
    samples_per_batch = num_samples // num_batches
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * samples_per_batch
        batch_end = batch_start + samples_per_batch
        batch_samples = shuffled_samples[batch_start:batch_end]
        
        # Process in gradient accumulation steps of 128 traces
        gradient_accum_size = 128
        
        for step_start in range(0, len(batch_samples), gradient_accum_size):
            step_end = min(step_start + gradient_accum_size, len(batch_samples))
            step_traces = batch_samples[step_start:step_end]
            
            optimizer.zero_grad()
            
            # Accumulate gradients over individual traces
            for trace in step_traces:
                mse = compute_single_trace_mse_with_dtype(model, trace, device, model_dtype)
                scaled_mse = mse / float(len(step_traces))
                scaled_mse.backward()  # Scale gradient by 1/num_traces
            
            optimizer.step()
        
        print(f"    Batch {batch_idx + 1}/20 completed")
    
    # Calculate training MSE on all samples at the end of the epoch
    model.eval()
    total_mse = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for trace in shuffled_samples:
            mse = compute_single_trace_mse_with_dtype(model, trace, device, model_dtype)
            total_mse += mse.item()
            total_samples += 1
    
    epoch_mse = total_mse / total_samples if total_samples > 0 else 0.0
    
    return epoch_mse

def format_test_results_for_file(test_results):
    """Format test results for writing to file"""
    lines = []
    for key in sorted(test_results.keys()):
        lines.append(f"    {key}: {test_results[key]:.6f}")
    return "\n".join(lines)

def write_results_to_file(filepath, content):
    """Append content to results file"""
    with open(filepath, 'a') as f:
        f.write(content)
        f.flush()

def initialize_results_file(filepath):
    """Initialize results file with header"""
    with open(filepath, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TRAIN READ-IN + READ-OUT LAYERS FROM GPT2 PURE CHECKPOINT\n")
        f.write("="*80 + "\n")
        f.write("Configuration:\n")
        f.write("  - Loaded GPT2 Pure checkpoint (step%3D99000.ckpt) for backbone\n")
        f.write("  - Read-in layer: Scaled random init from Experiment1\n")
        f.write("    W_scaled = W_random * (||W_pure||_F / ||W_random||_F)\n")
        f.write("  - Read-out layer: Random init with std = (1/sqrt(128)) * sqrt(5/128) * sqrt(2) / 2\n")
        f.write("    then scaled to match ||W_pure_readout||_F\n")
        f.write("  - Frozen: Backbone only\n")
        f.write("  - Training: Read-in + Read-out layers\n")
        f.write("  - Optimizer: Adam with lr = 1e-4 * sqrt(128/57)\n")
        f.write("  - Batch structure: 20 batches, 10000 traces, 128 gradient accumulation\n")
        f.write("="*80 + "\n\n")
        f.flush()

def save_checkpoint(model, optimizer, epoch, mse, output_dir, iteration=None):
    """Save checkpoint"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'mse': mse
    }
    
    if iteration == 0:
        filename = 'initial_random_model.ckpt'
    elif iteration is not None:
        filename = f'iteration_{iteration:06d}.ckpt'
    else:
        filename = f'epoch_{epoch:06d}.ckpt'
    
    checkpoint_path = os.path.join(output_dir, filename) 
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path

def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint in the output directory"""
    if not os.path.exists(output_dir):
        return None, 0
    
    checkpoints = []
    for filename in os.listdir(output_dir):
        if filename.endswith('.ckpt'):
            if filename == 'initial_random_model.ckpt':
                checkpoints.append((0, os.path.join(output_dir, filename)))
            elif filename.startswith('iteration_'):
                try:
                    iteration = int(filename.replace('iteration_', '').replace('.ckpt', ''))
                    checkpoints.append((iteration, os.path.join(output_dir, filename)))
                except ValueError:
                    continue
    
    if not checkpoints:
        return None, 0
    
    # Sort by iteration number and return the latest
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    latest_iteration, latest_path = checkpoints[0]
    return latest_path, latest_iteration

def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load model and optimizer from checkpoint
    
    Loads both read-in and read-out layer weights from checkpoint.
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract state dict
    if 'model_state_dict' in checkpoint:
        checkpoint_state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        checkpoint_state_dict = checkpoint['state_dict']
    else:
        checkpoint_state_dict = checkpoint
    
    # Only load read-in and read-out layer weights (backbone stays from GPT2 Pure)
    io_state_dict = {}
    for key, value in checkpoint_state_dict.items():
        if key.startswith('_read_in') or key.startswith('_read_out'):
            io_state_dict[key] = value
    
    # Load read-in and read-out layers
    if io_state_dict:
        model.load_state_dict(io_state_dict, strict=False)
        print(f"  Loaded read-in and read-out layers from checkpoint ({len(io_state_dict)} parameters)")
    
    # Load optimizer state if available
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"  Loaded optimizer state")
    else:
        print(f"  No optimizer state found in checkpoint")
    
    # Get epoch number
    epoch = checkpoint.get('epoch', 0)
    mse = checkpoint.get('mse', 0.0)
    
    # Move model to device
    model.to(device)
    
    return epoch, mse

def main():
    # Configuration
    config = Config()
    config.override("model_type", "GPT2")
    
    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA (NVIDIA GPU) with float64 - GPU ACCELERATED")
        model_dtype = torch.float64
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple GPU) with float32 - GPU ACCELERATED")
        model_dtype = torch.float32
    else:
        device = torch.device("cpu")
        print("Using device: CPU with float64")
        model_dtype = torch.float64
    
    # Training parameters
    num_pretraining_traces = 10000
    num_pretraining_batches = 20
    # Learning rate: 10^-4 * sqrt(128/57)
    learning_rate = 1e-4 * math.sqrt(128 / 57)
    max_epochs = 10000
    checkpoint_interval = 25  # Save checkpoint every 25 epochs
    
    # Read-out initialization std: (1/sqrt(128)) * sqrt(5/128) * sqrt(2) / 2
    readout_init_std = (1.0 / math.sqrt(128)) * math.sqrt(5.0 / 128) * math.sqrt(2) / 2
    
    # Output directory
    output_dir = os.path.join(
        os.path.dirname(__file__),
        "gpt2_pure_readin_readout_training_results"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    results_file = os.path.join(output_dir, 'training_results.txt')
    
    print("\n" + "="*80)
    print("TRAIN READ-IN + READ-OUT LAYERS FROM GPT2 PURE CHECKPOINT")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Learning rate: {learning_rate}")
    print(f"Max epochs: {max_epochs}")
    print(f"Pretraining traces per epoch: {num_pretraining_traces}")
    print(f"Pretraining batches per epoch: {num_pretraining_batches}")
    print(f"Read-out init std: {readout_init_std:.6f} (= (1/sqrt(128)) * sqrt(5/128) * sqrt(2) / 2)")
    print("="*80 + "\n")
    
    # Check for existing checkpoints first
    latest_checkpoint, latest_iteration = find_latest_checkpoint(output_dir)
    start_epoch = 0
    
    if latest_checkpoint is not None:
        print(f"\nFound existing checkpoint at iteration {latest_iteration}")
        print(f"Will resume from: {latest_checkpoint}")
    else:
        # Only initialize results file if starting fresh
        initialize_results_file(results_file)
    
    # =========================================================================
    # LOAD GPT2 PURE CHECKPOINT
    # =========================================================================
    
    print("Loading GPT2 Pure checkpoint...")
    gpt2_checkpoint_path = os.path.join(
        os.path.dirname(__file__),
        "Set-up Data",
        "step%3D99000.ckpt"
    )
    
    if not os.path.exists(gpt2_checkpoint_path):
        raise FileNotFoundError(f"GPT2 Pure checkpoint not found: {gpt2_checkpoint_path}")
    
    print(f"Loading from: {gpt2_checkpoint_path}")
    gpt2_checkpoint = torch.load(gpt2_checkpoint_path, map_location='cpu')
    
    # Create model
    model = GPT2(
        n_dims_in=config.n_dims_in,
        n_positions=config.n_positions,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_dims_out=config.n_dims_out
    )
    
    # Extract state dict (handle different checkpoint formats)
    if 'model_state_dict' in gpt2_checkpoint:
        gpt2_state_dict = gpt2_checkpoint['model_state_dict']
    elif 'state_dict' in gpt2_checkpoint:
        gpt2_state_dict = gpt2_checkpoint['state_dict']
    else:
        gpt2_state_dict = gpt2_checkpoint
    
    # Load ONLY backbone from GPT2 checkpoint (exclude read-in AND read-out)
    backbone_state_dict = {}
    for key, value in gpt2_state_dict.items():
        if not key.startswith('_read_in') and not key.startswith('_read_out'):
            backbone_state_dict[key] = value
    
    # Load backbone layers from GPT2 checkpoint
    model.load_state_dict(backbone_state_dict, strict=False)
    print(f"  Loaded {len(backbone_state_dict)} layers from GPT2 Pure checkpoint (backbone only)")
    
    # =========================================================================
    # INITIALIZE READ-IN LAYER (same as train_readin_from_gpt2_pure.py)
    # =========================================================================
    
    if latest_checkpoint is None:
        # Load random init from Experiment1
        random_ckpt_path = os.path.join(
            os.path.dirname(__file__),
            "gpt2_pure_readin_training_results",
            "Experiment1",
            "initial_random_model.ckpt"
        )
        print(f"\n  Loading random init for read-in from: {random_ckpt_path}")
        random_ckpt = torch.load(random_ckpt_path, map_location='cpu')
        random_state = random_ckpt.get('model_state_dict', random_ckpt.get('state_dict', random_ckpt))
        W_random_readin = random_state['_read_in.weight']
        b_random_readin = random_state['_read_in.bias']
        
        # Get GPT2 Pure read-in norms for scaling
        W_pure_readin = gpt2_state_dict['_read_in.weight']
        b_pure_readin = gpt2_state_dict['_read_in.bias']
        W_pure_readin_norm = torch.norm(W_pure_readin, p='fro').item()
        W_random_readin_norm = torch.norm(W_random_readin, p='fro').item()
        b_pure_readin_norm = torch.norm(b_pure_readin, p=2).item()
        b_random_readin_norm = torch.norm(b_random_readin, p=2).item()
        
        # Compute scaling factors to match pure norms
        scale_W_readin = W_pure_readin_norm / W_random_readin_norm
        scale_b_readin = b_pure_readin_norm / b_random_readin_norm
        
        # Scale the random init: W_scaled = W_random * scale_W (to match pure norm)
        W_scaled_readin = W_random_readin * scale_W_readin
        b_scaled_readin = b_random_readin * scale_b_readin
        
        # Set the scaled weights
        with torch.no_grad():
            model._read_in.weight.copy_(W_scaled_readin)
            model._read_in.bias.copy_(b_scaled_readin)
        
        print(f"  Read-in initialized with scaled random init from Experiment1:")
        print(f"    W_scaled = W_random * ({W_pure_readin_norm:.4f}/{W_random_readin_norm:.4f}) = W_random * {scale_W_readin:.6f}")
        print(f"    b_scaled = b_random * ({b_pure_readin_norm:.4f}/{b_random_readin_norm:.4f}) = b_random * {scale_b_readin:.6f}")
        print(f"    W_scaled norm: {torch.norm(W_scaled_readin, p='fro').item():.6f} (target: {W_pure_readin_norm:.6f})")
        print(f"    b_scaled norm: {torch.norm(b_scaled_readin, p=2).item():.6f} (target: {b_pure_readin_norm:.6f})")
        
        # =========================================================================
        # INITIALIZE READ-OUT LAYER (custom initialization)
        # =========================================================================
        
        print(f"\n  Initializing read-out layer...")
        print(f"    Random init std: {readout_init_std:.6f} (= (1/sqrt(128)) * sqrt(5/128) * sqrt(2) / 2)")
        
        # Initialize read-out with custom std
        W_random_readout = torch.randn_like(model._read_out.weight) * readout_init_std
        b_random_readout = torch.randn_like(model._read_out.bias) * readout_init_std
        
        # Get GPT2 Pure read-out norms for scaling
        W_pure_readout = gpt2_state_dict['_read_out.weight']
        b_pure_readout = gpt2_state_dict['_read_out.bias']
        W_pure_readout_norm = torch.norm(W_pure_readout, p='fro').item()
        W_random_readout_norm = torch.norm(W_random_readout, p='fro').item()
        b_pure_readout_norm = torch.norm(b_pure_readout, p=2).item()
        b_random_readout_norm = torch.norm(b_random_readout, p=2).item()
        
        # Compute scaling factors to match pure norms
        scale_W_readout = W_pure_readout_norm / W_random_readout_norm
        scale_b_readout = b_pure_readout_norm / b_random_readout_norm
        
        # Scale the random init: W_scaled = W_random * scale_W (to match pure norm)
        W_scaled_readout = W_random_readout * scale_W_readout
        b_scaled_readout = b_random_readout * scale_b_readout
        
        # Set the scaled weights
        with torch.no_grad():
            model._read_out.weight.copy_(W_scaled_readout)
            model._read_out.bias.copy_(b_scaled_readout)
        
        print(f"  Read-out initialized with scaled random init:")
        print(f"    W_random_readout norm (before scaling): {W_random_readout_norm:.6f}")
        print(f"    W_scaled = W_random * ({W_pure_readout_norm:.4f}/{W_random_readout_norm:.4f}) = W_random * {scale_W_readout:.6f}")
        print(f"    b_scaled = b_random * ({b_pure_readout_norm:.4f}/{b_random_readout_norm:.4f}) = b_random * {scale_b_readout:.6f}")
        print(f"    W_scaled norm: {torch.norm(W_scaled_readout, p='fro').item():.6f} (target: {W_pure_readout_norm:.6f})")
        print(f"    b_scaled norm: {torch.norm(b_scaled_readout, p=2).item():.6f} (target: {b_pure_readout_norm:.6f})")
    
    # If resuming, read-in and read-out layers will be loaded from checkpoint later
    
    # Move model to device and set dtype
    if model_dtype == torch.float64:
        model.double().to(device)
    else:
        model.float().to(device)
    
    # =========================================================================
    # FREEZE BACKBONE, TRAIN READ-IN + READ-OUT
    # =========================================================================
    
    print("\nFreezing backbone, training read-in + read-out layers...")
    for name, param in model.named_parameters():
        if name.startswith('_read_in') or name.startswith('_read_out'):
            param.requires_grad = True
            print(f"  Training: {name}")
        else:
            param.requires_grad = False
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    # =========================================================================
    # SETUP OPTIMIZER
    # =========================================================================
    
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    print(f"\nOptimizer: Adam with lr={learning_rate:.6e} (= 1e-4 * sqrt(128/57))")
    
    # =========================================================================
    # LOAD EXISTING CHECKPOINT OR SAVE INITIAL CHECKPOINT
    # =========================================================================
    
    if latest_checkpoint is not None:
        # Resume from existing checkpoint
        print(f"\nResuming from checkpoint: {latest_checkpoint}")
        start_epoch, _ = load_checkpoint(model, optimizer, latest_checkpoint, device)
        print(f"  Resuming from epoch {start_epoch}")
        
        # Ensure model is in correct dtype
        if model_dtype == torch.float64:
            model.double()
        else:
            model.float()
    else:
        # Save initial checkpoint
        print("\nSaving initial checkpoint...")
        initial_checkpoint_path = save_checkpoint(model, optimizer, 0, 0.0, output_dir, iteration=0)
        print(f"  Initial checkpoint saved: {initial_checkpoint_path}")
    
    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    
    print("\n" + "="*80)
    print("TRAINING LOOP")
    print("="*80)
    print(f"Training for {max_epochs} epochs...")
    print(f"Checkpoint interval: Every {checkpoint_interval} epochs\n")
    
    best_mse = float('inf')
    best_epoch = start_epoch
    
    # If resuming, try to load best MSE from results file
    if start_epoch > 0:
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                content = f.read()
                mse_matches = re.findall(r'Training MSE: ([\d.]+)', content)
                if mse_matches:
                    try:
                        best_mse = min(float(m) for m in mse_matches)
                        print(f"  Loaded best MSE from results file: {best_mse:.6f}")
                    except:
                        pass
    
    for epoch in range(start_epoch, max_epochs):
        epoch_start_time = time.time()
        
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{max_epochs}")
        print(f"{'='*80}")
        
        # Generate fresh pretraining batches for this epoch
        random_seed = epoch  # Different seed for each epoch
        batches = load_pretraining_batches(
            num_traces=num_pretraining_traces,
            num_batches=num_pretraining_batches,
            batch_idx=None,
            random_seed=random_seed,
            cache_suffix="_readin_readout"
        )
        
        # Flatten batches into single list of traces
        all_traces = []
        for batch in batches:
            all_traces.extend(batch)
        
        print(f"  Generated {len(all_traces)} pretraining traces")
        
        # Train for one epoch
        print("  Training...")
        epoch_mse = train_epoch(model, optimizer, all_traces, batch_size=128, device=device, model_dtype=model_dtype)
        
        epoch_time = time.time() - epoch_start_time
        
        print(f"  Epoch {epoch + 1} Training MSE: {epoch_mse:.6f} (Time: {epoch_time:.2f}s)")
        
        # Write to results file
        content = f"Epoch {epoch + 1}:\n"
        content += f"  Training MSE: {epoch_mse:.6f}\n"
        content += f"  Time: {epoch_time:.2f}s\n"
        content += "\n"
        write_results_to_file(results_file, content)
        
        # Track best model
        if epoch_mse < best_mse:
            best_mse = epoch_mse
            best_epoch = epoch + 1
            print(f"  *** New best training MSE: {best_mse:.6f} (Epoch {best_epoch}) ***")
        
        # Save checkpoint
        if (epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == max_epochs:
            checkpoint_path = save_checkpoint(model, optimizer, epoch + 1, epoch_mse, output_dir, iteration=epoch + 1)
            print(f"  Checkpoint saved: {checkpoint_path}")
        
        sys.stdout.flush()
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Total epochs: {max_epochs}")
    print(f"Best training MSE: {best_mse:.6f} (Epoch {best_epoch})")
    print(f"Results saved to: {results_file}")
    print(f"Checkpoints saved to: {output_dir}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
