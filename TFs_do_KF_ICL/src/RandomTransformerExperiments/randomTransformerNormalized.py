"""
Train read-in and read-out layers with a FULLY RANDOM backbone.
Backbone is initialized randomly (not from ckpt9900) using normalized init.

Initialization (per parameter):
- For each matrix/bias: get corresponding param from stepckpt9900
- F = squared Frobenius norm (matrices) or squared L2 norm (biases/vectors)
- n = number of elements
- Initialize each element as N(0, F/n) i.e. variance F/n, std = sqrt(F/n)
- Exception: LayerNorm weights (ln_1, ln_2, ln_f) are initialized to 1 + N(0, epsilon) with epsilon=1e-5

Training: Same as train_readin_readout_from_gpt2_pure.py
- Freeze backbone (middle layers), train read-in + read-out only
- Same batch structure, optimizer, etc.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import math
import time
import random
import re

from core import Config
from models import GPT2
from inputOutputLayerGPT2ModifiedAdamBothPretraining import (
    load_pretraining_batches,
)


def init_param_from_pure_norm(param, pure_param):
    """
    Initialize param so each element is N(0, F/n) where:
    - F = squared Frobenius norm of pure_param (for matrices) or squared L2 norm (for vectors)
    - n = number of elements
    - variance = F/n, so std = sqrt(F/n)
    """
    n = pure_param.numel()
    if n == 0:
        return

    if pure_param.dim() >= 2:
        F = (torch.norm(pure_param, p='fro').item()) ** 2
    else:
        F = (torch.norm(pure_param, p=2).item()) ** 2

    std = (F / n) ** 0.5
    with torch.no_grad():
        param.copy_(torch.randn_like(param) * std)


LAYERNORM_WEIGHT_EPSILON = 1e-5


def _is_layernorm_weight(key):
    """LayerNorm scale params: ln_1.weight, ln_2.weight, ln_f.weight"""
    return key.endswith("ln_1.weight") or key.endswith("ln_2.weight") or key.endswith("ln_f.weight")


def initialize_model_from_pure_norms(model, pure_state_dict):
    """Initialize all model parameters using N(0, F/n) where F from ckpt9900.
    LayerNorm weights are initialized to 1 + epsilon instead."""
    model_state = model.state_dict()
    for key in model_state.keys():
        if key in pure_state_dict:
            pure_val = pure_state_dict[key]
            if pure_val.shape != model_state[key].shape:
                print(f"  Skip {key}: shape mismatch {pure_val.shape} vs {model_state[key].shape}")
                continue
            if _is_layernorm_weight(key):
                with torch.no_grad():
                    model_state[key].copy_(1.0 + torch.randn_like(model_state[key]) * LAYERNORM_WEIGHT_EPSILON)
                print(f"  {key}: LayerNorm weight -> 1 + N(0, {LAYERNORM_WEIGHT_EPSILON})")
            else:
                init_param_from_pure_norm(model_state[key], pure_val)
                n = pure_val.numel()
                F = (torch.norm(pure_val, p='fro').item()) ** 2 if pure_val.dim() >= 2 else (torch.norm(pure_val, p=2).item()) ** 2
                print(f"  {key}: F={F:.4f}, n={n}, std=sqrt(F/n)={(F/n)**0.5:.6e}")
        else:
            print(f"  Skip {key}: not in pure checkpoint")


def compute_single_trace_mse_with_dtype(model, trace, device, model_dtype):
    """Compute MSE for a single trace."""
    if model_dtype == torch.float64:
        trace_tensor = torch.from_numpy(trace).double().to(device).unsqueeze(0)
    else:
        trace_tensor = torch.from_numpy(trace).float().to(device).unsqueeze(0)

    embeds = model._read_in(trace_tensor)
    hidden = model._backbone(inputs_embeds=embeds).last_hidden_state
    preds = model._read_out(hidden)

    targets = trace_tensor[:, 1:, -5:]
    preds_shifted = preds[:, :-1, :]
    mask = trace_tensor[:, 1:, 51] != 0

    if mask.sum() == 0:
        return torch.tensor(0.0, device=device, dtype=model_dtype, requires_grad=True)

    diff = (preds_shifted - targets) ** 2
    diff_sum = diff.sum(dim=-1)
    mask_dtype = mask.type_as(diff_sum)
    mse = (diff_sum * mask_dtype).sum() / mask_dtype.sum()
    return mse


def train_epoch(model, optimizer, samples, batch_size, device, model_dtype):
    """Train read-in and read-out for one epoch (20 batches, 128 grad accum)."""
    model.train()
    model.to(device)
    if model_dtype == torch.float64:
        model.double()
    else:
        model.float()

    shuffled_samples = samples.copy()
    np.random.shuffle(shuffled_samples)

    num_batches = 20
    samples_per_batch = len(shuffled_samples) // num_batches

    for batch_idx in range(num_batches):
        batch_start = batch_idx * samples_per_batch
        batch_end = batch_start + samples_per_batch
        batch_samples = shuffled_samples[batch_start:batch_end]

        gradient_accum_size = 128
        for step_start in range(0, len(batch_samples), gradient_accum_size):
            step_end = min(step_start + gradient_accum_size, len(batch_samples))
            step_traces = batch_samples[step_start:step_end]

            optimizer.zero_grad()
            for trace in step_traces:
                mse = compute_single_trace_mse_with_dtype(model, trace, device, model_dtype)
                scaled_mse = mse / float(len(step_traces))
                scaled_mse.backward()
            optimizer.step()

        print(f"    Batch {batch_idx + 1}/20 completed")

    model.eval()
    total_mse = 0.0
    with torch.no_grad():
        for trace in shuffled_samples:
            mse = compute_single_trace_mse_with_dtype(model, trace, device, model_dtype)
            total_mse += mse.item()
    return total_mse / len(shuffled_samples) if shuffled_samples else 0.0


def save_checkpoint(model, optimizer, epoch, mse, output_dir, iteration=None):
    """Save checkpoint"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'mse': mse
    }
    filename = 'initial_random_model.ckpt' if iteration == 0 else f'iteration_{iteration:06d}.ckpt'
    path = os.path.join(output_dir, filename)
    torch.save(checkpoint, path)
    return path


def find_latest_checkpoint(output_dir):
    """Find latest checkpoint"""
    if not os.path.exists(output_dir):
        return None, 0
    checkpoints = []
    for f in os.listdir(output_dir):
        if f.endswith('.ckpt'):
            if f == 'initial_random_model.ckpt':
                checkpoints.append((0, os.path.join(output_dir, f)))
            elif f.startswith('iteration_'):
                try:
                    it = int(f.replace('iteration_', '').replace('.ckpt', ''))
                    checkpoints.append((it, os.path.join(output_dir, f)))
                except ValueError:
                    pass
    if not checkpoints:
        return None, 0
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1], checkpoints[0][0]


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load full model from checkpoint (backbone + read-in + read-out must match original run)"""
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    # Load full model: backbone was frozen but has specific random init; read-in was trained for that backbone
    model.load_state_dict(state, strict=True)
    if 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt.get('epoch', 0), ckpt.get('mse', 0.0)


def main():
    config = Config()
    config.override("model_type", "GPT2")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model_dtype = torch.float64
        print("Using device: CUDA with float64")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model_dtype = torch.float32
        print("Using device: MPS (Apple GPU) with float32")
    else:
        device = torch.device("cpu")
        model_dtype = torch.float64
        print("Using device: CPU with float64")

    learning_rate = 1e-4 * math.sqrt(128 / 57)
    max_epochs = 10000
    checkpoint_interval = 25
    num_pretraining_traces = 10000
    num_pretraining_batches = 20

    output_dir = os.path.join(os.path.dirname(__file__), "random_transformer_normalized_results")
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, 'training_results.txt')

    print("\n" + "="*80)
    print("RANDOM TRANSFORMER (NORMALIZED INIT) - Train Read-in + Read-out")
    print("="*80)
    print(f"Init: Each param ~ N(0, F/n) where F=squared Frobenius norm from ckpt9900, n=num_elements")
    print(f"Backbone: RANDOM (not from ckpt). Train: read-in + read-out only")
    print(f"Output: {output_dir}")
    print("="*80 + "\n")

    latest_ckpt, latest_it = find_latest_checkpoint(output_dir)
    start_epoch = 0

    if latest_ckpt:
        print(f"Found checkpoint at iteration {latest_it}, will resume")
    else:
        with open(results_file, 'w') as f:
            f.write("Random Transformer Normalized Init - Train Read-in + Read-out\n")
            f.write("Init: N(0, F/n) per param, F=squared Frobenius norm from ckpt9900\n\n")

    # Load pure checkpoint for norms only
    gpt2_path = os.path.join(os.path.dirname(__file__), "Set-up Data", "step%3D99000.ckpt")
    if not os.path.exists(gpt2_path):
        raise FileNotFoundError(f"Checkpoint not found: {gpt2_path}")

    gpt2_ckpt = torch.load(gpt2_path, map_location='cpu')
    pure_state = gpt2_ckpt.get('model_state_dict', gpt2_ckpt.get('state_dict', gpt2_ckpt))

    # Create model (default init)
    model = GPT2(
        n_dims_in=config.n_dims_in,
        n_positions=config.n_positions,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_dims_out=config.n_dims_out
    )

    if latest_ckpt is None:
        print("Initializing ALL parameters with N(0, F/n) from ckpt9900 norms...")
        initialize_model_from_pure_norms(model, pure_state)

    # Freeze backbone, train read-in + read-out
    print("\nFreezing backbone, training read-in + read-out...")
    for name, param in model.named_parameters():
        if name.startswith('_read_in') or name.startswith('_read_out'):
            param.requires_grad = True
            print(f"  Training: {name}")
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,}")

    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    if model_dtype == torch.float64:
        model.double().to(device)
    else:
        model.float().to(device)

    if latest_ckpt:
        print(f"\nResuming from {latest_ckpt}")
        start_epoch, _ = load_checkpoint(model, optimizer, latest_ckpt, device)
        if model_dtype == torch.float64:
            model.double()
        else:
            model.float()
    else:
        save_checkpoint(model, optimizer, 0, 0.0, output_dir, iteration=0)
        print("Saved initial checkpoint")

    print(f"\nTraining for {max_epochs} epochs, lr={learning_rate:.6e}")
    best_mse = float('inf')

    if start_epoch > 0 and os.path.exists(results_file):
        with open(results_file) as f:
            m = re.findall(r'Training MSE: ([\d.]+)', f.read())
            if m:
                best_mse = min(float(x) for x in m)
                print(f"Loaded best MSE: {best_mse:.6f}")

    for epoch in range(start_epoch, max_epochs):
        t0 = time.time()
        print(f"\n{'='*80}\nEpoch {epoch + 1}/{max_epochs}\n{'='*80}")

        batches = load_pretraining_batches(
            num_traces=num_pretraining_traces,
            num_batches=num_pretraining_batches,
            batch_idx=None,
            random_seed=epoch,
            cache_suffix="_random_normalized"
        )
        all_traces = []
        for b in batches:
            all_traces.extend(b)

        print(f"  {len(all_traces)} traces")
        epoch_mse = train_epoch(model, optimizer, all_traces, 128, device, model_dtype)
        elapsed = time.time() - t0

        print(f"  Epoch {epoch + 1} Training MSE: {epoch_mse:.6f} (Time: {elapsed:.2f}s)")

        with open(results_file, 'a') as f:
            f.write(f"Epoch {epoch + 1}:\n  Training MSE: {epoch_mse:.6f}\n  Time: {elapsed:.2f}s\n\n")
            f.flush()

        if epoch_mse < best_mse:
            best_mse = epoch_mse
            print(f"  *** New best: {best_mse:.6f} ***")

        if (epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == max_epochs:
            save_checkpoint(model, optimizer, epoch + 1, epoch_mse, output_dir, iteration=epoch + 1)
            print(f"  Checkpoint saved")

        sys.stdout.flush()

    print("\n" + "="*80)
    print("DONE")
    print("="*80)


if __name__ == "__main__":
    main()
