"""
Compare GPT2 Pure vs GPT2 Modified (outer layer optimized via least squares).

GPT2 Pure: Loaded checkpoint without modification
GPT2 Modified: Loaded checkpoint with outer layer optimized via least squares on training data

Training: Process in 20 batches (2 configs per batch)
Uses running sums of YA^T and AA^T for efficient least squares computation
All computations in float64
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import numpy as np
import torch
import torch.nn as nn
from core import Config
from models import GPT2
# import matplotlib.pyplot as plt  # Disabled plotting
from copy import deepcopy
from tqdm import tqdm

def load_haystack_file(haystack_len):
    """Load a haystack file by length"""
    filepath = os.path.join(
        os.path.dirname(__file__),
        "Training:Test Data",
        "Haystack",
        f"val_interleaved_traces_ortho_haar_ident_C_haystack_len_{haystack_len}.pkl"
    )
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    return data

def find_open_brackets(trace):
    """
    Find positions where open brackets occur (even indices 0-49 are nonzero).
    
    trace: shape [seq_len, 57]
    Returns: dict mapping position -> bracket_index (which even index is hot)
    """
    open_brackets = {}
    even_indices = np.arange(0, 50, 2)  # [0, 2, 4, ..., 48]
    
    for pos in range(len(trace)):
        for bracket_idx in even_indices:
            if trace[pos, bracket_idx] != 0:
                open_brackets[pos] = bracket_idx
                break
    
    return open_brackets

def get_or_compute_activations(model, haystack_len, device='cpu', force_recompute=False):
    """
    Get cached activations or compute and cache them.
    
    Returns:
        activations_dict: Dict mapping (config_idx, trace_idx, example_idx) -> activations array [seq_len, n_embd]
    """
    cache_path = os.path.join(
        os.path.dirname(__file__),
        "Training:Test Data",
        "Haystack",
        "Activations",
        f"activations_haystack_{haystack_len}.pkl"
    )
    
    # Try to load from cache
    if os.path.exists(cache_path) and not force_recompute:
        print(f"Loading cached activations from {cache_path}...")
        with open(cache_path, 'rb') as f:
            activations_dict = pickle.load(f)
        print(f"Loaded cached activations (float64)\n")
        return activations_dict
    
    # Compute activations
    print(f"Computing activations for haystack length {haystack_len}...")
    model.eval()
    # Convert model to float64 for full precision
    model.double()
    model.to(device)
    
    data = load_haystack_file(haystack_len)
    multi_sys_ys = data['multi_sys_ys']  # shape: (50, 1, 1000, seq_len, 57)
    
    activations_dict = {}
    
    total_traces = multi_sys_ys.shape[0] * multi_sys_ys.shape[1] * multi_sys_ys.shape[2]
    batch_size = 100  # Process 100 traces at once for speed
    
    with tqdm(total=total_traces, desc=f"Haystack {haystack_len}") as pbar:
        for config_idx in range(multi_sys_ys.shape[0]):  # 50 configs
            for trace_idx in range(multi_sys_ys.shape[1]):  # 1 trace
                # Process all 1000 examples in batches
                num_examples = multi_sys_ys.shape[2]
                for batch_start in range(0, num_examples, batch_size):
                    batch_end = min(batch_start + batch_size, num_examples)
                    batch_traces = multi_sys_ys[config_idx, trace_idx, batch_start:batch_end]  # [batch, seq_len, 57]
                    
                    # Process batch at once
                    with torch.no_grad():
                        input_tensor = torch.from_numpy(batch_traces).double().to(device)  # [batch, seq_len, 57] float64
                        
                        # Forward through read_in and backbone
                        embeds = model._read_in(input_tensor)
                        hidden = model._backbone(inputs_embeds=embeds).last_hidden_state  # [batch, seq_len, n_embd]
                        
                    batch_activations = hidden.cpu().numpy().astype(np.float64)  # Convert to float64 after computation
                    
                    # Store individual activations
                    for i, example_idx in enumerate(range(batch_start, batch_end)):
                        activations = batch_activations[i]  # [seq_len, n_embd]
                        
                        # Check and clean activations
                        if not np.isfinite(activations).all():
                            activations = np.nan_to_num(activations, nan=0.0, posinf=1e10, neginf=-1e10)
                        
                        activations_dict[(config_idx, trace_idx, example_idx)] = activations
                        pbar.update(1)
    
    # Cache the activations
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(activations_dict, f)
    print(f"Cached activations to {cache_path}\n")
    
    return activations_dict

def extract_training_pairs_batch(data, activations_dict, config_indices):
    """
    Extract training pairs from a batch of configs.
    
    Returns A and Y as matrices where:
    - A has COLUMNS that are activations: A shape = [n_embd, num_samples]
    - Y has COLUMNS that are targets: Y shape = [n_dims_out, num_samples]
    
    So the least squares solution is: W_opt = Y @ A^T @ (A @ A^T)^{-1}
    """
    multi_sys_ys = data['multi_sys_ys']
    
    A_columns = []  # List of column vectors (activations)
    Y_columns = []  # List of column vectors (targets)
    
    total_traces = len(config_indices) * multi_sys_ys.shape[1] * multi_sys_ys.shape[2]
    
    with tqdm(total=total_traces, desc="Extracting pairs", leave=False) as pbar:
        for config_idx in config_indices:
            for trace_idx in range(multi_sys_ys.shape[1]):
                for example_idx in range(multi_sys_ys.shape[2]):
                    trace = multi_sys_ys[config_idx, trace_idx, example_idx]  # [seq_len, 57]
                    activations = activations_dict[(config_idx, trace_idx, example_idx)]  # [seq_len, n_embd]
                    
                    # Extract training pairs: activation[i] -> target[i+1]
                    # Only include if trace[i+1, 51] != 0 (target has payload flag)
                    for i in range(len(trace) - 1):
                        if trace[i+1, 51] != 0:  # Next token has payload flag
                            # Convert float32 activations to float64 for least squares precision
                            activation = activations[i].astype(np.float64)  # [n_embd]
                            target = trace[i+1, -5:].astype(np.float64)  # [5]
                            
                            A_columns.append(activation)
                            Y_columns.append(target)
                    
                    pbar.update(1)
    
    # Convert to matrices with samples as columns
    A = np.column_stack(A_columns)  # [n_embd, num_samples]
    Y = np.column_stack(Y_columns)  # [n_dims_out, num_samples]
    
    # Check for inf/nan values
    if not np.isfinite(A).all():
        print(f"  WARNING: A contains inf/nan values! Replacing with 0.")
        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    
    if not np.isfinite(Y).all():
        print(f"  WARNING: Y contains inf/nan values! Replacing with 0.")
        Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
    
    return A, Y

def compute_least_squares_incremental(YAT_sum, AAT_sum):
    """
    Compute least squares solution from accumulated matrices.
    
    When samples are columns (A: [n_embd, num_samples], Y: [n_dims_out, num_samples]):
    - Minimize: ||Y - W @ A||^2 where W: [n_dims_out, n_embd]
    - Solution: W = Y @ A^T @ (A @ A^T)^{-1}
    
    Args:
        YAT_sum: Y @ A^T accumulated matrix [n_dims_out, n_embd]
        AAT_sum: A @ A^T accumulated matrix [n_embd, n_embd]
    
    Returns:
        W: Weight matrix [n_dims_out, n_embd] (ready for PyTorch Linear layer)
    """
    # Check for inf/nan
    if not np.isfinite(YAT_sum).all():
        print("  WARNING: YAT_sum contains inf/nan! Cleaning...")
        YAT_sum = np.nan_to_num(YAT_sum, nan=0.0, posinf=1e10, neginf=-1e10)
    
    if not np.isfinite(AAT_sum).all():
        print("  WARNING: AAT_sum contains inf/nan! Cleaning...")
        AAT_sum = np.nan_to_num(AAT_sum, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # W = YAT @ (AAT)^{-1}
    AAT_inv = np.linalg.inv(AAT_sum)  # [n_embd, n_embd]
    
    if not np.isfinite(AAT_inv).all():
        print("  WARNING: AAT_inv contains inf/nan after inversion!")
        AAT_inv = np.nan_to_num(AAT_inv, nan=0.0, posinf=1e10, neginf=-1e10)
    
    W = YAT_sum @ AAT_inv  # [n_dims_out, n_embd] @ [n_embd, n_embd] = [n_dims_out, n_embd]
    
    if not np.isfinite(W).all():
        print("  WARNING: W contains inf/nan after computation!")
        W = np.nan_to_num(W, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return W

def create_modified_model(model_pure, W):
    """
    Create a modified model with updated outer layer weights.
    No bias in this formulation.
    Everything in float64.
    
    Args:
        model_pure: Original model
        W: Weight matrix [n_dims_out, n_embd] from least squares
    """
    model_modified = deepcopy(model_pure)
    
    # Update the outer layer weights (float64)
    # PyTorch Linear layer expects weight in shape [out_features, in_features]
    model_modified._read_out.weight.data = torch.from_numpy(W).double()  # [n_dims_out, n_embd] in float64
    model_modified._read_out.bias.data = torch.zeros_like(model_modified._read_out.bias.data).double()  # Bias to zero in float64
    
    return model_modified

def evaluate_models_per_config(model_pure, model_modified, data, activations_dict, config_indices, k_values, device='cpu'):
    """
    Evaluate both models on test data.
    
    For each trace:
    1. Find the FIRST open bracket in the trace
    2. Find all occurrences of that same bracket
    3. k_after_initial: k positions after the first occurrence
    4. k_after_final: k positions after the last occurrence
    
    Each trace contributes 1 squared error per k value.
    Compute median over 1000 traces per config.
    
    Returns:
        config_medians_pure: dict[metric] = list of 10 medians (one per config)
        config_medians_modified: dict[metric] = list of 10 medians (one per config)
    """
    model_pure.eval()
    model_modified.eval()
    model_pure.double().to(device)  # float64
    model_modified.double().to(device)  # float64
    
    multi_sys_ys = data['multi_sys_ys']
    
    # Store squared errors per config for each metric
    config_errors_pure = {config_idx: {f'{k}_after_initial': [] for k in k_values} for config_idx in config_indices}
    config_errors_modified = {config_idx: {f'{k}_after_initial': [] for k in k_values} for config_idx in config_indices}
    
    for config_idx in config_errors_pure:
        for k in k_values:
            config_errors_pure[config_idx][f'{k}_after_final'] = []
            config_errors_modified[config_idx][f'{k}_after_final'] = []
    
    # Track valid predictions per trace for each config
    config_valid_counts = {config_idx: [] for config_idx in config_indices}
    
    total_traces = len(config_indices) * multi_sys_ys.shape[1] * multi_sys_ys.shape[2]
    
    with tqdm(total=total_traces, desc="Evaluating", leave=False) as pbar:
        for config_idx in config_indices:
            for trace_idx in range(multi_sys_ys.shape[1]):
                for example_idx in range(multi_sys_ys.shape[2]):
                    trace = multi_sys_ys[config_idx, trace_idx, example_idx]  # [seq_len, 57]
                    activations = activations_dict[(config_idx, trace_idx, example_idx)]  # [seq_len, n_embd]
                    
                    # Track valid predictions for this trace
                    valid_count_this_trace = 0
                    
                    # Find all open brackets
                    open_brackets = find_open_brackets(trace)
                
                    if len(open_brackets) == 0:
                        config_valid_counts[config_idx].append(0)
                        pbar.update(1)
                        continue
                    
                    # Get the FIRST open bracket in the trace
                    first_bracket_pos = min(open_brackets.keys())
                    first_bracket_idx = open_brackets[first_bracket_pos]
                    
                    # Find ALL occurrences of this same bracket
                    same_bracket_positions = [pos for pos, idx in open_brackets.items() 
                                             if idx == first_bracket_idx]
                    
                    if len(same_bracket_positions) == 0:
                        config_valid_counts[config_idx].append(0)
                        pbar.update(1)
                        continue
                    
                    initial_pos = same_bracket_positions[0]  # First occurrence
                    final_pos = same_bracket_positions[-1]   # Last occurrence
                    
                    # Evaluate k positions after initial and final
                    for k in k_values:
                        # k after initial
                        eval_pos_initial = initial_pos + k
                        if eval_pos_initial < len(trace) and trace[eval_pos_initial, 51] != 0:
                            activation = activations[eval_pos_initial - 1].astype(np.float64)  # [n_embd] float64
                            target = trace[eval_pos_initial, -5:].astype(np.float64)  # [5]
                            
                            # Predictions using cached activations (float64)
                            with torch.no_grad():
                                activation_tensor = torch.from_numpy(activation).double().to(device)  # float64
                                
                                pred_pure = model_pure._read_out(activation_tensor).detach().cpu().numpy()
                                pred_mod = model_modified._read_out(activation_tensor).detach().cpu().numpy()
                            
                            # Squared error (sum over 5 dimensions)
                            se_pure = np.sum((pred_pure - target) ** 2)
                            se_mod = np.sum((pred_mod - target) ** 2)
                            
                            config_errors_pure[config_idx][f'{k}_after_initial'].append(se_pure)
                            config_errors_modified[config_idx][f'{k}_after_initial'].append(se_mod)
                            valid_count_this_trace += 1
                        
                        # k after final
                        eval_pos_final = final_pos + k
                        if eval_pos_final < len(trace) and trace[eval_pos_final, 51] != 0:
                            activation = activations[eval_pos_final - 1].astype(np.float64)  # [n_embd] float64
                            target = trace[eval_pos_final, -5:].astype(np.float64)  # [5]
                            
                            # Predictions using cached activations (float64)
                            with torch.no_grad():
                                activation_tensor = torch.from_numpy(activation).double().to(device)  # float64
                                
                                pred_pure = model_pure._read_out(activation_tensor).detach().cpu().numpy()
                                pred_mod = model_modified._read_out(activation_tensor).detach().cpu().numpy()
                            
                            # Squared error (sum over 5 dimensions)
                            se_pure = np.sum((pred_pure - target) ** 2)
                            se_mod = np.sum((pred_mod - target) ** 2)
                            
                            config_errors_pure[config_idx][f'{k}_after_final'].append(se_pure)
                            config_errors_modified[config_idx][f'{k}_after_final'].append(se_mod)
                            valid_count_this_trace += 1
                    
                    config_valid_counts[config_idx].append(valid_count_this_trace)
                    pbar.update(1)
    
    # Print valid predictions statistics per config
    print(f"      Valid predictions per trace statistics:")
    for config_idx in config_indices:
        valid_counts = config_valid_counts[config_idx]
        if len(valid_counts) > 0:
            min_valid = np.min(valid_counts)
            max_valid = np.max(valid_counts)
            mean_valid = np.mean(valid_counts)
            print(f"        Config {config_idx}: min={min_valid}, max={max_valid}, mean={mean_valid:.1f}")
        else:
            print(f"        Config {config_idx}: NO traces")
    
    # Compute median over traces for each config (return list of config medians)
    config_medians_pure = {metric: [] for metric in [f'{k}_after_initial' for k in k_values] + [f'{k}_after_final' for k in k_values]}
    config_medians_modified = {metric: [] for metric in [f'{k}_after_initial' for k in k_values] + [f'{k}_after_final' for k in k_values]}
    
    # Check for configs with no valid evaluation positions
    for metric in config_medians_pure.keys():
        for config_idx in config_indices:
            num_errors = len(config_errors_pure[config_idx][metric])
            if num_errors == 0:
                print(f"\nWARNING: Config {config_idx} has NO valid positions for metric '{metric}'!")
                print(f"   This will cause NaN values. Check trace structure.")
                raise ValueError(f"Config {config_idx} has no valid evaluation positions for {metric}")
            
            config_medians_pure[metric].append(np.median(config_errors_pure[config_idx][metric]))
            config_medians_modified[metric].append(np.median(config_errors_modified[config_idx][metric]))
    
    return config_medians_pure, config_medians_modified

def save_results_to_text(results_dict, k_values, output_path, batch_num=None):
    """
    Save results to a text file instead of plotting.
    
    Args:
        results_dict: Dictionary mapping num_train_configs -> {'pure': {...}, 'modified': {...}}
        k_values: List of k values
        output_path: Path to save text file
        batch_num: Optional batch number for incremental updates
    """
    mode = 'a' if batch_num is not None else 'w'
    
    with open(output_path, mode) as f:
        if batch_num is not None:
            f.write(f"\n{'='*80}\n")
            f.write(f"BATCH {batch_num}/20\n")
            f.write(f"{'='*80}\n")
        
        for n_train_configs in sorted(results_dict.keys()):
            f.write(f"\n--- Total Training Configs: {n_train_configs} ---\n")
            
            pure_results = results_dict[n_train_configs]['pure']
            modified_results = results_dict[n_train_configs]['modified']
            
            f.write("\nk_after_initial:\n")
            for k in k_values:
                metric = f'{k}_after_initial'
                f.write(f"  k={k}: Pure={pure_results[metric]:.8f}, Modified={modified_results[metric]:.8f}\n")
            
            f.write("\nk_after_final:\n")
            for k in k_values:
                metric = f'{k}_after_final'
                f.write(f"  k={k}: Pure={pure_results[metric]:.8f}, Modified={modified_results[metric]:.8f}\n")

def main():
    print("\n" + "="*80)
    print("COMPARING GPT2 PURE VS MODIFIED (ALL HAYSTACKS COMBINED)")
    print("="*80 + "\n")
    
    # Configuration
    config = Config()
    config.override("model_type", "GPT2")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Haystack lengths to process
    haystack_lengths = [1, 2, 3, 4, 5, 6]
    
    # k values for evaluation
    k_values = [1, 2, 3, 4, 5, 6, 7, 8]
    
    # Load pretrained model
    print("Loading pretrained GPT2 model...")
    checkpoint_path = os.path.join(
        os.path.dirname(__file__),
        "Set-up Data",
        "step%3D99000.ckpt"
    )
    
    model_pure = GPT2(
        n_dims_in=config.n_dims_in,
        n_positions=config.n_positions,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_dims_out=config.n_dims_out
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model_pure.load_state_dict(state_dict, strict=False)
    
    # Convert model to float64 for full precision
    model_pure.double()
    print("Model loaded and converted to float64\n")
    
    # Get dimensions
    n_embd = config.n_embd
    n_dims_out = config.n_dims_out
    
    # Cache activations for ALL haystacks first
    print("="*80)
    print("STEP 1: CACHING ACTIVATIONS FOR ALL HAYSTACKS")
    print("="*80 + "\n")
    
    all_activations = {}
    all_data = {}
    
    for haystack_len in haystack_lengths:
        print(f"\nProcessing haystack {haystack_len}...")
        all_activations[haystack_len] = get_or_compute_activations(model_pure, haystack_len, device)
        all_data[haystack_len] = load_haystack_file(haystack_len)
    
    # Results storage: results[num_train_configs]['pure'/'modified'][metric]
    results_dict = {}
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "comparison_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Clear any existing text results file
    text_path = os.path.join(output_dir, 'gpt2_comparison_results.txt')
    if os.path.exists(text_path):
        os.remove(text_path)
        print(f"Cleared existing results file\n")
    
    # Training in batches: 20 batches × 2 configs per haystack = 240 total configs
    num_batches = 20
    configs_per_batch = 2  # Per haystack
    
    print(f"\n{'='*80}")
    print("STEP 2: TRAINING WITH COMBINED HAYSTACKS")
    print("="*80 + "\n")
    print(f"Total batches: {num_batches}")
    print(f"Configs per batch per haystack: {configs_per_batch}")
    print(f"Total configs per batch: {configs_per_batch * len(haystack_lengths)} (across all haystacks)")
    print(f"Total training configs: {num_batches * configs_per_batch * len(haystack_lengths)}\n")
    
    # Initialize accumulated matrices
    YAT_sum = np.zeros((n_dims_out, n_embd), dtype=np.float64)
    AAT_sum = np.zeros((n_embd, n_embd), dtype=np.float64)
    
    # Progressive training
    for batch_idx in range(num_batches):
        # This batch processes NEW configs only
        batch_start = batch_idx * configs_per_batch
        batch_end = (batch_idx + 1) * configs_per_batch
        total_configs_so_far = batch_end * len(haystack_lengths)
        
        print(f"\n{'='*80}")
        print(f"BATCH {batch_idx + 1}/20: Processing NEW configs {batch_start}-{batch_end-1} per haystack")
        print(f"Total configs so far: {batch_end} per haystack ({total_configs_so_far} across all)")
        print(f"{'='*80}\n")
        
        # Process each haystack for this batch (only NEW configs)
        for haystack_len in haystack_lengths:
            # Get ONLY the NEW configs for this batch
            batch_config_indices = list(range(batch_start, batch_end))
            
            print(f"Haystack {haystack_len}: Extracting NEW configs {batch_config_indices}...")
            
            # Extract training pairs for ONLY this batch from this haystack
            A_batch, Y_batch = extract_training_pairs_batch(
                all_data[haystack_len], 
                all_activations[haystack_len], 
                batch_config_indices
            )
            
            print(f"  Extracted {A_batch.shape[1]} NEW samples from haystack {haystack_len}")
            
            # Check for NaN/inf and extreme values
        
            YAT_sum += Y_batch @ A_batch.T  # [n_dims_out, n_embd]
            AAT_sum += A_batch @ A_batch.T  # [n_embd, n_embd]
        
        print(f"\nAfter batch {batch_idx + 1}:")
        print(f"  Total training configs used: {total_configs_so_far}")
        print(f"  YAT_sum shape: {YAT_sum.shape}")
        print(f"  AAT_sum shape: {AAT_sum.shape}")
        
        # Clean accumulated sums
        YAT_sum = np.nan_to_num(YAT_sum, nan=0.0, posinf=0.0, neginf=0.0)
        AAT_sum = np.nan_to_num(AAT_sum, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Compute least squares solution
        print(f"  Computing least squares solution...")
        W_opt = compute_least_squares_incremental(YAT_sum, AAT_sum)
        print(f"  W_opt shape: {W_opt.shape}")
        
        # Create modified model
        model_modified = create_modified_model(model_pure, W_opt)
        
        # Evaluate on test data from ALL haystacks (configs 40-49 from each)
        print(f"\n  Evaluating on test data (configs 40-49 from all haystacks)...")
        
        test_config_indices = list(range(40, 50))
        
        # Collect per-config medians from all 60 test configs (10 per haystack × 6 haystacks)
        all_config_medians_pure = {f'{k}_after_initial': [] for k in k_values}
        all_config_medians_pure.update({f'{k}_after_final': [] for k in k_values})
        
        all_config_medians_modified = {f'{k}_after_initial': [] for k in k_values}
        all_config_medians_modified.update({f'{k}_after_final': [] for k in k_values})
        
        for haystack_len in haystack_lengths:
            print(f"    Testing on haystack {haystack_len}...")
            
            # Get per-config medians for this haystack (10 configs)
            config_medians_pure, config_medians_modified = evaluate_models_per_config(
                model_pure,
                model_modified,
                all_data[haystack_len],
                all_activations[haystack_len],
                test_config_indices,
                k_values,
                device
            )
            
            # Collect all config medians (no intermediate haystack median)
            for metric in config_medians_pure.keys():
                all_config_medians_pure[metric].extend(config_medians_pure[metric])
                all_config_medians_modified[metric].extend(config_medians_modified[metric])
        
        # Take median over ALL 60 test configs directly (60 values -> 1 final value per metric)
        aggregated_pure = {metric: np.median(values) for metric, values in all_config_medians_pure.items()}
        aggregated_modified = {metric: np.median(values) for metric, values in all_config_medians_modified.items()}
        
        # Store results
        results_dict[total_configs_so_far] = {
            'pure': aggregated_pure,
            'modified': aggregated_modified
        }
        
        # Print summary
        print(f"\n  Results after {total_configs_so_far} total training configs:")
        print("  k_after_initial:")
        for k in k_values:
            pure_init = aggregated_pure[f'{k}_after_initial']
            mod_init = aggregated_modified[f'{k}_after_initial']
            print(f"    {k}: Pure={pure_init:.6f}, Modified={mod_init:.6f}")
        
        print("  k_after_final:")
        for k in k_values:
            pure_final = aggregated_pure[f'{k}_after_final']
            mod_final = aggregated_modified[f'{k}_after_final']
            print(f"    {k}: Pure={pure_final:.6f}, Modified={mod_final:.6f}")
        
        # Save results to text file after each batch
        print(f"\n  Saving results to text file...")
        text_path = os.path.join(output_dir, "gpt2_comparison_results.txt")
        save_results_to_text(results_dict, k_values, text_path, batch_num=batch_idx + 1)
    
    # Save final numerical results
    print(f"\n{'='*80}")
    print("SAVING FINAL RESULTS")
    print(f"{'='*80}\n")
    
    results_path = os.path.join(output_dir, "gpt2_comparison_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(results_dict, f)
    print(f"Results saved to: {results_path}")
    
    text_path = os.path.join(output_dir, "gpt2_comparison_results.txt")
    print(f"Text results saved to: {text_path}")
    
    print(f"\n{'='*80}")
    print("COMPARISON COMPLETE!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
