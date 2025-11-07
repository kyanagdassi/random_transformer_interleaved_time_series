"""
Compare GPT2 Pure vs GPT2 Modified: Train on last 10 predictions, test on last 10 configs.

Training data: Last 10 valid predictions per trace from configs 0-39 of haystacks 1-6
Test data: Configs 40-49 of haystacks 1-6 (last 10 configs per haystack)
Training: 20 batches, 2 configs per haystack per batch (12 configs total per batch)
Modified model: Outer layer replaced by inverse method solution

Evaluation: On held-out test configs 40-49 after each training batch
Error at k=1,2,3,4,5,6,7,8 positions after initial/final bracket
Aggregation: median over traces -> median over configs (10 configs × 6 haystacks = 60 total)
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
from tqdm import tqdm
from copy import deepcopy

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

def load_activations(haystack_len):
    """Load cached activations for a haystack"""
    filepath = os.path.join(
        os.path.dirname(__file__),
        "Training:Test Data",
        "Haystack",
        "Activations",
        f"activations_haystack_{haystack_len}.pkl"
    )
    
    with open(filepath, 'rb') as f:
        activations_dict = pickle.load(f)
    
    return activations_dict

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

def extract_last10_training_pairs(data, activations_dict, config_indices):
    """
    Extract the LAST 10 valid prediction pairs from each trace.
    
    For each trace:
    - Find all valid positions where trace[i+1, 51] != 0 (payload flag)
    - Take only the LAST 10 of these positions
    - Extract pairs: (activation[i], target[i+1])
    
    Returns A and Y as matrices where:
    - A has COLUMNS that are activations: A shape = [n_embd, num_samples]
    - Y has COLUMNS that are targets: Y shape = [n_dims_out, num_samples]
    """
    multi_sys_ys = data['multi_sys_ys']
    
    A_columns = []  # List of column vectors (activations)
    Y_columns = []  # List of column vectors (targets)
    
    total_traces = len(config_indices) * multi_sys_ys.shape[1] * multi_sys_ys.shape[2]
    
    with tqdm(total=total_traces, desc="Extracting last 10 pairs", leave=False) as pbar:
        for config_idx in config_indices:
            for trace_idx in range(multi_sys_ys.shape[1]):
                for example_idx in range(multi_sys_ys.shape[2]):
                    trace = multi_sys_ys[config_idx, trace_idx, example_idx]  # [seq_len, 57]
                    activations = activations_dict[(config_idx, trace_idx, example_idx)]  # [seq_len, n_embd]
                    
                    # Find all valid positions
                    valid_positions = []
                    for i in range(len(trace) - 1):
                        if trace[i+1, 51] != 0:  # Next token has payload flag
                            valid_positions.append(i)
                    
                    # Take the LAST 10 valid positions
                    last10_positions = valid_positions[-10:] if len(valid_positions) >= 10 else valid_positions
                    
                    # Extract pairs from these positions
                    for i in last10_positions:
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

def evaluate_models_on_test_configs(model_pure, model_modified, all_data, all_activations, 
                                    haystack_lengths, k_values, device='cpu'):
    """
    Evaluate both models on test data (configs 40-49 from all haystacks).
    
    For each trace:
    1. Find the FIRST open bracket in the trace
    2. Find all occurrences of that same bracket
    3. k_after_initial: k positions after the first occurrence
    4. k_after_final: k positions after the last occurrence
    
    Aggregation: median over traces -> median over configs (10 configs × 6 haystacks = 60 total)
    
    Returns:
        aggregated_pure: dict[metric] = median error across all test configs
        aggregated_modified: dict[metric] = median error across all test configs
    """
    model_pure.eval()
    model_modified.eval()
    model_pure.double().to(device)
    model_modified.double().to(device)
    
    # Test configs: 40-49 from each haystack
    test_config_indices = list(range(40, 50))
    
    # Collect per-config medians from all 60 test configs (10 per haystack × 6 haystacks)
    all_config_medians_pure = {f'{k}_after_initial': [] for k in k_values}
    all_config_medians_pure.update({f'{k}_after_final': [] for k in k_values})
    
    all_config_medians_modified = {f'{k}_after_initial': [] for k in k_values}
    all_config_medians_modified.update({f'{k}_after_final': [] for k in k_values})
    
    for haystack_len in haystack_lengths:
        data = all_data[haystack_len]
        activations_dict = all_activations[haystack_len]
        multi_sys_ys = data['multi_sys_ys']
        
        # Evaluate each config separately to get per-config medians
        for config_idx in test_config_indices:
            # Store squared errors for this config
            config_errors_pure = {f'{k}_after_initial': [] for k in k_values}
            config_errors_pure.update({f'{k}_after_final': [] for k in k_values})
            
            config_errors_modified = {f'{k}_after_initial': [] for k in k_values}
            config_errors_modified.update({f'{k}_after_final': [] for k in k_values})
            
            for trace_idx in range(multi_sys_ys.shape[1]):
                for example_idx in range(multi_sys_ys.shape[2]):
                    trace = multi_sys_ys[config_idx, trace_idx, example_idx]  # [seq_len, 57]
                    activations = activations_dict[(config_idx, trace_idx, example_idx)]  # [seq_len, n_embd]
                    
                    # Find all open brackets
                    open_brackets = find_open_brackets(trace)
                    
                    if len(open_brackets) == 0:
                        continue
                    
                    # Get the FIRST open bracket in the trace
                    first_bracket_pos = min(open_brackets.keys())
                    first_bracket_idx = open_brackets[first_bracket_pos]
                    
                    # Find ALL occurrences of this same bracket
                    same_bracket_positions = [pos for pos, idx in open_brackets.items() 
                                             if idx == first_bracket_idx]
                    
                    if len(same_bracket_positions) == 0:
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
                            
                            config_errors_pure[f'{k}_after_initial'].append(se_pure)
                            config_errors_modified[f'{k}_after_initial'].append(se_mod)
                        
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
                            
                            config_errors_pure[f'{k}_after_final'].append(se_pure)
                            config_errors_modified[f'{k}_after_final'].append(se_mod)
            
            # Compute median over traces for this config
            for metric in config_errors_pure.keys():
                if len(config_errors_pure[metric]) > 0:
                    all_config_medians_pure[metric].append(np.median(config_errors_pure[metric]))
                    all_config_medians_modified[metric].append(np.median(config_errors_modified[metric]))
    
    # Take median over all 60 test configs
    aggregated_pure = {metric: np.median(values) for metric, values in all_config_medians_pure.items()}
    aggregated_modified = {metric: np.median(values) for metric, values in all_config_medians_modified.items()}
    
    return aggregated_pure, aggregated_modified

def save_results_to_text(results_dict, k_values, output_path, batch_num=None):
    """
    Save results to a text file.
    
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
    print("TRAIN ON LAST 10, TEST ON LAST 10 CONFIGS")
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
    
    # Load data and activations for ALL haystacks
    print("="*80)
    print("LOADING DATA AND ACTIVATIONS FOR ALL HAYSTACKS")
    print("="*80 + "\n")
    
    all_activations = {}
    all_data = {}
    
    for haystack_len in haystack_lengths:
        print(f"Loading haystack {haystack_len}...")
        all_data[haystack_len] = load_haystack_file(haystack_len)
        all_activations[haystack_len] = load_activations(haystack_len)
    
    print("\nAll data loaded\n")
    
    # Results storage
    results_dict = {}
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "test_last10configs_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Clear any existing text results file
    text_path = os.path.join(output_dir, 'test_last10configs_results.txt')
    if os.path.exists(text_path):
        os.remove(text_path)
        print(f"Cleared existing results file\n")
    
    # Training in batches: 20 batches × 2 configs per haystack = 240 total configs
    num_batches = 20
    configs_per_batch = 2  # Per haystack
    
    print(f"\n{'='*80}")
    print("TRAINING WITH LAST 10 PREDICTIONS PER TRACE")
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
            
            print(f"Haystack {haystack_len}: Extracting LAST 10 predictions from NEW configs {batch_config_indices}...")
            
            # Extract training pairs for ONLY this batch from this haystack (last 10 per trace)
            A_batch, Y_batch = extract_last10_training_pairs(
                all_data[haystack_len], 
                all_activations[haystack_len], 
                batch_config_indices
            )
            
            print(f"  Extracted {A_batch.shape[1]} samples from haystack {haystack_len}")
            
            # Accumulate
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
        
        aggregated_pure, aggregated_modified = evaluate_models_on_test_configs(
            model_pure,
            model_modified,
            all_data,
            all_activations,
            haystack_lengths,
            k_values,
            device
        )
        
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
        save_results_to_text(results_dict, k_values, text_path, batch_num=batch_idx + 1)
    
    # Save final numerical results
    print(f"\n{'='*80}")
    print("SAVING FINAL RESULTS")
    print(f"{'='*80}\n")
    
    results_path = os.path.join(output_dir, "test_last10configs_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(results_dict, f)
    print(f"Results saved to: {results_path}")
    
    print(f"Text results saved to: {text_path}")
    
    print(f"\n{'='*80}")
    print("COMPARISON COMPLETE!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()





