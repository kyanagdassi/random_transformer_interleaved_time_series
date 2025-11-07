"""
Cheat Feature Training Error Test

Similar to cheat_feature_numerical_stability_test.py but evaluates on TRAINING data instead of held-out test data.

Strategy:
1. Augment each 128-dim activation with its 5-dim target → 133-dim vector [activation, target]
2. Apply a single random orthogonal rotation U ∈ R^{133×133} to ALL augmented vectors
3. Train least squares on these rotated features (configs 0-39 from all 6 haystacks)
4. Measure error on TRAINING data accumulated so far (batches 1 through i)

Expected result:
- Since the target is embedded in the features (after rotation), a perfect least squares 
  solution should exist (training MSE ≈ 0)
- Training error should decrease as we see more data
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pickle
import numpy as np
from tqdm import tqdm

# Import the orthogonal matrix generator
from dyn_models.filtering_lti import gen_rand_ortho_haar_real

def load_haystack_file(haystack_len):
    """Load haystack activations and targets"""
    # Load GPT2 activations
    act_filepath = os.path.join(
        os.path.dirname(__file__),
        "Training:Test Data",
        "Haystack",
        "Activations",
        f"activations_haystack_{haystack_len}.pkl"
    )
    
    with open(act_filepath, 'rb') as f:
        activations = pickle.load(f)
    
    # Load targets
    tgt_filepath = os.path.join(
        os.path.dirname(__file__),
        "Training:Test Data",
        "Haystack",
        f"val_interleaved_traces_ortho_haar_ident_C_haystack_len_{haystack_len}.pkl"
    )
    
    with open(tgt_filepath, 'rb') as f:
        data = pickle.load(f)
        targets = data['multi_sys_ys']
    
    return activations, targets

def extract_cheat_feature_pairs(activations, targets, config_indices, U, haystack_len):
    """
    Extract training pairs with cheat features using ACTUAL GPT2 activations.
    
    Args:
        activations: Dict with keys (config_idx, batch_idx, trace_idx) -> (trace_len, 128)
        targets: Array of shape (n_configs, 1, n_traces, trace_len, 57)
        config_indices: List of config indices to extract
        U: Orthogonal matrix (133, 133)
        haystack_len: Length of this haystack (for dictionary keys)
    
    Returns:
        A: Concatenated feature matrix [num_samples, 133]
        Y: Concatenated target matrix [num_samples, 5]
    """
    A_rows = []  # All samples concatenated
    Y_rows = []  # All samples concatenated
    
    # Count total traces for progress bar
    total_traces = 0
    for config_idx in config_indices:
        batch_idx = 0
        for key in activations.keys():
            if key[0] == config_idx and key[1] == batch_idx:
                total_traces += 1
    
    with tqdm(total=total_traces, desc=f"  Haystack {haystack_len}", leave=False) as pbar:
        for config_idx in config_indices:
            batch_idx = 0
            
            # Find all trace indices for this config
            trace_indices = []
            for key in activations.keys():
                if key[0] == config_idx and key[1] == batch_idx:
                    trace_indices.append(key[2])
            trace_indices = sorted(trace_indices)
            
            # Process each trace
            for trace_idx in trace_indices:
                # Get GPT2 activations (128-dim)
                act_trace = activations[(config_idx, batch_idx, trace_idx)]  # [trace_len, 128]
                
                # Get targets (5-dim from last 5 of 57-dim)
                tgt_trace = targets[config_idx, batch_idx, trace_idx]  # [trace_len, 57]
                
                trace_len = len(act_trace)
                
                # Extract pairs: activation[i] -> target[i+1, -5:]
                for i in range(trace_len - 1):
                    activation = act_trace[i].astype(np.float64)  # [128] - ACTUAL GPT2 activation
                    target = tgt_trace[i + 1, -5:].astype(np.float64)  # [5]
                    
                    # Check for valid target
                    if not np.all(target == 0.0):
                        # Create augmented vector: [activation, target] → 133-dim
                        augmented = np.concatenate([activation, target])  # [133]
                        
                        # Apply orthogonal rotation
                        rotated_feature = U @ augmented  # [133]
                        
                        A_rows.append(rotated_feature)
                        Y_rows.append(target)
                
                pbar.update(1)
    
    # Convert to matrices with samples as rows
    A = np.array(A_rows, dtype=np.float64)  # [num_samples, 133]
    Y = np.array(Y_rows, dtype=np.float64)  # [num_samples, 5]
    
    return A, Y

def compute_least_squares_inverse(YAT_sum, AAT_sum):
    """
    Compute least squares using matrix inversion.
    
    With samples as rows:
    - A is [n_samples, n_features]
    - Y is [n_samples, n_target]
    - Normal equations: A^T @ A @ W = A^T @ Y
    - Solution: W = (A^T @ A)^{-1} @ A^T @ Y
    
    Args:
        YAT_sum: Y^T @ A accumulated matrix [n_target, n_features]
        AAT_sum: A^T @ A accumulated matrix [n_features, n_features]
    
    Returns:
        W: Weight matrix [n_target, n_features]
    """
    AAT_inv = np.linalg.inv(AAT_sum)  # [n_features, n_features]
    W = YAT_sum @ AAT_inv  # [n_target, n_features]
    return W

def compute_batch_mse(A_batch, Y_batch, W):
    """
    Compute MSE on a given batch.
    
    Args:
        A_batch: Feature matrix for batch [num_samples, 133]
        Y_batch: Target matrix for batch [num_samples, 5]
        W: Weight matrix [5, 133]
    
    Returns:
        mse: MSE on batch
    """
    Y_pred = A_batch @ W.T
    mse = np.mean((Y_batch - Y_pred) ** 2)
    return mse

# Removed checkpoint functions - always start from scratch

def append_results_to_file(batch_num, n_samples, mse_inverse_train, mse_true_train, 
                          mse_inverse_last10, mse_true_last10, w_err_inv, w_frob_inv):
    """Append results for a single batch to the results file"""
    results_file = "cheat_train_error_results.txt"
    
    # Create header if file doesn't exist
    if not os.path.exists(results_file):
        with open(results_file, 'w') as f:
            f.write("batch,n_samples,mse_inverse_train,mse_true_train,mse_inverse_last10,mse_true_last10,w_error_inverse_pct,w_frob_norm_inv\n")
    
    # Append this batch's results
    with open(results_file, 'a') as f:
        f.write(f"{batch_num},{n_samples},{mse_inverse_train:.6e},{mse_true_train:.6e},{mse_inverse_last10:.6e},{mse_true_last10:.6e},{w_err_inv:.6e},{w_frob_inv:.6e}\n")

def main():
    print("\n" + "="*80)
    print("CHEAT FEATURE TRAINING ERROR TEST")
    print("="*80 + "\n")
    
    print("This test embeds targets into features via orthogonal rotation.")
    print("Evaluates on TRAINING data accumulated so far.\n")
    
    # Set random seed for reproducibility (SAME AS TEST FILE)
    np.random.seed(42)
    
    # Haystack lengths to process
    haystack_lengths = [1, 2, 3, 4, 5, 6]
    
    # Dimensions
    n_activation = 128
    n_target = 5
    n_features = n_activation + n_target  # 133
    
    # Generate ONE random orthogonal matrix to use for ALL samples
    print(f"Generating random orthogonal matrix U ∈ R^{{{n_features}×{n_features}}}...")
    U = gen_rand_ortho_haar_real(n_features)
    print(f"Generated U with shape {U.shape}")
    print(f"  Checking orthogonality: ||U^T U - I||_F = {np.linalg.norm(U.T @ U - np.eye(n_features), 'fro'):.2e}")
    
    # Construct the extraction matrix X (5×133)
    X = np.zeros((n_target, n_features), dtype=np.float64)
    for i in range(n_target):
        X[i, n_activation + i] = 1.0  # Set diagonal to extract target positions
    
    # Ground truth W
    W_opt_true = X @ U.T  
    print(f"  Ground truth W_opt shape: {W_opt_true.shape}\n")
    
    # Training in batches: 20 batches × 2 configs per haystack
    num_batches = 20
    configs_per_batch = 2
    
    print(f"Training with incremental least squares:")
    print(f"  Total batches: {num_batches}")
    print(f"  Configs per batch per haystack: {configs_per_batch}")
    print(f"  Total training configs: {num_batches * configs_per_batch * len(haystack_lengths)}\n")
    
    # Initialize accumulated matrices (always start from scratch)
    YAT_sum = np.zeros((n_target, n_features), dtype=np.float64)
    AAT_sum = np.zeros((n_features, n_features), dtype=np.float64)
    
    # Progressive training
    for batch_idx in range(num_batches):
        
        batch_start = batch_idx * configs_per_batch
        batch_end = (batch_idx + 1) * configs_per_batch
        total_configs_so_far = batch_end * len(haystack_lengths)
        
        print(f"\n{'='*80}")
        print(f"BATCH {batch_idx + 1}/{num_batches}: Processing configs {batch_start}-{batch_end-1} per haystack")
        print(f"Total configs so far: {batch_end} per haystack ({total_configs_so_far} across all)")
        print(f"{'='*80}\n")
        
        # Collect data from all haystacks for this batch
        batch_A_all = []
        batch_Y_all = []
        
        for haystack_len in haystack_lengths:
            batch_config_indices = list(range(batch_start, batch_end))
            
            # Load haystack on-demand
            activations, targets = load_haystack_file(haystack_len)
            
            # Extract cheat feature pairs
            A_batch, Y_batch = extract_cheat_feature_pairs(
                activations,
                targets,
                batch_config_indices,
                U,
                haystack_len
            )
            
            del activations, targets
            
            print(f"    Haystack {haystack_len}: {A_batch.shape[0]:,} samples")
            
            batch_A_all.append(A_batch)
            batch_Y_all.append(Y_batch)
            
            # Accumulate
            YAT_sum += Y_batch.T @ A_batch  # [n_target, n_features]
            AAT_sum += A_batch.T @ A_batch  # [n_features, n_features]
        
        print(f"\nAfter batch {batch_idx + 1}:")
        print(f"  YAT_sum shape: {YAT_sum.shape}")
        print(f"  AAT_sum shape: {AAT_sum.shape}")
        
        # Compute least squares solution
        print(f"  Computing W_opt using inverse method...")
        W_opt_inverse = compute_least_squares_inverse(YAT_sum, AAT_sum)
        print(f"  W_opt shapes: inverse={W_opt_inverse.shape}, true={W_opt_true.shape}")
        
        # Evaluate on ALL TRAINING DATA seen so far (batches 1 through batch_idx+1)
        # Aggregation: 1) median over traces, 2) median over configs, 3) mean over trace length, 4) mean over haystacks
        print(f"  Computing errors on ALL training data (batches 1-{batch_idx + 1})...")
        
        haystack_errors_inverse = []
        haystack_errors_true = []
        
        total_samples = 0
        
        for haystack_len in haystack_lengths:
            # For this haystack, collect errors per position per trace per config
            # Structure: config_idx -> trace_idx -> list of position errors (length ~= trace_len)
            config_trace_position_errors_inverse = {}
            config_trace_position_errors_true = {}
            
            for eval_batch_idx in range(batch_idx + 1):
                eval_batch_start = eval_batch_idx * configs_per_batch
                eval_batch_end = (eval_batch_idx + 1) * configs_per_batch
                eval_config_indices = list(range(eval_batch_start, eval_batch_end))
                
                activations, targets = load_haystack_file(haystack_len)
                
                for config_idx in eval_config_indices:
                    if config_idx not in config_trace_position_errors_inverse:
                        config_trace_position_errors_inverse[config_idx] = {}
                        config_trace_position_errors_true[config_idx] = {}
                    
                    batch_idx_key = 0
                    trace_indices = []
                    for key in activations.keys():
                        if key[0] == config_idx and key[1] == batch_idx_key:
                            trace_indices.append(key[2])
                    trace_indices = sorted(trace_indices)
                    
                    for trace_idx in trace_indices:
                        act_trace = activations[(config_idx, batch_idx_key, trace_idx)]
                        tgt_trace = targets[config_idx, batch_idx_key, trace_idx]
                        
                        trace_len = len(act_trace)
                        position_errors_inverse = []
                        position_errors_true = []
                        
                        for i in range(trace_len - 1):
                            activation = act_trace[i].astype(np.float64)
                            target = tgt_trace[i + 1, -5:].astype(np.float64)
                            
                            if not np.all(target == 0.0):
                                augmented = np.concatenate([activation, target])
                                rotated_feature = U @ augmented
                                
                                pred_inverse = rotated_feature @ W_opt_inverse.T
                                pred_true = rotated_feature @ W_opt_true.T
                                
                                se_inverse = np.sum((target - pred_inverse) ** 2)
                                se_true = np.sum((target - pred_true) ** 2)
                                
                                position_errors_inverse.append(se_inverse)
                                position_errors_true.append(se_true)
                                total_samples += 1
                        
                        # Store position errors for this trace
                        if len(position_errors_inverse) > 0:
                            config_trace_position_errors_inverse[config_idx][trace_idx] = position_errors_inverse
                            config_trace_position_errors_true[config_idx][trace_idx] = position_errors_true
                
                del activations, targets
            
            # Now aggregate for this haystack:
            # 1) Median over traces for each config and each position
            # Result: config_idx -> vector of length ~trace_len
            config_position_medians_inverse = {}
            config_position_medians_true = {}
            
            for config_idx in config_trace_position_errors_inverse.keys():
                traces_data_inverse = config_trace_position_errors_inverse[config_idx]
                traces_data_true = config_trace_position_errors_true[config_idx]
                
                # Determine max length across traces for this config
                max_len = max(len(traces_data_inverse[t]) for t in traces_data_inverse.keys())
                
                # For each position, collect errors across traces and take median
                position_medians_inverse = []
                position_medians_true = []
                
                for pos in range(max_len):
                    pos_errors_inverse = [traces_data_inverse[t][pos] for t in traces_data_inverse.keys() if pos < len(traces_data_inverse[t])]
                    pos_errors_true = [traces_data_true[t][pos] for t in traces_data_true.keys() if pos < len(traces_data_true[t])]
                    
                    if len(pos_errors_inverse) > 0:
                        position_medians_inverse.append(np.median(pos_errors_inverse))
                        position_medians_true.append(np.median(pos_errors_true))
                
                config_position_medians_inverse[config_idx] = position_medians_inverse
                config_position_medians_true[config_idx] = position_medians_true
            
            # 2) Median over configs for each position
            # Result: vector of length ~trace_len
            max_len_haystack = max(len(config_position_medians_inverse[c]) for c in config_position_medians_inverse.keys())
            
            haystack_position_medians_inverse = []
            haystack_position_medians_true = []
            
            for pos in range(max_len_haystack):
                pos_errors_inverse = [config_position_medians_inverse[c][pos] for c in config_position_medians_inverse.keys() if pos < len(config_position_medians_inverse[c])]
                pos_errors_true = [config_position_medians_true[c][pos] for c in config_position_medians_true.keys() if pos < len(config_position_medians_true[c])]
                
                if len(pos_errors_inverse) > 0:
                    haystack_position_medians_inverse.append(np.median(pos_errors_inverse))
                    haystack_position_medians_true.append(np.median(pos_errors_true))
            
            # 3) Mean over trace length (mean of the position vector)
            haystack_error_inverse = np.mean(haystack_position_medians_inverse)
            haystack_error_true = np.mean(haystack_position_medians_true)
            
            haystack_errors_inverse.append(haystack_error_inverse)
            haystack_errors_true.append(haystack_error_true)
        
        # 4) Mean over haystacks (final aggregation)
        mse_inverse_train = np.mean(haystack_errors_inverse)
        mse_true_train = np.mean(haystack_errors_true)
        
        print(f"    Total training samples: {total_samples:,}")
        
        # ALSO evaluate on LAST 10 CONFIGS (configs 40-49) as a held-out set
        # Use same aggregation method
        print(f"\n  Computing errors on LAST 10 CONFIGS (configs 40-49, held-out)...")
        
        last10_haystack_errors_inverse = []
        last10_haystack_errors_true = []
        last10_total_samples = 0
        
        for haystack_len in haystack_lengths:
            config_trace_position_errors_inverse = {}
            config_trace_position_errors_true = {}
            
            activations, targets = load_haystack_file(haystack_len)
            
            for config_idx in range(40, 50):
                config_trace_position_errors_inverse[config_idx] = {}
                config_trace_position_errors_true[config_idx] = {}
                
                batch_idx_key = 0
                trace_indices = []
                for key in activations.keys():
                    if key[0] == config_idx and key[1] == batch_idx_key:
                        trace_indices.append(key[2])
                trace_indices = sorted(trace_indices)
                
                for trace_idx in trace_indices:
                    act_trace = activations[(config_idx, batch_idx_key, trace_idx)]
                    tgt_trace = targets[config_idx, batch_idx_key, trace_idx]
                    
                    trace_len = len(act_trace)
                    position_errors_inverse = []
                    position_errors_true = []
                    
                    for i in range(trace_len - 1):
                        activation = act_trace[i].astype(np.float64)
                        target = tgt_trace[i + 1, -5:].astype(np.float64)
                        
                        if not np.all(target == 0.0):
                            augmented = np.concatenate([activation, target])
                            rotated_feature = U @ augmented
                            
                            pred_inverse = rotated_feature @ W_opt_inverse.T
                            pred_true = rotated_feature @ W_opt_true.T
                            
                            se_inverse = np.sum((target - pred_inverse) ** 2)
                            se_true = np.sum((target - pred_true) ** 2)
                            
                            position_errors_inverse.append(se_inverse)
                            position_errors_true.append(se_true)
                            last10_total_samples += 1
                    
                    if len(position_errors_inverse) > 0:
                        config_trace_position_errors_inverse[config_idx][trace_idx] = position_errors_inverse
                        config_trace_position_errors_true[config_idx][trace_idx] = position_errors_true
            
            del activations, targets
            
            # 1) Median over traces for each config and position
            config_position_medians_inverse = {}
            config_position_medians_true = {}
            
            for config_idx in config_trace_position_errors_inverse.keys():
                traces_data_inverse = config_trace_position_errors_inverse[config_idx]
                traces_data_true = config_trace_position_errors_true[config_idx]
                
                if len(traces_data_inverse) == 0:
                    continue
                    
                max_len = max(len(traces_data_inverse[t]) for t in traces_data_inverse.keys())
                
                position_medians_inverse = []
                position_medians_true = []
                
                for pos in range(max_len):
                    pos_errors_inverse = [traces_data_inverse[t][pos] for t in traces_data_inverse.keys() if pos < len(traces_data_inverse[t])]
                    pos_errors_true = [traces_data_true[t][pos] for t in traces_data_true.keys() if pos < len(traces_data_true[t])]
                    
                    if len(pos_errors_inverse) > 0:
                        position_medians_inverse.append(np.median(pos_errors_inverse))
                        position_medians_true.append(np.median(pos_errors_true))
                
                config_position_medians_inverse[config_idx] = position_medians_inverse
                config_position_medians_true[config_idx] = position_medians_true
            
            # 2) Median over configs for each position
            if len(config_position_medians_inverse) == 0:
                continue
                
            max_len_haystack = max(len(config_position_medians_inverse[c]) for c in config_position_medians_inverse.keys())
            
            haystack_position_medians_inverse = []
            haystack_position_medians_true = []
            
            for pos in range(max_len_haystack):
                pos_errors_inverse = [config_position_medians_inverse[c][pos] for c in config_position_medians_inverse.keys() if pos < len(config_position_medians_inverse[c])]
                pos_errors_true = [config_position_medians_true[c][pos] for c in config_position_medians_true.keys() if pos < len(config_position_medians_true[c])]
                
                if len(pos_errors_inverse) > 0:
                    haystack_position_medians_inverse.append(np.median(pos_errors_inverse))
                    haystack_position_medians_true.append(np.median(pos_errors_true))
            
            # 3) Mean over trace length
            haystack_error_inverse = np.mean(haystack_position_medians_inverse)
            haystack_error_true = np.mean(haystack_position_medians_true)
            
            last10_haystack_errors_inverse.append(haystack_error_inverse)
            last10_haystack_errors_true.append(haystack_error_true)
        
        # 4) Mean over haystacks
        mse_inverse_last10 = np.mean(last10_haystack_errors_inverse)
        mse_true_last10 = np.mean(last10_haystack_errors_true)
        
        print(f"    Last 10 configs samples: {last10_total_samples:,}")
        
        # Compute relative errors vs true W_opt
        W_error_inverse = np.linalg.norm(W_opt_inverse - W_opt_true, 'fro')
        W_norm_true = np.linalg.norm(W_opt_true, 'fro')
        W_rel_error_inverse = W_error_inverse / W_norm_true
        
        print(f"\n  Training MSE (on all data seen so far, batches 1-{batch_idx + 1}):")
        print(f"    Inverse method:  {mse_inverse_train:.6e}")
        print(f"    True W_opt:      {mse_true_train:.6e}")
        
        print(f"\n  Held-out MSE (on last 10 configs, configs 40-49):")
        print(f"    Inverse method:  {mse_inverse_last10:.6e}")
        print(f"    True W_opt:      {mse_true_last10:.6e}")
        
        print(f"\n  W_opt relative error vs true:")
        print(f"    Inverse method:  {W_rel_error_inverse:.6e} ({W_rel_error_inverse:.2%})")
        
        print(f"\n  Frobenius norm of W difference vs true:")
        print(f"    Inverse method:  {W_error_inverse:.6e}")
        
        # Append results to file
        print(f"  Saving results...")
        
        append_results_to_file(
            batch_num=batch_idx + 1,
            n_samples=total_samples,
            mse_inverse_train=mse_inverse_train,
            mse_true_train=mse_true_train,
            mse_inverse_last10=mse_inverse_last10,
            mse_true_last10=mse_true_last10,
            w_err_inv=W_rel_error_inverse * 100,  # Convert to percentage
            w_frob_inv=W_error_inverse  # Frobenius norm
        )
        
        print(f"  Results saved (batch {batch_idx + 1} complete)")
    
    print(f"\n{'='*80}")
    print("TEST COMPLETE!")
    print(f"{'='*80}\n")
    
    print("Summary:")
    print(f"  Total batches processed: {num_batches}")
    print(f"  Total configs processed: {num_batches * configs_per_batch * len(haystack_lengths)}")
    print(f"  Results saved to: cheat_train_error_results.txt")
    
    # Read final results from file
    results_file = "cheat_train_error_results.txt"
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:  # Skip header
                last_line = lines[-1].strip().split(',')
                if len(last_line) >= 6:
                    print(f"\nFinal batch results (from file):")
                    print(f"  Batch: {last_line[0]}")
                    print(f"  Total samples: {last_line[1]}")
                    print(f"  MSE - Inverse: {last_line[2]}")
                    print(f"  MSE - True: {last_line[3]}")
                    print(f"  W Error - Inverse: {float(last_line[4]):.2f}%")
                    print(f"  W Frobenius Norm - Inverse: {last_line[5]}")
    
    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()

