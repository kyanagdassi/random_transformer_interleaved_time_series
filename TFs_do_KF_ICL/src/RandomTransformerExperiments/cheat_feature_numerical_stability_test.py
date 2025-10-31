"""
Cheat Feature Numerical Stability Test

This test checks if numerical instability is causing least squares to fail.

Strategy:
1. Augment each 128-dim activation with its 5-dim target → 133-dim vector [activation, target]
2. Apply a single random orthogonal rotation U ∈ R^{133×133} to ALL augmented vectors
3. Train least squares on these rotated features (configs 0-39 from all 6 haystacks)
4. Measure TRAINING error on the same data

Expected result:
- Since the target is embedded in the features (after rotation), a perfect least squares 
  solution should exist (training MSE ≈ 0)
- If we get high training error, it indicates numerical instability in the least squares solver
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
    
    print(f"    Extracting cheat features from {len(config_indices)} configs...")
    
    with tqdm(total=total_traces, desc="  Extracting pairs", leave=False) as pbar:
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

def compute_least_squares_lstsq(A, Y):
    """
    Compute least squares using numpy's lstsq solver directly on data.
    
    With samples as rows:
    - A is [n_samples, n_features]
    - Y is [n_samples, n_target]
    - Solves: A @ W^T = Y
    - lstsq(A, Y) gives W^T, so we transpose to get W
    
    Args:
        A: Feature matrix [n_samples, n_features]
        Y: Target matrix [n_samples, n_target]
    
    Returns:
        W: Weight matrix [n_target, n_features]
    """
    W_T, residuals, rank, s = np.linalg.lstsq(A, Y, rcond=None)
    W = W_T.T  # [n_target, n_features]
    return W

def compute_simple_mse(A_list, Y_list, W):
    """
    Compute simple MSE averaged over all batches.
    
    At batch i:
    - For W_inverse and W_lstsq: Compute MSE on batches 1..i, average them
    - For W_true: Only compute MSE on batch i (since W_true doesn't change)
    
    Args:
        A_list: List of A matrices for each batch
        Y_list: List of Y matrices for each batch
        W: Weight matrix [5, 133]
    
    Returns:
        avg_mse: Average MSE over all batches
    """
    batch_mses = []
    
    for A_batch, Y_batch in zip(A_list, Y_list):
        Y_pred = A_batch @ W.T
        mse = np.mean((Y_batch - Y_pred) ** 2)
        batch_mses.append(mse)
    
    return np.mean(batch_mses) if len(batch_mses) > 0 else 0.0

def load_checkpoint():
    """Load checkpoint to resume from last completed batch"""
    checkpoint_file = "stability_checkpoint.txt"
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            last_batch = int(f.read().strip())
        print(f"Found checkpoint: resuming from batch {last_batch + 1}")
        return last_batch
    return -1


def save_checkpoint(batch_idx):
    """Save checkpoint after completing a batch"""
    checkpoint_file = "stability_checkpoint.txt"
    with open(checkpoint_file, 'w') as f:
        f.write(str(batch_idx))


def append_results_to_file(batch_num, mse_inverse, mse_lstsq, mse_true, w_err_inv, w_err_lstsq, n_samples):
    """Append results for a single batch to the results file"""
    results_file = "stability_results.txt"
    
    # Create header if file doesn't exist
    if not os.path.exists(results_file):
        with open(results_file, 'w') as f:
            f.write("batch,n_samples,mse_inverse,mse_lstsq,mse_true,w_error_inverse_pct,w_error_lstsq_pct\n")
    
    # Append this batch's results
    with open(results_file, 'a') as f:
        f.write(f"{batch_num},{n_samples},{mse_inverse:.6e},{mse_lstsq:.6e},{mse_true:.6e},{w_err_inv:.6e},{w_err_lstsq:.6e}\n")


def main():
    print("\n" + "="*80)
    print("CHEAT FEATURE NUMERICAL STABILITY TEST")
    print("="*80 + "\n")
    
    print("This test embeds targets into features via orthogonal rotation.")
    print("If least squares is numerically stable, training MSE should be near zero.\n")
    
    # Check for checkpoint
    last_completed_batch = load_checkpoint()
    
    # Set random seed for reproducibility
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
    # X extracts the last 5 dimensions (positions 128-132) which contain the target
    X = np.zeros((n_target, n_features), dtype=np.float64)
    for i in range(n_target):
        X[i, n_activation + i] = 1.0  # Set diagonal to extract target positions
    
    # With samples as rows layout:
    # - A is [n_samples, 133], Y is [n_samples, 5]
    # - W is [n_target, n_features] = [5, 133]
    # - Prediction: Y_pred = A @ W^T
    # For cheat feature: augmented = [activation, target], rotated = U @ augmented
    # To extract target from rotated: W @ rotated = target
    # Since rotated = U @ [activation, target], and X extracts target from original
    # W_true = X @ U^T (shape [5, 133])
    W_opt_true = X @ U.T  
    print(f"  Ground truth W_opt shape: {W_opt_true.shape}\n")
    
    # Note: We'll load haystacks on-demand to save memory
    print("Haystack files will be loaded on-demand to save memory.\n")
    
    # Training in batches: 20 batches × 1 config per haystack
    num_batches = 20
    configs_per_batch = 1
    
    print(f"Training with incremental least squares:")
    print(f"  Total batches: {num_batches}")
    print(f"  Configs per batch per haystack: {configs_per_batch}")
    print(f"  Total training configs: {num_batches * configs_per_batch * len(haystack_lengths)}\n")
    
    # Initialize accumulated matrices
    YAT_sum = np.zeros((n_target, n_features), dtype=np.float64)
    AAT_sum = np.zeros((n_features, n_features), dtype=np.float64)
    
    # Store results for plotting (3 methods)
    batch_numbers = []
    training_mse_inverse_list = []    # Using matrix inverse
    training_mse_lstsq_list = []      # Using numpy lstsq
    training_mse_true_list = []       # Using true W_opt
    w_relative_error_inverse_list = []
    w_relative_error_lstsq_list = []
    
    # Store all batch data (list of batches, each batch has concatenated data from all 6 haystacks)
    all_batch_A_list = []
    all_batch_Y_list = []
    
    # For lstsq, need all data concatenated
    all_A_concat_list = []
    all_Y_concat_list = []
    
    # If resuming, need to rebuild accumulated matrices
    if last_completed_batch >= 0:
        print(f"\nWARNING: Resuming requires reprocessing batches 1-{last_completed_batch + 1} to rebuild YAT_sum, AAT_sum")
        print(f"   (This is necessary for the incremental inverse method)")
        print(f"   Processing will be fast (no lstsq computation for already-completed batches)\n")
        
        for rebuild_idx in range(last_completed_batch + 1):
            batch_start = rebuild_idx * configs_per_batch
            batch_end = (rebuild_idx + 1) * configs_per_batch
            
            print(f"  Rebuilding batch {rebuild_idx + 1}...")
            
            rebuild_batch_A_all = []
            rebuild_batch_Y_all = []
            
            for haystack_len in haystack_lengths:
                batch_config_indices = list(range(batch_start, batch_end))
                activations, targets = load_haystack_file(haystack_len)
                A_batch, Y_batch = extract_cheat_feature_pairs(
                    activations, targets, batch_config_indices, U, haystack_len
                )
                del activations, targets
                
                # Accumulate
                YAT_sum += Y_batch.T @ A_batch
                AAT_sum += A_batch.T @ A_batch
                all_A_concat_list.append(A_batch)
                all_Y_concat_list.append(Y_batch)
                
                rebuild_batch_A_all.append(A_batch)
                rebuild_batch_Y_all.append(Y_batch)
            
            # Store this batch
            rebuild_batch_concat = np.vstack(rebuild_batch_A_all)
            rebuild_Y_concat = np.vstack(rebuild_batch_Y_all)
            all_batch_A_list.append(rebuild_batch_concat)
            all_batch_Y_list.append(rebuild_Y_concat)
        
        print(f"  Rebuilt accumulators through batch {last_completed_batch + 1}\n")
    
    # Progressive training
    for batch_idx in range(num_batches):
        # Skip batches that were already completed
        if batch_idx <= last_completed_batch:
            continue
        
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
            
            # Load haystack on-demand (to save memory)
            activations, targets = load_haystack_file(haystack_len)
            
            # Extract cheat feature pairs using ACTUAL GPT2 activations
            A_batch, Y_batch = extract_cheat_feature_pairs(
                activations,
                targets,
                batch_config_indices,
                U,
                haystack_len
            )
            
            # Delete haystack data to free memory
            del activations, targets
            
            print(f"    Haystack {haystack_len}: {A_batch.shape[0]:,} samples")
            
            batch_A_all.append(A_batch)
            batch_Y_all.append(Y_batch)
            
            # ADD to accumulated Y^T @ A and A^T @ A
            YAT_sum += Y_batch.T @ A_batch  # [n_target, n_features]
            AAT_sum += A_batch.T @ A_batch  # [n_features, n_features]
            
            # Store for lstsq (need all data)
            all_A_concat_list.append(A_batch)
            all_Y_concat_list.append(Y_batch)
        
        # Concatenate current batch data (vertically - stacking rows)
        A_batch_concat = np.vstack(batch_A_all)
        Y_batch_concat = np.vstack(batch_Y_all)
        
        # Store this batch for computing average error over batches 1..i
        all_batch_A_list.append(A_batch_concat)
        all_batch_Y_list.append(Y_batch_concat)
        
        print(f"\nAfter batch {batch_idx + 1}:")
        print(f"  YAT_sum shape: {YAT_sum.shape}")
        print(f"  AAT_sum shape: {AAT_sum.shape}")
        
        # Compute least squares solution using THREE methods
        print(f"  Computing W_opt using 3 methods...")
        
        # Method 1: Matrix inverse (using accumulated YAT, AAT)
        W_opt_inverse = compute_least_squares_inverse(YAT_sum, AAT_sum)
        
        # Method 2: Numpy lstsq (using ALL accumulated data)
        A_all = np.vstack(all_A_concat_list)
        Y_all = np.vstack(all_Y_concat_list)
        print(f"  Running lstsq on ALL data so far: {A_all.shape[0]:,} samples...")
        W_opt_lstsq = compute_least_squares_lstsq(A_all, Y_all)
        
        # Method 3: True W_opt (ground truth)
        # W_opt_true is already computed at the start
        
        print(f"  W_opt shapes: inverse={W_opt_inverse.shape}, lstsq={W_opt_lstsq.shape}, true={W_opt_true.shape}")
        
        # Compute errors
        print(f"  Computing errors...")
        
        # For W_inverse and W_lstsq: Average MSE over batches 1..i (they change each batch)
        mse_inverse = compute_simple_mse(all_batch_A_list, all_batch_Y_list, W_opt_inverse)
        mse_lstsq = compute_simple_mse(all_batch_A_list, all_batch_Y_list, W_opt_lstsq)
        
        # For W_true: Only compute on current batch (it doesn't change)
        Y_pred_true = A_batch_concat @ W_opt_true.T
        mse_true = np.mean((Y_batch_concat - Y_pred_true) ** 2)
        
        # Compute relative errors vs true W_opt
        W_error_inverse = np.linalg.norm(W_opt_inverse - W_opt_true, 'fro')
        W_error_lstsq = np.linalg.norm(W_opt_lstsq - W_opt_true, 'fro')
        W_norm_true = np.linalg.norm(W_opt_true, 'fro')
        W_rel_error_inverse = W_error_inverse / W_norm_true
        W_rel_error_lstsq = W_error_lstsq / W_norm_true
        
        print(f"\n  Training MSE:")
        print(f"    Inverse method:  {mse_inverse:.6e}")
        print(f"    Lstsq method:    {mse_lstsq:.6e}")
        print(f"    True W_opt:      {mse_true:.6e}")
        
        print(f"\n  W_opt relative error vs true:")
        print(f"    Inverse method:  {W_rel_error_inverse:.6e} ({W_rel_error_inverse:.2%})")
        print(f"    Lstsq method:    {W_rel_error_lstsq:.6e} ({W_rel_error_lstsq:.2%})")
        
        # Store results for plotting
        batch_numbers.append(total_configs_so_far)
        training_mse_inverse_list.append(mse_inverse)
        training_mse_lstsq_list.append(mse_lstsq)
        training_mse_true_list.append(mse_true)
        w_relative_error_inverse_list.append(W_rel_error_inverse)
        w_relative_error_lstsq_list.append(W_rel_error_lstsq)
        
        # Append results to file immediately after each batch
        print(f"  Saving results...")
        append_results_to_file(
            batch_num=batch_idx + 1,
            mse_inverse=mse_inverse,
            mse_lstsq=mse_lstsq,
            mse_true=mse_true,
            w_err_inv=W_rel_error_inverse * 100,  # Convert to percentage
            w_err_lstsq=W_rel_error_lstsq * 100,  # Convert to percentage
            n_samples=len(A_all)
        )
        
        # Save checkpoint
        save_checkpoint(batch_idx)
        print(f"  Checkpoint saved (batch {batch_idx + 1} complete)")
        
        # Free memory from this batch
        del A_batch_concat, Y_batch_concat, batch_A_all, batch_Y_all
    
    # Clean up checkpoint file after successful completion
    if os.path.exists("stability_checkpoint.txt"):
        os.remove("stability_checkpoint.txt")
    
    print(f"\n{'='*80}")
    print("TEST COMPLETE!")
    print(f"{'='*80}\n")
    
    print("Summary:")
    print(f"  Total batches: {len(batch_numbers)}")
    print(f"  Total configs processed: {num_batches * configs_per_batch * len(haystack_lengths)}")
    print(f"  Results saved to: stability_results.txt")
    
    if len(batch_numbers) > 0:
        print(f"\nFinal batch results:")
        print(f"  MSE - Inverse: {training_mse_inverse_list[-1]:.6e}")
        print(f"  MSE - Lstsq: {training_mse_lstsq_list[-1]:.6e}")
        print(f"  MSE - True: {training_mse_true_list[-1]:.6e}")
        print(f"  W Error - Inverse: {w_relative_error_inverse_list[-1]:.2%}")
        print(f"  W Error - Lstsq: {w_relative_error_lstsq_list[-1]:.2%}")
        

    
    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()

