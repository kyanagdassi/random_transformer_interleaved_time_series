"""
Compare GPT2 Pure vs Randomly Initialized GPT2 (seed=0): Train on last 10 predictions,
test on last 10 configs.

Training data: Last 10 valid predictions per trace from configs 0-39 of haystacks 1-6
Test data: Configs 40-49 of haystacks 1-6 (last 10 configs per haystack)
Training: 20 batches, 2 configs per haystack per batch (12 configs total per batch)
Random model: Outer layer replaced by least squares solution (features from random model)

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


def load_activations(haystack_len, cache_tag="pure"):
    """Load cached activations for a haystack under given cache tag."""
    if cache_tag == "pure":
        filename = f"activations_haystack_{haystack_len}.pkl"
    else:
        filename = f"activations_{cache_tag}_haystack_{haystack_len}.pkl"

    filepath = os.path.join(
        os.path.dirname(__file__),
        "Training:Test Data",
        "Haystack",
        "Activations",
        filename,
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
    Extract the LAST 10 valid prediction pairs from each trace using provided activations.

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
    """
    if not np.isfinite(YAT_sum).all():
        print("  WARNING: YAT_sum contains inf/nan! Cleaning...")
        YAT_sum = np.nan_to_num(YAT_sum, nan=0.0, posinf=1e10, neginf=-1e10)

    if not np.isfinite(AAT_sum).all():
        print("  WARNING: AAT_sum contains inf/nan! Cleaning...")
        AAT_sum = np.nan_to_num(AAT_sum, nan=0.0, posinf=1e10, neginf=-1e10)

    AAT_inv = np.linalg.inv(AAT_sum)

    if not np.isfinite(AAT_inv).all():
        print("  WARNING: AAT_inv contains inf/nan after inversion!")
        AAT_inv = np.nan_to_num(AAT_inv, nan=0.0, posinf=1e10, neginf=-1e10)

    W = YAT_sum @ AAT_inv

    if not np.isfinite(W).all():
        print("  WARNING: W contains inf/nan after computation!")
        W = np.nan_to_num(W, nan=0.0, posinf=1e10, neginf=-1e10)

    return W


def create_random_model(config):
    """Instantiate a randomly initialized GPT2 model (seed=0) in float64."""
    torch.manual_seed(0)
    np.random.seed(0)

    model_random = GPT2(
        n_dims_in=config.n_dims_in,
        n_positions=config.n_positions,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_dims_out=config.n_dims_out,
    )

    model_random.double()
    return model_random


def evaluate_models_on_test_configs(model_pure, model_random, all_data,
                                    activations_pure, activations_random,
                                    haystack_lengths, k_values, device='cpu'):
    """
    Evaluate both models on test data (configs 40-49 from all haystacks), using the
    activation caches corresponding to each model.
    """
    model_pure.eval()
    model_random.eval()
    model_pure.double().to(device)
    model_random.double().to(device)

    test_config_indices = list(range(40, 50))

    all_config_medians_pure = {f'{k}_after_initial': [] for k in k_values}
    all_config_medians_pure.update({f'{k}_after_final': [] for k in k_values})

    all_config_medians_random = {f'{k}_after_initial': [] for k in k_values}
    all_config_medians_random.update({f'{k}_after_final': [] for k in k_values})

    for haystack_len in haystack_lengths:
        data = all_data[haystack_len]
        activations_dict_pure = activations_pure[haystack_len]
        activations_dict_random = activations_random[haystack_len]
        multi_sys_ys = data['multi_sys_ys']

        for config_idx in test_config_indices:
            config_errors_pure = {f'{k}_after_initial': [] for k in k_values}
            config_errors_pure.update({f'{k}_after_final': [] for k in k_values})

            config_errors_random = {f'{k}_after_initial': [] for k in k_values}
            config_errors_random.update({f'{k}_after_final': [] for k in k_values})

            for trace_idx in range(multi_sys_ys.shape[1]):
                for example_idx in range(multi_sys_ys.shape[2]):
                    trace = multi_sys_ys[config_idx, trace_idx, example_idx]
                    activations_p = activations_dict_pure[(config_idx, trace_idx, example_idx)]
                    activations_r = activations_dict_random[(config_idx, trace_idx, example_idx)]

                    open_brackets = find_open_brackets(trace)

                    if len(open_brackets) == 0:
                        continue

                    first_bracket_pos = min(open_brackets.keys())
                    first_bracket_idx = open_brackets[first_bracket_pos]

                    same_bracket_positions = [pos for pos, idx in open_brackets.items()
                                              if idx == first_bracket_idx]

                    if len(same_bracket_positions) == 0:
                        continue

                    initial_pos = same_bracket_positions[0]
                    final_pos = same_bracket_positions[-1]

                    for k in k_values:
                        eval_pos_initial = initial_pos + k
                        if eval_pos_initial < len(trace) and trace[eval_pos_initial, 51] != 0:
                            activation_pure = activations_p[eval_pos_initial - 1].astype(np.float64)
                            activation_random = activations_r[eval_pos_initial - 1].astype(np.float64)
                            target = trace[eval_pos_initial, -5:].astype(np.float64)

                            with torch.no_grad():
                                act_pure_tensor = torch.from_numpy(activation_pure).double().to(device)
                                act_random_tensor = torch.from_numpy(activation_random).double().to(device)

                                pred_pure = model_pure._read_out(act_pure_tensor).detach().cpu().numpy()
                                pred_random = model_random._read_out(act_random_tensor).detach().cpu().numpy()

                            se_pure = np.sum((pred_pure - target) ** 2)
                            se_random = np.sum((pred_random - target) ** 2)

                            config_errors_pure[f'{k}_after_initial'].append(se_pure)
                            config_errors_random[f'{k}_after_initial'].append(se_random)

                        eval_pos_final = final_pos + k
                        if eval_pos_final < len(trace) and trace[eval_pos_final, 51] != 0:
                            activation_pure = activations_p[eval_pos_final - 1].astype(np.float64)
                            activation_random = activations_r[eval_pos_final - 1].astype(np.float64)
                            target = trace[eval_pos_final, -5:].astype(np.float64)

                            with torch.no_grad():
                                act_pure_tensor = torch.from_numpy(activation_pure).double().to(device)
                                act_random_tensor = torch.from_numpy(activation_random).double().to(device)

                                pred_pure = model_pure._read_out(act_pure_tensor).detach().cpu().numpy()
                                pred_random = model_random._read_out(act_random_tensor).detach().cpu().numpy()

                            se_pure = np.sum((pred_pure - target) ** 2)
                            se_random = np.sum((pred_random - target) ** 2)

                            config_errors_pure[f'{k}_after_final'].append(se_pure)
                            config_errors_random[f'{k}_after_final'].append(se_random)

            for metric in config_errors_pure.keys():
                if len(config_errors_pure[metric]) > 0:
                    all_config_medians_pure[metric].append(np.median(config_errors_pure[metric]))
                    all_config_medians_random[metric].append(np.median(config_errors_random[metric]))

    aggregated_pure = {metric: np.median(values) for metric, values in all_config_medians_pure.items()}
    aggregated_random = {metric: np.median(values) for metric, values in all_config_medians_random.items()}

    return aggregated_pure, aggregated_random


def save_results_to_text(results_dict, k_values, output_path, batch_num=None):
    """
    Save results to a text file.

    Args:
        results_dict: Dictionary mapping num_train_configs -> {'pure': {...}, 'random': {...}}
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
            random_results = results_dict[n_train_configs]['random']

            f.write("\nk_after_initial:\n")
            for k in k_values:
                metric = f'{k}_after_initial'
                f.write(f"  k={k}: Pure={pure_results[metric]:.8f}, Random={random_results[metric]:.8f}\n")

            f.write("\nk_after_final:\n")
            for k in k_values:
                metric = f'{k}_after_final'
                f.write(f"  k={k}: Pure={pure_results[metric]:.8f}, Random={random_results[metric]:.8f}\n")


def main():
    print("\n" + "="*80)
    print("TRAIN ON LAST 10, TEST ON LAST 10 CONFIGS (RANDOM MODEL)")
    print("="*80 + "\n")

    config = Config()
    config.override("model_type", "GPT2")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    haystack_lengths = [1, 2, 3, 4, 5, 6]
    k_values = [1, 2, 3, 4, 5, 6, 7, 8]

    print("Loading pretrained GPT2 model for baseline...")
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
    model_pure.double()
    print("Pure model loaded and converted to float64\n")

    print("Instantiating randomly initialized GPT2 model (seed=0)...")
    model_random = create_random_model(config)
    print("Random model ready\n")

    n_embd = config.n_embd
    n_dims_out = config.n_dims_out

    print("="*80)
    print("LOADING DATA AND ACTIVATIONS FOR ALL HAYSTACKS")
    print("="*80 + "\n")

    all_data = {}
    all_activations_pure = {}
    all_activations_random = {}

    for haystack_len in haystack_lengths:
        print(f"Loading haystack {haystack_len}...")
        all_data[haystack_len] = load_haystack_file(haystack_len)
        all_activations_pure[haystack_len] = load_activations(haystack_len, cache_tag="pure")
        all_activations_random[haystack_len] = load_activations(haystack_len, cache_tag="random_seed0_fresh")

    print("\nAll data and activations loaded\n")

    results_dict = {}

    output_dir = os.path.join(os.path.dirname(__file__), "test_last10configs_random_results")
    os.makedirs(output_dir, exist_ok=True)

    text_path = os.path.join(output_dir, 'test_last10configs_random_results.txt')
    if os.path.exists(text_path):
        os.remove(text_path)
        print(f"Cleared existing results file\n")

    num_batches = 20
    configs_per_batch = 2

    print(f"\n{'='*80}")
    print("TRAINING WITH LAST 10 PREDICTIONS PER TRACE (RANDOM MODEL)")
    print("="*80 + "\n")
    print(f"Total batches: {num_batches}")
    print(f"Configs per batch per haystack: {configs_per_batch}")
    print(f"Total configs per batch: {configs_per_batch * len(haystack_lengths)} (across all haystacks)")
    print(f"Total training configs: {num_batches * configs_per_batch * len(haystack_lengths)}\n")

    YAT_sum = np.zeros((n_dims_out, n_embd), dtype=np.float64)
    AAT_sum = np.zeros((n_embd, n_embd), dtype=np.float64)

    for batch_idx in range(num_batches):
        batch_start = batch_idx * configs_per_batch
        batch_end = (batch_idx + 1) * configs_per_batch
        total_configs_so_far = batch_end * len(haystack_lengths)

        print(f"\n{'='*80}")
        print(f"BATCH {batch_idx + 1}/20: Processing NEW configs {batch_start}-{batch_end-1} per haystack")
        print(f"Total configs so far: {batch_end} per haystack ({total_configs_so_far} across all)")
        print(f"{'='*80}\n")

        for haystack_len in haystack_lengths:
            batch_config_indices = list(range(batch_start, batch_end))

            print(f"Haystack {haystack_len}: Extracting LAST 10 predictions from NEW configs {batch_config_indices}...")

            A_batch, Y_batch = extract_last10_training_pairs(
                all_data[haystack_len],
                all_activations_random[haystack_len],
                batch_config_indices
            )

            print(f"  Extracted {A_batch.shape[1]} samples from haystack {haystack_len}")

            YAT_sum += Y_batch @ A_batch.T
            AAT_sum += A_batch @ A_batch.T

        print(f"\nAfter batch {batch_idx + 1}:")
        print(f"  Total training configs used: {total_configs_so_far}")
        print(f"  YAT_sum shape: {YAT_sum.shape}")
        print(f"  AAT_sum shape: {AAT_sum.shape}")

        YAT_sum = np.nan_to_num(YAT_sum, nan=0.0, posinf=0.0, neginf=0.0)
        AAT_sum = np.nan_to_num(AAT_sum, nan=0.0, posinf=0.0, neginf=0.0)

        print(f"  Computing least squares solution for random model...")
        W_opt = compute_least_squares_incremental(YAT_sum, AAT_sum)
        print(f"  W_opt shape: {W_opt.shape}")

        with torch.no_grad():
            model_random._read_out.weight.data = torch.from_numpy(W_opt).double()
            if isinstance(model_random._read_out, nn.Linear) and model_random._read_out.bias is not None:
                model_random._read_out.bias.data.zero_()

        print(f"\n  Evaluating on test data (configs 40-49 from all haystacks)...")

        aggregated_pure, aggregated_random = evaluate_models_on_test_configs(
            model_pure,
            model_random,
            all_data,
            all_activations_pure,
            all_activations_random,
            haystack_lengths,
            k_values,
            device
        )

        results_dict[total_configs_so_far] = {
            'pure': aggregated_pure,
            'random': aggregated_random,
        }

        print(f"\n  Results after {total_configs_so_far} total training configs:")
        print("  k_after_initial:")
        for k in k_values:
            pure_init = aggregated_pure[f'{k}_after_initial']
            rand_init = aggregated_random[f'{k}_after_initial']
            print(f"    {k}: Pure={pure_init:.6f}, Random={rand_init:.6f}")

        print("  k_after_final:")
        for k in k_values:
            pure_final = aggregated_pure[f'{k}_after_final']
            rand_final = aggregated_random[f'{k}_after_final']
            print(f"    {k}: Pure={pure_final:.6f}, Random={rand_final:.6f}")

        print(f"\n  Saving results to text file...")
        save_results_to_text(results_dict, k_values, text_path, batch_num=batch_idx + 1)

    print(f"\n{'='*80}")
    print("SAVING FINAL RESULTS")
    print(f"{'='*80}\n")

    results_path = os.path.join(output_dir, "test_last10configs_random_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(results_dict, f)
    print(f"Results saved to: {results_path}")

    print(f"Text results saved to: {text_path}")

    print(f"\n{'='*80}")
    print("COMPARISON COMPLETE!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

