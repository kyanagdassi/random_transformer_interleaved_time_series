"""
Compare GPT2 Pure vs GPT2 Modified on TRAINING data when training on ALL valid
predictions (full traces).

Training data: All valid prediction pairs from configs 0-39 of haystacks 1-6
Training: 20 batches, 2 configs per haystack per batch (12 configs total per batch)
Modified model: Outer layer replaced by inverse-method (least squares) solution

Evaluation: On training data accumulated so far (batches 1 through i)
Error at k=1..8 after initial/final bracket
Aggregation: median over traces -> median over all configs (across haystacks)
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import numpy as np
import torch
from tqdm import tqdm
from copy import deepcopy

from core import Config
from models import GPT2


def load_haystack_file(haystack_len):
    filepath = os.path.join(
        os.path.dirname(__file__),
        "Training:Test Data",
        "Haystack",
        f"val_interleaved_traces_ortho_haar_ident_C_haystack_len_{haystack_len}.pkl",
    )
    with open(filepath, "rb") as f:
        return pickle.load(f)


def load_activations(haystack_len):
    filepath = os.path.join(
        os.path.dirname(__file__),
        "Training:Test Data",
        "Haystack",
        "Activations",
        f"activations_haystack_{haystack_len}.pkl",
    )
    with open(filepath, "rb") as f:
        return pickle.load(f)


def find_open_brackets(trace):
    open_brackets = {}
    even_indices = np.arange(0, 50, 2)
    for pos in range(len(trace)):
        for idx in even_indices:
            if trace[pos, idx] != 0:
                open_brackets[pos] = idx
                break
    return open_brackets


def extract_full_training_pairs(data, activations_dict, config_indices):
    """Extract ALL valid (activation, target) pairs from the selected configs."""

    multi_sys_ys = data["multi_sys_ys"]
    A_columns = []
    Y_columns = []

    total_traces = len(config_indices) * multi_sys_ys.shape[1] * multi_sys_ys.shape[2]

    with tqdm(total=total_traces, desc="Extracting full-trace pairs", leave=False) as pbar:
        for config_idx in config_indices:
            for trace_idx in range(multi_sys_ys.shape[1]):
                for example_idx in range(multi_sys_ys.shape[2]):
                    trace = multi_sys_ys[config_idx, trace_idx, example_idx]
                    activations = activations_dict[(config_idx, trace_idx, example_idx)]

                    for i in range(len(trace) - 1):
                        if trace[i + 1, 51] == 0:
                            continue
                        activation = activations[i].astype(np.float64)
                        target = trace[i + 1, -5:].astype(np.float64)
                        A_columns.append(activation)
                        Y_columns.append(target)

                    pbar.update(1)

    A = np.column_stack(A_columns)
    Y = np.column_stack(Y_columns)

    if not np.isfinite(A).all():
        print("  WARNING: A contains inf/nan values! Replacing with 0.")
        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

    if not np.isfinite(Y).all():
        print("  WARNING: Y contains inf/nan values! Replacing with 0.")
        Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

    return A, Y


def compute_least_squares_incremental(YAT_sum, AAT_sum):
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


def create_modified_model(model_pure, W):
    model_modified = deepcopy(model_pure)
    model_modified._read_out.weight.data = torch.from_numpy(W).double()
    model_modified._read_out.bias.data = torch.zeros_like(
        model_modified._read_out.bias.data
    ).double()
    return model_modified


def evaluate_models_on_training_data(
    model_pure,
    model_modified,
    all_data,
    all_activations,
    haystack_lengths,
    batches_so_far,
    k_values,
    device="cpu",
):
    model_pure.eval()
    model_modified.eval()
    model_pure.double().to(device)
    model_modified.double().to(device)

    train_config_indices = list(range(batches_so_far * 2))

    all_config_medians_pure = {f"{k}_after_initial": [] for k in k_values}
    all_config_medians_pure.update({f"{k}_after_final": [] for k in k_values})

    all_config_medians_modified = {f"{k}_after_initial": [] for k in k_values}
    all_config_medians_modified.update({f"{k}_after_final": [] for k in k_values})

    for haystack_len in haystack_lengths:
        data = all_data[haystack_len]
        activations_dict = all_activations[haystack_len]
        multi_sys_ys = data["multi_sys_ys"]

        for config_idx in train_config_indices:
            config_errors_pure = {f"{k}_after_initial": [] for k in k_values}
            config_errors_pure.update({f"{k}_after_final": [] for k in k_values})

            config_errors_modified = {f"{k}_after_initial": [] for k in k_values}
            config_errors_modified.update({f"{k}_after_final": [] for k in k_values})

            for trace_idx in range(multi_sys_ys.shape[1]):
                for example_idx in range(multi_sys_ys.shape[2]):
                    trace = multi_sys_ys[config_idx, trace_idx, example_idx]
                    activations = activations_dict[(config_idx, trace_idx, example_idx)]

                    open_brackets = find_open_brackets(trace)
                    if not open_brackets:
                        continue

                    first_bracket_pos = min(open_brackets.keys())
                    first_bracket_idx = open_brackets[first_bracket_pos]
                    same_bracket_positions = [
                        pos for pos, idx in open_brackets.items() if idx == first_bracket_idx
                    ]
                    if not same_bracket_positions:
                        continue

                    initial_pos = same_bracket_positions[0]
                    final_pos = same_bracket_positions[-1]

                    for k in k_values:
                        eval_pos_initial = initial_pos + k
                        if (
                            eval_pos_initial < len(trace)
                            and trace[eval_pos_initial, 51] != 0
                        ):
                            activation = activations[eval_pos_initial - 1].astype(np.float64)
                            target = trace[eval_pos_initial, -5:].astype(np.float64)

                            with torch.no_grad():
                                activation_tensor = (
                                    torch.from_numpy(activation).double().to(device)
                                )
                                pred_pure = (
                                    model_pure._read_out(activation_tensor)
                                    .detach()
                                    .cpu()
                                    .numpy()
                                )
                                pred_mod = (
                                    model_modified._read_out(activation_tensor)
                                    .detach()
                                    .cpu()
                                    .numpy()
                                )

                            se_pure = np.sum((pred_pure - target) ** 2)
                            se_mod = np.sum((pred_mod - target) ** 2)

                            config_errors_pure[f"{k}_after_initial"].append(se_pure)
                            config_errors_modified[f"{k}_after_initial"].append(se_mod)

                        eval_pos_final = final_pos + k
                        if eval_pos_final < len(trace) and trace[eval_pos_final, 51] != 0:
                            activation = activations[eval_pos_final - 1].astype(np.float64)
                            target = trace[eval_pos_final, -5:].astype(np.float64)

                            with torch.no_grad():
                                activation_tensor = (
                                    torch.from_numpy(activation).double().to(device)
                                )
                                pred_pure = (
                                    model_pure._read_out(activation_tensor)
                                    .detach()
                                    .cpu()
                                    .numpy()
                                )
                                pred_mod = (
                                    model_modified._read_out(activation_tensor)
                                    .detach()
                                    .cpu()
                                    .numpy()
                                )

                            se_pure = np.sum((pred_pure - target) ** 2)
                            se_mod = np.sum((pred_mod - target) ** 2)

                            config_errors_pure[f"{k}_after_final"].append(se_pure)
                            config_errors_modified[f"{k}_after_final"].append(se_mod)

            for metric in config_errors_pure.keys():
                if config_errors_pure[metric]:
                    all_config_medians_pure[metric].append(
                        np.median(config_errors_pure[metric])
                    )
                    all_config_medians_modified[metric].append(
                        np.median(config_errors_modified[metric])
                    )

    aggregated_pure = {
        metric: np.median(values) for metric, values in all_config_medians_pure.items()
    }
    aggregated_modified = {
        metric: np.median(values) for metric, values in all_config_medians_modified.items()
    }

    return aggregated_pure, aggregated_modified


def save_results_to_text(results_dict, k_values, output_path, batch_num=None):
    mode = "a" if batch_num is not None else "w"

    with open(output_path, mode) as f:
        if batch_num is not None:
            f.write(f"\n{'='*80}\n")
            f.write(f"BATCH {batch_num}/20\n")
            f.write(f"{'='*80}\n")

        for n_train_configs in sorted(results_dict.keys()):
            f.write(f"\n--- Total Training Configs: {n_train_configs} ---\n")

            pure_results = results_dict[n_train_configs]["pure"]
            modified_results = results_dict[n_train_configs]["modified"]

            f.write("\nk_after_initial:\n")
            for k in k_values:
                metric = f"{k}_after_initial"
                f.write(
                    f"  k={k}: Pure={pure_results[metric]:.8f}, Modified={modified_results[metric]:.8f}\n"
                )

            f.write("\nk_after_final:\n")
            for k in k_values:
                metric = f"{k}_after_final"
                f.write(
                    f"  k={k}: Pure={pure_results[metric]:.8f}, Modified={modified_results[metric]:.8f}\n"
                )


def main():
    print("\n" + "=" * 80)
    print("TRAIN ERROR COMPARISON: GPT2 PURE VS MODIFIED (FULL TRACES)")
    print("=" * 80 + "\n")

    config = Config()
    config.override("model_type", "GPT2")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    haystack_lengths = [1, 2, 3, 4, 5, 6]
    k_values = [1, 2, 3, 4, 5, 6, 7, 8]

    print("Loading pretrained GPT2 model...")
    checkpoint_path = os.path.join(
        os.path.dirname(__file__),
        "Set-up Data",
        "step%3D99000.ckpt",
    )

    model_pure = GPT2(
        n_dims_in=config.n_dims_in,
        n_positions=config.n_positions,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_dims_out=config.n_dims_out,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model_pure.load_state_dict(state_dict, strict=False)
    model_pure.double()
    print("Model loaded and converted to float64\n")

    print("=" * 80)
    print("LOADING DATA AND ACTIVATIONS FOR ALL HAYSTACKS")
    print("=" * 80 + "\n")

    all_activations = {}
    all_data = {}
    for haystack_len in haystack_lengths:
        print(f"Loading haystack {haystack_len}...")
        all_data[haystack_len] = load_haystack_file(haystack_len)
        all_activations[haystack_len] = load_activations(haystack_len)

    print("\nAll data loaded\n")

    results_dict = {}
    output_dir = os.path.join(os.path.dirname(__file__), "train_error_results")
    os.makedirs(output_dir, exist_ok=True)

    text_path = os.path.join(output_dir, "train_error_fulltrace_results.txt")
    if os.path.exists(text_path):
        os.remove(text_path)
        print("Cleared existing results file\n")

    num_batches = 20
    configs_per_batch = 2

    print("\n" + "=" * 80)
    print("TRAINING WITH FULL TRACES")
    print("=" * 80 + "\n")
    print(f"Total batches: {num_batches}")
    print(f"Configs per batch per haystack: {configs_per_batch}")
    print(f"Total configs per batch: {configs_per_batch * len(haystack_lengths)}")
    print(f"Total training configs: {num_batches * configs_per_batch * len(haystack_lengths)}\n")

    YAT_sum = np.zeros((config.n_dims_out, config.n_embd), dtype=np.float64)
    AAT_sum = np.zeros((config.n_embd, config.n_embd), dtype=np.float64)

    for batch_idx in range(num_batches):
        batch_start = batch_idx * configs_per_batch
        batch_end = (batch_idx + 1) * configs_per_batch
        total_configs_so_far = batch_end * len(haystack_lengths)

        print("\n" + "=" * 80)
        print(
            f"BATCH {batch_idx + 1}/20: Processing NEW configs {batch_start}-{batch_end - 1} per haystack"
        )
        print(
            f"Total configs so far: {batch_end} per haystack ({total_configs_so_far} across all)"
        )
        print("=" * 80 + "\n")

        for haystack_len in haystack_lengths:
            batch_config_indices = list(range(batch_start, batch_end))
            print(
                f"Haystack {haystack_len}: Extracting FULL traces from NEW configs {batch_config_indices}..."
            )

            A_batch, Y_batch = extract_full_training_pairs(
                all_data[haystack_len],
                all_activations[haystack_len],
                batch_config_indices,
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

        print("  Computing least squares solution...")
        W_opt = compute_least_squares_incremental(YAT_sum, AAT_sum)
        print(f"  W_opt shape: {W_opt.shape}")

        model_modified = create_modified_model(model_pure, W_opt)

        print(
            f"\n  Evaluating on training data (batches 1 through {batch_idx + 1})..."
        )

        aggregated_pure, aggregated_modified = evaluate_models_on_training_data(
            model_pure,
            model_modified,
            all_data,
            all_activations,
            haystack_lengths,
            batch_idx + 1,
            k_values,
            device,
        )

        results_dict[total_configs_so_far] = {
            "pure": aggregated_pure,
            "modified": aggregated_modified,
        }

        print(f"\n  Results after {total_configs_so_far} total training configs:")
        print("  k_after_initial:")
        for k in k_values:
            pure_init = aggregated_pure[f"{k}_after_initial"]
            mod_init = aggregated_modified[f"{k}_after_initial"]
            print(f"    {k}: Pure={pure_init:.6f}, Modified={mod_init:.6f}")

        print("  k_after_final:")
        for k in k_values:
            pure_final = aggregated_pure[f"{k}_after_final"]
            mod_final = aggregated_modified[f"{k}_after_final"]
            print(f"    {k}: Pure={pure_final:.6f}, Modified={mod_final:.6f}")

        print("\n  Saving results to text file...")
        save_results_to_text(results_dict, k_values, text_path, batch_num=batch_idx + 1)

    print("\n" + "=" * 80)
    print("SAVING FINAL RESULTS")
    print("=" * 80 + "\n")

    results_path = os.path.join(output_dir, "train_error_fulltrace_results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results_dict, f)
    print(f"Results saved to: {results_path}")
    print(f"Text results saved to: {text_path}")

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()


