#!/usr/bin/env python
"""Utility to regenerate interleaved pretraining traces via `generate_interleaved_traces`.

This script reproduces (and slightly tidies up) the workflow the user previously
ran by hand:

1. Load a cached bank of ortho_haar training systems (`ys`, `sim_objs`).
2. Build a `Config` that matches the desired multi-system setup.
3. Optionally delete the cached interleaved file to force fresh sampling.
4. Call `generate_interleaved_traces` so that the canonical cache is created.
5. Copy the cache to a unique filename under `DataRandomTransformer` for
   downstream experiments, and print shape diagnostics.

Example:
    python generate_pretraining_interleaved.py --num-configs 512 --copy 3
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Ensure local packages resolve exactly as the interactive notebook/script did.
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = WORKSPACE_ROOT / "src"
LOCAL_INTERLEAVED_ROOT = Path(__file__).resolve().parent / "LocalInterleavedCache"

import sys

if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.append(str(WORKSPACE_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

LOCAL_INTERLEAVED_ROOT.mkdir(parents=True, exist_ok=True)

from core import Config  # type: ignore  # noqa: E402
from data_train import generate_interleaved_traces  # type: ignore  # noqa: E402


@dataclass(frozen=True)
class GenerationSpec:
    """Parameters required to regenerate a cached interleaved-trace file."""

    dataset_typ: str = "ortho_haar"
    datasource: str = "train"
    c_dist_suffix: str = "_ident_C"
    data_dir: Optional[str] = None
    ny: int = 5
    nx: int = 5
    n_positions: int = 250
    num_tasks: int = 25
    num_trials_train: int = 1
    num_trials_val: int = 1
    num_test_configs: int = 240000
    len_seg_haystack: int = 10
    copy_count: int = 1
    force_regenerate: bool = True
    random_seed: Optional[int] = None
    additional_seed_offset: int = 0


def _default_data_dir() -> Path:
    return SRC_ROOT / "RandomTransformerExperiments" / "Set-up Data"


def _pretraining_output_dir() -> Path:
    return (
        SRC_ROOT
        / "RandomTransformerExperiments"
        / "Training:Test Data"
        / "Pretraining"
    )


def _load_training_bank(
    data_dir: Path, spec: GenerationSpec
) -> Tuple[np.ndarray, Sequence[dict]]:
    """Load cached training systems and truncate to `spec.num_tasks`."""
    data_path = data_dir / f"train_{spec.dataset_typ}{spec.c_dist_suffix}_state_dim_{spec.nx}.pkl"
    sim_path = data_dir / f"train_{spec.dataset_typ}{spec.c_dist_suffix}_state_dim_{spec.nx}_sim_objs.pkl"

    with open(data_path, "rb") as fh:
        raw_samples: Sequence[Dict[str, np.ndarray]] = pickle.load(fh)

    with open(sim_path, "rb") as fh:
        sim_objs: Sequence[dict] = pickle.load(fh)

    trimmed_samples = raw_samples[: spec.num_tasks]
    trimmed_sim_objs = sim_objs[: spec.num_tasks]

    ys_list: List[np.ndarray] = []
    expected_trace_len = spec.n_positions + 1
    for entry in trimmed_samples:
        obs = entry["obs"]
        if obs.shape[0] < expected_trace_len:
            raise ValueError(
                f"Observed trace shorter than expected {expected_trace_len}: {obs.shape}"
            )
        ys_list.append(obs[:expected_trace_len])

    ys = np.stack(ys_list, axis=0).astype(np.float32)  # (num_tasks, n_positions+1, ny)
    ys = ys.reshape(
        spec.num_tasks,
        1,
        expected_trace_len,
        spec.ny,
    )

    return ys, trimmed_sim_objs


def _build_config(spec: GenerationSpec) -> Config:
    cfg = Config()

    cfg.override("datasource", spec.datasource)
    cfg.override("dataset_typ", spec.dataset_typ)
    cfg.override("C_dist", spec.c_dist_suffix)
    cfg.override("nx", spec.nx)
    cfg.override("ny", spec.ny)
    cfg.override("n_positions", spec.n_positions)
    cfg.override("num_traces", {"train": spec.num_trials_train, "val": spec.num_trials_val})
    cfg.override("num_tasks", spec.num_tasks)
    cfg.override("max_sys_trace", spec.num_tasks)
    cfg.override("multi_sys_trace", True)
    cfg.override("needle_in_haystack", False)
    cfg.override("num_test_traces_configs", spec.num_test_configs)
    cfg.override("num_haystack_examples", 1)
    cfg.override("len_seg_haystack", spec.len_seg_haystack)
    cfg.override("num_val_tasks", spec.num_tasks)
    cfg.override("val_dataset_typ", spec.dataset_typ)
    cfg.override("late_start", None)
    cfg.override("single_system", False)

    # Flags referenced by path construction inside `generate_interleaved_traces`.
    cfg.override("fix_needle", False)
    cfg.override("opposite_ortho", False)
    cfg.override("irrelevant_tokens", False)
    cfg.override("same_tokens", False)
    cfg.override("new_hay_insert", False)
    cfg.override("paren_swap", False)
    cfg.override("zero_cut", False)
    cfg.override("needle_final_seg_extended", False)

    return cfg


def _cached_interleave_path(cfg: Config) -> Path:
    if cfg.needle_in_haystack:
        interleaving = f"haystack_len_{cfg.num_sys_haystack}"
        if cfg.num_test_traces_configs > 1:
            interleaving += f"_trace_configs_{cfg.num_test_traces_configs}"
        if cfg.needle_final_seg_extended:
            interleaving += "_needle_final_seg_extended"
    else:
        interleaving = "multi_cut"

    filename = (
        f"{cfg.datasource}_"
        + ("ortho_sync_" if cfg.val_dataset_typ == "ortho_sync" else "")
        + ("fix_needle_" if cfg.fix_needle else "")
        + ("opposite_ortho_" if cfg.opposite_ortho else "")
        + ("irrelevant_tokens_" if cfg.irrelevant_tokens else "")
        + ("same_tokens_" if cfg.same_tokens else "")
        + ("new_hay_insert_" if cfg.new_hay_insert else "")
        + ("paren_swap_" if cfg.paren_swap else "")
        + ("zero_cut_" if cfg.zero_cut else "")
        + f"interleaved_traces_{cfg.dataset_typ}{cfg.C_dist}_{interleaving}.pkl"
    )

    base_dir = LOCAL_INTERLEAVED_ROOT / cfg.dataset_typ
    return base_dir / filename


def _set_seed(seed: Optional[int]) -> int:
    if seed is None:
        seed = int(time.time()) % (2**32 - 1)
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    return seed


def _copy_with_suffix(src: Path, dst_dir: Path, suffix: str) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / suffix
    shutil.copy2(src, dst_path)
    return dst_path


def _split_pretraining_batches(full_cache_path: Path, output_dir: Path, batch_size: int = 20000, batch_count: int = 12) -> List[Path]:
    if not full_cache_path.exists():
        raise FileNotFoundError(f"Cannot split non-existent cache: {full_cache_path}")

    with open(full_cache_path, "rb") as fh:
        payload = pickle.load(fh)

    multi_sys_ys = payload["multi_sys_ys"]
    num_configs_total = multi_sys_ys.shape[1]

    expected_total = batch_size * batch_count
    if num_configs_total != expected_total:
        raise ValueError(
            f"Cache holds {num_configs_total} configs but batch_size * batch_count = {expected_total}. "
            "Adjust the split parameters to evenly cover the cache."
        )

    written_paths: List[Path] = []
    for batch_idx in range(batch_count):
        start = batch_idx * batch_size
        end = start + batch_size

        batch_payload = {
            "multi_sys_ys": multi_sys_ys[:, start:end].copy(),
        }

        for key, value in payload.items():
            if key == "multi_sys_ys":
                continue
            batch_payload[key] = value[start:end]

        batch_path = output_dir / f"pretraining_batch_{batch_idx + 1}.pkl"
        if batch_path.exists():
            batch_path.unlink()
        with open(batch_path, "wb") as fh:
            pickle.dump(batch_payload, fh)
        written_paths.append(batch_path)
        print(f"  Saved {batch_path} with configs [{start}, {end})")

    return written_paths


def generate_pretraining_data(spec: GenerationSpec) -> List[Path]:
    data_dir = Path(spec.data_dir) if spec.data_dir else _default_data_dir()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    ys, sim_objs = _load_training_bank(data_dir, spec)

    output_dir = _pretraining_output_dir()

    cfg = _build_config(spec)
    cache_path = _cached_interleave_path(cfg)

    if spec.force_regenerate and cache_path.exists():
        cache_path.unlink()
        print(f"Deleted cached file to force regeneration: {cache_path}")

    # Establish deterministic randomness per invocation.
    base_seed = _set_seed(spec.random_seed)
    print(f"Using RNG seed: {base_seed}")

    generated_paths: List[Path] = []
    for copy_index in range(spec.copy_count):
        seed_with_offset = (base_seed + copy_index + spec.additional_seed_offset) & 0xFFFFFFFF
        _set_seed(seed_with_offset)
        print(f"\n=== Generating copy {copy_index + 1}/{spec.copy_count} with seed {seed_with_offset}")

        if cache_path.exists():
            cache_path.unlink()
            print(f"  Deleted cached interleave file prior to regeneration: {cache_path}")

        path = generate_interleaved_traces(cfg, ys, sim_objs, num_trials=spec.num_trials_train)
        src_path = Path(path)
        if not src_path.exists():
            raise FileNotFoundError(f"generate_interleaved_traces returned non-existent path: {src_path}")

        suffix = (
            f"train_interleaved_traces_{spec.num_test_configs}_from_seed_{seed_with_offset}.pkl"
        )
        copied_path = _copy_with_suffix(src_path, output_dir, suffix)
        print(f"✓ Copied cache to {copied_path}")
        generated_paths.append(copied_path)

    print("\n=== Dimension summary ===")
    for path in generated_paths:
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        multi_sys = payload["multi_sys_ys"]
        print(
            f"{path.name}: multi_sys_ys shape {multi_sys.shape}, dtype {multi_sys.dtype}, "
            f"min {multi_sys.min():.3e}, max {multi_sys.max():.3e}"
        )

    if generated_paths:
        print("\nSplitting canonical cache into 12 batches of 20,000 traces each...")
        _split_pretraining_batches(generated_paths[0], output_dir)

    return generated_paths


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-configs", type=int, default=240000, help="Number of interleaved trace configs to produce.")
    parser.add_argument("--copies", type=int, default=1, help="How many distinct copies (with offset seeds) to create.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducibility; defaults to time-based.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing train_ortho_haar_ident_C_state_dim_5*.pkl (defaults to DataRandomTransformer under repo).",
    )
    parser.add_argument(
        "--allow-cache",
        action="store_true",
        help="Reuse existing interleaved cache instead of regenerating it.",
    )
    parser.add_argument(
        "--dump-spec",
        action="store_true",
        help="Print the resolved GenerationSpec and exit without generating.",
    )
    args = parser.parse_args(argv)

    spec = GenerationSpec(
        num_test_configs=args.num_configs,
        copy_count=args.copies,
        random_seed=args.seed,
        force_regenerate=not args.allow_cache,
        data_dir=args.data_dir,
    )

    if args.dump_spec:
        print(json.dumps(spec.__dict__, indent=2))
        return

    generate_pretraining_data(spec)


if __name__ == "__main__":
    main()

