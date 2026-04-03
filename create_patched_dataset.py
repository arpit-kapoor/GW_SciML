#!/usr/bin/env python3
"""Create patch metadata and patch datasets from FEFLOW variable-density outputs.

This script combines two steps into one terminal-friendly pipeline:
1) Build patch metadata and write ``patches.json``.
2) Build per-patch numpy arrays under ``patch_all_ts`` (or custom output dir).

The output format is compatible with existing training/inference loaders.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from tqdm import tqdm


DEFAULT_BASE_DATA_DIR = "/srv/scratch/z5370003/projects/data/groundwater/FEFLOW/coastal/variable_density"

TARGET_COLS = ["mass_concentration", "head", "pressure"]
FORCING_COLS = ["mass_concentration_bc", "head_bc", "recharge_forcing", "sea_level_forcing"]
COORD_COLS = ["X", "Y", "Z"]
NODE_COL = "node"
SLICE_COL = "slice"


@dataclass
class PipelineConfig:
    base_data_dir: str
    raw_data_subdir: str
    forcings_data_subdir: str
    patch_data_subdir: str
    patches_json_name: str
    n_patches: int
    slice_split: int
    ghost_points_ratio: float
    n_ghost_points: int | None
    neighbour_k: int
    max_workers: int
    use_normalized_coords_for_patching: bool
    clip_negative_mass_concentration: bool

    @property
    def raw_data_dir(self) -> str:
        return os.path.join(self.base_data_dir, self.raw_data_subdir)

    @property
    def forcings_data_dir(self) -> str:
        return os.path.join(self.base_data_dir, self.forcings_data_subdir)

    @property
    def patch_data_dir(self) -> str:
        return os.path.join(self.base_data_dir, self.patch_data_subdir)

    @property
    def patches_json_path(self) -> str:
        return os.path.join(self.base_data_dir, self.patches_json_name)


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(
        description="Create patches.json and patch_* numpy datasets from FEFLOW time series CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--base-data-dir", type=str, default=DEFAULT_BASE_DATA_DIR)
    parser.add_argument("--raw-data-subdir", type=str, default="all")
    parser.add_argument("--forcings-data-subdir", type=str, default="forcings_corrected_all")
    parser.add_argument("--patch-data-subdir", type=str, default="patch_all_ts")
    parser.add_argument("--patches-json-name", type=str, default="patches.json")

    parser.add_argument("--n-patches", type=int, default=20)
    parser.add_argument("--slice-split", type=int, default=4)
    parser.add_argument("--ghost-points-ratio", type=float, default=0.05)
    parser.add_argument("--n-ghost-points", type=int, default=None)
    parser.add_argument("--neighbour-k", type=int, default=10)

    parser.add_argument("--max-workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--use-normalized-coords-for-patching", action="store_true")
    parser.add_argument("--no-clip-negative-mass-concentration", action="store_true")

    args = parser.parse_args()

    if args.n_patches <= 0:
        raise ValueError("--n-patches must be > 0")
    if args.slice_split <= 0:
        raise ValueError("--slice-split must be > 0")
    if args.ghost_points_ratio < 0:
        raise ValueError("--ghost-points-ratio must be >= 0")
    if args.n_ghost_points is not None and args.n_ghost_points < 0:
        raise ValueError("--n-ghost-points must be >= 0 when provided")
    if args.neighbour_k < 2:
        raise ValueError("--neighbour-k must be >= 2 (includes self neighbor)")
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be > 0")

    return PipelineConfig(
        base_data_dir=args.base_data_dir,
        raw_data_subdir=args.raw_data_subdir,
        forcings_data_subdir=args.forcings_data_subdir,
        patch_data_subdir=args.patch_data_subdir,
        patches_json_name=args.patches_json_name,
        n_patches=args.n_patches,
        slice_split=args.slice_split,
        ghost_points_ratio=args.ghost_points_ratio,
        n_ghost_points=args.n_ghost_points,
        neighbour_k=args.neighbour_k,
        max_workers=args.max_workers,
        use_normalized_coords_for_patching=args.use_normalized_coords_for_patching,
        clip_negative_mass_concentration=not args.no_clip_negative_mass_concentration,
    )


def list_time_series_files(raw_data_dir: str) -> list[str]:
    if not os.path.isdir(raw_data_dir):
        raise FileNotFoundError(f"Raw data directory not found: {raw_data_dir}")

    ts_files = sorted(
        f for f in os.listdir(raw_data_dir) if f.lower().endswith(".csv") and not f.startswith(".")
    )
    if not ts_files:
        raise RuntimeError(f"No CSV files found in {raw_data_dir}")
    return ts_files


def load_reference_frame(raw_data_dir: str, first_file: str) -> pd.DataFrame:
    ref_df = pd.read_csv(
        os.path.join(raw_data_dir, first_file),
        usecols=[NODE_COL, SLICE_COL] + COORD_COLS,
        low_memory=False,
    )
    required = {NODE_COL, SLICE_COL, *COORD_COLS}
    missing = required - set(ref_df.columns)
    if missing:
        raise RuntimeError(f"Reference CSV is missing required columns: {sorted(missing)}")
    return ref_df


def _normalize_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        cmin = out[col].min()
        cmax = out[col].max()
        if cmax > cmin:
            out[col] = (out[col] - cmin) / (cmax - cmin)
        else:
            out[col] = 0.0
    return out


def create_patches(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    slice_id: np.ndarray,
    n_patches: int,
    slice_split: int,
    ghost_points_ratio: float,
    n_ghost_points: int | None,
    neighbour_k: int,
) -> tuple[np.ndarray, np.ndarray, dict[int, set[int]], dict[int, dict[int, np.ndarray]]]:
    slices = np.sort(np.unique(slice_id))
    slices_split = np.array_split(slices, slice_split)
    n_patches_per_slice = max(1, n_patches // slice_split)

    patch_ids = np.empty(x.shape[0], dtype=int)
    slice_groups = np.empty(x.shape[0], dtype=int)

    patch_id_offset = 1
    for i, slice_group in enumerate(slices_split):
        slice_idx = np.where(np.isin(slice_id, slice_group))[0]
        slice_groups[slice_idx] = i + 1

        coords = np.stack([x[slice_idx], y[slice_idx], z[slice_idx]], axis=1)
        n_clusters = min(n_patches_per_slice, coords.shape[0])
        if n_clusters < 1:
            continue

        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(coords)

        patch_ids[slice_idx] = cluster_labels + patch_id_offset
        patch_id_offset += n_patches_per_slice

    coords_all = np.stack([x, y, z], axis=1)
    unique_patches = np.unique(patch_ids)
    patch_indices = {pid: np.where(patch_ids == pid)[0] for pid in unique_patches}

    tree = cKDTree(coords_all)
    patch_neighbours: dict[int, set[int]] = defaultdict(set)
    for pid in unique_patches:
        idx = patch_indices[pid]
        k_val = min(neighbour_k, len(coords_all))
        _, nbrs = tree.query(coords_all[idx], k=k_val)
        nbrs_2d = np.atleast_2d(nbrs)
        for row, nbr_row in zip(idx, nbrs_2d):
            for nbr_idx in np.atleast_1d(nbr_row):
                if nbr_idx == row:
                    continue
                nbr_pid = patch_ids[nbr_idx]
                if nbr_pid != pid:
                    patch_neighbours[int(pid)].add(int(nbr_pid))

    patch_ghost_points: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    dynamic_n_ghost_points = n_ghost_points is None

    for pid in unique_patches:
        idx = patch_indices[pid]
        coords_patch = coords_all[idx]

        current_n_ghost = n_ghost_points
        if dynamic_n_ghost_points:
            current_n_ghost = max(1, int(len(idx) * ghost_points_ratio))

        for nbr_pid in patch_neighbours[int(pid)]:
            nbr_idx = patch_indices[nbr_pid]
            coords_nbr = coords_all[nbr_idx]
            nbr_tree = cKDTree(coords_nbr)

            k_val = min(current_n_ghost, len(nbr_idx))
            dists, ghost_indices = nbr_tree.query(coords_patch, k=k_val)

            dists_flat = dists.flatten() if np.ndim(dists) > 1 else np.asarray(dists)
            ghost_indices_flat = (
                ghost_indices.flatten() if np.ndim(ghost_indices) > 1 else np.asarray(ghost_indices)
            )

            sorted_idx = np.argsort(ghost_indices_flat)
            ghost_indices_sorted = ghost_indices_flat[sorted_idx]
            dists_sorted = dists_flat[sorted_idx]

            unique_ghost_indices, unique_pos = np.unique(ghost_indices_sorted, return_index=True)
            if len(unique_ghost_indices) > 0:
                unique_dists = dists_sorted[unique_pos]
                dist_order = np.argsort(unique_dists)
                selected_indices = unique_ghost_indices[dist_order][:current_n_ghost]
            else:
                selected_indices = np.array([], dtype=int)

            patch_ghost_points[int(pid)][int(nbr_pid)] = np.array(nbr_idx)[selected_indices]

    return patch_ids, slice_groups, patch_neighbours, patch_ghost_points


def build_patch_metadata(ref_df: pd.DataFrame, cfg: PipelineConfig) -> dict[str, dict[str, Any]]:
    patch_input_df = ref_df
    if cfg.use_normalized_coords_for_patching:
        patch_input_df = _normalize_columns(ref_df, COORD_COLS)

    patch_ids, slice_groups, patch_neighbours, patch_ghost_points = create_patches(
        x=patch_input_df[COORD_COLS[0]].to_numpy(),
        y=patch_input_df[COORD_COLS[1]].to_numpy(),
        z=patch_input_df[COORD_COLS[2]].to_numpy(),
        slice_id=patch_input_df[SLICE_COL].to_numpy(),
        n_patches=cfg.n_patches,
        slice_split=cfg.slice_split,
        ghost_points_ratio=cfg.ghost_points_ratio,
        n_ghost_points=cfg.n_ghost_points,
        neighbour_k=cfg.neighbour_k,
    )

    patch_data: dict[str, dict[str, Any]] = {}
    nodes = ref_df[NODE_COL].to_numpy()

    for patch_id in np.unique(patch_ids):
        patch_mask = patch_ids == patch_id
        core_nodes = nodes[patch_mask]

        ghost_nodes_list: list[int] = []
        for _, ghost_idx in patch_ghost_points[int(patch_id)].items():
            ghost_nodes_list.extend(nodes[ghost_idx].tolist())

        # Preserve stable ordering while removing duplicates.
        ghost_nodes = list(dict.fromkeys(ghost_nodes_list))

        slice_group_vals = np.unique(slice_groups[patch_mask])
        slice_group = int(slice_group_vals[0]) if len(slice_group_vals) > 0 else 0

        patch_data[str(int(patch_id))] = {
            "slice_group": slice_group,
            "neighbour_patches": sorted(int(p) for p in patch_neighbours[int(patch_id)]),
            "ghost_nodes": [int(n) for n in ghost_nodes],
            "core_nodes": [int(n) for n in core_nodes.tolist()],
        }

    return patch_data


def save_patch_metadata(patch_data: dict[str, dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(patch_data, f, indent=2)


def _build_node_index(ref_df: pd.DataFrame) -> tuple[dict[int, int], np.ndarray]:
    node_arr = ref_df[NODE_COL].to_numpy()
    node_to_idx = {int(node): idx for idx, node in enumerate(node_arr)}
    return node_to_idx, node_arr


def _load_timestep_for_patch(
    ts_file: str,
    raw_data_dir: str,
    forcings_data_dir: str,
    core_idx: np.ndarray,
    ghost_idx: np.ndarray,
    clip_negative_mass_concentration: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ts_df = pd.read_csv(
        os.path.join(raw_data_dir, ts_file),
        usecols=TARGET_COLS,
        low_memory=False,
    )
    ts_vals = ts_df.to_numpy()

    if clip_negative_mass_concentration:
        neg_mask = ts_vals[:, 0] < 0
        if np.any(neg_mask):
            ts_vals = ts_vals.copy()
            ts_vals[neg_mask, 0] = 0

    forcing_df = pd.read_csv(
        os.path.join(forcings_data_dir, ts_file),
        usecols=FORCING_COLS,
        low_memory=False,
    )
    forcing_vals = forcing_df.to_numpy()

    return (
        ts_vals[core_idx, :],
        ts_vals[ghost_idx, :],
        forcing_vals[core_idx, :],
        forcing_vals[ghost_idx, :],
    )


def create_patch_arrays(
    patch_config: dict[str, dict[str, Any]],
    ref_df: pd.DataFrame,
    ts_files: list[str],
    cfg: PipelineConfig,
) -> None:
    os.makedirs(cfg.patch_data_dir, exist_ok=True)

    node_to_idx, _ = _build_node_index(ref_df)
    first_coords_df = ref_df[[NODE_COL] + COORD_COLS].copy()

    for patch_id_str in sorted(patch_config.keys(), key=lambda x: int(x)):
        config = patch_config[patch_id_str]

        core_nodes = [int(n) for n in config["core_nodes"]]
        ghost_nodes = [int(n) for n in config["ghost_nodes"]]

        core_idx = np.array([node_to_idx[n] for n in core_nodes], dtype=np.int64)
        ghost_idx = np.array([node_to_idx[n] for n in ghost_nodes], dtype=np.int64)

        print(f"\nProcessing patch {patch_id_str}")
        print(f"Patch {patch_id_str} has {len(core_idx)} core nodes and {len(ghost_idx)} ghost nodes")
        print(f"Patch {patch_id_str} has {len(config['neighbour_patches'])} neighbour patches")
        print(f"Patch {patch_id_str} has {config['slice_group']} slice group")

        worker = partial(
            _load_timestep_for_patch,
            raw_data_dir=cfg.raw_data_dir,
            forcings_data_dir=cfg.forcings_data_dir,
            core_idx=core_idx,
            ghost_idx=ghost_idx,
            clip_negative_mass_concentration=cfg.clip_negative_mass_concentration,
        )

        with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
            results = list(tqdm(executor.map(worker, ts_files), total=len(ts_files), leave=False))

        core_patch_data = np.nan_to_num(np.array([r[0] for r in results]))
        ghost_patch_data = np.nan_to_num(np.array([r[1] for r in results]))
        core_forcings_data = np.nan_to_num(np.array([r[2] for r in results]))
        ghost_forcings_data = np.nan_to_num(np.array([r[3] for r in results]))

        core_coords = first_coords_df.iloc[core_idx][COORD_COLS].to_numpy()
        ghost_coords = first_coords_df.iloc[ghost_idx][COORD_COLS].to_numpy()

        patch_dir_path = os.path.join(cfg.patch_data_dir, f"patch_{int(patch_id_str):03d}")
        os.makedirs(patch_dir_path, exist_ok=True)

        np.save(os.path.join(patch_dir_path, "core_obs.npy"), core_patch_data)
        np.save(os.path.join(patch_dir_path, "ghost_obs.npy"), ghost_patch_data)
        np.save(os.path.join(patch_dir_path, "core_coords.npy"), core_coords)
        np.save(os.path.join(patch_dir_path, "ghost_coords.npy"), ghost_coords)
        np.save(os.path.join(patch_dir_path, "core_forcings.npy"), core_forcings_data)
        np.save(os.path.join(patch_dir_path, "ghost_forcings.npy"), ghost_forcings_data)


def main() -> None:
    cfg = parse_args()

    print("=" * 80)
    print("Creating patch metadata and patch datasets")
    print("=" * 80)
    print(f"Base data directory: {cfg.base_data_dir}")
    print(f"Raw data directory: {cfg.raw_data_dir}")
    print(f"Forcings data directory: {cfg.forcings_data_dir}")
    print(f"Patch output directory: {cfg.patch_data_dir}")
    print(f"Patch metadata path: {cfg.patches_json_path}")

    ts_files = list_time_series_files(cfg.raw_data_dir)
    print(f"Total number of files: {len(ts_files)}")
    print(f"First 3 files: {ts_files[:3]}")
    print(f"Last 3 files: {ts_files[-3:]}")

    ref_df = load_reference_frame(cfg.raw_data_dir, ts_files[0])

    print("\nStep 1/2: Building patch metadata...")
    patch_metadata = build_patch_metadata(ref_df, cfg)
    save_patch_metadata(patch_metadata, cfg.patches_json_path)
    print(f"Saved patch metadata for {len(patch_metadata)} patches to {cfg.patches_json_path}")

    print("\nStep 2/2: Building patch arrays...")
    create_patch_arrays(patch_metadata, ref_df, ts_files, cfg)

    print("\nDone.")


if __name__ == "__main__":
    main()
