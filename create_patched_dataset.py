#!/usr/bin/env python3
"""Create patch metadata and patch datasets from FEFLOW variable-density outputs.

This script combines two steps into one terminal-friendly pipeline:
1) Build patch metadata and write ``patches.json``.
2) Build per-patch numpy arrays under ``patch_all_ts`` (or custom output dir).

Patches are KMeans clusters within groups of slices. Neighbours and ghost nodes come from the mesh
connectivity, rebuilt from the prismatic FEFLOW mesh: a Delaunay triangulation of one slice repeated on every
slice, plus vertical edges between consecutive slices. The ghost nodes of a patch are all non-core nodes within
``ghost_hops`` mesh edges of its core.

The output format is compatible with existing training/inference loaders.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import Delaunay
from sklearn.cluster import KMeans
from tqdm import tqdm


# DEFAULT_BASE_DATA_DIR = "/srv/scratch/z5370003/projects/data/groundwater/FEFLOW/coastal/variable_density"

DEFAULT_BASE_DATA_DIR = "/scratch/yl75/ak4177/data/feflow_data"

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
    ghost_hops: int
    max_edge_factor: float
    max_workers: int
    use_normalized_coords_for_patching: bool
    clip_negative_mass_concentration: bool
    metadata_only: bool

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
    parser.add_argument("--ghost-hops", type=int, default=2,
                        help="Ghost nodes are all non-core nodes within this many mesh edges of the patch core")
    parser.add_argument("--max-edge-factor", type=float, default=3.0,
                        help="Drop Delaunay edges longer than this multiple of the median edge length "
                             "(removes edges across concave parts of the domain boundary)")

    parser.add_argument("--max-workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--use-normalized-coords-for-patching", action="store_true")
    parser.add_argument("--no-clip-negative-mass-concentration", action="store_true")
    parser.add_argument("--metadata-only", action="store_true", help="Write the patch metadata JSON and skip the arrays")

    args = parser.parse_args()

    if args.n_patches <= 0:
        raise ValueError("--n-patches must be > 0")
    if args.slice_split <= 0:
        raise ValueError("--slice-split must be > 0")
    if args.ghost_hops < 0:
        raise ValueError("--ghost-hops must be >= 0")
    if args.max_edge_factor <= 0:
        raise ValueError("--max-edge-factor must be > 0")
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
        ghost_hops=args.ghost_hops,
        max_edge_factor=args.max_edge_factor,
        max_workers=args.max_workers,
        use_normalized_coords_for_patching=args.use_normalized_coords_for_patching,
        clip_negative_mass_concentration=not args.no_clip_negative_mass_concentration,
        metadata_only=args.metadata_only,
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


def build_mesh_adjacency(
    x: np.ndarray,
    y: np.ndarray,
    slice_id: np.ndarray,
    max_edge_factor: float = 3.0,
) -> sparse.csr_matrix:
    """Node adjacency of a prismatic mesh (every slice has the same nodes in XY).

    Horizontal edges come from a Delaunay triangulation of one slice, without edges longer than
    ``max_edge_factor`` times the median edge length, and are repeated on every slice. Vertical edges join
    the same column in consecutive slices.

    Returns:
        Symmetric boolean CSR matrix of shape [N, N] in the row order of the inputs.
    """
    slices = np.sort(np.unique(slice_id))
    slice_rows = [np.where(slice_id == s)[0] for s in slices]
    n_per_slice = {len(rows) for rows in slice_rows}
    if len(n_per_slice) != 1:
        raise ValueError(f"Mesh is not prismatic: nodes per slice differ ({sorted(n_per_slice)})")

    # columns[k, c]: row of column c in slice k, with columns ordered by (X, Y)
    columns = np.stack([rows[np.lexsort((y[rows], x[rows]))] for rows in slice_rows])
    xy = np.stack([x[columns[0]], y[columns[0]]], axis=1)
    for k in range(1, len(slices)):
        xy_k = np.stack([x[columns[k]], y[columns[k]]], axis=1)
        if not np.allclose(xy_k, xy, rtol=0.0, atol=1e-6):
            raise ValueError(f"Mesh is not prismatic: XY positions of slice {slices[k]} differ from slice {slices[0]}")

    simplices = Delaunay(xy).simplices
    edges = np.concatenate([simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    lengths = np.linalg.norm(xy[edges[:, 0]] - xy[edges[:, 1]], axis=1)
    edges = edges[lengths <= max_edge_factor * np.median(lengths)]

    src = [columns[:, edges[:, 0]].ravel(), columns[:-1].ravel()]
    dst = [columns[:, edges[:, 1]].ravel(), columns[1:].ravel()]
    src, dst = np.concatenate(src), np.concatenate(dst)
    n = x.shape[0]
    adjacency = sparse.coo_matrix(
        (np.ones(2 * len(src), dtype=bool), (np.concatenate([src, dst]), np.concatenate([dst, src]))), shape=(n, n)
    ).tocsr()
    return adjacency


def ghost_rings(adjacency: sparse.csr_matrix, core_mask: np.ndarray, n_hops: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes within ``n_hops`` edges of a core set, excluding the core.

    Returns:
        (rows, hops): rows of the ghost nodes ordered by hop then row, and the hop distance of each.
    """
    hop = np.full(core_mask.shape[0], -1, dtype=int)
    hop[core_mask] = 0
    frontier = core_mask.astype(np.float32)
    for h in range(1, n_hops + 1):
        reached = (adjacency @ frontier > 0) & (hop < 0)
        hop[reached] = h
        frontier = reached.astype(np.float32)
    rows = np.where(hop > 0)[0]
    rows = rows[np.argsort(hop[rows], kind="stable")]
    return rows, hop[rows]


def create_patches(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    slice_id: np.ndarray,
    n_patches: int,
    slice_split: int,
    adjacency: sparse.csr_matrix,
    ghost_hops: int,
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

    # Ghost nodes: every node within ghost_hops mesh edges of the core, grouped by the patch that owns it.
    # The neighbours of a patch are the patches its ghost nodes belong to.
    patch_neighbours: dict[int, set[int]] = defaultdict(set)
    patch_ghost_points: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    for pid in np.unique(patch_ids):
        ghost_rows, _ = ghost_rings(adjacency, patch_ids == pid, ghost_hops)
        owners = patch_ids[ghost_rows]
        for nbr_pid in np.unique(owners):
            patch_neighbours[int(pid)].add(int(nbr_pid))
            patch_ghost_points[int(pid)][int(nbr_pid)] = ghost_rows[owners == nbr_pid]

    return patch_ids, slice_groups, patch_neighbours, patch_ghost_points


def build_patch_metadata(ref_df: pd.DataFrame, cfg: PipelineConfig) -> dict[str, dict[str, Any]]:
    patch_input_df = ref_df
    if cfg.use_normalized_coords_for_patching:
        patch_input_df = _normalize_columns(ref_df, COORD_COLS)

    # Connectivity depends only on the mesh, so it is built from the raw coordinates
    adjacency = build_mesh_adjacency(
        x=ref_df[COORD_COLS[0]].to_numpy(),
        y=ref_df[COORD_COLS[1]].to_numpy(),
        slice_id=ref_df[SLICE_COL].to_numpy(),
        max_edge_factor=cfg.max_edge_factor,
    )
    print(f"Mesh adjacency: {ref_df.shape[0]} nodes, {adjacency.nnz // 2} edges")

    patch_ids, slice_groups, patch_neighbours, patch_ghost_points = create_patches(
        x=patch_input_df[COORD_COLS[0]].to_numpy(),
        y=patch_input_df[COORD_COLS[1]].to_numpy(),
        z=patch_input_df[COORD_COLS[2]].to_numpy(),
        slice_id=patch_input_df[SLICE_COL].to_numpy(),
        n_patches=cfg.n_patches,
        slice_split=cfg.slice_split,
        adjacency=adjacency,
        ghost_hops=cfg.ghost_hops,
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
        print(
            f"Patch {int(patch_id)}: {len(core_nodes)} core, {len(ghost_nodes)} ghost, "
            f"neighbours {patch_data[str(int(patch_id))]['neighbour_patches']}"
        )

    return patch_data


def save_patch_metadata(patch_data: dict[str, dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        # Keep the first backup made on a given day rather than overwriting it on a rerun
        backup_path = f"{out_path}.bak_{datetime.date.today():%Y%m%d}"
        if not os.path.exists(backup_path):
            shutil.copy2(out_path, backup_path)
            print(f"Backed up existing patch metadata to {backup_path}")
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
        low_memory=False,
    )[TARGET_COLS]
    ts_vals = ts_df.to_numpy()

    if clip_negative_mass_concentration:
        neg_mask = ts_vals[:, 0] < 0
        if np.any(neg_mask):
            ts_vals = ts_vals.copy()
            ts_vals[neg_mask, 0] = 0

    forcing_df = pd.read_csv(
        os.path.join(forcings_data_dir, ts_file),
        low_memory=False,
    )[FORCING_COLS]
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

        # Write to a temporary directory and swap it in, so an interrupted run never leaves a patch
        # with a mix of old and new files
        patch_dir_path = os.path.join(cfg.patch_data_dir, f"patch_{int(patch_id_str):03d}")
        tmp_dir_path = f"{patch_dir_path}.tmp"
        shutil.rmtree(tmp_dir_path, ignore_errors=True)
        os.makedirs(tmp_dir_path)

        np.save(os.path.join(tmp_dir_path, "core_obs.npy"), core_patch_data)
        np.save(os.path.join(tmp_dir_path, "ghost_obs.npy"), ghost_patch_data)
        np.save(os.path.join(tmp_dir_path, "core_coords.npy"), core_coords)
        np.save(os.path.join(tmp_dir_path, "ghost_coords.npy"), ghost_coords)
        np.save(os.path.join(tmp_dir_path, "core_forcings.npy"), core_forcings_data)
        np.save(os.path.join(tmp_dir_path, "ghost_forcings.npy"), ghost_forcings_data)

        if os.path.exists(patch_dir_path):
            shutil.rmtree(patch_dir_path)
        os.rename(tmp_dir_path, patch_dir_path)


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

    if cfg.metadata_only:
        print("\n--metadata-only: skipping patch arrays.")
        return

    print("\nStep 2/2: Building patch arrays...")
    create_patch_arrays(patch_metadata, ref_df, ts_files, cfg)

    print("\nDone.")


if __name__ == "__main__":
    main()
