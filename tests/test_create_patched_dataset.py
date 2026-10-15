"""Mesh adjacency, neighbour patches and ghost rings built by create_patched_dataset."""
from collections import deque

import numpy as np
import pytest

from create_patched_dataset import build_mesh_adjacency, create_patches, ghost_rings

NX, NY, NS = 12, 10, 8


def _mesh(seed=0):
    """Prismatic mesh: a jittered NX x NY grid (spacing 40 m) on NS slices (Z spacing 2 m), rows shuffled."""
    rng = np.random.default_rng(seed)
    gx, gy = np.meshgrid(np.arange(NX) * 40.0, np.arange(NY) * 40.0, indexing="ij")
    xy = np.stack([gx.ravel(), gy.ravel()], 1) + rng.uniform(-2, 2, size=(NX * NY, 2))
    x = np.tile(xy[:, 0], NS)
    y = np.tile(xy[:, 1], NS)
    slice_id = np.repeat(np.arange(1, NS + 1), NX * NY)
    z = -2.0 * slice_id + rng.uniform(-0.1, 0.1, size=slice_id.shape)
    col = np.tile(np.arange(NX * NY), NS)  # column = ix * NY + iy
    perm = rng.permutation(len(x))
    return x[perm], y[perm], z[perm], slice_id[perm], col[perm]


def _patches(adjacency, ghost_hops=2):
    x, y, z, s, _ = _mesh()
    return create_patches(x, y, z, s, n_patches=4, slice_split=2, adjacency=adjacency, ghost_hops=ghost_hops)


def _bfs_ring(adjacency, core_rows, n_hops):
    dist = {int(r): 0 for r in core_rows}
    queue = deque(dist)
    while queue:
        i = queue.popleft()
        if dist[i] == n_hops:
            continue
        for j in adjacency.indices[adjacency.indptr[i]:adjacency.indptr[i + 1]]:
            if int(j) not in dist:
                dist[int(j)] = dist[i] + 1
                queue.append(int(j))
    return {i for i, d in dist.items() if d > 0}


def test_adjacency_matches_mesh():
    x, y, _, s, col = _mesh()
    a = build_mesh_adjacency(x, y, s)
    assert (a != a.T).nnz == 0 and a.diagonal().sum() == 0
    rows, cols = a.nonzero()
    ds = np.abs(s[rows] - s[cols])
    # Vertical edges join the same column in consecutive slices; horizontal edges stay in one slice
    assert set(np.unique(ds)) == {0, 1}
    assert np.all(col[rows[ds == 1]] == col[cols[ds == 1]])
    # Every grid side is an edge. Interior edges are no longer than a cell diagonal; along the domain boundary
    # Delaunay slivers can join nodes a few cells apart, bounded by the max_edge_factor cut
    ix, iy = col // NY, col % NY
    lookup = {(int(c), int(sl)): r for r, (c, sl) in enumerate(zip(col, s))}
    for r in range(len(x)):
        for dx, dy in ((1, 0), (0, 1)):
            if ix[r] + dx < NX and iy[r] + dy < NY:
                assert a[r, lookup[(int((ix[r] + dx) * NY + iy[r] + dy), int(s[r]))]]
    h = ds == 0
    length = np.hypot(x[rows] - x[cols], y[rows] - y[cols])
    on_boundary = (ix == 0) | (ix == NX - 1) | (iy == 0) | (iy == NY - 1)
    interior = h & ~(on_boundary[rows] & on_boundary[cols])
    assert length[interior].max() < 40 * np.sqrt(2) + 6
    assert length[h].max() <= 3.0 * 47


def test_non_prismatic_mesh_rejected():
    x, y, _, s, _ = _mesh()
    with pytest.raises(ValueError, match="not prismatic"):
        build_mesh_adjacency(x[1:], y[1:], s[1:])
    y_moved = y.copy()
    y_moved[np.where(s == 3)[0][0]] += 5.0
    with pytest.raises(ValueError, match="not prismatic"):
        build_mesh_adjacency(x, y_moved, s)


def test_ghost_rings_match_bfs():
    x, y, _, s, _ = _mesh()
    a = build_mesh_adjacency(x, y, s)
    core = (x < 200) & (s <= 4)
    for n_hops in (0, 1, 2, 3):
        rows, hops = ghost_rings(a, core, n_hops)
        assert set(rows.tolist()) == _bfs_ring(a, np.where(core)[0], n_hops)
        assert len(rows) == len(set(rows.tolist())) and not core[rows].any()
        assert np.all(np.diff(hops) >= 0) and (n_hops == 0 or hops.max() == n_hops)


def test_patch_ghosts_and_neighbours():
    x, y, _, s, _ = _mesh()
    a = build_mesh_adjacency(x, y, s)
    patch_ids, slice_groups, neighbours, ghosts = _patches(a)
    assert sorted(np.unique(patch_ids)) == [1, 2, 3, 4]
    for pid in (1, 2, 3, 4):
        core = patch_ids == pid
        ghost_rows = np.concatenate(list(ghosts[pid].values()))
        assert set(ghost_rows.tolist()) == _bfs_ring(a, np.where(core)[0], 2)
        assert len(ghost_rows) == len(set(ghost_rows.tolist())) and not core[ghost_rows].any()
        for nbr_pid, rows in ghosts[pid].items():
            assert np.all(patch_ids[rows] == nbr_pid)
        assert neighbours[pid] == set(ghosts[pid])
        # Lateral neighbour in the same slice group and a vertical neighbour in the other group
        same_group = {q for q in (1, 2, 3, 4) if q != pid and slice_groups[patch_ids == q][0] == slice_groups[core][0]}
        assert same_group and same_group <= neighbours[pid]
        assert neighbours[pid] - same_group


def test_create_patches_deterministic():
    x, y, _, s, _ = _mesh()
    a = build_mesh_adjacency(x, y, s)
    first, second = _patches(a), _patches(a)
    assert np.array_equal(first[0], second[0])
    for pid in first[3]:
        assert first[3][pid].keys() == second[3][pid].keys()
        for q in first[3][pid]:
            assert np.array_equal(first[3][pid][q], second[3][pid][q])
