# GINO rollout blow-up: findings and changes

Status as of 2026-10-15. All experiments ran on Katana. Unless stated otherwise:

- Training data is resolution 0.3.
- Rollouts are full resolution (61,360 core nodes over 20 patches) over the validation period: 564 time steps when one step is fed back per call, 565 when five are.
- A node is **blown up** if any of its predictions becomes non-finite (NaN or inf) during the rollout. "Bad" adds nodes whose head error exceeds 1 m at some step.

## Summary

- **The blow-up comes from the trained model's dynamics under autoregressive feedback.** The rollout code and the normalisation are not the cause.
- **Long rollouts are not yet reliably stable, under any setting tested.** The 500-epoch 5-out run with the fixed grid and radius 0.24 still blows up:
  - 5.6% of nodes when five steps are fed back per call (`--val-stride 5`), all in patch 1;
  - 31–41% of nodes when one step is fed back per call.
- **What makes it worse:**
  - training with a one-step output window;
  - rolling out at a different GNO radius from the one used in training;
  - feeding back one step per call instead of the whole output window;
  - longer training.
- **What is fixed so far:**
  - The latent grid now stays in the same place when the resolution changes.
  - The neighbourhood reduction is about 100× faster.
  - The learning-rate schedule now decays.
  - Prediction saves only the steps the rollout actually feeds back.
  - Metrics now exclude and log blown-up nodes instead of crashing.
- **Still open:**
  - Patch 1 diverges in the stride-5 rollout.
  - Rollout-aware fine-tuning with three unrolled windows did not fix it.

## 1. What was ruled out

### Rollout implementation

The repo's `generate_rolling_predictions` (`src/inference/inference_utils.py`) was compared with an independently written rollout that stays in normalised space throughout. The two used the same weights, all 20 patches and both strides. Predictions agree to about 1e-6 (normalised) at every step until a patch is already diverging; after that, float rounding is amplified. This held for:
- a 5-out model trained for 100 epochs;
- the Gadi 1-out checkpoint;
- a 1-out control model;
- the 500-epoch 5-out run.

The rollout loop never denormalises or renormalises: the buffer holds normalised values, and denormalisation happens once, after the loop. The 3D FNO uses the same function and is stable.

### Normalisation

- **Mismatched stats don't cause it.** Observation statistics are not stored in checkpoints. `calculate_obs_transform` recomputes them from `all/`, and the 2026-06-10 commit changed them: head mean 0.396 → 0.343, std 0.213 → 0.234. Rolling out with the old statistics on purpose does not create a blow-up:
  - Gadi checkpoint: 64.9% bad nodes with either set of statistics.
  - 5-out models trained for 100 epochs: 0% bad with either set.
- **Katana's own checkpoints are stale.** Every checkpoint trained on Katana predates that fix and the patch regeneration. On current data their one-step relative L2 is 7–9, so they were not used. The training-time statistics should be saved in the checkpoint and reused at prediction time; this is not done yet.

### Forcing offset

At sea-boundary nodes, `head[t+1] == head_bc[t]` to 1e-16, and consecutive steps alternate between high and low tide. The model is therefore given the opposite-phase tide for the step it predicts. This is by design, and FNO and F-FNO are stable with the same pairing, so it is not the cause.

## 2. What drives the blow-up

### GNO radius at inference must match training

The output GNO sums over latent neighbours without normalising (`gino.py`, `reduction='sum'`). Changing the radius at prediction time changes the size of that sum.

| Model | Radius 0.21 (= training) | 0.24 | 0.28 |
|---|---|---|---|
| 5-out, 100 epochs, trained at 0.21 | 0% bad | 5.6% (patch 1 to NaN) | 40% bad, 15% non-finite |

The Gadi checkpoint (`standard/training_20260715_181457`) was trained at 0.21, confirmed from the checkpoint arguments, and blows up at 0.21 as well. Other radii (0.18 to 0.28) change which patches fail but don't stabilise it. `--gno-radius` has been removed from `PRED_ARGS` so prediction always uses the checkpoint's radius.

### Output window: one step against five

Both runs used the same `standard.sh` settings, 100 epochs, seed 42, radius 0.21 and the original grid. Both rollouts feed back one step per call.

| | 5-in/1-out | 5-in/5-out |
|---|---|---|
| Bad nodes | 64.9% (13 patches; 8 reach NaN) | 0% |
| Gadi 1-out checkpoint, 500 epochs | 64.9% | n/a |

The 5-out model was stable even feeding back only its first predicted step. So the training target, not just the number of feedback calls, made the difference.

### Longer training makes 5-out models fragile again

Run `GINO/standard/training_20261015_011834`: 500 epochs, 5-out, radius 0.24, grid fix, cosine schedule, final validation loss 0.097.

| Rollout | Nodes blown up | Patches reaching NaN |
|---|---|---|
| 5 steps fed back per call (`--val-stride 5`) | 3,409 (5.6%) | 1 (head error > 1 m from step about 92, NaN at step 446) |
| 1 step fed back per call | 22,436–25,300 (36.6–41.3%) | 1, 2, 3, 5, 8, 11, (16), 20 |

With five steps per call, the other 19 patches stay finite, with final-step head errors of 0.02–0.16 m. With one step per call, head error passes 1 m around steps 20–150 and NaNs appear at steps 114–484.

The 100-epoch 5-out model had a validation loss of about 0.17 and was stable. This model has a lower one-step loss but tolerates its own errors much less.

### Rollout-aware fine-tuning (first attempt)

The 500-epoch checkpoint was fine-tuned for 20 epochs, lr 1e-4 decaying to 1e-5, batch 256.
- **K = 3 unrolled windows (15 steps).** Window k+1 receives the model's own detached prediction as its history.
- **Control: K = 1.** Ordinary continued training.

| | Before | K = 1 control | K = 3 unrolled |
|---|---|---|---|
| Teacher-forced val loss (mean of 20 patches) | 0.196 | 0.196 | 0.198 |
| 5 steps per call: nodes blown up | 5.6% (patch 1) | 5.6% (patch 1) | 5.6% (patch 1) |
| 1 step per call: nodes blown up | 36.6% (7 patches) | 31.1% (6 patches) | 15.3% (3 patches) |

The unrolled run cuts one-step-per-call failures by more than the control does. It does not fix patch 1 in the stride-5 rollout. Patch 1 drifts slowly: an error of 1 m takes about 18 rollout calls, longer than a 3-window unroll exposes.

## 3. Latent grid, coordinate normalisation and radius

### Run-time latent grid moved with resolution (fixed)

The latent grid (16×16×8) used to be built from the bounding box of whichever points were loaded.
- Resolution 0.3 drops some top or bottom nodes, so in 13 of 20 patches the latent z-planes at prediction (full resolution) sat 1.0–1.7 z-cells from where they were in training.
- As a result, 57–87% of nodes in those patches had a different set of output-GNO neighbours.
- With old checkpoints, building the prediction grid from the training-resolution points cut the one-step loss by 33–41%.
- Stability was unaffected: 1-out models blow up even when rolled out entirely at resolution 0.3.

### Anisotropic reach of an isotropic radius

Coordinates are normalised per axis: std 569 m (x), 566 m (y), 15.3 m (z).

| Radius | Horizontal reach | Vertical reach | Vertical reach / z-spacing (4.3–7.2 m) |
|---|---|---|---|
| 0.21 | 120 m (1.1–2.2 cells) | 3.2 m | 0.45–0.75 |
| 0.24 | 137 m | 3.7 m | 0.51–0.85 |
| 0.27 | 154 m | 4.1 m | 0.57–0.96 |

Each mesh node effectively reads one latent z-plane. At 0.21, 9–15% of nodes in patches 6, 7, 13, 14 and 15 have no latent neighbour at all, so their output ignores the input. About half the latent nodes (33–67% per patch) lie outside the irregular mesh.

### Radius sweep (coverage, on the fixed grid)

The input-GNO column counts only latent nodes inside the mesh.

| Radius | Inside latent nodes with no input point (res 0.3 / 1.0) | Mesh nodes with no latent neighbour (mean / worst patch) | Edges vs 0.21 |
|---|---|---|---|
| 0.18 | 43% / 25% | 15% / 35% | 0.6× |
| 0.21 | 35% / 18% | 3.1% / 15% | 1.0× |
| 0.24 | 28% / 13% | 0.3% / 6.7% | 1.5× |
| 0.27 | 22% / 8% | 0 / 0 | 2.1× |
| 0.30 | 17% / 5% | 0 / 0 | 2.9× |

### Radius comparison (training, `standard.sh` with grid fix, 100 epochs)

| | 0.21 | 0.24 | 0.27 |
|---|---|---|---|
| One-step val loss at full resolution (mean of 20 patches) | 0.287 | **0.214** | 0.226 |
| Same, excluding patch 11 | 0.276 | 0.203 | 0.204 |
| 564-step rollout, nodes blown up | 0% | 0% | 0% |

0.24 was chosen: as accurate as 0.27 and cheaper. Radius 0.27 needed the batch split in two to fit on a 32 GB V100.

## 4. Code and configuration changes

None of these are committed yet.

| File | Change |
|---|---|
| `src/data/patch_dataset_multi_col.py` | Stores each patch's full-resolution bounding box (`latent_bbox`), computed before subsampling. |
| `src/data/data_utils.py` | `_latent_grid_bounds` builds the latent grid from `latent_bbox` in `make_collate_fn` and `make_spacetime_collate_fn`; it falls back to the points' box for other datasets. This also affects FNO and F-FNO. |
| `src/models/neuralop/segment_csr.py` | The native per-neighbourhood Python loop is replaced by one `index_add`. Outputs and gradients match the loop to about 6e-7 relative; a forward plus backward pass drops from 4.8–8.6 s to 0.05–0.09 s; one rollout call from 644 ms to 6 ms. Not bit-reproducible on GPU unless `torch.use_deterministic_algorithms(True)` is set. |
| `src/inference/metrics.py` | `compute_metrics` logs `BLOWN-UP NODES: n of N (x%) … first non-finite at step …`, excludes those nodes, and logs finite nodes far outside the target range. `metrics.csv` gains `n_nodes_evaluated`, `n_nodes_nonfinite` and `pct_nodes_nonfinite`. |
| `gino_train.py` | The post-training `train_/val_preds.npy` and `_targets.npy` keep only the first output step (6.2 GB → 1.2 GB). |
| `gino_predict.py` | Saves, evaluates and plots the steps the rollout advances by: the first step of each window at stride 1, all 5 steps at stride 5. Output shape is `[T, N, 1, C]`; `metadata.json` records `saved_steps_per_window`. |
| `hpc/configs/gino/standard.sh` | 500 epochs; batch 256 (your change); cosine schedule (1e-3 → 1e-5; the old exponential setting decayed only 18% over 100 epochs); radius 0.24; no `--gno-radius` in `PRED_ARGS`; `--val-stride 5` (your change). |
| `hpc/gino_train.pbs`, `hpc/gino_predict.pbs` | `export NUMPY_MADVISE_HUGEPAGE=0`. On fragmented nodes NumPy's huge-page requests made allocations of about 2 GB crawl (one save stalled for over 20 min). |
| `tests/test_segment_csr.py` | New: reduction against the original loop (sum/mean, batched/unbatched, empty neighbourhoods, forward and gradients, CPU/GPU) and a full GINO forward/backward. |
| `tests/test_metrics.py` | New: non-finite nodes are excluded, counted and logged. |

`torch-scatter` was tried and removed: the PyG wheel needs glibc 2.32, and Katana has 2.28.

`gino_train.py` ignores `--lr-warmup-epochs`: cosine is plain `CosineAnnealingLR`.

## 5. Other observations

- **Run-to-run variation near the stability edge.** Two prediction processes on the same checkpoint differ at a single node by up to about 0.15% (mass) at step 0. My explanation (not directly verified): a latent node almost exactly at the radius falls in or out of the hard neighbour cutoff because of float rounding. Within one process, results agree to 1e-6. Blown-up node counts for one step per call therefore vary by a few percent between runs (36.6–41.3%).
- **Prediction log `gino_standard_predict_9519107`** still ends with `ValueError: Input contains NaN`. It ran before the metrics change.
- **Disk.** `/srv/scratch/z5370003` was full (128/128 GB) during testing, which made `np.save` fail with `problem writing element …`.

## 6. Open issues and next steps

1. **Patch 1 diverges in the stride-5 rollout** of the 500-epoch model. Candidates:
   - longer unrolls (K = 8–10);
   - noise on the input history during training;
   - a diagnostic of where in patch 1, and in which variable, the drift starts. Patch 1 has the densest latent neighbourhoods and the lowest one-step loss (0.053).
2. **Pick checkpoints by rollout stability, not validation loss.** Longer training lowered the one-step loss but reduced stability.
3. **Store normalisation statistics in the checkpoint** and reuse them at prediction time.
4. **Optional:**
   - normalising the output-GNO sum (`reduction='mean'`) so its scale doesn't depend on neighbour count;
   - anisotropic neighbour search or more latent z-planes. All of these need retraining.
5. **GINO-level `segment_csr` test fails on an H200 but passes on a V100.** Suspected TF32 on newer GPUs amplifying float-rounding differences; to be confirmed.

## 7. Where things are

Experiment scripts, logs and `results.json` summaries are in `/srv/scratch/z5370003/projects/results/04_groundwater/variable_density/GINO_stability_exp/`. Large outputs (checkpoints, arrays) were deleted to free disk space; the scripts reproduce them.

| Script | Purpose |
|---|---|
| `rollout_audit.py` | Train (`standard.sh` settings) or load a checkpoint; independent vs repo rollout; mismatched stats; other radii. |
| `grid_diag.py` | Latent-grid geometry and rollouts with the training-resolution grid. |
| `radius_sweep.py` | Neighbour coverage and cost against radius; grid-fix check. |
| `pushforward_finetune.py` | Rollout-aware fine-tuning (K unrolled windows) with before/after evaluation. |
| `segment_csr_real_check.py`, `determinism_check.py`, `step0_check.py` | Reduction equivalence and run-to-run checks. |

| Result folder | Contents |
|---|---|
| `audit_gadi_ckpt/` | Gadi 1-out checkpoint |
| `audit_standard/` | 5-out, 100 epochs, old grid |
| `audit_standard_wout1/` | 1-out control |
| `grid_diag/` | Grid diagnostics |
| `radius_sweep/` | Coverage sweep |
| `audit_fix_r021/`, `audit_fix_r024/`, `audit_fix_r027/` | Radius comparison with the grid fix |
| `audit_std500/` | 500-epoch run |
| `pushforward_k1/`, `pushforward_k3/` | Fine-tuning |
