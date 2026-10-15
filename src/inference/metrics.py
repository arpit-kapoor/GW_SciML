"""
Metrics computation utilities for model evaluation.

Provides functions for:
- Computing performance metrics (KGE, R², Relative L2 Error)
- Denormalizing predictions
- Saving metrics to files
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def compute_normalized_mae(predictions, observations):
    """
    Compute Mean Absolute Error normalized by standard deviation of observations.
    
    NMAE = MAE / std(observations)
    
    Args:
        predictions (np.ndarray): Predicted values (1D array)
        observations (np.ndarray): Observed/target values (1D array)
        
    Returns:
        float: Normalized MAE
    """
    # Ensure inputs are 1D
    pred_flat = predictions.flatten()
    obs_flat = observations.flatten()
    
    # Compute MAE
    mae = np.mean(np.abs(pred_flat - obs_flat))
    
    # Normalize by std of observations
    obs_std = np.std(obs_flat)
    
    # Avoid division by zero
    nmae = mae / (obs_std)
    
    return nmae


def compute_r2_score(predictions, observations):
    """
    Compute R² score (coefficient of determination).
    
    R² = 1 - SS_res/SS_tot = 1 - Σ(y - ŷ)² / Σ(y - ȳ)²
    
    Args:
        predictions (np.ndarray): Predicted values
        observations (np.ndarray): Observed/target values
        
    Returns:
        float: R² score
    """
    pred_flat = predictions.flatten()
    obs_flat = observations.flatten()
    return r2_score(obs_flat, pred_flat)


def compute_kge(simulations, observations):
    """
    Compute Kling-Gupta Efficiency (KGE) and its components.
    
    KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)
    where:
        r = correlation coefficient
        alpha = std(sim) / std(obs)  (variability ratio)
        beta = mean(sim) / mean(obs)  (bias ratio)
    
    Args:
        simulations (np.ndarray): Simulated/predicted values (1D array)
        observations (np.ndarray): Observed/target values (1D array)
        
    Returns:
        dict: Dictionary with kge, r, alpha, beta
    """
    # Ensure inputs are 1D
    sim_flat = simulations.flatten()
    obs_flat = observations.flatten()
    
    # Correlation coefficient (r)
    r = np.corrcoef(sim_flat, obs_flat)[0, 1]
    
    # Variability ratio (alpha)
    alpha = np.std(sim_flat) / (np.std(obs_flat) + 1e-10)
    
    # Bias ratio (beta)
    beta = np.mean(sim_flat) / (np.mean(obs_flat) + 1e-10)
    
    # KGE
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    
    return {
        'kge': kge,
        'r': r,
        'alpha': alpha,
        'beta': beta
    }


def denormalize_observations(normalized_data, obs_transform, target_col_indices):
    """
    Denormalize observations from normalized scale back to original scale.
    
    Args:
        normalized_data: Array of shape [N_samples, N_points, output_window_size, n_target_cols]
        obs_transform: Normalize transform object with mean and std attributes
        target_col_indices: List of target column indices
        
    Returns:
        Array in original scale with same shape
    """
    # Get mean and std for target columns
    mean_all = obs_transform.mean
    std_all = obs_transform.std
    
    mean = mean_all[target_col_indices]
    std = std_all[target_col_indices]
    
    # Convert to numpy if they are tensors
    import torch
    if isinstance(mean, torch.Tensor):
        mean = mean.cpu().numpy()
    if isinstance(std, torch.Tensor):
        std = std.cpu().numpy()
    
    # Reshape mean and std for broadcasting
    # Shape: [1, 1, 1, n_target_cols]
    mean = mean.reshape(1, 1, 1, -1)
    std = std.reshape(1, 1, 1, -1)
    
    # Denormalize: data = normalized * std + mean
    denormalized = normalized_data * std + mean
    
    return denormalized


def compute_metrics(results_dict, target_cols, target_col_indices, obs_transform=None):
    """
    Compute relative L2 error, R² score, and KGE for each target column on train and val sets.
    
    All metrics are computed on denormalized (original scale) data. Nodes with non-finite predictions at any
    step (a blown-up rollout) are logged and excluded; their count is returned with the metrics.

    Args:
        results_dict: Dictionary containing train and val predictions and targets
        target_cols: List of target column names
        target_col_indices: List of target column indices
        obs_transform: Normalize transform object for denormalization (None if data already denormalized)
        
    Returns:
        Dictionary containing metrics for each dataset and target column
    """
    metrics = {}
    
    for dataset_name in [d for d in ['train', 'val'] if d in results_dict]:
        # Get data (may be normalized or denormalized)
        predictions_data = results_dict[dataset_name]['predictions']  # [N_samples, N_points, output_window_size, n_target_cols]
        targets_data = results_dict[dataset_name]['targets']
        
        # Denormalize to original scale if transform provided
        if obs_transform is not None:
            predictions = denormalize_observations(predictions_data, obs_transform, target_col_indices)
            targets = denormalize_observations(targets_data, obs_transform, target_col_indices)
        else:
            # Data is already denormalized
            predictions = predictions_data
            targets = targets_data
        
        metrics[dataset_name] = {}

        # Nodes whose predictions became non-finite at any time step (a blown-up rollout) are counted, logged
        # and excluded, so the metrics describe the remaining nodes instead of failing on NaN
        node_counts = _report_blown_up_nodes(predictions, dataset_name)
        node_ok = node_counts.pop('_mask')
        predictions = predictions[:, node_ok]
        targets = targets[:, node_ok]

        # Compute metrics for each target column
        for col_idx, col_name in enumerate(target_cols):
            print(f"Computing metrics for {col_name} ({dataset_name}) on {node_counts['n_nodes_evaluated']} nodes...")
            col_predictions = predictions[:, :, :, col_idx]  # [N_samples, N_points, output_window_size]
            col_targets = targets[:, :, :, col_idx]
            if col_predictions.size == 0:
                metrics[dataset_name][col_name] = dict.fromkeys(
                    ['nmae', 'rel_l2_error', 'r2', 'kge', 'kge_r', 'kge_alpha', 'kge_beta'], np.nan)
                metrics[dataset_name][col_name].update(node_counts)
                continue
            _report_diverging_nodes(col_predictions, col_targets, col_name, dataset_name)
            
            # Flatten for metric computation
            col_predictions_flat = col_predictions.flatten()
            col_targets_flat = col_targets.flatten()
            
            # 1. Normalized MAE
            nmae = compute_normalized_mae(col_predictions_flat, col_targets_flat)
            
            # 2. Relative L2 Error (using numpy implementation)
            # Relative L2 = ||pred - target||_2 / ||target||_2
            diff = col_predictions_flat - col_targets_flat
            l2_error = np.linalg.norm(diff)
            l2_norm_target = np.linalg.norm(col_targets_flat)
            rel_l2_error = l2_error / (l2_norm_target + 1e-10)
            
            # 3. Kling-Gupta Efficiency (KGE)
            kge_results = compute_kge(col_predictions_flat, col_targets_flat)
            
            # 4. R² score (coefficient of determination)
            r2 = compute_r2_score(col_predictions_flat, col_targets_flat)
            
            metrics[dataset_name][col_name] = {
                'nmae': nmae,
                'rel_l2_error': rel_l2_error,
                'r2': r2,
                'kge': kge_results['kge'],
                'kge_r': kge_results['r'],
                'kge_alpha': kge_results['alpha'],
                'kge_beta': kge_results['beta'],
                **node_counts,
            }

    return metrics


def _report_blown_up_nodes(predictions, dataset_name):
    """
    Count and log the nodes whose predictions are non-finite at any time step.

    Args:
        predictions: Array of shape [N_samples, N_points, output_window_size, n_target_cols]
        dataset_name: Name used in the log message

    Returns:
        dict: node counts, plus '_mask' (boolean [N_points], True for nodes kept in the metrics)
    """
    finite_per_step = np.isfinite(predictions).all(axis=(2, 3))   # [N_samples, N_points]
    node_ok = finite_per_step.all(axis=0)
    n_nodes = int(node_ok.size)
    n_bad = n_nodes - int(node_ok.sum())
    if n_bad:
        # first sample (time step) at which each blown-up node became non-finite, 1-based
        first_bad = np.argmax(~finite_per_step[:, ~node_ok], axis=0) + 1
        print(f"[{dataset_name}] BLOWN-UP NODES: {n_bad} of {n_nodes} nodes ({100 * n_bad / n_nodes:.2f}%) have "
              f"non-finite predictions; first non-finite at step {int(first_bad.min())} "
              f"(median {int(np.median(first_bad))}, last {int(first_bad.max())}) of {predictions.shape[0]}. "
              f"These nodes are excluded from the metrics.")
    else:
        print(f"[{dataset_name}] All {n_nodes} nodes have finite predictions at every step.")
    return {'n_nodes_evaluated': n_nodes - n_bad, 'n_nodes_nonfinite': n_bad,
            'pct_nodes_nonfinite': 100.0 * n_bad / n_nodes, '_mask': node_ok}


def _report_diverging_nodes(col_predictions, col_targets, col_name, dataset_name, factor=10.0):
    """
    Log finite nodes whose predictions leave the target range by more than `factor` times its width at some
    step: they are still included in the metrics but usually dominate them.
    """
    t_min, t_max = np.nanmin(col_targets), np.nanmax(col_targets)
    width = max(t_max - t_min, 1e-12)
    outside = ((col_predictions < t_min - factor * width) | (col_predictions > t_max + factor * width))
    n_div = int(outside.reshape(outside.shape[0], outside.shape[1], -1).any(axis=(0, 2)).sum())
    if n_div:
        print(f"[{dataset_name}] {col_name}: {n_div} of {col_predictions.shape[1]} remaining nodes have finite "
              f"predictions more than {factor:g}x the target range outside it at some step (still included).")


def save_metrics(metrics, target_cols, results_dir):
    """
    Save computed metrics to a CSV file in long format.
    
    Format (Option B - Long Format):
    metric_name | variable | train | val
    
    Args:
        metrics: Dictionary containing metrics for each dataset and target column
        target_cols: List of target column names
        results_dir: Directory to save metrics file
    """
    metrics_file = os.path.join(results_dir, 'metrics.csv')
    
    # Build list of rows for DataFrame
    rows = []
    
    # Metrics to extract (in desired order)
    metric_names = ['nmae', 'rel_l2_error', 'r2', 'kge', 'kge_r', 'kge_alpha', 'kge_beta',
                    'n_nodes_evaluated', 'n_nodes_nonfinite', 'pct_nodes_nonfinite']
    
    for metric_name in metric_names:
        for col_name in target_cols:
            row = {
                'metric_name': metric_name,
                'variable': col_name,
                'train': metrics.get('train', {}).get(col_name, {}).get(metric_name, None),
                'val': metrics.get('val', {}).get(col_name, {}).get(metric_name, None)
            }
            rows.append(row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(metrics_file, index=False, float_format='%.6f')
    
    print(f"Saved metrics to {metrics_file}")
    print("\nMetrics Preview:")
    print(df.head(10).to_string(index=False))
