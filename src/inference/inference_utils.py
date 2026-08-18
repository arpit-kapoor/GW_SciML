"""
Generic inference utilities for neural operator models.

Provides model loading, prediction generation, and configuration setup
that works with any model architecture (GINO, FNO, UNO, etc.).
"""

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.batch_sampler import PatchBatchSampler
from ..training.config import configure_device


def setup_inference_arguments(description, default_base_data_dir, default_results_dir,
                              add_model_specific_args=None):
    """
    Setup argument parser for inference with common parameters.
    
    Args:
        description (str): Description for argument parser
        default_base_data_dir (str): Default base data directory
        default_results_dir (str): Default results directory
        add_model_specific_args (callable, optional): Function to add model-specific args
        
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description=description)
    
    # Model and data paths
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--base-data-dir', type=str, default=default_base_data_dir,
                       help='Base data directory')
    parser.add_argument('--raw-data-subdir', type=str, default='all',
                       help='Raw data subdirectory name')
    parser.add_argument('--patch-data-subdir', type=str, default='filter_patch',
                       help='Patch data subdirectory name')
    parser.add_argument('--results-dir', type=str, default=default_results_dir,
                       help='Directory to save predictions and plots')
    
    # Model parameters
    parser.add_argument('--target-col', type=str, default='head',
                       help='Target observation column name')
    parser.add_argument('--input-window-size', type=int, default=10,
                       help='Input sequence length')
    parser.add_argument('--output-window-size', type=int, default=10,
                       help='Output sequence length')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for inference')
    
    # Resolution parameters for testing at different spatial resolutions
    parser.add_argument('--resolution-ratio', type=float, default=1.0,
                       help='Ratio of nodes to keep in each patch (0 < ratio <= 1.0). Default is 1.0 (no subsampling)')
    parser.add_argument('--min-resolution-ratio', type=float, default=0.20,
                       help='Minimum per-patch ratio used by dynamic subsampling (0 < min_ratio <= resolution_ratio)')
    parser.add_argument('--sampling-strategy', type=str, default='dynamic',
                       choices=['dynamic', 'static'],
                       help='Subsampling mode: dynamic adjusts ratio per patch by variability, static uses a uniform ratio for all patches')
    
    # Output control
    parser.add_argument('--metrics-only', action='store_true', default=False,
                       help='Only save metrics and metadata (no arrays or plots) to save disk space')
    parser.add_argument('--create-3d-plots', action='store_true', default=False,
                       help='Create 3D scatter plots and videos (disabled by default to save time/storage)')
    
    # Device configuration
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, or auto)')
    
    # Add model-specific arguments if provided
    if add_model_specific_args is not None:
        parser = add_model_specific_args(parser)
    
    args = parser.parse_args()
    
    # Construct full paths
    args.raw_data_dir = f"{args.base_data_dir}/{args.raw_data_subdir}"
    args.patch_data_dir = f"{args.base_data_dir}/{args.patch_data_subdir}"
    
    # Configure device
    args = configure_device(args)
    
    print(f"Model path: {args.model_path}")
    print(f"Raw data directory: {args.raw_data_dir}")
    print(f"Patch data directory: {args.patch_data_dir}")
    
    return args


def load_checkpoint(checkpoint_path, device='cpu'):
    """
    Load checkpoint file and extract components.
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        device (str): Device to map checkpoint to
        
    Returns:
        dict: Checkpoint dictionary containing model_state_dict, args, etc.
    """
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Print checkpoint info
    if 'args' in checkpoint:
        print("\nCheckpoint configuration found")
    if 'epoch' in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch'] + 1}")
    if 'train_losses' in checkpoint and len(checkpoint['train_losses']) > 0:
        print(f"Final train loss: {checkpoint['train_losses'][-1]:.4f}")
    if 'val_losses' in checkpoint and len(checkpoint['val_losses']) > 0:
        print(f"Final val loss: {checkpoint['val_losses'][-1]:.4f}")
    
    return checkpoint


def create_model_from_checkpoint(checkpoint, model_factory, device='cpu'):
    """
    Create and initialize model from checkpoint using provided factory function.
    
    Args:
        checkpoint (dict): Checkpoint dictionary
        model_factory (callable): Function that takes (checkpoint, device) and returns model
        device (str): Device to load model on
        
    Returns:
        torch.nn.Module: Initialized model in eval mode
    """
    # Use factory function to create model with checkpoint config
    model = model_factory(checkpoint, device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded successfully and set to eval mode")
    return model


def generate_predictions(model, dataset, args, dataset_name='dataset',
                        forward_fn=None, collate_fn=None):
    """
    Generate predictions for an entire dataset.
    
    Args:
        model (torch.nn.Module): Model to use for prediction
        dataset: PyTorch dataset
        args (argparse.Namespace): Arguments with batch_size and device
        dataset_name (str): Name for logging
        forward_fn (callable, optional): Custom forward function(model, batch, args)
        collate_fn (callable, optional): Custom collate function
        
    Returns:
        dict: Dictionary with 'predictions', 'targets', 'coords', 'metadata' (normalized)
    """
    print(f"\nGenerating predictions for {dataset_name} dataset...")
    
    # Create sampler and data loader
    sampler = PatchBatchSampler(
        dataset,
        batch_size=args.batch_size,
        shuffle_within_batches=False,
        shuffle_patches=False,
        seed=None
    )
    
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)
    
    all_predictions = {}
    all_targets = {}
    all_coords = {}
    all_patch_metadata = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Processing {dataset_name}")):
            # Use custom forward function if provided, otherwise use default
            if forward_fn is not None:
                outputs, core_output, core_target, core_coords = forward_fn(model, batch, args)
            else:
                # Default forward pass (assumes GINO-style interface)
                outputs, core_output, core_target, core_coords = _default_forward(model, batch, args)
            
            patch_id = batch['patch_id']
            
            # Store results per patch
            if patch_id not in all_predictions:
                all_predictions[patch_id] = []
                all_targets[patch_id] = []
                all_coords[patch_id] = []
            
            all_predictions[patch_id].append(core_output)
            all_targets[patch_id].append(core_target)
            all_coords[patch_id].append(core_coords)
            
            # Store metadata
            batch_size = core_output.shape[0]
            for i in range(batch_size):
                all_patch_metadata.append({
                    'batch_idx': batch_idx,
                    'sample_idx': i,
                    'patch_id': patch_id,
                    'dataset': dataset_name,
                    'core_len': batch.get('core_len', core_output.shape[1])
                })
    
    # Concatenate results per patch
    for patch_id in all_predictions.keys():
        all_predictions[patch_id] = np.concatenate(all_predictions[patch_id], axis=0)
        all_targets[patch_id] = np.concatenate(all_targets[patch_id], axis=0)
        all_coords[patch_id] = np.concatenate(all_coords[patch_id], axis=0)
    
    # Concatenate all patches
    predictions = np.concatenate(list(all_predictions.values()), axis=1)
    targets = np.concatenate(list(all_targets.values()), axis=1)
    coords = np.concatenate(list(all_coords.values()), axis=1)
    
    print(f"{dataset_name} predictions shape: {predictions.shape}")
    print(f"{dataset_name} targets shape: {targets.shape}")
    print(f"{dataset_name} coords shape: {coords.shape}")
    
    return {
        'predictions': predictions,
        'targets': targets,
        'coords': coords,
        'metadata': all_patch_metadata
    }


# ---------------------------------------------------------------------------
# Rolling (autoregressive) prediction helpers
# ---------------------------------------------------------------------------

def _roll_buffer(buffer, prev_pred, input_window_size, output_window_size, n_obs_features):
    """
    Update the rolling observation buffer by shifting left and appending the
    latest model prediction.

    The buffer stores the last `input_window_size` time-steps worth of
    (normalised) observations, laid out as
    ``[N_points, input_window_size * n_obs_features]`` (row-major, same layout
    produced by ``_concat_sequence`` in the dataset).

    Args:
        buffer (np.ndarray): Current buffer ``[N_points, W_in * F]``.
        prev_pred (np.ndarray): Model output from the previous step
            ``[N_points, W_out * F]`` (core points only).
        input_window_size (int): W_in — number of time-steps in the input.
        output_window_size (int): W_out — number of time-steps in the output.
        n_obs_features (int): F — number of observation features per time-step.

    Returns:
        np.ndarray: Updated buffer ``[N_points, W_in * F]``.
    """
    n_points = buffer.shape[0]
    # Reshape to [N_points, W_in, F] for easy slicing along the time axis
    buf_3d = buffer.reshape(n_points, input_window_size, n_obs_features)

    # Shift: drop the oldest W_out steps, keep steps [W_out:]
    kept = buf_3d[:, output_window_size:, :]          # [N_points, W_in - W_out, F]

    # Append the prediction as the new trailing W_out steps
    pred_3d = prev_pred.reshape(n_points, output_window_size, n_obs_features)  # [N_points, W_out, F]
    updated = np.concatenate([kept, pred_3d], axis=1)  # [N_points, W_in, F]

    return updated.reshape(n_points, input_window_size * n_obs_features)


def generate_rolling_predictions(model, dataset, args, dataset_name='dataset',
                                  forward_fn=None, collate_fn=None,
                                  obs_transform=None):
    """
    Generate autoregressive (rolling-sequence) predictions for an entire dataset.

    This implements the initial-value-problem (IVP) evaluation setting:

    * **First window per patch** — the model receives the ground-truth input
      window (the true initial condition).
    * **Subsequent windows** — the model's previous output is rolled into the
      trailing ``output_window_size`` slots of the input buffer, replacing the
      corresponding ground-truth observations.  Leading slots that are not
      covered by the most-recent prediction continue to hold predictions from
      earlier steps (or the original ground truth for the very first window),
      which is the correct behaviour when ``input_window_size > output_window_size``.
    * **Forcings** — when ``forcings_required=True`` the dataset concatenates
      forcings channels onto ``x``.  In rolling mode only the *observation*
      portion of ``x`` is replaced; future forcings are still read from the
      current sample so that externally-prescribed boundary conditions remain
      accurate.

    The function returns the same dict shape as :func:`generate_predictions` so
    all downstream processing (reshape, denormalise, metrics, visualisation)
    works without modification.

    Args:
        model (torch.nn.Module): Trained model in eval mode.
        dataset: PyTorch dataset (must expose ``get_all_patch_ids()``).
        args (argparse.Namespace): Must contain ``batch_size``, ``device``,
            ``input_window_size``, ``output_window_size``, and optionally
            ``forcings_required``.
        dataset_name (str): Name used for logging.
        forward_fn (callable, optional): Custom ``(model, batch, args)`` →
            ``(outputs, core_output, core_target, core_coords)`` function.  If
            *None* the default GINO-style forward pass is used.
        collate_fn (callable, optional): Collate function passed to DataLoader.
        obs_transform (callable, optional): Normalization transform used for
            clamping physical constraints (like mass_concentration >= 0).

    Returns:
        dict: Same keys as :func:`generate_predictions`:
            ``predictions``, ``targets``, ``coords``, ``metadata``.
    """
    print(f"\nGenerating ROLLING predictions for {dataset_name} dataset...")
    print(f"  input_window_size  = {args.input_window_size}")
    print(f"  output_window_size = {args.output_window_size}")

    from torch.utils.data import DataLoader
    from ..data.batch_sampler import PatchBatchSampler

    W_in  = args.input_window_size
    W_out = args.output_window_size
    forcings_required = getattr(args, 'forcings_required', False)

    # -----------------------------------------------------------------------
    # Group dataset indices by patch_id (preserving temporal order)
    # -----------------------------------------------------------------------
    all_patch_ids = dataset.get_all_patch_ids()       # [N_samples]
    unique_patches = sorted(set(all_patch_ids.tolist()))

    # Map each patch_id → list of dataset indices in temporal order
    patch_index_map = {pid: [] for pid in unique_patches}
    for idx, pid in enumerate(all_patch_ids.tolist()):
        patch_index_map[pid].append(idx)

    all_predictions = {}   # patch_id → list of arrays [B, N_core, W_out*F]
    all_targets     = {}   # patch_id → list of arrays [B, N_core, W_out*F]
    all_coords      = {}   # patch_id → list of arrays [B, N_core, coord_dim]
    all_patch_metadata = []

    model.eval()
    with torch.no_grad():
        for pid in tqdm(unique_patches, desc=f"Rolling {dataset_name}"):
            indices = patch_index_map[pid]  # temporal order

            # Initialise rolling buffer (will be set on the first step)
            rolling_buffer = None   # [N_core, W_in * n_obs_feat]
            prev_pred_core = None   # [N_core, W_out * n_obs_feat]

            all_predictions[pid] = []
            all_targets[pid]     = []
            all_coords[pid]      = []

            for step_idx, ds_idx in enumerate(indices):
                # Fetch single sample and collate into a batch of 1
                sample = dataset[ds_idx]
                batch  = collate_fn([sample]) if collate_fn is not None else _simple_collate([sample])

                # ---- Determine sizes from first step -------------------------
                if step_idx == 0:
                    core_len = batch['core_len']
                    x_full   = batch['x']  # [1, N_total, C_total]
                    n_total_feat = x_full.shape[-1]  # W_in * n_obs_feat [+ W_out * n_forc_feat]

                    if forcings_required:
                        # obs channels: W_in * n_obs_feat
                        # forc channels: remaining
                        n_obs_feat_flat  = W_in * (n_total_feat // W_in - (
                            n_total_feat % W_in != 0))  # heuristic fallback
                        # Safer: dataset tells us n_target_cols via args
                        n_target_cols    = len(getattr(args, 'target_cols', ['head']))
                        n_obs_feat_flat  = W_in * n_target_cols    # W_in * n_obs_feat
                        n_forc_feat_flat = n_total_feat - n_obs_feat_flat
                    else:
                        n_target_cols   = len(getattr(args, 'target_cols', ['head']))
                        n_obs_feat_flat = n_total_feat              # all features are obs
                        n_forc_feat_flat = 0

                    n_obs_feat = n_target_cols  # features per time-step

                    # Seed buffer from ground-truth core observations (first W_in steps)
                    rolling_buffer = x_full[0, :core_len, :n_obs_feat_flat].cpu().numpy()
                    # shape: [N_core, W_in * n_obs_feat]

                # ---- Build modified x with rolling buffer -------------------
                x_current = batch['x'].clone()   # [1, N_total, C_total]

                if step_idx == 0:
                    # First window: use true inputs unmodified
                    pass
                else:
                    # Replace obs portion of core points with rolling buffer
                    x_current[0, :core_len, :n_obs_feat_flat] = torch.from_numpy(rolling_buffer)
                    # Ghost point obs channels are NOT updated (they act as
                    # boundary conditions and remain ground-truth)

                batch['x'] = x_current

                # ---- Forward pass -------------------------------------------
                if forward_fn is not None:
                    outputs, core_output, core_target, core_coords = forward_fn(model, batch, args)
                else:
                    outputs, core_output, core_target, core_coords = _default_forward(model, batch, args)
                # core_output: [1, N_core, W_out * n_obs_feat]  (numpy)

                # ---- Apply Physical Constraints (Clamping) ------------------
                # Clamp mass_concentration >= 0 to prevent unphysical negative mass
                # which causes exponential blowout in PDE surrogates.
                if obs_transform is not None and hasattr(args, 'target_cols') and hasattr(args, 'target_col_indices'):
                    try:
                        mass_idx = args.target_cols.index('mass_concentration')
                        # Get normalization parameters for this specific column
                        global_idx = args.target_col_indices[mass_idx]
                        mean_mass = obs_transform.mean[global_idx]
                        std_mass = obs_transform.std[global_idx]
                        
                        if isinstance(mean_mass, torch.Tensor):
                            mean_mass = mean_mass.item()
                            std_mass = std_mass.item()
                            
                        # Calculate what 0.0 maps to in normalized space
                        norm_threshold = (0.0 - mean_mass) / std_mass
                        
                        # Clamp the predicted values for mass_concentration
                        for w in range(W_out):
                            col_idx = w * n_obs_feat + mass_idx
                            core_output[:, :, col_idx] = np.maximum(core_output[:, :, col_idx], norm_threshold)
                    except ValueError:
                        pass # 'mass_concentration' not in target_cols

                # ---- Update rolling buffer ----------------------------------
                prev_pred_core = core_output[0]   # [N_core, W_out * n_obs_feat]
                rolling_buffer = _roll_buffer(
                    rolling_buffer, prev_pred_core,
                    W_in, W_out, n_obs_feat
                )

                # ---- Accumulate results ------------------------------------
                all_predictions[pid].append(core_output)
                all_targets[pid].append(core_target)
                all_coords[pid].append(core_coords)

                all_patch_metadata.append({
                    'batch_idx': step_idx,
                    'sample_idx': 0,
                    'patch_id': pid,
                    'dataset': dataset_name,
                    'core_len': core_len,
                    'rolling_step': step_idx,
                })

    # -----------------------------------------------------------------------
    # Concatenate results in the same layout as generate_predictions
    # -----------------------------------------------------------------------
    for pid in unique_patches:
        all_predictions[pid] = np.concatenate(all_predictions[pid], axis=0)
        all_targets[pid]     = np.concatenate(all_targets[pid],     axis=0)
        all_coords[pid]      = np.concatenate(all_coords[pid],      axis=0)

    predictions = np.concatenate(list(all_predictions.values()), axis=1)
    targets     = np.concatenate(list(all_targets.values()),     axis=1)
    coords      = np.concatenate(list(all_coords.values()),      axis=1)

    print(f"{dataset_name} rolling predictions shape: {predictions.shape}")
    print(f"{dataset_name} rolling targets shape:     {targets.shape}")
    print(f"{dataset_name} rolling coords shape:      {coords.shape}")

    return {
        'predictions': predictions,
        'targets':     targets,
        'coords':      coords,
        'metadata':    all_patch_metadata,
    }


def _simple_collate(samples):
    """Minimal collate that stacks torch tensors and passes other values through."""
    batch = {}
    for key in samples[0].keys():
        vals = [s[key] for s in samples]
        if isinstance(vals[0], torch.Tensor):
            batch[key] = torch.stack(vals, dim=0)
        else:
            batch[key] = vals[0]
    return batch


def _default_forward(model, batch, args):
    """
    Default forward pass for GINO-style models.
    
    Args:
        model: Model to run forward pass
        batch (dict): Batch dictionary
        args: Arguments with device
        
    Returns:
        tuple: (outputs, core_output, core_target, core_coords)
    """
    # Move data to device
    point_coords = batch['point_coords'].to(args.device).float()
    latent_queries = batch['latent_queries'].to(args.device).float()
    x = batch['x'].to(args.device).float()
    y = batch['y'].to(args.device).float()
    
    # Generate predictions
    outputs = model(
        input_geom=point_coords,
        latent_queries=latent_queries,
        x=x,
        output_queries=point_coords,
    )
    
    # Extract core points only (exclude ghost points)
    core_len = batch['core_len']
    core_output = outputs[:, :core_len].cpu().numpy()
    core_target = y[:, :core_len].cpu().numpy()
    core_coords = point_coords[:core_len].cpu().numpy()
    
    # Repeat coords for batch dimension
    core_coords = np.repeat(
        np.expand_dims(core_coords, axis=0),
        core_output.shape[0],
        axis=0
    )
    
    return outputs, core_output, core_target, core_coords
