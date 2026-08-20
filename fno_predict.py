"""
Generate predictions using trained FNOInterpolate model (refactored version).

This is a streamlined version that uses modular components from src/inference,
src/data, and src/models modules.

Usage:
    python generate_fno_predictions_refactored.py --model-path /path/to/model.pth
"""

import torch
import numpy as np
import functools
import os
from src.data.data_utils import (
    calculate_coord_transform,
    calculate_obs_transform,
    create_patch_datasets,
    make_collate_fn,
    reshape_multi_col_predictions,
    calculate_forcings_transform,
)
from src.data.patch_dataset_multi_col import GWPatchDatasetMultiCol
from src.models.neuralop.fno import FNOInterpolate
from src.inference import (
    setup_inference_arguments,
    load_checkpoint,
    create_model_from_checkpoint,
    generate_predictions,
    generate_rolling_predictions,
    organize_and_save_results,
    create_results_directory,
)
from src.inference.visualization import (
    create_per_column_visualizations,
)
from src.inference.metrics import (
    compute_metrics,
    save_metrics,
    denormalize_observations,
)
from src.training.parallel_utils import unwrap_dp, broadcast_static_inputs_for_dp
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data.batch_sampler import PatchBatchSampler


def add_fno_model_args(parser):
    """Add FNO-specific model arguments."""
    parser.add_argument('--rolling-sequence', action='store_true', default=False,
                        help='Enable autoregressive / rolling-sequence prediction mode. '
                             'In this mode the model solves an initial-value problem: '
                             'the first window of each patch uses ground-truth inputs, '
                             'then subsequent windows roll the model outputs back into '
                             'the input buffer (replacing ground-truth observations). '
                             'Results are saved to a rolling_sequence/ sub-directory.')
    parser.add_argument('--val-only', action='store_true', default=False,
                        help='Only run and save predictions for the validation set. '
                             'Skips the training set entirely to save time and disk space.')
    return parser


def configure_model_parameters_from_checkpoint(checkpoint):
    """Extract model configuration from checkpoint."""
    saved_args = checkpoint['args']
    
    print("\nModel configuration from checkpoint:")
    print(f"- FNO modes: {saved_args.fno_n_modes}")
    print(f"- FNO layers: {saved_args.fno_n_layers}")
    print(f"- Hidden channels: {saved_args.fno_hidden_channels}")
    print(f"- Latent query dims: {saved_args.latent_query_dims}")
    print(f"- Align corners: {getattr(saved_args, 'align_corners', False)}")
    print(f"- Padding mode: {getattr(saved_args, 'padding_mode', 'border')}")
    
    return saved_args


def define_fno_from_checkpoint(checkpoint, device):
    """
    Create FNOInterpolate model instance from checkpoint configuration.
    
    Args:
        checkpoint (dict): Loaded checkpoint dictionary
        device (str): Device to load model on
        
    Returns:
        FNOInterpolate: Instantiated FNOInterpolate model
    """
    saved_args = checkpoint['args']
    
    model = FNOInterpolate(
        # Coordinate and channel configuration
        coord_dim=saved_args.coord_dim,
        in_channels=saved_args.in_channels,
        out_channels=saved_args.out_channels,
        
        # Latent grid configuration
        latent_query_dims=saved_args.latent_query_dims,
        latent_feature_channels=getattr(saved_args, 'latent_feature_channels', None),
        
        # FNO configuration
        fno_n_layers=saved_args.fno_n_layers,
        fno_n_modes=saved_args.fno_n_modes,
        fno_hidden_channels=saved_args.fno_hidden_channels,
        lifting_channels=saved_args.lifting_channels,
        
        # Projection configuration
        projection_channel_ratio=saved_args.projection_channel_ratio,
        
        # Interpolation settings
        align_corners=getattr(saved_args, 'align_corners', False),
        padding_mode=getattr(saved_args, 'padding_mode', 'border'),
    ).to(device)
    
    return model


def configure_target_col_indices(args):
    """Map target column names to indices."""
    names_to_idx = {
        'mass_concentration': 0,
        'head': 1,
        'pressure': 2
    }
    
    # Handle multi-column prediction
    if hasattr(args, 'target_cols') and args.target_cols:
        args.target_col_indices = [names_to_idx[col] for col in args.target_cols]
        print(f"Target columns: {args.target_cols}")
        print(f"Target column indices: {args.target_col_indices}")
    else:
        # Single column for backward compatibility
        args.target_col_indices = [names_to_idx[args.target_col]]
        print(f"Target column: {args.target_col} (index: {args.target_col_indices[0]})")
    
    return args


def main():
    """Main inference pipeline."""
    # Setup arguments
    args = setup_inference_arguments(
        description='Generate predictions using trained FNOInterpolate model',
        default_base_data_dir='/srv/scratch/z5370003/projects/data/groundwater/FEFLOW/coastal/variable_density',
        default_results_dir='/srv/scratch/z5370003/projects/results/04_groundwater/variable_density/FNO_predictions',
        add_model_specific_args=add_fno_model_args
    )
    
    # Create results directory
    args.results_dir = create_results_directory(args.results_dir, 'fno_predictions')
    
    # Load checkpoint first to get target_cols and window sizes
    checkpoint = load_checkpoint(args.model_path, args.device)
    saved_args = checkpoint['args']
    
    # Set window sizes from checkpoint
    if hasattr(saved_args, 'input_window_size'):
        args.input_window_size = saved_args.input_window_size
        print(f"Using input window size from checkpoint: {args.input_window_size}")
    
    if hasattr(saved_args, 'output_window_size'):
        args.output_window_size = saved_args.output_window_size
        print(f"Using output window size from checkpoint: {args.output_window_size}")
    
    # Set target columns from checkpoint
    if hasattr(saved_args, 'target_cols'):
        args.target_cols = saved_args.target_cols
        print(f"Using target columns from checkpoint: {args.target_cols}")

    # Set forcings flag from checkpoint
    if hasattr(saved_args, 'forcings_required'):
        args.forcings_required = saved_args.forcings_required
        print(f"Using forcings_required from checkpoint: {args.forcings_required}")
    else:
        args.forcings_required = False
        print("forcings_required not found in checkpoint, defaulting to False")

    
    # Configure target columns
    args = configure_target_col_indices(args)
    
    # Calculate data transforms (same as training)
    print("\nPreparing data transforms...")
    coord_transform = calculate_coord_transform(args.raw_data_dir)
    obs_transform = calculate_obs_transform(
        args.raw_data_dir,
        target_obs_cols=['mass_concentration', 'head', 'pressure']
    )
    forcings_transform = calculate_forcings_transform()
    
    # Create datasets with multi-column support
    print("\nCreating datasets...")
    if args.resolution_ratio < 1.0:
        print(f"Using sampling strategy: {args.sampling_strategy}")
        print(f"Using resolution ratio: {args.resolution_ratio} (subsampling to {args.resolution_ratio*100:.1f}% of nodes)")
        print(f"Using minimum resolution ratio: {args.min_resolution_ratio}")
    train_ds, val_ds = create_patch_datasets(
        dataset_class=GWPatchDatasetMultiCol,
        patch_data_dir=args.patch_data_dir,
        coord_transform=coord_transform,
        obs_transform=obs_transform,
        target_col_indices=args.target_col_indices,
        input_window_size=args.input_window_size,
        output_window_size=args.output_window_size,
        forcings_transform=forcings_transform,
        forcings_required=args.forcings_required,
        resolution_ratio=args.resolution_ratio,
        min_resolution_ratio=args.min_resolution_ratio,
        sampling_strategy=args.sampling_strategy,
        val_stride=getattr(args, 'val_stride', 1),
        val_only=args.val_only,
    )
    
    if train_ds is not None:
        print(f"Train dataset length: {len(train_ds)}")
    print(f"Val dataset length: {len(val_ds)}")
    
    # Store time_values on args for make_collate_fn
    args._time_values = getattr(train_ds, 'time_values', None) if train_ds is not None else getattr(val_ds, 'time_values', None)
    
    # Create model
    model = create_model_from_checkpoint(
        checkpoint,
        model_factory=define_fno_from_checkpoint,
        device=args.device
    )
    
    # Create collate function
    args.coord_dim = checkpoint['args'].coord_dim
    args.latent_query_dims = checkpoint['args'].latent_query_dims
    collate_fn = make_collate_fn(args, coord_dim=args.coord_dim)
    
    # Generate predictions (standard teacher-forcing OR rolling autoregressive mode)
    if args.rolling_sequence:
        print("\n" + "="*60)
        print("EVALUATION MODE: Rolling Sequence (Initial-Value Problem) for 4D FNO")
        print("="*60)
        _predict_fn = functools.partial(_fno_4d_rolling_predict, obs_transform=obs_transform)
    else:
        print("\nEVALUATION MODE: Standard (Teacher Forcing) for 4D FNO")
        _predict_fn = _fno_4d_predict

    if args.val_only:
        print("\nDataset scope: VAL ONLY (--val-only flag set, skipping train set)")

    # --- Train set ---
    if not args.val_only:
        train_results = _predict_fn(
            model, train_ds, args,
            dataset_name='train',
            collate_fn=collate_fn
        )
    else:
        train_results = None

    # --- Val set ---
    val_results = _predict_fn(
        model, val_ds, args,
        dataset_name='val',
        collate_fn=collate_fn
    )
    
    # Reshape multi-column predictions to separate target columns
    # From [N_samples, N_points, output_window_size * n_cols]
    # To [N_samples, N_points, output_window_size, n_cols]
    n_target_cols = len(args.target_cols)
    print(f"Using {n_target_cols} target columns for reshaping")
    
    # Always reshape regardless of output_window_size
    # For output_window_size=1, this changes [N, P, C] to [N, P, 1, C]
    # For output_window_size>1, this de-interleaves [t0_v0, t0_v1, t1_v0, t1_v1, ...]
    if train_results is not None:
        train_results['predictions'] = reshape_multi_col_predictions(
            train_results['predictions'], args.output_window_size, n_target_cols
        )
        train_results['targets'] = reshape_multi_col_predictions(
            train_results['targets'], args.output_window_size, n_target_cols
        )
    val_results['predictions'] = reshape_multi_col_predictions(
        val_results['predictions'], args.output_window_size, n_target_cols
    )
    val_results['targets'] = reshape_multi_col_predictions(
        val_results['targets'], args.output_window_size, n_target_cols
    )
    
    if train_results is not None:
        print(f"Reshaped train predictions to: {train_results['predictions'].shape}")
        print(f"Reshaped train targets to:     {train_results['targets'].shape}")
    print(f"Reshaped val predictions to:   {val_results['predictions'].shape}")
    print(f"Reshaped val targets to:       {val_results['targets'].shape}")
    
    # Denormalize predictions and targets after reshaping
    print("\nDenormalizing predictions and targets...")
    if train_results is not None:
        train_results['predictions'] = denormalize_observations(
            train_results['predictions'], obs_transform, args.target_col_indices
        )
        train_results['targets'] = denormalize_observations(
            train_results['targets'], obs_transform, args.target_col_indices
        )
    val_results['predictions'] = denormalize_observations(
        val_results['predictions'], obs_transform, args.target_col_indices
    )
    val_results['targets'] = denormalize_observations(
        val_results['targets'], obs_transform, args.target_col_indices
    )
    if train_results is not None:
        print(f"Denormalized train predictions range: [{train_results['predictions'].min():.3f}, {train_results['predictions'].max():.3f}]")
        print(f"Denormalized train targets range:     [{train_results['targets'].min():.3f}, {train_results['targets'].max():.3f}]")
    
    results_dict = {'val': val_results}
    if train_results is not None:
        results_dict['train'] = train_results
    
    # Save results (redirect to sub-directory when in rolling mode)
    results_dir_final = args.results_dir
    if args.rolling_sequence:
        import os
        results_dir_final = os.path.join(args.results_dir, 'rolling_sequence')
        os.makedirs(results_dir_final, exist_ok=True)
        args.results_dir = results_dir_final
        print(f"\nRolling-sequence results will be saved to: {results_dir_final}")

    organize_and_save_results(results_dict, args)
    
    # Compute and save metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(results_dict, args.target_cols, args.target_col_indices, obs_transform=None)
    save_metrics(metrics, args.target_cols, args.results_dir)
    
    # Create visualizations only if not metrics_only
    if not args.metrics_only:
        create_per_column_visualizations(
            results_dict, 
            args.target_cols, 
            args.target_col_indices,
            args.output_window_size,
            args.results_dir,
            create_3d_plots=getattr(args, 'create_3d_plots', False),
            coord_transform=coord_transform
        )
    else:
        print("\nSkipping visualizations (--metrics-only flag enabled)")
    
    print("\n" + "="*60)
    print("Prediction generation complete!")
    print(f"Results saved to: {args.results_dir}")
    if args.metrics_only:
        print("Mode: Metrics only (arrays and plots skipped)")
    if args.rolling_sequence:
        print("Evaluation mode: Rolling Sequence (Initial-Value Problem)")
    if args.val_only:
        print("Dataset scope: Val only")
    print("="*60)


def _fno_4d_predict(model, dataset, args, dataset_name, collate_fn):
    """Standard 4D prediction loop without rolling."""
    print(f"\nGenerating predictions for {dataset_name} dataset...")
    sampler = PatchBatchSampler(dataset, batch_size=args.batch_size, shuffle_within_batches=False, shuffle_patches=False)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)
    
    all_predictions, all_targets, all_coords = {}, {}, {}
    all_patch_metadata = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Processing {dataset_name}")):
            input_coords = batch['input_coords'].to(args.device).float()
            output_coords = batch['output_coords'].to(args.device).float()
            latent_queries = batch['latent_queries'].to(args.device).float()
            x = batch['x'].to(args.device).float()
            
            batch_size = x.shape[0]
            if not model.training:
                model = unwrap_dp(model)
                
            input_geom_b, latent_queries_b, output_queries_b = broadcast_static_inputs_for_dp(
                input_coords, latent_queries, batch_size, output_queries=output_coords
            )
            
            outputs = model(input_geom_b, latent_queries_b, x, output_queries_b)
            
            # Extract core points
            y = batch['y'].to(args.device).float()
            core_len = batch['core_len']
            T_out = batch['T_out']
            C_obs = args.n_target_cols
            
            outputs_reshaped = reshape_multi_col_predictions(outputs, T_out, C_obs)
            y_reshaped = reshape_multi_col_predictions(y, T_out, C_obs)
            
            core_outputs = outputs_reshaped[:, :core_len, :, :].cpu().numpy()
            core_targets = y_reshaped[:, :core_len, :, :].cpu().numpy()
            core_coords_sp = batch['spatial_coords'][:core_len].unsqueeze(0).expand(batch_size, -1, -1).numpy()
            
            # Re-flatten for legacy compatibility in metrics and plotting
            B = batch_size
            core_outputs_flat = core_outputs.reshape(B, core_len, T_out * C_obs)
            core_targets_flat = core_targets.reshape(B, core_len, T_out * C_obs)
            
            patch_id = batch['patch_id']
            if patch_id not in all_predictions:
                all_predictions[patch_id] = []
                all_targets[patch_id] = []
                all_coords[patch_id] = []
            
            all_predictions[patch_id].append(core_outputs_flat)
            all_targets[patch_id].append(core_targets_flat)
            all_coords[patch_id].append(core_coords_sp)
            
            for i in range(batch_size):
                all_patch_metadata.append({
                    'batch_idx': batch_idx, 'sample_idx': i, 'patch_id': patch_id,
                    'dataset': dataset_name, 'core_len': core_len
                })
                
    for patch_id in all_predictions.keys():
        all_predictions[patch_id] = np.concatenate(all_predictions[patch_id], axis=0)
        all_targets[patch_id] = np.concatenate(all_targets[patch_id], axis=0)
        all_coords[patch_id] = np.concatenate(all_coords[patch_id], axis=0)
        
    return {
        'predictions': np.concatenate(list(all_predictions.values()), axis=1),
        'targets': np.concatenate(list(all_targets.values()), axis=1),
        'coords': np.concatenate(list(all_coords.values()), axis=1),
        'metadata': all_patch_metadata
    }

def _fno_4d_rolling_predict(model, dataset, args, dataset_name, collate_fn, obs_transform=None):
    """4D autoregressive prediction."""
    print(f"\nGenerating ROLLING predictions for {dataset_name} dataset...")
    
    W_in = args.input_window_size
    W_out = args.output_window_size
    n_target_cols = len(getattr(args, 'target_cols', ['head']))
    
    all_patch_ids = dataset.get_all_patch_ids()
    unique_patches = sorted(set(all_patch_ids.tolist()))
    
    patch_index_map = {pid: [] for pid in unique_patches}
    for idx, pid in enumerate(all_patch_ids.tolist()):
        patch_index_map[pid].append(idx)
        
    all_predictions, all_targets, all_coords = {}, {}, {}
    all_patch_metadata = []
    
    model.eval()
    if not model.training:
        model = unwrap_dp(model)
        
    with torch.no_grad():
        for pid in tqdm(unique_patches, desc=f"Rolling {dataset_name}"):
            indices = patch_index_map[pid]
            
            rolling_buffer = None  # [N_core, T_in, C_obs]
            
            all_predictions[pid] = []
            all_targets[pid] = []
            all_coords[pid] = []
            
            for step_idx, ds_idx in enumerate(indices):
                sample = dataset[ds_idx]
                batch = collate_fn([sample])
                
                core_len = batch['core_len']
                n_pts = batch['n_pts']
                T_in = batch['T_in']
                
                # batch['x'] is (1, N_pts * T_in, C_in)
                x_full = batch['x'].clone()
                C_in = x_full.shape[-1]
                
                # Reshape to (N_pts, T_in, C_in)
                x_3d = x_full[0].reshape(n_pts, T_in, C_in)
                
                if step_idx == 0:
                    # Seed buffer from ground-truth core observations
                    rolling_buffer = x_3d[:core_len, :, :n_target_cols].cpu().numpy()
                else:
                    # Replace obs portion of core points with rolling buffer
                    x_3d[:core_len, :, :n_target_cols] = torch.from_numpy(rolling_buffer)
                    
                # Reshape back to flat for model
                batch['x'] = x_3d.reshape(n_pts * T_in, C_in).unsqueeze(0)
                
                # Forward pass
                input_coords = batch['input_coords'].to(args.device).float()
                output_coords = batch['output_coords'].to(args.device).float()
                latent_queries = batch['latent_queries'].to(args.device).float()
                x = batch['x'].to(args.device).float()
                
                input_geom_b, latent_queries_b, output_queries_b = broadcast_static_inputs_for_dp(
                    input_coords, latent_queries, 1, output_queries=output_coords
                )
                
                outputs = model(input_geom_b, latent_queries_b, x, output_queries_b)
                
                y = batch['y'].to(args.device).float()
                outputs_reshaped = reshape_multi_col_predictions(outputs, W_out, n_target_cols)
                y_reshaped = reshape_multi_col_predictions(y, W_out, n_target_cols)
                
                core_outputs = outputs_reshaped[:, :core_len, :, :].cpu().numpy()  # (1, N_core, T_out, C_obs)
                core_targets = y_reshaped[:, :core_len, :, :].cpu().numpy()
                core_coords_sp = batch['spatial_coords'][:core_len].unsqueeze(0).numpy()
                
                # Clamp mass_concentration
                if obs_transform is not None and hasattr(args, 'target_cols') and hasattr(args, 'target_col_indices'):
                    try:
                        mass_idx = args.target_cols.index('mass_concentration')
                        global_idx = args.target_col_indices[mass_idx]
                        mean_mass = float(obs_transform.mean[global_idx])
                        std_mass = float(obs_transform.std[global_idx])
                        norm_threshold = (0.0 - mean_mass) / std_mass
                        core_outputs[0, :, :, mass_idx] = np.maximum(core_outputs[0, :, :, mass_idx], norm_threshold)
                    except ValueError:
                        pass
                        
                # Update rolling buffer
                rollout_step = min(getattr(dataset, 'stride', W_out), W_out)
                prev_pred = core_outputs[0, :, :rollout_step, :]  # (N_core, rollout_step, C_obs)
                
                if rollout_step >= W_in:
                    rolling_buffer = prev_pred[:, -W_in:, :]
                else:
                    kept = rolling_buffer[:, rollout_step:, :]
                    rolling_buffer = np.concatenate([kept, prev_pred], axis=1)
                
                # Store
                core_outputs_flat = core_outputs.reshape(1, core_len, W_out * n_target_cols)
                core_targets_flat = core_targets.reshape(1, core_len, W_out * n_target_cols)
                
                all_predictions[pid].append(core_outputs_flat)
                all_targets[pid].append(core_targets_flat)
                all_coords[pid].append(core_coords_sp)
                
                all_patch_metadata.append({
                    'batch_idx': step_idx, 'sample_idx': 0, 'patch_id': pid,
                    'dataset': dataset_name, 'core_len': core_len
                })
                
    for patch_id in all_predictions.keys():
        all_predictions[patch_id] = np.concatenate(all_predictions[patch_id], axis=0)
        all_targets[patch_id] = np.concatenate(all_targets[patch_id], axis=0)
        all_coords[patch_id] = np.concatenate(all_coords[patch_id], axis=0)
        
    return {
        'predictions': np.concatenate(list(all_predictions.values()), axis=1),
        'targets': np.concatenate(list(all_targets.values()), axis=1),
        'coords': np.concatenate(list(all_coords.values()), axis=1),
        'metadata': all_patch_metadata
    }

if __name__ == "__main__":
    main()
