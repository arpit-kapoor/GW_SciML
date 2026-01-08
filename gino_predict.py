"""
Generate predictions using trained GINO model (refactored version).

This is a streamlined version that uses modular components from src/inference,
src/data, and src/models modules.

Usage:
    python generate_gino_predictions_refactored.py --model-path /path/to/model.pth
"""

import torch
import numpy as np
from src.data.data_utils import (
    calculate_coord_transform,
    calculate_obs_transform,
    create_patch_datasets,
    make_collate_fn,
    reshape_multi_col_predictions,
)
from src.data.patch_dataset_multi_col import GWPatchDatasetMultiCol
from src.models.neuralop.gino import GINO
from src.inference import (
    setup_inference_arguments,
    load_checkpoint,
    create_model_from_checkpoint,
    generate_predictions,
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


def add_gino_model_args(parser):
    """Add GINO-specific model arguments (if needed)."""
    return parser


def configure_model_parameters_from_checkpoint(checkpoint):
    """Extract model configuration from checkpoint."""
    saved_args = checkpoint['args']
    
    print("\nModel configuration from checkpoint:")
    print(f"- FNO modes: {saved_args.fno_n_modes}")
    print(f"- FNO layers: {saved_args.fno_n_layers}")
    print(f"- Hidden channels: {saved_args.fno_hidden_channels}")
    print(f"- GNO radius: {saved_args.gno_radius}")
    
    return saved_args


def define_gino_from_checkpoint(checkpoint, device):
    """
    Create GINO model instance from checkpoint configuration.
    
    Args:
        checkpoint (dict): Loaded checkpoint dictionary
        device (str): Device to load model on
        
    Returns:
        GINO: Instantiated GINO model
    """
    saved_args = checkpoint['args']
    
    model = GINO(
        # Input GNO configuration
        in_gno_coord_dim=saved_args.coord_dim,
        in_gno_radius=saved_args.gno_radius,
        in_gno_out_channels=saved_args.in_gno_out_channels,
        in_gno_channel_mlp_layers=saved_args.in_gno_channel_mlp_layers,
        
        # FNO configuration
        fno_n_layers=saved_args.fno_n_layers,
        fno_n_modes=saved_args.fno_n_modes,
        fno_hidden_channels=saved_args.fno_hidden_channels,
        lifting_channels=saved_args.lifting_channels,
        
        # Output GNO configuration
        out_gno_coord_dim=saved_args.coord_dim,
        out_gno_radius=saved_args.gno_radius,
        out_gno_channel_mlp_layers=saved_args.out_gno_channel_mlp_layers,
        projection_channel_ratio=saved_args.projection_channel_ratio,
        out_channels=saved_args.out_channels,
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
        description='Generate predictions using trained GINO model',
        default_base_data_dir='/srv/scratch/z5370003/projects/data/groundwater/FEFLOW/coastal/variable_density',
        default_results_dir='/srv/scratch/z5370003/projects/results/04_groundwater/variable_density/GINO_predictions',
        add_model_specific_args=add_gino_model_args
    )
    
    # Create results directory
    args.results_dir = create_results_directory(args.results_dir, 'gino_predictions')
    
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
    
    # Configure target columns
    args = configure_target_col_indices(args)
    
    # Calculate data transforms (same as training)
    print("\nPreparing data transforms...")
    coord_transform = calculate_coord_transform(args.raw_data_dir)
    obs_transform = calculate_obs_transform(
        args.raw_data_dir,
        target_obs_cols=['mass_concentration', 'head', 'pressure']
    )
    
    # Create datasets with multi-column support
    print("\nCreating datasets...")
    train_ds, val_ds = create_patch_datasets(
        dataset_class=GWPatchDatasetMultiCol,
        patch_data_dir=args.patch_data_dir,
        coord_transform=coord_transform,
        obs_transform=obs_transform,
        target_col_indices=args.target_col_indices,
        input_window_size=args.input_window_size,
        output_window_size=args.output_window_size,
    )
    
    print(f"Train dataset length: {len(train_ds)}")
    print(f"Val dataset length: {len(val_ds)}")
    
    # Create model
    model = create_model_from_checkpoint(
        checkpoint,
        model_factory=define_gino_from_checkpoint,
        device=args.device
    )
    
    # Create collate function
    # We need to extract latent_query_dims from checkpoint
    args.coord_dim = checkpoint['args'].coord_dim
    args.latent_query_dims = checkpoint['args'].latent_query_dims
    collate_fn = make_collate_fn(args, coord_dim=args.coord_dim)
    
    # Generate predictions
    train_results = generate_predictions(
        model, train_ds, args,
        dataset_name='train',
        collate_fn=collate_fn
    )
    
    val_results = generate_predictions(
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
    
    print(f"Reshaped predictions to: {train_results['predictions'].shape}")
    print(f"Reshaped targets to: {train_results['targets'].shape}")
    
    results_dict = {
        'train': train_results,
        'val': val_results
    }
    
    # Save results
    organize_and_save_results(results_dict, args)
    
    # Compute and save metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(results_dict, args.target_cols, args.target_col_indices, obs_transform)
    save_metrics(metrics, args.target_cols, args.results_dir)
    
    # Create visualizations with per-column analysis
    create_per_column_visualizations(
        results_dict, 
        args.target_cols, 
        args.target_col_indices,
        args.output_window_size,
        args.results_dir,
        obs_transform,
        create_3d_plots=getattr(args, 'create_3d_plots', False)
    )
    
    print("\n" + "="*60)
    print("Prediction generation complete!")
    print(f"Results saved to: {args.results_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
