"""
Generate predictions using trained GINO model (refactored version).

This is a streamlined version that uses modular components from src/inference,
src/data, and src/models modules.

Usage:
    python generate_gino_predictions_refactored.py --model-path /path/to/model.pth
"""

import torch
from src.data.data_utils import (
    calculate_coord_transform,
    calculate_obs_transform,
    create_patch_datasets,
    make_collate_fn,
)
from src.models.neuralop.gino import GINO
from src.inference import (
    setup_inference_arguments,
    load_checkpoint,
    create_model_from_checkpoint,
    generate_predictions,
    organize_and_save_results,
    create_all_visualizations,
    create_results_directory,
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


def configure_target_col_idx(args):
    """Map target column name to index."""
    names_to_idx = {
        'mass_concentration': 0,
        'head': 1,
        'pressure': 2
    }
    args.target_col_idx = names_to_idx[args.target_col]
    print(f"Target column: {args.target_col} (index: {args.target_col_idx})")
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
    
    # Configure target column
    args = configure_target_col_idx(args)
    
    # Calculate data transforms (same as training)
    print("\nPreparing data transforms...")
    coord_transform = calculate_coord_transform(args.raw_data_dir)
    obs_transform = calculate_obs_transform(args.raw_data_dir)
    
    # Create datasets
    print("\nCreating datasets...")
    train_ds, val_ds = create_patch_datasets(
        dataset_class=None,  # Will use default GWPatchDataset
        patch_data_dir=args.patch_data_dir,
        coord_transform=coord_transform,
        obs_transform=obs_transform,
        target_col_idx=args.target_col_idx,
        input_window_size=args.input_window_size,
        output_window_size=args.output_window_size,
    )
    
    print(f"Train dataset length: {len(train_ds)}")
    print(f"Val dataset length: {len(val_ds)}")
    
    # Load checkpoint and create model
    checkpoint = load_checkpoint(args.model_path, args.device)
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
    
    results_dict = {
        'train': train_results,
        'val': val_results
    }
    
    # Save results
    organize_and_save_results(results_dict, args)
    
    # Create visualizations
    create_all_visualizations(results_dict, args)
    
    print("\n" + "="*60)
    print("Prediction generation complete!")
    print(f"Results saved to: {args.results_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
