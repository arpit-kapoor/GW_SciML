"""
Train FNOInterpolate on variable-density groundwater patches with multi-column support.

FNOInterpolate uses interpolation (instead of GNO blocks) for encoding/decoding between
point clouds and regular grids, with FNO blocks operating in the latent grid space.
"""

import sys
import argparse
import os
import pkgutil
import importlib
import inspect
import torch
from torch.utils.data import DataLoader

from src.data.patch_dataset_multi_col import GWPatchDatasetMultiCol
from src.data.batch_sampler import PatchBatchSampler
from src.data.data_utils import (
    calculate_coord_transform,
    calculate_forcings_transform,
    calculate_obs_transform,
    create_patch_datasets,
    make_collate_fn,
)
from src.models.neuralop.fno import FNOInterpolate
from src.models.neuralop.losses import variance_aware_multicol_loss
from src.training import (
    setup_training_arguments,
    configure_target_col_indices,
    DataParallelAdapter,
    unwrap_model_for_state_dict,
    train_model,
)

# --- Begin DP compatibility patch for tltorch complex buffers ---
try:
    import sys
    import importlib
    import inspect
    import torch.nn as _nn
    import tltorch

    def _safe_register_buffer(self, name, value, persistent=True):
        # Only convert actual complex tensors; real tensors pass through unchanged
        if torch.is_tensor(value) and value.is_complex():
            value = torch.view_as_real(value)
        # Use base nn.Module.register_buffer to avoid any custom overrides
        return _nn.Module.register_buffer(self, name, value, persistent=persistent)

    patched = []

    # Walk all tltorch submodules and patch classes in modules likely to hold complex/factorized tensors
    for _finder, _modname, _ispkg in pkgutil.walk_packages(tltorch.__path__, tltorch.__name__ + '.'):
        if not any(key in _modname for key in ('factorized', 'complex')):
            continue
        try:
            _m = importlib.import_module(_modname)
        except Exception:
            continue

        for _name, _obj in inspect.getmembers(_m, inspect.isclass):
            # Only patch torch.nn.Module subclasses
            try:
                if issubclass(_obj, _nn.Module):
                    # Overwrite register_buffer unconditionally with the safe version
                    setattr(_obj, 'register_buffer', _safe_register_buffer)
                    patched.append(f'{_modname}:{_name}')
            except Exception:
                pass

    print(f"Patched tltorch DP compatibility on {len(patched)} classes. Examples: {patched[:6]}")
except Exception as _e:
    print(f"Warning: DP compatibility patch failed: {_e}")
# --- End patch ---


def add_fno_interpolate_model_args(parser):
    """Add FNOInterpolate-specific model arguments to the parser."""
    parser.add_argument('--align-corners', action='store_true', default=False,
                        dest='align_corners',
                        help='Whether to align corners in grid_sample interpolation')
    parser.add_argument('--padding-mode', type=str, default='border',
                        dest='padding_mode',
                        help='Padding mode for grid_sample (zeros, border, reflection)')
    return parser


def define_model_parameters(args):
    """Define FNOInterpolate architecture parameters based on input configuration."""
    args.coord_dim = 3
    args.n_target_cols = len(args.target_cols)
    
    # FNO configuration
    args.fno_n_layers = 4
    args.fno_n_modes = (10, 10, 6, 6)
    args.fno_hidden_channels = 64
    args.lifting_channels = 64
    args.projection_channel_ratio = 2
    # 4D space-time FNO packs channels per point-time pair.
    # Base channels: observations at current step
    args.in_channels = args.n_target_cols
    
    if getattr(args, 'forcings_required', False):
        # We pack 4 current forcings and 4 future forcings per point-time pair
        args.forcings_dim = 4
        args.in_channels += 2 * args.forcings_dim
        
    args.out_channels = args.n_target_cols
    # The latent grid covers the input window size
    args.latent_query_dims = (24, 24, 12, args.input_window_size)
    
    return args


def define_fno_interpolate_model(args):
    """Instantiate FNOInterpolate model with configured parameters."""
    model = FNOInterpolate(
        latent_query_dims=args.latent_query_dims,
        coord_dim=args.coord_dim,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        lifting_channels=args.lifting_channels,
        projection_channel_ratio=args.projection_channel_ratio,
        fno_n_modes=args.fno_n_modes,
        fno_hidden_channels=args.fno_hidden_channels,
        fno_n_layers=args.fno_n_layers,
        align_corners=args.align_corners,
        padding_mode=args.padding_mode,
    ).to(args.device)
    return model


def _fno_4d_forward(model, batch, args):
    """
    Custom forward pass for 4D space-time FNO.
    Handles separate input and output coordinates and DataParallel broadcasting.
    """
    from src.training.parallel_utils import unwrap_dp, broadcast_static_inputs_for_dp
    
    input_coords = batch['input_coords'].to(args.device).float()
    output_coords = batch['output_coords'].to(args.device).float()
    latent_queries = batch['latent_queries'].to(args.device).float()
    x = batch['x'].to(args.device).float()
    
    batch_size = x.shape[0]

    if not model.training:
        model = unwrap_dp(model)
    
    # Add fake batch dimensions to static inputs for DataParallel
    # We broadcast input_coords and output_coords. We use broadcast_static_inputs_for_dp
    # but pass output_coords as the third argument.
    input_geom_b, latent_queries_b, output_queries_b = broadcast_static_inputs_for_dp(
        input_coords, latent_queries, batch_size, output_queries=output_coords
    )
    
    # Call model (DataParallelAdapter uses positional args: input_geom, latent_queries, x, output_queries)
    outputs = model(input_geom_b, latent_queries_b, x, output_queries_b)
    
    return outputs


def _fno_4d_extract_core(outputs, batch, args):
    """
    Custom core point extraction for 4D space-time FNO.
    The output is (B, N_pts * T_out, C). We need to extract the core points
    for each time step.
    """
    from src.data.data_utils import reshape_multi_col_predictions
    
    y = batch['y'].to(args.device).float()
    core_len = batch['core_len']
    T_out = batch['T_out']
    C_obs = args.n_target_cols
    
    # Reshape outputs and targets to separate spatial and temporal dims
    # (B, N_pts * T_out, C) -> (B, N_pts, T_out, C)
    outputs_reshaped = reshape_multi_col_predictions(outputs, T_out, C_obs)
    y_reshaped = reshape_multi_col_predictions(y, T_out, C_obs)
    
    # Extract core points (first core_len points along the spatial dimension)
    core_outputs = outputs_reshaped[:, :core_len, :, :]  # (B, N_core, T_out, C)
    core_targets = y_reshaped[:, :core_len, :, :]        # (B, N_core, T_out, C)
    
    # The updated variance_aware_multicol_loss now expects [B, N_core, T_out, C]
    
    # Extract weights safely. batch['weights'] is tiled to (N_core * T_out).
    # We reshape to (N_pts, T_out) and take the first column for spatial weights up to core_len.
    weights = batch['weights'].to(args.device).float()
    core_weights = weights.reshape((-1, T_out))[:core_len, 0]
    
    return core_outputs, core_targets, core_weights


def create_data_loaders(train_ds, val_ds, args):
    """Create train and validation data loaders with patch-based batching."""
    train_sampler = PatchBatchSampler(
        train_ds, batch_size=args.batch_size,
        shuffle_within_batches=args.shuffle_within_batches,
        shuffle_patches=args.shuffle_patches,
        seed=args.seed
    )
    val_sampler = PatchBatchSampler(
        val_ds, batch_size=args.batch_size,
        shuffle_within_batches=False,
        shuffle_patches=False,
        seed=None
    )
    collate_fn = make_collate_fn(args, coord_dim=args.coord_dim)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler, collate_fn=collate_fn)
    return train_loader, val_loader


if __name__ == "__main__":
    print("\n" + "="*60)
    print("FNOInterpolate Training Script - Starting")
    print("="*60)
    
    # Setup arguments with defaults
    print("Setting up training arguments...")
    args = setup_training_arguments(
        description='Train FNOInterpolate model on groundwater patches with multi-column support',
        default_base_data_dir='/srv/scratch/z5370003/projects/data/groundwater/FEFLOW/coastal/variable_density',
        default_results_dir='/srv/scratch/z5370003/projects/results/04_groundwater/variable_density/FNO/interpolate/mass_conc_head',
        add_model_specific_args=add_fno_interpolate_model_args
    )
    
    # Configure FNOInterpolate model parameters and target columns
    args = define_model_parameters(args)
    args = configure_target_col_indices(args)
    print("Training configuration:")
    for k in sorted(vars(args)):
        print(f"  {k}: {getattr(args, k)}")
    print("="*60 + "\n")

    # Set random seeds
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)
    
    # Prepare data transforms and datasets
    print("Calculating coordinate transform...")
    coord_transform = calculate_coord_transform(args.raw_data_dir)
    print("Calculating observation transform...")
    obs_transform = calculate_obs_transform(
        args.raw_data_dir,
        target_obs_cols=['mass_concentration', 'head', 'pressure']
    )
    print("Creating datasets...")
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
        forcings_required=args.forcings_required,
        forcings_transform=calculate_forcings_transform(),
        resolution_ratio=args.resolution_ratio,
        min_resolution_ratio=args.min_resolution_ratio,
        sampling_strategy=args.sampling_strategy,
        train_stride=args.train_stride,
    )
    
    print(f"Dataset sizes - Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Store time_values on args for make_collate_fn
    args._time_values = getattr(train_ds, 'time_values', None) if train_ds is not None else getattr(val_ds, 'time_values', None)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(train_ds, val_ds, args)
    print(f"Data loaders - Train: {len(train_loader)} batches, Val: {len(val_loader)} batches\n")

    # Initialize model
    base_model = define_fno_interpolate_model(args)
    model = DataParallelAdapter(base_model)
    
    # Wrap with DataParallel if multiple GPUs available
    if args.device.startswith("cuda") and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if args.scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.01
        )
    else:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    
    # Define loss function
    loss_fn = lambda y_pred, y_true, weights: variance_aware_multicol_loss(
        y_pred, y_true, weights,
        output_window_size=args.output_window_size,
        target_cols=args.target_cols,
        lambda_conc_focus=args.lambda_conc_focus,
    )
    
    # Train model
    print("Starting training...\n")
    model = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        args=args,
        forward_fn=_fno_4d_forward,
        extract_core_fn=_fno_4d_extract_core,
    )

    # Save final model
    model_path = os.path.join(args.results_dir, 'fno_interpolate_model.pth')
    torch.save(unwrap_model_for_state_dict(model).state_dict(), model_path)
    print(f"\n{'='*60}")
    print(f"Training complete! Model saved to: {model_path}")
    print(f"{'='*60}")
