"""
Train GINO on variable-density groundwater patches with multi-column support.
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
    calculate_obs_transform,
    create_patch_datasets,
    make_collate_fn,
)
from src.models.neuralop.gino import GINO
from src.models.neuralop.losses import variance_aware_multicol_loss
from src.training import (
    setup_training_arguments,
    configure_target_col_indices,
    GINODataParallelAdapter,
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


def add_gino_model_args(parser):
    """Add GINO-specific model arguments to the parser."""
    # All GINO parameters are computed automatically
    return parser


def define_model_parameters(args):
    """Define GINO architecture parameters based on input configuration."""
    args.coord_dim = 3
    args.n_target_cols = len(args.target_cols)
    args.gno_radius = 0.15
    args.in_gno_out_channels = args.input_window_size * args.n_target_cols
    args.in_gno_channel_mlp_layers = [32, 64, 32]
    args.fno_n_layers = 4
    args.fno_n_modes = (12, 12, 8)
    args.fno_hidden_channels = 64
    args.lifting_channels = 64
    args.out_gno_channel_mlp_layers = [32, 64, 32]
    args.projection_channel_ratio = 2
    args.out_channels = args.output_window_size * args.n_target_cols
    args.latent_query_dims = (32, 32, 24)
    return args


def define_gino_model(args):
    """Instantiate GINO model with configured parameters."""
    model = GINO(
        in_gno_coord_dim=args.coord_dim,
        in_gno_radius=args.gno_radius,
        in_gno_out_channels=args.in_gno_out_channels,
        in_gno_channel_mlp_layers=args.in_gno_channel_mlp_layers,
        fno_n_layers=args.fno_n_layers,
        fno_n_modes=args.fno_n_modes,
        fno_hidden_channels=args.fno_hidden_channels,
        lifting_channels=args.lifting_channels,
        out_gno_coord_dim=args.coord_dim,
        out_gno_radius=args.gno_radius,
        out_gno_channel_mlp_layers=args.out_gno_channel_mlp_layers,
        projection_channel_ratio=args.projection_channel_ratio,
        out_channels=args.out_channels,
    ).to(args.device)
    return model


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
    print("GINO Training Script - Starting")
    print("="*60)
    
    # Setup arguments with defaults
    print("Setting up training arguments...")
    args = setup_training_arguments(
        description='Train GINO model on groundwater patches with multi-column support',
        default_base_data_dir='/srv/scratch/z5370003/projects/data/groundwater/FEFLOW/coastal/variable_density',
        default_results_dir='/srv/scratch/z5370003/projects/results/04_groundwater/variable_density/GINO/multi_col/mass_conc_head',
        add_model_specific_args=add_gino_model_args
    )
    
    # Configure GINO model parameters and target columns
    args = define_model_parameters(args)
    args = configure_target_col_indices(args)
    print("Training configuration:")
    for k in sorted(vars(args)):
        print(f"  {k}: {getattr(args, k)}")
    print("="*60 + "\n")
    
    # print(f"\n{'='*60}")
    # print(f"Training Configuration Summary")
    # print(f"{'='*60}")
    # print(f"Device: {args.device}")
    # print(f"Target columns: {args.target_cols} (indices: {args.target_col_indices})")
    # print(f"Input/Output windows: {args.input_window_size}/{args.output_window_size}")
    # print(f"Input/Output channels: {args.in_gno_out_channels}/{args.out_channels}")
    # print(f"Batch size: {args.batch_size}, Epochs: {args.epochs}")
    # print(f"Learning rate: {args.learning_rate}, Scheduler: {args.scheduler_type}")
    # print(f"{'='*60}\n")

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
    train_ds, val_ds = create_patch_datasets(
        dataset_class=GWPatchDatasetMultiCol,
        patch_data_dir=args.patch_data_dir,
        coord_transform=coord_transform,
        obs_transform=obs_transform,
        target_col_indices=args.target_col_indices,
        input_window_size=args.input_window_size,
        output_window_size=args.output_window_size,
    )
    
    print(f"Dataset sizes - Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(train_ds, val_ds, args)
    print(f"Data loaders - Train: {len(train_loader)} batches, Val: {len(val_loader)} batches\n")

    # Initialize model
    base_model = define_gino_model(args)
    model = GINODataParallelAdapter(base_model)
    
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
    )

    # Save final model
    model_path = os.path.join(args.results_dir, 'gino_model.pth')
    torch.save(unwrap_model_for_state_dict(model).state_dict(), model_path)
    print(f"\n{'='*60}")
    print(f"Training complete! Model saved to: {model_path}")
    print(f"{'='*60}")
