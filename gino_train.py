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
    calculate_forcings_transform,
)
from src.models.neuralop.gino import GINO
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


def add_gino_model_args(parser):
    """Add GINO-specific model arguments to the parser."""
    parser.add_argument('--gno-radius', type=float, default=None,
                        help='Override the default GNO radius (default: 0.18). '
                             'Also overrides the value read from a checkpoint when '
                             'resuming training, so the model is re-instantiated '
                             'with this radius rather than the one saved in the checkpoint.')
    return parser


def define_model_parameters(args):
    """Define GINO architecture parameters based on input configuration."""
    args.coord_dim = 3
    args.n_target_cols = len(args.target_cols)
    # Preserve CLI-supplied --gno-radius; fall back to the hardcoded default.
    if not hasattr(args, 'gno_radius') or args.gno_radius is None:
        args.gno_radius = 0.18
    args.in_gno_out_channels = args.input_window_size * args.n_target_cols
    if args.forcings_required:
        args.forcings_dim = 4  # Number of forcings features
        args.in_gno_out_channels += args.forcings_dim
    args.in_gno_channel_mlp_layers = [32, 64, 32]
    args.fno_n_layers = 4
    args.fno_n_modes = (8, 8, 6)
    args.fno_hidden_channels = 128
    args.lifting_channels = 64
    args.out_gno_channel_mlp_layers = [32, 64, 32]
    args.projection_channel_ratio = 2
    args.out_channels = args.output_window_size * args.n_target_cols
    args.latent_query_dims = (16, 16, 8)
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
    
    # Capture CLI-supplied --gno-radius before define_model_parameters may set
    # the default (None means the user did not pass the flag).
    cli_gno_radius = getattr(args, 'gno_radius', None)

    # Configure GINO model parameters and target columns
    args = define_model_parameters(args)

    # Log effective gno_radius and whether it came from the CLI or the default.
    if cli_gno_radius is not None:
        print(f"GNO radius: {args.gno_radius} (overridden via --gno-radius; "
              "also applies when resuming from checkpoint)")
    else:
        print(f"GNO radius: {args.gno_radius} (default)")
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
        forcings_transform=calculate_forcings_transform(),
        forcings_required=args.forcings_required,
        resolution_ratio=args.resolution_ratio,
        min_resolution_ratio=args.min_resolution_ratio,
        sampling_strategy=args.sampling_strategy,
    )
    
    print(f"Dataset sizes - Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(train_ds, val_ds, args)
    print(f"Data loaders - Train: {len(train_loader)} batches, Val: {len(val_loader)} batches\n")

    # Initialize model
    base_model = define_gino_model(args)
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

    # batch shapes
    for batch_idx, batch in enumerate(train_loader):
        print(f"Batch {batch_idx}: point_coords: {batch['point_coords'].shape}, \
                latent_queries: {batch['latent_queries'].shape}, x: {batch['x'].shape}, \
                y: {batch['y'].shape}, weights: {batch['weights'].shape}")
        break
    
    
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

    # ---------------------------------------------------------
    # INJECTED CODE: Compute & save raw targets and predictions
    # ---------------------------------------------------------
    import numpy as np
    from src.training.trainer import _default_gino_forward, _default_extract_core_points

    print("\nComputing final predictions for train and val datasets...")

    # Recreate the train loader without shuffling for 1:1 consistent mapping
    eval_train_sampler = PatchBatchSampler(
        train_ds, batch_size=args.batch_size,
        shuffle_within_batches=False,
        shuffle_patches=False,
        seed=None
    )
    eval_train_loader = DataLoader(
        train_ds, 
        batch_sampler=eval_train_sampler, 
        collate_fn=train_loader.collate_fn
    )

    def collect_predictions(loader):
        model.eval()
        all_preds = {}
        all_targets = {}
        patch_id = 0
        prev_nodes = -1

        
        with torch.no_grad():
            for batch in loader:
                # Forward pass
                outputs = _default_gino_forward(model, batch, args)
                # Filter out the boundary 'ghost' patches
                core_outputs, core_targets, _ = _default_extract_core_points(outputs, batch, args)

                batch_size = core_outputs.shape[0]
                num_nodes = core_outputs.shape[1]

                if num_nodes != prev_nodes:
                    patch_id += 1
                    print(f"Processing patch {patch_id} with {num_nodes} nodes (prev: {prev_nodes})")
                    all_preds[patch_id] = []
                    all_targets[patch_id] = []

                all_preds[patch_id].append(core_outputs.cpu().numpy())
                all_targets[patch_id].append(core_targets.cpu().numpy())

                prev_nodes = num_nodes
            
        for patch_id in all_preds.keys():
            all_preds[patch_id] = np.concatenate(all_preds[patch_id], axis=0)
            all_targets[patch_id] = np.concatenate(all_targets[patch_id], axis=0)

        all_preds_array = np.transpose(np.concatenate(list(all_preds.values()), axis=1), (1, 0, 2))
        all_targets_array = np.transpose(np.concatenate(list(all_targets.values()), axis=1), (1, 0, 2))

        print(f"Collected predictions and targets for {len(all_preds)} nodes. Final shapes - Preds: {all_preds_array.shape}, Targets: {all_targets_array.shape}")
        return all_preds_array, all_targets_array


    # Compute mean and std for inverse normalization
    obs_mean = obs_transform.mean[args.target_col_indices].cpu().numpy()
    obs_std = obs_transform.std[args.target_col_indices].cpu().numpy()
    print(f"Applying inverse normalization to predictions and targets using mean: {obs_mean}, std: {obs_std}")
    
    print("Processing Validation Set...")
    val_preds, val_targets = collect_predictions(val_loader)

    # Inverse transform predictions and targets to original scale
    val_preds = val_preds * obs_std + obs_mean
    val_targets = val_targets * obs_std + obs_mean
    
    # Save the predictions and targets as .npy files for later analysis
    np.save(os.path.join(args.results_dir, 'val_preds.npy'), val_preds)
    np.save(os.path.join(args.results_dir, 'val_targets.npy'), val_targets)

    print("Processing Training Set...")
    train_preds, train_targets = collect_predictions(eval_train_loader)

    # Inverse transform predictions and targets to original scale
    train_preds = train_preds * obs_std + obs_mean
    train_targets = train_targets * obs_std + obs_mean
    
    # Save the predictions and targets as .npy files for later analysis
    np.save(os.path.join(args.results_dir, 'train_preds.npy'), train_preds)
    np.save(os.path.join(args.results_dir, 'train_targets.npy'), train_targets)

    print(f"Evaluations complete! Saved train/val '.npy' predictions and targets to {args.results_dir}")

