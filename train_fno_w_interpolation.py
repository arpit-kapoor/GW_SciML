#!/usr/bin/env python
# coding: utf-8

# Import required libraries
import os, sys
import argparse
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import custom modules
from src.data.dataset import GWDataset, GWGridDataset, Normalize
from src.model.handler import ModelHandler
from src.model.neuralop.fno import FNO
from src.model.neuralop.losses import LpLoss, H1Loss

# Argument parser
parser = argparse.ArgumentParser(description='Train FNO model for groundwater flow')

# Add arguments
parser.add_argument('--data_dir', type=str, default='/srv/scratch/z5370003/projects/data/groundwater/FEFLOW/coastal/variable_density/')
parser.add_argument('--output_dir', type=str, default='/srv/scratch/z5370003/projects/04_groundwater/variable_density/results/FNO')
parser.add_argument('--in_window_size', type=int, default=5)
parser.add_argument('--out_window_size', type=int, default=5)
parser.add_argument('--val_ratio', type=float, default=0.3)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_modes', type=int, default=16)
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--projection_channels', type=int, default=64)
parser.add_argument('--scheduler_interval', type=int, default=10)
parser.add_argument('--n_epochs', type=int, default=500)
args = parser.parse_args()


def plot_2d_projection(x_grid, y_grid, z_grid, values, vmin=None, vmax=None):
    """
    Create 2D projections of 3D data along each axis.
    
    Args:
        x_grid, y_grid, z_grid: Grid coordinates
        values: 3D array of values to plot
        vmin, vmax: Optional min/max values for color scaling
    """
    # Create a figure with 3 subplots for different slices
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Set default min/max values if not provided
    if vmin is None:
        vmin = np.nanmin(values)
    if vmax is None:
        vmax = np.nanmax(values)
    
    # Plot YZ plane (constant X)
    for ix in range(values.shape[0]):
        im1 = ax1.imshow(values[ix,:,:].T, aspect='auto', 
                        extent=[y_grid[0], y_grid[-1], z_grid[0], z_grid[-1]],
                        origin='lower', cmap='viridis', alpha=0.1, vmin=vmin, vmax=vmax)
    ax1.set_title('YZ plane (constant X)')
    ax1.set_xlabel('Y')
    ax1.set_ylabel('Z')
    plt.colorbar(im1, ax=ax1)
    
    # Plot XZ plane (constant Y)
    for iy in range(values.shape[1]):
        im2 = ax2.imshow(values[:,iy,:].T, aspect='auto',
                        extent=[x_grid[0], x_grid[-1], z_grid[0], z_grid[-1]],
                        origin='lower', cmap='viridis', alpha=0.1, vmin=vmin, vmax=vmax)
    ax2.set_title('XZ plane (constant Y)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    plt.colorbar(im2, ax=ax2)
    
    # Plot XY plane (constant Z)
    for iz in range(values.shape[2]):
        im3 = ax3.imshow(values[:,:,iz].T, aspect='auto',
                        extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]],
                        origin='lower', cmap='viridis', alpha=0.1, vmin=vmin, vmax=vmax)
    ax3.set_title('XY plane (constant Z)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    plt.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    plt.show()

    return fig

# Set up data directories
base_data_dir = args.data_dir
interpolated_data_dir = os.path.join(base_data_dir, 'interpolated')

# Define normalization parameters (currently disabled)
_mean = np.array([-0.5474])
_std = np.array([0.6562])
input_transform = Normalize(mean=_mean, std=_std)
output_transform = Normalize(mean=_mean, std=_std)
fill_value = -1


# Create and configure training dataset
train_ds = GWGridDataset(data_path=interpolated_data_dir,
                         dataset='train', val_ratio=args.val_ratio,
                         in_window_size=args.in_window_size,
                         out_window_size=args.out_window_size,
                         input_transform=input_transform,
                         output_transform=output_transform,
                         fillval=fill_value)

train_dl = DataLoader(train_ds, batch_size=args.batch_size, 
                      shuffle=True, pin_memory=True)

# Create and configure validation dataset
val_ds = GWGridDataset(data_path=interpolated_data_dir,
                         dataset='val', val_ratio=args.val_ratio,
                         in_window_size=args.in_window_size,
                         out_window_size=args.out_window_size,
                         input_transform=input_transform,
                         output_transform=output_transform,
                         fillval=fill_value)

val_dl = DataLoader(val_ds, batch_size=args.batch_size, 
                      shuffle=False, pin_memory=True)

# Set up device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Found following device: {device}")

# Configure loss functions
l2loss = LpLoss(d=3, p=2)
h1loss = H1Loss(d=3)


def loss_fn(preds, targets, loss_type='h1'):

    # # Apply mask to targets
    # mask = targets != fill_value
    # preds[mask] = fill_value

    # Calculate loss
    if loss_type == 'h1':
        return h1loss(preds, targets)
    elif loss_type == 'l2':
        return l2loss(preds, targets)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")
    


# Initialize and configure FNO model
model = FNO(n_modes=(args.n_modes, args.n_modes, args.n_modes), 
            in_channels=args.in_window_size, 
            out_channels=args.out_window_size,
            hidden_channels=args.hidden_channels, 
            projection_channels=args.projection_channels).double()

model = model.to(device)

# Configure optimizer and learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(),
                            lr=1e-2,
                            weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.985)

# Initialize model handler
model_handler = ModelHandler(model=model, device=device, optimizer=optimizer, 
                             criterion=loss_fn, scheduler=scheduler, scheduler_interval=args.scheduler_interval)

# Train the model
model_handler.train(train_dl, num_epochs=args.n_epochs, val_loader=val_dl)

# Evaluate model performance
print(f"Train loss: {model_handler.evaluate(train_dl):.3f}")
print(f"Test loss: {model_handler.evaluate(val_dl):.3f}")

# Generate and process predictions
preds = np.array(model_handler.predict(val_dl))
targets = model_handler.get_targets(val_dl)

# Set up output directory
output_dir = os.path.join(args.output_dir, 
                          dt.datetime.now().strftime('%Y%m%d_%H%M%S'))

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Replace -999 with NaN in targets and predictions
mask = targets == fill_value
preds[mask] = np.nan
targets[mask] = np.nan

# Plot and save targets
target_fig = plot_2d_projection(val_ds.x_grid, val_ds.y_grid, val_ds.z_grid, 
                                targets[0, 0], vmin=targets[0, 0].min(), 
                                vmax=targets[0, 0].max())
target_fig.savefig(os.path.join(output_dir, 'target.png'), bbox_inches='tight')

# Plot and save predictions
pred_fig = plot_2d_projection(val_ds.x_grid, val_ds.y_grid, val_ds.z_grid,
                              preds[0, 0], vmin=targets[0, 0].min(), 
                              vmax=targets[0, 0].max())
pred_fig.savefig(os.path.join(output_dir, 'prediction.png'), 
                 bbox_inches='tight')


# Save trained model
torch.save(model.state_dict(), os.path.join(output_dir, 'savedmodel_fno'))
