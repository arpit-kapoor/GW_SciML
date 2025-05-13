# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from tqdm import tqdm

# %%
# Define data directories
base_data_dir = '/srv/scratch/z5370003/projects/data/groundwater/FEFLOW/coastal/variable_density/'
# base_data_dir = '/Users/akap5486/FEFLOW/variable_density'
raw_data_dir = os.path.join(base_data_dir, 'all')
filtered_data_dir = os.path.join(base_data_dir, 'filtered')

# %%
# Get and sort time series files
ts_files = sorted(os.listdir(filtered_data_dir))
print(f"First 3 files: {ts_files[:3]}")
print(f"Last 3 files: {ts_files[-3:]}")

# %%
def interpolate(X, Y, Z, values):
    """Interpolate 3D data onto a regular grid.
    
    Args:
        X, Y, Z: Input coordinates
        values: Values to interpolate
        
    Returns:
        tuple: (interpolated data, x_grid, y_grid, z_grid)
    """
    # Get min/max bounds for each dimension
    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()
    z_min, z_max = Z.min(), Z.max()
    
    # Create regular 3D grid
    x_grid = np.linspace(x_min, x_max, 40)
    y_grid = np.linspace(y_min, y_max, 40)
    z_grid = np.linspace(z_min, z_max, 40)
    
    # Create meshgrid for interpolation points
    X_interp, Y_interp, Z_interp = np.meshgrid(x_grid, y_grid, z_grid)
    points = np.column_stack((X, Y, Z))
    
    # Interpolate values onto regular grid
    interpolated = griddata(
        points, 
        values,
        (X_interp, Y_interp, Z_interp),
        method='linear',
        fill_value=np.nan
    )
    
    return np.transpose(interpolated, (1, 0, 2)), x_grid, y_grid, z_grid

# %%
# Process all time series files
interpolated_data = []
for ts_file in tqdm(ts_files):
    res_df = pd.read_csv(os.path.join(filtered_data_dir, ts_file))
    origin = res_df[['X', 'Y', 'Z']].min().values
    
    inter_res, x_grid, y_grid, z_grid = interpolate(
        res_df.X, res_df.Y, res_df.Z, res_df['head']
    )
    interpolated_data.append(inter_res)

interpolated_data = np.array(interpolated_data)
print(f"Interpolated data shape: {interpolated_data.shape}")

# Create interpolated directory if it doesn't exist
interpolated_dir = os.path.join(base_data_dir, 'interpolated')
os.makedirs(interpolated_dir, exist_ok=True)

# Save interpolated data and grids
np.save(os.path.join(interpolated_dir, 'data.npy'), interpolated_data)
np.save(os.path.join(interpolated_dir, 'x_grid.npy'), x_grid)
np.save(os.path.join(interpolated_dir, 'y_grid.npy'), y_grid)
np.save(os.path.join(interpolated_dir, 'z_grid.npy'), z_grid)

print(f"Saved interpolated data and grids to {interpolated_dir}")


# %%
def plot_3d_slices(inter_res, x_grid, y_grid, z_grid):
    """Plot 3D slices of interpolated data."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    vmin, vmax = np.nanmin(inter_res), np.nanmax(inter_res)
    
    # YZ plane (constant X)
    for ix in range(inter_res.shape[0]):
        im1 = ax1.imshow(
            inter_res[ix,:,:].T,
            aspect='auto',
            extent=[y_grid[0], y_grid[-1], z_grid[0], z_grid[-1]],
            origin='lower',
            cmap='viridis',
            alpha=0.1,
            vmin=vmin,
            vmax=vmax
        )
    ax1.set_title('YZ plane (constant X)')
    ax1.set_xlabel('Y')
    ax1.set_ylabel('Z')
    plt.colorbar(im1, ax=ax1)
    
    # XZ plane (constant Y)
    for iy in range(inter_res.shape[1]):
        im2 = ax2.imshow(
            inter_res[:,iy,:].T,
            aspect='auto',
            extent=[x_grid[0], x_grid[-1], z_grid[0], z_grid[-1]],
            origin='lower',
            cmap='viridis',
            alpha=0.1,
            vmin=vmin,
            vmax=vmax
        )
    ax2.set_title('XZ plane (constant Y)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    plt.colorbar(im2, ax=ax2)
    
    # XY plane (constant Z)
    for iz in range(inter_res.shape[2]):
        im3 = ax3.imshow(
            inter_res[:,:,iz].T,
            aspect='auto',
            extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]],
            origin='lower',
            cmap='viridis',
            alpha=0.1,
            vmin=vmin,
            vmax=vmax
        )
    ax3.set_title('XY plane (constant Z)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    plt.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    plt.show()

# %%
def plot_scatter_slices(res_df):
    """Plot scatter plots of raw data in 3D slices."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # YZ plane (constant X)
    ax1.scatter(res_df.Y, res_df.Z, c=res_df['head'], cmap='viridis', alpha=0.3)
    ax1.set_title('YZ plane (constant X)')
    plt.colorbar(ax1.collections[0], ax=ax1)
    
    # XZ plane (constant Y)
    ax2.scatter(res_df.X, res_df.Z, c=res_df['head'], cmap='viridis', alpha=0.3)
    ax2.set_title('XZ plane (constant Y)')
    plt.colorbar(ax2.collections[0], ax=ax2)
    
    # XY plane (constant Z)
    ax3.scatter(res_df.X, res_df.Y, c=res_df['head'], cmap='viridis', alpha=0.3)
    ax3.set_title('XY plane (constant Z)')
    plt.colorbar(ax3.collections[0], ax=ax3)
    
    plt.tight_layout()
    plt.show()

# %%
# Plot the results
plot_3d_slices(inter_res, x_grid, y_grid, z_grid)
plot_scatter_slices(res_df)
