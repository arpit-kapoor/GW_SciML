#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import required libraries
import os
import json
import pandas as pd
import numpy as np

from tqdm import tqdm


# In[8]:


# Define data directories
base_data_dir = '/srv/scratch/z5370003/projects/data/groundwater/FEFLOW/coastal/variable_density/'
# base_data_dir = '/Users/arpitkapoor/Library/CloudStorage/OneDrive-UNSW/Shared/Projects/01_PhD/05_groundwater/data/FEFLOW/variable_density'  # Uncomment for local testing
raw_data_dir = os.path.join(base_data_dir, 'all')
filtered_data_dir = os.path.join(base_data_dir, 'filter_all_ts')
# forcings_data_dir = os.path.join(base_data_dir, 'forcings_corrected')
forcings_data_dir = os.path.join(base_data_dir, 'forcings_corrected_all')

print(f"Base data directory: {base_data_dir}")
print(f"Raw data directory: {raw_data_dir}")
print(f"Filtered data directory: {filtered_data_dir}")
print(f"Forcings data directory: {forcings_data_dir}")


# In[9]:


# Get and sort time series files
ts_files = sorted(os.listdir(raw_data_dir))
print(f"Total number of files: {len(ts_files)}")
print(f"First 3 files: {ts_files[:3]}")
print(f"Last 3 files: {ts_files[-3:]}")


# In[10]:


# Define json file path
patch_config_json = os.path.join(base_data_dir, 'patches.json')

with open(patch_config_json, 'r') as f:
    patch_config = json.load(f)


# In[11]:


# patch_data_dir = os.path.join(base_data_dir, 'filter_patch_all_ts')
patch_data_dir = os.path.join(base_data_dir, 'patch_all_ts')

# data_dir = filtered_data_dir
data_dir = raw_data_dir


# In[ ]:


target_cols = ['mass_concentration', 'head', 'pressure']
forcing_cols = ['mass_concentration_bc', 'head_bc', 'recharge_forcing', 'sea_level_forcing']
coords_cols = ['X', 'Y', 'Z']

patch_data = {}

fillval = -999

for k, v in patch_config.items():
    
    # Get the patch configuration
    config = patch_config[k]

    # Print patch information
    print(f"\nProcessing patch {k}")
    print(f"Patch {k} has {len(config['core_nodes'])} core nodes and {len(config['ghost_nodes'])} ghost nodes")
    print(f"Patch {k} has {len(config['neighbour_patches'])} neighbour patches")
    print(f"Patch {k} has {config['slice_group']} slice group")

    # Initialize lists to store data
    core_patch_data = []
    ghost_patch_data = []

    core_forcings_data = []
    ghost_forcings_data = []

    # Load the data#
    for ts_file in ts_files:
        ts_df = pd.read_csv(os.path.join(data_dir, ts_file))
        core_patch_data.append(ts_df.loc[ts_df['node'].isin(config['core_nodes']), target_cols].values)
        ghost_patch_data.append(ts_df.loc[ts_df['node'].isin(config['ghost_nodes']), target_cols].values)
        
        ts_forcings_df = pd.read_csv(os.path.join(forcings_data_dir, ts_file))
        core_forcings_data.append(ts_forcings_df.loc[ts_forcings_df['node'].isin(config['core_nodes']), forcing_cols].values)
        ghost_forcings_data.append(ts_forcings_df.loc[ts_forcings_df['node'].isin(config['ghost_nodes']), forcing_cols].values)

        
    # Convert to numpy arrays
    core_patch_data = np.nan_to_num(np.array(core_patch_data))
    ghost_patch_data = np.nan_to_num(np.array(ghost_patch_data))
    core_forcings_data = np.nan_to_num(np.array(core_forcings_data))
    ghost_forcings_data = np.nan_to_num(np.array(ghost_forcings_data))
    
    core_coords = ts_df.loc[ts_df['node'].isin(config['core_nodes']), coords_cols].values
    ghost_coords = ts_df.loc[ts_df['node'].isin(config['ghost_nodes']), coords_cols].values

    # Create directory for patch data
    patch_dir_path = os.path.join(patch_data_dir, f'patch_{int(k):03d}')
    os.makedirs(patch_dir_path, exist_ok=True)

    # Save the data
    np.save(os.path.join(patch_dir_path, 'core_obs.npy'), core_patch_data)
    np.save(os.path.join(patch_dir_path, 'ghost_obs.npy'), ghost_patch_data)
    np.save(os.path.join(patch_dir_path, 'core_coords.npy'), core_coords)
    np.save(os.path.join(patch_dir_path, 'ghost_coords.npy'), ghost_coords)
    np.save(os.path.join(patch_dir_path, 'core_forcings.npy'), core_forcings_data)
    np.save(os.path.join(patch_dir_path, 'ghost_forcings.npy'), ghost_forcings_data)


# In[41]:


# concat_arr = []


# for k, v in patch_config.items():

#     patch_dir_path = os.path.join(filter_patch_data_dir, f'patch_{int(k):03d}')

#     core_forcings_data = np.load(os.path.join(patch_dir_path, 'core_forcings.npy'))
#     ghost_forcings_data = np.load(os.path.join(patch_dir_path, 'ghost_forcings.npy'))

#     # print((~np.isnan(core_forcings_data)).sum()/core_forcings_data.flatten().shape[0])
#     concat_arr.append(core_forcings_data.reshape(-1, 4))

# concat_arr = np.concatenate(concat_arr, axis=0)


# In[ ]:




