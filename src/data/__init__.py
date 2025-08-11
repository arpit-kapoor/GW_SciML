# This file makes the src/data directory a Python package

# Import main dataset classes
from .points_dataset import GWDataset
from .grid_dataset import GWGridDataset
from .patch_dataset import GWPatchDataset

# Import transform utilities
from .transform import Normalize

# Define what gets imported with "from src.data import *"
__all__ = [
    # Dataset classes
    'GWDataset',
    'GWGridDataset', 
    'GWPatchDataset',
    
    # Transform classes
    'Normalize',
]

# Version information
__version__ = '1.0.0'

# Package description
__description__ = 'Data loading and preprocessing utilities for groundwater variable density modeling'