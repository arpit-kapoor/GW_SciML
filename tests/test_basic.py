import pytest
import sys

def test_core_imports():
    # Attempt to import core modules heavily relied upon to ensure
    # environment consistency between macOS and Linux.
    try:
        import open3d
        import numpy
        import pandas
        import scipy
        import torch
        import torchvision
        import torch_geometric
        import torch_harmonics
        import torchinfo
        import tensorly
    except ImportError as e:
        pytest.fail(f"Core module missing or failed to load: {e}")

def test_pytorch_versions():
    import torch
    # Print out torch version to log
    print(f"Torch Version: {torch.__version__}")
    
    # Simple tensor creation to ensure it's functional
    x = torch.rand(5, 3)
    assert x.shape == (5, 3)
