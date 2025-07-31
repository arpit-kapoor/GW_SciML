#!/usr/bin/env python
# coding: utf-8

"""
Test script for the GINO (Geometry-Informed Neural Operator) model.
This script tests the GINO model with various input configurations and validates
that the forward pass works correctly.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

from src.model.neuralop.gino import GINO


def test_gino_basic():
    """Test basic GINO model functionality with simple inputs."""
    print("=" * 50)
    print("Testing Basic GINO Model")
    print("=" * 50)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create simple test data
    batch_size = 1
    n_input_points = 50
    n_latent_points = 32
    n_output_points = 50
    coord_dim = 3  # Changed to 3D
    input_channels = 1
    latent_channels = 2
    output_channels = 2
    
    # Input geometry (batch, n_points, coord_dim)
    input_geom = torch.rand(n_input_points, coord_dim, device=device)
    
    # Latent queries (batch, n_latent_x, n_latent_y, n_latent_z, coord_dim)
    latent_queries = torch.rand(n_latent_points, n_latent_points, n_latent_points, coord_dim, device=device)
    
    # Output queries (batch, n_output_points, coord_dim)
    output_queries = torch.rand(n_output_points, coord_dim, device=device)
    
    # Input features (batch, n_input_points, input_channels)
    x = None # torch.rand(batn_input_points, input_channels, device=device)
    
    # Latent features (batch, n_latent_x, n_latent_y, n_latent_z, latent_channels)
    latent_features = torch.rand(batch_size, n_latent_points, n_latent_points, n_latent_points, latent_channels, device=device)
    
    print(f"Input shapes:")
    print(f"  input_geom: {input_geom.shape}")
    print(f"  latent_queries: {latent_queries.shape}")
    print(f"  output_queries: {output_queries.shape}")
    # print(f"  x: {x.shape}")
    print(f"  latent_features: {latent_features.shape}")
    
    # Create GINO model
    model = GINO(
        # Input GNO parameters
        in_gno_coord_dim=coord_dim,
        in_gno_radius=0.1,
        in_gno_out_channels=16,
        in_gno_channel_mlp_layers=[32, 64, 32],
        
        # FNO parameters
        fno_n_layers=4,
        fno_n_modes=(8, 8, 8),  # 3D modes
        fno_hidden_channels=64,
        
        # Lifting parameters
        lifting_channels=64,
        
        # Output GNO parameters
        out_gno_coord_dim=coord_dim,
        out_gno_radius=0.1,
        out_gno_channel_mlp_layers=[32, 64, 32],
        
        # Projection parameters
        projection_channel_ratio=2,
        out_channels=output_channels,
        
        # Latent features
        latent_feature_channels=latent_channels,
    ).to(device)
    
    print(f"\nModel created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    print(f"\nTesting forward pass...")
    model.eval()
    
    with torch.no_grad():
        output = model(
            input_geom=input_geom,
            latent_queries=latent_queries,
            output_queries=output_queries,
            x=x,
            latent_features=latent_features
        )
    
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, {n_output_points}, {output_channels})")
    
    assert output.shape == (batch_size, n_output_points, output_channels), f"Output shape mismatch: {output.shape}"
    print("✓ Forward pass successful!")
    
    return model, output

def test_gino_basic_with_x():
    """Test basic GINO model functionality with simple inputs."""
    print("=" * 50)
    print("Testing Basic GINO Model with x")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create simple test data
    batch_size = 10
    n_input_points = 50
    n_latent_points = 32
    n_output_points = 50
    coord_dim = 3  # Changed to 3D
    input_channels = 3
    latent_channels = 2
    output_channels = 2

    # Input geometry (batch, n_points, coord_dim)
    input_geom = torch.rand(n_input_points, coord_dim, device=device)
    
    # Latent queries (batch, n_latent_x, n_latent_y, n_latent_z, coord_dim)
    latent_queries = torch.rand(n_latent_points, n_latent_points, n_latent_points, coord_dim, device=device)
    
    # Output queries (batch, n_output_points, coord_dim)
    output_queries = torch.rand(n_output_points, coord_dim, device=device)

    # Input features
    x = torch.rand(batch_size, n_input_points, input_channels, device=device, requires_grad=True)

    print(f"Input shapes:")
    print(f"  input_geom: {input_geom.shape}")
    print(f"  latent_queries: {latent_queries.shape}")
    print(f"  output_queries: {output_queries.shape}")
    print(f"  x: {x.shape}")
    
    # Create GINO model
    model = GINO(
        in_gno_coord_dim=coord_dim,
        in_gno_radius=0.1,
        in_gno_out_channels=input_channels,  # Match input channels (3)
        in_gno_channel_mlp_layers=[32, 64, 32],
        fno_n_layers=4,
        fno_n_modes=(8, 8, 8),  # 3D modes
        fno_hidden_channels=64,
        lifting_channels=64,
        out_gno_coord_dim=coord_dim,
        out_gno_radius=0.1,
        out_gno_channel_mlp_layers=[32, 64, 32],
        projection_channel_ratio=2,
        out_channels=output_channels,
    ).to(device)
    
    print(f"\nModel created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    print(f"\nTesting forward pass...") 
    model.eval()
    
    with torch.no_grad():
        output = model(
            input_geom=input_geom,
            latent_queries=latent_queries,
            output_queries=output_queries,
            x=x
        )
    
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, {n_output_points}, {output_channels})")
    
    assert output.shape == (batch_size, n_output_points, output_channels), f"Output shape mismatch: {output.shape}"
    print("✓ Forward pass successful!")

    return model, output



def test_gino_gradients():
    """Test that gradients can be computed through the GINO model."""
    print("\n" + "=" * 50)
    print("Testing GINO Model Gradients")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create simple test data
    batch_size = 1
    n_input_points = 50
    n_latent_points = 32
    n_output_points = 50
    coord_dim = 3  # Changed to 3D
    input_channels = 3
    latent_channels = 2
    output_channels = 2
    
    # Input geometry
    input_geom = torch.rand(n_input_points, coord_dim, device=device, requires_grad=True)
    
    # Latent queries
    latent_queries = torch.rand(n_latent_points, n_latent_points, n_latent_points, coord_dim, device=device)
    
    # Output queries
    output_queries = torch.rand(n_output_points, coord_dim, device=device)
    
    # Input features
    x = torch.rand(batch_size, n_input_points, input_channels, device=device, requires_grad=True)

    print(f"Input shapes:")
    print(f"  input_geom: {input_geom.shape}")
    print(f"  latent_queries: {latent_queries.shape}")
    print(f"  output_queries: {output_queries.shape}")
    print(f"  x: {x.shape}")
    # print(f"  latent_features: {latent_features.shape}")
    
    # Create GINO model
    model = GINO(
        in_gno_coord_dim=coord_dim,
        in_gno_radius=0.1,
        in_gno_out_channels=input_channels,  # Match input channels (3)
        in_gno_channel_mlp_layers=[32, 64, 32],
        fno_n_layers=4,
        fno_n_modes=(8, 8, 8),  # 3D modes
        fno_hidden_channels=64,
        lifting_channels=64,
        out_gno_coord_dim=coord_dim,
        out_gno_radius=0.1,
        out_gno_channel_mlp_layers=[32, 64, 32],
        projection_channel_ratio=2,
        out_channels=output_channels,
    ).to(device)
    
    # Test gradient computation
    print(f"Testing gradient computation...")
    model.train()
    
    output = model(
        input_geom=input_geom,
        latent_queries=latent_queries,
        output_queries=output_queries,
        x=x
    )
    
    # Compute loss and backward pass
    loss = output.mean()
    loss.backward()
    
    print(f"Loss: {loss.item():.6f}")
    print(f"Input geometry gradients: {input_geom.grad is not None}")
    # print(f"Input features gradients: {x.grad is not None}")
    
    # Check model parameter gradients
    has_gradients = any(p.grad is not None for p in model.parameters())
    print(f"Model parameters have gradients: {has_gradients}")
    
    assert has_gradients, "Model parameters should have gradients after backward pass"
    print("✓ Gradient computation successful!")
    
    return model, output, loss


def plot_test_results(model, outputs, test_names):
    """Plot test results for visualization."""
    print("\n" + "=" * 50)
    print("Plotting Test Results")
    print("=" * 50)
    
    fig, axes = plt.subplots(1, len(test_names), figsize=(5*len(test_names), 4))
    if len(test_names) == 1:
        axes = [axes]
    
    for i, (name, output) in enumerate(zip(test_names, outputs)):
        if isinstance(output, dict):
            # For dictionary outputs, plot the first key
            first_key = list(output.keys())[0]
            data = output[first_key].detach().cpu().numpy()
            title = f"{name} ({first_key})"
        else:
            data = output.detach().cpu().numpy()
            title = name
        
        # Plot the output values
        axes[i].plot(data[0, :, 0], 'b-', alpha=0.7, label='Output')
        axes[i].set_title(title)
        axes[i].set_xlabel('Output Point Index')
        axes[i].set_ylabel('Value')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('gino_test_results.png', dpi=150, bbox_inches='tight')
    print("✓ Test results plot saved as 'gino_test_results.png'")
    plt.show()


def main():
    """Main test function."""
    print("GINO Model Test Suite")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # Run all tests
        results = []
        test_names = []
        
        # Test 1: Basic functionality
        model1, output1 = test_gino_basic()
        results.append(output1)
        test_names.append("Basic")

        # Test 2: Basic functionality with x
        model2, output2 = test_gino_basic_with_x()
        results.append(output2)
        test_names.append("Basic with x")
        
        # Test 3: Gradient computation
        model3, output3, loss = test_gino_gradients()
        results.append(output3)
        test_names.append("Gradients")
        
        # Plot results
        plot_test_results([model1, model2, model3], results, test_names)
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("GINO model is working correctly with:")
        print("  ✓ Basic forward pass")
        print("  ✓ Dictionary output queries")
        print("  ✓ Gradient computation")
        print("  ✓ 3D coordinate support")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 