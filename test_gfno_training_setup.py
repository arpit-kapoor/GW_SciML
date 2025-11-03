"""
Quick test script to verify GFNO training setup.

This script performs sanity checks on:
- Data loading
- Model initialization
- Forward pass
- Loss computation
- Multi-GPU wrapping (if available)
- Checkpoint saving/loading

Run this before submitting full training jobs to catch configuration issues early.
"""

import os
import sys
import torch
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.data.plane_dataset import GWPlaneDatasetFromFiles
from src.data.batch_sampler import PatchBatchSampler
from torch.utils.data import DataLoader
from src.models import GFNO
from src.models.neuralop.losses import LpLoss


def test_data_loading(data_dir, batch_size=4, val_ratio=0.2):
    """Test dataset and dataloader."""
    print("\n" + "="*80)
    print("TEST 1: Data Loading")
    print("="*80)
    
    try:
        # Create training dataset
        print(f"Loading training dataset from: {data_dir}")
        train_dataset = GWPlaneDatasetFromFiles(
            data_dir=data_dir, 
            dataset='train',
            val_ratio=val_ratio,
            fill_nan_value=-999.0
        )
        print(f"✓ Training dataset loaded: {len(train_dataset)} sequences")
        
        # Create validation dataset
        print(f"Loading validation dataset from: {data_dir}")
        val_dataset = GWPlaneDatasetFromFiles(
            data_dir=data_dir,
            dataset='val', 
            val_ratio=val_ratio,
            fill_nan_value=-999.0
        )
        print(f"✓ Validation dataset loaded: {len(val_dataset)} sequences")
        print(f"✓ Total sequences: {len(train_dataset) + len(val_dataset)}")
        print(f"✓ Train/Val split: {len(train_dataset)}/{len(val_dataset)} ({1-val_ratio:.1%}/{val_ratio:.1%})")
        
        # Use training dataset for further tests
        dataset = train_dataset
        
        # Create sampler
        sampler = PatchBatchSampler(dataset, batch_size=batch_size, shuffle_within_batches=False, shuffle_patches=False)
        print(f"✓ Sampler created: {len(sampler)} batches")
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_sampler=sampler)
        print(f"✓ DataLoader created")
        
        # Fetch first batch
        batch = next(iter(dataloader))
        print(f"\n✓ First batch loaded successfully")
        print(f"  Batch keys: {list(batch.keys())}")
        print(f"  Batch size: {batch['input_geom'].shape[0]}")
        print(f"  Input geom shape: {batch['input_geom'].shape}")
        print(f"  Input data shape: {batch['input_data'].shape}")
        print(f"  Latent geom shape: {batch['latent_geom'].shape}")
        print(f"  Latent features shape: {batch['latent_features'].shape}")
        print(f"  Output latent features shape: {batch['output_latent_features'].shape}")
        
        return True, dataset, dataloader
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def test_model_initialization():
    """Test GFNO model initialization."""
    print("\n" + "="*80)
    print("TEST 2: Model Initialization")
    print("="*80)
    
    try:
        model = GFNO(
            gno_coord_dim=3,
            gno_radius=0.15,
            gno_out_channels=16,
            gno_channel_mlp_layers=[32, 64, 32],
            latent_feature_channels=4,
            fno_n_layers=4,
            fno_n_modes=(6, 8, 8),
            fno_hidden_channels=64,
            lifting_channels=64,
            projection_channel_ratio=2,
            out_channels=2,
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Model initialized successfully")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        return True, model
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_forward_pass(model, dataloader, device):
    """Test forward pass through model."""
    print("\n" + "="*80)
    print("TEST 3: Forward Pass")
    print("="*80)
    
    try:
        model = model.to(device)
        model.eval()
        
        # Get batch
        batch = next(iter(dataloader))
        
        # Move to device
        input_geom = batch['input_geom'].to(device)
        input_data = batch['input_data'].to(device)
        latent_geom = batch['latent_geom'].to(device)
        latent_features = batch['latent_features'].to(device)
        output_latent_features = batch['output_latent_features'].to(device)
        
        print(f"✓ Batch moved to device: {device}")
        
        # Forward pass
        with torch.no_grad():
            predictions = model(
                input_geom=input_geom,
                latent_queries=latent_geom,
                x=input_data,
                latent_features=latent_features
            )
        
        print(f"✓ Forward pass successful")
        print(f"  Prediction shape: {predictions.shape}")
        print(f"  Expected shape: {output_latent_features[..., -2:].shape}")
        print(f"  Prediction stats: min={predictions.min().item():.4f}, max={predictions.max().item():.4f}, mean={predictions.mean().item():.4f}")
        
        return True, predictions, output_latent_features
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def test_loss_computation(predictions, targets):
    """Test loss computation."""
    print("\n" + "="*80)
    print("TEST 4: Loss Computation")
    print("="*80)
    
    try:
        loss_fn = LpLoss(d=1, p=2, reduce_dims=[0, 1], reductions='mean')
        
        # Extract target columns
        target_values = targets[..., -2:]  # Last 2 channels: head, mass_conc
        
        loss = loss_fn(predictions, target_values)
        
        print(f"✓ Loss computed successfully")
        print(f"  Loss value: {loss.item():.6f}")
        print(f"  Loss shape: {loss.shape}")
        
        return True, loss
    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_backward_pass(model, loss, device):
    """Test backward pass and gradient computation."""
    print("\n" + "="*80)
    print("TEST 5: Backward Pass")
    print("="*80)
    
    try:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
        
        avg_grad_norm = np.mean(grad_norms)
        max_grad_norm = np.max(grad_norms)
        
        print(f"✓ Backward pass successful")
        print(f"  Avg gradient norm: {avg_grad_norm:.6f}")
        print(f"  Max gradient norm: {max_grad_norm:.6f}")
        print(f"  Parameters with gradients: {len(grad_norms)}")
        
        # Optimizer step
        optimizer.step()
        print(f"✓ Optimizer step successful")
        
        return True
    except Exception as e:
        print(f"✗ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_gpu(model, dataloader, device):
    """Test multi-GPU wrapping if available."""
    print("\n" + "="*80)
    print("TEST 6: Multi-GPU Support")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("⊘ CUDA not available, skipping multi-GPU test")
        return True
    
    if torch.cuda.device_count() <= 1:
        print(f"⊘ Only {torch.cuda.device_count()} GPU available, skipping multi-GPU test")
        return True
    
    try:
        from train_gfno_2d_planes import GFNODataParallelAdapter
        
        print(f"Testing with {torch.cuda.device_count()} GPUs")
        
        # Wrap model
        model = model.to(device)
        wrapped_model = GFNODataParallelAdapter(model)
        wrapped_model = torch.nn.DataParallel(wrapped_model)
        
        print(f"✓ Model wrapped with DataParallel")
        
        # Test forward pass
        wrapped_model.eval()
        batch = next(iter(dataloader))
        
        input_geom = batch['input_geom'].to(device)
        input_data = batch['input_data'].to(device)
        latent_geom = batch['latent_geom'].to(device)
        latent_features = batch['latent_features'].to(device)
        
        with torch.no_grad():
            predictions = wrapped_model(
                input_geom=input_geom,
                latent_queries=latent_geom,
                x=input_data,
                latent_features=latent_features
            )
        
        print(f"✓ Multi-GPU forward pass successful")
        print(f"  Prediction shape: {predictions.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Multi-GPU test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checkpoint_save_load(model, optimizer, results_dir):
    """Test checkpoint saving and loading."""
    print("\n" + "="*80)
    print("TEST 7: Checkpoint Save/Load")
    print("="*80)
    
    try:
        import tempfile
        
        # Create temporary checkpoint directory
        checkpoint_dir = os.path.join(results_dir, 'test_checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, 'test_checkpoint.pth')
        checkpoint = {
            'epoch': 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': [0.5, 0.4],
            'val_losses': [0.6, 0.5],
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Checkpoint saved to: {checkpoint_path}")
        
        # Load checkpoint
        loaded_checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(loaded_checkpoint['model_state_dict'])
        optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
        
        print(f"✓ Checkpoint loaded successfully")
        print(f"  Epoch: {loaded_checkpoint['epoch']}")
        print(f"  Train losses: {loaded_checkpoint['train_losses']}")
        print(f"  Val losses: {loaded_checkpoint['val_losses']}")
        
        # Clean up
        os.remove(checkpoint_path)
        os.rmdir(checkpoint_dir)
        print(f"✓ Test checkpoint cleaned up")
        
        return True
    except Exception as e:
        print(f"✗ Checkpoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("GFNO TRAINING SETUP TEST")
    print("="*80)
    
    # Configuration
    data_dir = '/Users/arpitkapoor/data/GW/2d_plane_sequences'
    results_dir = '/tmp/gfno_test'
    batch_size = 4
    val_ratio = 0.2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nConfiguration:")
    print(f"  Data dir: {data_dir}")
    print(f"  Results dir: {results_dir}")
    print(f"  Batch size: {batch_size}")
    print(f"  Validation ratio: {val_ratio}")
    print(f"  Device: {device}")
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Track results
    results = {}
    
    # Test 1: Data loading
    success, dataset, dataloader = test_data_loading(data_dir, batch_size, val_ratio)
    results['Data Loading'] = success
    if not success:
        print("\n✗ Cannot proceed without data loading. Exiting.")
        return
    
    # Test 2: Model initialization
    success, model = test_model_initialization()
    results['Model Initialization'] = success
    if not success:
        print("\n✗ Cannot proceed without model. Exiting.")
        return
    
    # Test 3: Forward pass
    success, predictions, targets = test_forward_pass(model, dataloader, device)
    results['Forward Pass'] = success
    if not success:
        print("\n✗ Cannot proceed without forward pass. Exiting.")
        return
    
    # Test 4: Loss computation
    success, loss = test_loss_computation(predictions, targets)
    results['Loss Computation'] = success
    
    # Test 5: Backward pass
    if success:
        # Need to redo forward pass for backward
        model.train()
        batch = next(iter(dataloader))
        input_geom = batch['input_geom'].to(device)
        input_data = batch['input_data'].to(device)
        latent_geom = batch['latent_geom'].to(device)
        latent_features = batch['latent_features'].to(device)
        output_latent_features = batch['output_latent_features'].to(device)
        
        predictions = model(
            input_geom=input_geom,
            latent_queries=latent_geom,
            x=input_data,
            latent_features=latent_features
        )
        
        loss_fn = LpLoss(d=1, p=2, reduce_dims=[0, 1], reductions='mean')
        loss = loss_fn(predictions, output_latent_features[..., -2:])
        
        success = test_backward_pass(model, loss, device)
        results['Backward Pass'] = success
    
    # Test 6: Multi-GPU
    success = test_multi_gpu(model, dataloader, device)
    results['Multi-GPU'] = success
    
    # Test 7: Checkpoint save/load
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    success = test_checkpoint_save_load(model, optimizer, results_dir)
    results['Checkpoint Save/Load'] = success
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:10s} {test_name}")
        if not passed:
            all_passed = False
    
    print("="*80)
    if all_passed:
        print("✓ All tests passed! Training setup is ready.")
    else:
        print("✗ Some tests failed. Please fix issues before training.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
