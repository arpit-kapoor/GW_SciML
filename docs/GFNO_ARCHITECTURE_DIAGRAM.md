# GFNO Architecture Visualization

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INPUT DATA (Per Batch)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Boundary Conditions (Irregular Points)                                    │
│  ┌──────────────────────┐  ┌──────────────────────┐                       │
│  │ input_geom           │  │ input_data           │                       │
│  │ [B, N_bc, 3]         │  │ [B, N_bc, 2]         │                       │
│  │ (S, Z, T coords)     │  │ (head, mass_conc)    │                       │
│  └──────────────────────┘  └──────────────────────┘                       │
│                                                                             │
│  Latent Grid (Regular Grid)                                                │
│  ┌──────────────────────┐  ┌──────────────────────┐                       │
│  │ latent_geom          │  │ latent_features      │                       │
│  │ [B, α, H, W, 3]      │  │ [B, α, H, W, 4]      │                       │
│  │ (Grid coordinates)   │  │ (X, Y, head, conc)   │                       │
│  └──────────────────────┘  └──────────────────────┘                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                        STAGE 1: GNO ENCODING                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  GNOBlock (Boundary → Grid)                                                │
│  ┌───────────────────────────────────────────────────────────┐            │
│  │  • Neighbor search (radius=0.15)                          │            │
│  │  • Position embedding (transformer, 32 channels)          │            │
│  │  • Channel MLP: [32, 64, 32]                              │            │
│  │  • Aggregation: mean over neighbors                       │            │
│  └───────────────────────────────────────────────────────────┘            │
│                                                                             │
│  Input:  [B, N_bc, 2]   (boundary values)                                 │
│  Output: [B, α, H, W, 16]  (encoded features on grid)                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CONCATENATE WITH LATENT FEATURES                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  GNO output:        [B, α, H, W, 16]                                       │
│  + Latent features: [B, α, H, W, 4]                                        │
│  ─────────────────────────────────────────                                 │
│  = Combined:        [B, α, H, W, 20]                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                        STAGE 2: LIFTING                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ChannelMLP (2 layers)                                                     │
│  ┌───────────────────────────────────────────────────────────┐            │
│  │  20 → 64 → 64                                              │            │
│  └───────────────────────────────────────────────────────────┘            │
│                                                                             │
│  Input:  [B, α, H, W, 20]                                                  │
│  Output: [B, α, H, W, 64]                                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                        STAGE 3: FNO PROCESSING                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  FNOBlocks (4 layers)                                                      │
│  ┌───────────────────────────────────────────────────────────┐            │
│  │  Layer 1:                                                  │            │
│  │    • FFT to frequency domain                               │            │
│  │    • Spectral convolution (modes: 6×8×8)                   │            │
│  │    • IFFT back to spatial domain                           │            │
│  │    • Skip connection + activation                          │            │
│  │                                                            │            │
│  │  Layers 2-4: Repeat                                        │            │
│  └───────────────────────────────────────────────────────────┘            │
│                                                                             │
│  Input:  [B, 64, α, H, W]  (channels first)                               │
│  Output: [B, 64, α, H, W]                                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                        STAGE 4: PROJECTION                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ChannelMLP (2 layers)                                                     │
│  ┌───────────────────────────────────────────────────────────┐            │
│  │  64 → 128 → 2                                              │            │
│  └───────────────────────────────────────────────────────────┘            │
│                                                                             │
│  Input:  [B, α, H, W, 64]                                                  │
│  Output: [B, α, H, W, 2]   (head, mass_concentration)                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OUTPUT PREDICTIONS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  predictions: [B, α, H, W, 2]                                              │
│  targets:     [B, α, H, W, 2]  (from output_latent_features[..., -2:])    │
│                                                                             │
│  Loss: Relative L2 = ||pred - target||₂ / ||target||₂                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Shape Transformations

```
Shapes throughout the network (B=batch, α=5, H=32, W=32):

Input:
  input_geom:      [B, 200, 3]      Boundary coordinates
  input_data:      [B, 200, 2]      Boundary values
  latent_geom:     [B, 5, 32, 32, 3]   Grid coordinates
  latent_features: [B, 5, 32, 32, 4]   Grid features

GNO Encoding:
  [B, 200, 2] ──GNO──> [B, 5*32*32, 16] ──reshape──> [B, 5, 32, 32, 16]

Concatenation:
  [B, 5, 32, 32, 16] + [B, 5, 32, 32, 4] ──concat──> [B, 5, 32, 32, 20]

Lifting:
  [B, 5, 32, 32, 20] ──MLP──> [B, 5, 32, 32, 64]

FNO (channels first):
  [B, 5, 32, 32, 64] ──permute──> [B, 64, 5, 32, 32]
                     ──FNO───────> [B, 64, 5, 32, 32]
                     ──permute──> [B, 5, 32, 32, 64]

Projection:
  [B, 5, 32, 32, 64] ──MLP──> [B, 5, 32, 32, 2]

Output:
  predictions: [B, 5, 32, 32, 2]
```

## Multi-GPU Data Parallelism

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MULTI-GPU SETUP                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DataLoader produces batch of size B=128                                   │
│  ┌─────────────────────────────────────────────────────────┐              │
│  │ Batch: [128, ...]                                        │              │
│  └─────────────────────────────────────────────────────────┘              │
│                         ↓                                                   │
│  DataParallel splits across 2 GPUs                                         │
│  ┌───────────────────────────┐  ┌────────────────────────────┐            │
│  │ GPU 0: [64, ...]          │  │ GPU 1: [64, ...]           │            │
│  │                           │  │                            │            │
│  │ GFNO forward pass         │  │ GFNO forward pass          │            │
│  │ ↓                         │  │ ↓                          │            │
│  │ predictions: [64, ...]    │  │ predictions: [64, ...]     │            │
│  │ loss: scalar              │  │ loss: scalar               │            │
│  └───────────────────────────┘  └────────────────────────────┘            │
│                         ↓                     ↓                             │
│  Losses averaged and gradients aggregated                                  │
│  ┌─────────────────────────────────────────────────────────┐              │
│  │ Average loss                                             │              │
│  │ Gradients synchronized                                   │              │
│  └─────────────────────────────────────────────────────────┘              │
│                         ↓                                                   │
│  Optimizer step updates parameters (broadcast to all GPUs)                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Training Loop Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            TRAINING LOOP                                    │
└─────────────────────────────────────────────────────────────────────────────┘

For epoch in [0, epochs):
    │
    ├─> For batch in train_loader:
    │       │
    │       ├─> Load batch to device
    │       │   ├─> input_geom: GPU
    │       │   ├─> input_data: GPU
    │       │   ├─> latent_geom: GPU
    │       │   ├─> latent_features: GPU
    │       │   └─> output_latent_features: GPU
    │       │
    │       ├─> Forward pass
    │       │   predictions = model(input_geom, latent_geom, 
    │       │                        input_data, latent_features)
    │       │
    │       ├─> Compute loss
    │       │   targets = output_latent_features[..., -2:]
    │       │   loss = loss_fn(predictions, targets)
    │       │
    │       ├─> Backward pass
    │       │   loss.backward()
    │       │
    │       ├─> Clip gradients
    │       │   torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    │       │
    │       └─> Optimizer step
    │           optimizer.step()
    │
    ├─> Validation
    │   val_loss = evaluate_model(val_loader, model, loss_fn, args)
    │
    ├─> Learning rate scheduling
    │   if (epoch + 1) % lr_scheduler_interval == 0:
    │       scheduler.step()
    │
    ├─> Save checkpoint
    │   if (epoch + 1) % save_checkpoint_every == 0:
    │       save_checkpoint(model, optimizer, scheduler, epoch, ...)
    │
    └─> Plot training curves
        if (epoch + 1) % 5 == 0:
            plot_training_curves(train_losses, val_losses, args)
```

## Checkpoint Structure

```
checkpoint.pth
├─ epoch: int                    Current epoch number
├─ model_state_dict: OrderedDict    Model weights
│  ├─ gno.layer1.weight
│  ├─ gno.layer1.bias
│  ├─ lifting.layer0.weight
│  ├─ fno_blocks.layers.0...
│  └─ projection.layer0.weight
│
├─ optimizer_state_dict: dict    Optimizer state
│  ├─ state: dict                Per-parameter state
│  └─ param_groups: list         Learning rate, etc.
│
├─ scheduler_state_dict: dict    LR scheduler state
│  ├─ last_epoch: int
│  └─ _step_count: int
│
├─ train_losses: list[float]     Training loss history
├─ val_losses: list[float]       Validation loss history
└─ args: Namespace                Training configuration
```

## File Organization

```
GW_SciML/
│
├─ train_gfno_2d_planes.py          Main training script
├─ test_gfno_training_setup.py      Test suite
├─ GFNO_TRAINING_COMPLETE.md        This summary
│
├─ hpc/
│  └─ train_gfno_2d_planes.pbs      HPC job script
│
├─ docs/
│  ├─ GFNO_TRAINING_README.md       Complete documentation
│  ├─ GFNO_IMPLEMENTATION_SUMMARY.md Implementation details
│  ├─ GFNO_QUICK_REFERENCE.md       Quick reference
│  └─ GFNO_ARCHITECTURE_DIAGRAM.md  This file
│
├─ src/
│  ├─ models/
│  │  └─ gfno.py                    GFNO model definition
│  └─ data/
│     ├─ plane_dataset.py           Dataset classes
│     └─ batch_sampler.py           Batch sampling logic
│
└─ results/
   └─ run_YYYYMMDD_HHMMSS/
      ├─ checkpoints/
      │  ├─ checkpoint_epoch_0010.pth
      │  ├─ latest_checkpoint.pth
      │  └─ loss_history.json
      ├─ training_curves.png
      └─ loss_history.json
```

## Legend

- `B`: Batch size
- `N_bc`: Number of boundary condition points (~200)
- `α` (alpha): Temporal window size (typically 5)
- `H`: Grid height (typically 32)
- `W`: Grid width (typically 32)
- `[...]`: Tensor shape notation

---

**Created**: November 3, 2025
