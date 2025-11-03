# Train/Validation Split Implementation

## Overview
This document describes the train/validation split functionality added to the GFNO training pipeline. The implementation allows for proper model validation during training by splitting each plane's sequences into training and validation sets.

## Implementation Details

### Dataset Parameter: `val_ratio`
- **Default value**: `0.2` (20% validation, 80% training)
- **Range**: `0.0` to `1.0`
- **Effect**: Controls the proportion of sequences allocated to validation set

### Splitting Strategy: Per-Plane Temporal Split
The split is performed **per plane** using a temporal approach:
- **Training set**: First `(1 - val_ratio)` sequences of each plane
- **Validation set**: Last `val_ratio` sequences of each plane

**Example** with `val_ratio=0.2` and 100 sequences per plane:
```
Plane 1: sequences 0-79   → Training
         sequences 80-99  → Validation
         
Plane 2: sequences 0-79   → Training
         sequences 80-99  → Validation
         
... and so on for all planes
```

### Why Temporal Split?
1. **Realistic evaluation**: Tests model's ability to predict future timesteps
2. **Avoids data leakage**: Validation data is truly unseen during training
3. **Consistent with physical simulations**: Respects temporal ordering

## Modified Files

### 1. `src/data/plane_dataset.py`
Both dataset classes updated:

#### `GWPlaneDataset`
```python
def __init__(self, 
             data,
             dataset='train',  # NEW: 'train' or 'val'
             val_ratio=0.2,    # NEW: validation split ratio
             # ... other params
            ):
```

#### `GWPlaneDatasetFromFiles`
```python
def __init__(self, 
             data_dir, 
             dataset='train',  # NEW: 'train' or 'val'
             val_ratio=0.2,    # NEW: validation split ratio
             # ... other params
            ):
```

**Split Implementation**:
```python
# Calculate train/val split
n_train = int(len(sequences) * (1 - val_ratio))

if dataset == 'train':
    sequences = sequences[:n_train]
elif dataset == 'val':
    sequences = sequences[n_train:]
else:
    raise ValueError(f"Invalid dataset '{dataset}'. Must be 'train' or 'val'")
```

### 2. `train_gfno_2d_planes.py`

#### New Command-Line Argument
```python
parser.add_argument('--val-ratio', type=float, default=0.2,
                   help='Validation set ratio (default: 0.2)')
```

#### Updated `create_datasets()` Function
```python
def create_datasets(data_dir, val_ratio=0.2):
    """Create training and validation datasets."""
    # Training dataset
    train_ds = GWPlaneDatasetFromFiles(
        data_dir=data_dir,
        dataset='train',      # NEW
        val_ratio=val_ratio,  # NEW
        fill_nan_value=-999.0
    )
    
    # Validation dataset
    val_ds = GWPlaneDatasetFromFiles(
        data_dir=data_dir,
        dataset='val',        # NEW
        val_ratio=val_ratio,  # NEW
        fill_nan_value=-999.0
    )
    
    return train_ds, val_ds
```

#### Training Configuration Display
Now shows validation ratio:
```python
print(f"Validation Ratio: {args.val_ratio:.1%}")
print(f"Training sequences: {len(train_dataset)}")
print(f"Validation sequences: {len(val_dataset)}")
```

### 3. `hpc/train_gfno_2d_planes.pbs`

#### New Variable
```bash
VAL_RATIO="${VAL_RATIO:-0.2}"
```

#### Updated Training Command
```bash
python train_gfno_2d_planes.py \
    --data-dir $DATA_DIR \
    # ... other args ...
    --val-ratio $VAL_RATIO
```

#### Hyperparameters Display
```bash
echo "Validation ratio : $VAL_RATIO"
```

### 4. `test_gfno_training_setup.py`

#### Updated Test Function
```python
def test_data_loading(data_dir, batch_size=4, val_ratio=0.2):
    """Test dataset and dataloader."""
    # Create training dataset
    train_dataset = GWPlaneDatasetFromFiles(
        data_dir=data_dir,
        dataset='train',
        val_ratio=val_ratio,
        fill_nan_value=-999.0
    )
    
    # Create validation dataset
    val_dataset = GWPlaneDatasetFromFiles(
        data_dir=data_dir,
        dataset='val',
        val_ratio=val_ratio,
        fill_nan_value=-999.0
    )
    
    print(f"✓ Train/Val split: {len(train_dataset)}/{len(val_dataset)} "
          f"({1-val_ratio:.1%}/{val_ratio:.1%})")
```

## Usage Examples

### 1. Default (80/20 split)
```bash
# Training script
python train_gfno_2d_planes.py \
    --data-dir /path/to/2d_plane_sequences \
    --epochs 100

# PBS job
qsub hpc/train_gfno_2d_planes.pbs
```

### 2. Custom Split (90/10)
```bash
# Training script
python train_gfno_2d_planes.py \
    --data-dir /path/to/2d_plane_sequences \
    --val-ratio 0.1 \
    --epochs 100

# PBS job - set environment variable
export VAL_RATIO=0.1
qsub hpc/train_gfno_2d_planes.pbs
```

### 3. Testing
```bash
python test_gfno_training_setup.py \
    --data-dir /path/to/2d_plane_sequences
```

## Expected Output

### Training Start
```
================================================================================
GFNO Training Configuration
================================================================================
Data Directory: /path/to/2d_plane_sequences
Training sequences: 1600
Validation sequences: 400
Validation Ratio: 20.0%
Learning Rate: 0.001
Batch Size: 128
...
```

### During Training
```
Epoch [1/100]
Train Loss: 0.1234
Val Loss: 0.1456
Time: 45.2s
...
```

## Validation During Training

The validation loop runs after each epoch:
```python
def validate_epoch(model, val_loader, loss_fn, device):
    """Validate model on validation set."""
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            # ... validation logic ...
            val_loss += loss.item()
    
    return val_loss / len(val_loader)
```

## Benefits

1. **Proper Generalization Assessment**: Monitor if model overfits
2. **Early Stopping**: Can stop training when validation loss stops improving
3. **Hyperparameter Tuning**: Compare different configurations objectively
4. **Model Selection**: Choose best checkpoint based on validation performance

## Best Practices

### Recommended Split Ratios
- **Small datasets** (<1000 sequences): `val_ratio=0.1` (10%)
- **Medium datasets** (1000-5000): `val_ratio=0.2` (20%)
- **Large datasets** (>5000): `val_ratio=0.2-0.3` (20-30%)

### When to Use Different Ratios
- **More validation data** (`val_ratio > 0.2`):
  - When you need more reliable validation metrics
  - For hyperparameter tuning experiments
  
- **Less validation data** (`val_ratio < 0.2`):
  - When training data is limited
  - When validation is just for monitoring (not tuning)

### What to Monitor
1. **Training vs Validation Loss Gap**:
   - Small gap: Good generalization
   - Large gap: Overfitting (reduce model capacity or add regularization)

2. **Validation Loss Trend**:
   - Decreasing: Model is learning
   - Plateauing: Consider stopping or reducing LR
   - Increasing: Overfitting (implement early stopping)

## Verification Checklist

- [ ] Training and validation datasets have different sizes
- [ ] Training sequences + validation sequences = total sequences
- [ ] Split ratio matches expected (e.g., 80/20)
- [ ] Validation loss is computed each epoch
- [ ] No overlap between training and validation data
- [ ] All planes contribute to both sets (temporal split per plane)

## Testing the Implementation

Run the test suite to verify correct splitting:
```bash
python test_gfno_training_setup.py \
    --data-dir /path/to/2d_plane_sequences
```

Expected output:
```
================================================================================
TEST 1: Data Loading
================================================================================
✓ Training dataset loaded: 1600 sequences
✓ Validation dataset loaded: 400 sequences
✓ Total sequences: 2000
✓ Train/Val split: 1600/400 (80.0%/20.0%)
```

## Troubleshooting

### Issue: Validation set is empty
**Cause**: `val_ratio` too small or sequences per plane < 5
**Solution**: 
- Increase `val_ratio` (e.g., to 0.2)
- Check that each plane has sufficient sequences

### Issue: Train and val losses are identical
**Cause**: Both datasets might be using same data (bug in dataset code)
**Solution**: 
- Verify `dataset='train'` vs `dataset='val'` parameters
- Check split logic in `plane_dataset.py`
- Add debug prints to confirm different sequences

### Issue: Validation loss much higher than training
**Cause**: This is normal! Validation uses unseen future timesteps
**Solution**: 
- This indicates model has not seen val data (good!)
- Monitor the gap - if too large, might be overfitting

## Future Enhancements

Potential improvements for consideration:
1. **Stratified splitting**: Ensure balanced representation of different planes
2. **K-fold cross-validation**: For more robust evaluation
3. **Separate test set**: Additional holdout for final evaluation
4. **Random splitting**: Option for non-temporal splits
5. **Early stopping**: Automatic training termination based on validation loss

## References

- Main training script: `train_gfno_2d_planes.py`
- Dataset implementation: `src/data/plane_dataset.py`
- PBS script: `hpc/train_gfno_2d_planes.pbs`
- Test suite: `test_gfno_training_setup.py`
- GFNO documentation: `docs/GFNO_TRAINING_README.md`
