# GINO Training for Variable-Density Groundwater Modeling

Graph-Informed Neural Operator (GINO) training for 3D groundwater modeling with multi-variable support.

## Quick Start

### Run Training Directly (Python)

```bash
# Default training
python train_gino_on_patches_multi_col_refactored.py

# Custom arguments
python train_gino_on_patches_multi_col_refactored.py \
    --epochs 50 \
    --batch-size 256 \
    --learning-rate 1e-3 \
    --target-cols mass_concentration head
```

### Submit PBS Job (HPC)

```bash
# View available configs
./hpc/submit.sh

# Submit with config
./hpc/submit.sh default

# Submit with custom args
./hpc/submit.sh adhoc --epochs 50 --batch-size 256
```

### Monitor Training

```bash
# Check job status
qstat -u $USER

# View logs
tail -f logs/train_gino_*.log

# Check results
ls -lt results/GINO/
```

## Available Configs

Located in `hpc/configs/`:

- **default** - Standard training (mass_concentration + head)
- **high_lr** - Higher learning rate experiment
- **large_batch** - Large batch size experiment  
- **three_vars** - Train on all three variables
- **long_horizon** - Longer prediction windows

## Directory Structure

```
results/GINO/
├── default/
│   └── training_TIMESTAMP/
│       ├── checkpoints/
│       ├── training_curves.png
│       └── loss_history.json
├── high_lr/
└── three_vars/
```


## Key Features

- Modular training system with reusable components
- Config-based experiment organization
- Automatic checkpointing and resume
- Multi-variable prediction support
- DataParallel GPU training

## Common Arguments

```
--epochs              Number of training epochs (default: 40)
--batch-size         Batch size (default: 128)
--learning-rate      Learning rate (default: 5e-4)
--target-cols        Target variables (default: mass_concentration head)
--input-window-size  Input timesteps (default: 5)
--output-window-size Output timesteps (default: 5)
--resume-from        Checkpoint path to resume from
```
