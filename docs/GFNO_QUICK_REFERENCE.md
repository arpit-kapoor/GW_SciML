# GFNO Training Quick Reference

Quick reference for common GFNO training operations.

## Quick Start

### 1. Test Setup (Recommended First Step)
```bash
python test_gfno_training_setup.py
```

### 2. Local Training (CPU/Single GPU)
```bash
python train_gfno_2d_planes.py \
    --data-dir /Users/arpitkapoor/data/GW/2d_plane_sequences \
    --results-dir ./results \
    --batch-size 32 \
    --epochs 50
```

### 3. HPC Training (Multi-GPU)
```bash
# Submit job
qsub hpc/train_gfno_2d_planes.pbs

# Check status
qstat -u $USER

# Monitor output
tail -f train_gfno_2d_planes.o*

# Cancel job
qdel <job_id>
```

## Common Commands

### Check Data
```bash
# Verify data directory structure
ls -lh /path/to/2d_plane_sequences/

# Check plane directories
ls /path/to/2d_plane_sequences/ | head

# Check files in a plane
ls -lh /path/to/2d_plane_sequences/plane_00/
```

### Monitor Training
```bash
# Real-time log monitoring
tail -f train_gfno_2d_planes.o*

# GPU utilization
nvidia-smi

# Watch GPU continuously
watch -n 1 nvidia-smi

# Check disk usage
du -sh /path/to/results/
```

### Checkpoint Management
```bash
# List checkpoints
ls -lht /path/to/results/run_*/checkpoints/

# Check checkpoint contents
python -c "import torch; ckpt = torch.load('checkpoint.pth'); print(ckpt.keys())"

# Get checkpoint epoch
python -c "import torch; print(f\"Epoch: {torch.load('checkpoint.pth')['epoch']}\")"

# Copy checkpoint for backup
cp /path/to/checkpoint.pth /path/to/backup/checkpoint_backup_$(date +%Y%m%d).pth
```

### Resume Training
```bash
# Resume from latest checkpoint
python train_gfno_2d_planes.py \
    --resume-from /path/to/results/run_20251103_103000/checkpoints/latest_checkpoint.pth \
    --epochs 200

# Resume from specific epoch
python train_gfno_2d_planes.py \
    --resume-from /path/to/results/run_20251103_103000/checkpoints/checkpoint_epoch_0050.pth \
    --epochs 200
```

## Configuration Presets

### Quick Test (Fast)
```bash
python train_gfno_2d_planes.py \
    --batch-size 16 \
    --epochs 10 \
    --learning-rate 5e-3 \
    --save-checkpoint-every 5
```

### Standard Training (Balanced)
```bash
python train_gfno_2d_planes.py \
    --batch-size 64 \
    --epochs 100 \
    --learning-rate 1e-3 \
    --lr-gamma 0.95 \
    --save-checkpoint-every 10
```

### Long Training (Thorough)
```bash
python train_gfno_2d_planes.py \
    --batch-size 128 \
    --epochs 500 \
    --learning-rate 5e-4 \
    --lr-gamma 0.98 \
    --save-checkpoint-every 20 \
    --use-multi-gpu
```

### Small Memory (GPU < 8GB)
```bash
python train_gfno_2d_planes.py \
    --batch-size 16 \
    --epochs 100 \
    --learning-rate 1e-3
```

### Large Memory (GPU > 16GB)
```bash
python train_gfno_2d_planes.py \
    --batch-size 256 \
    --epochs 100 \
    --learning-rate 2e-3 \
    --use-multi-gpu
```

## PBS Script Modifications

### Change GPU Count
```bash
# Edit line in train_gfno_2d_planes.pbs:
#PBS -l select=1:ncpus=12:mem=64GB:ngpus=4  # Change from 2 to 4

# Adjust batch size proportionally
BATCH_SIZE=256  # Was 64 per GPU, now 4 GPUs
```

### Change Walltime
```bash
# Edit line in train_gfno_2d_planes.pbs:
#PBS -l walltime=48:00:00  # Change from 24 to 48 hours
```

### Change Memory
```bash
# Edit line in train_gfno_2d_planes.pbs:
#PBS -l select=1:ncpus=12:mem=128GB:ngpus=2  # Change from 64GB to 128GB
```

## Analyze Results

### View Training Curves
```bash
# Open PNG file
open /path/to/results/run_*/training_curves.png

# On HPC (copy to local)
scp username@katana:/path/to/results/run_*/training_curves.png ./
```

### Extract Loss Values
```bash
# View loss history
python -c "import json; data = json.load(open('loss_history.json')); print(f'Final loss: {data[\"train_losses\"][-1]:.6f}')"

# Plot custom curves
python << EOF
import json
import matplotlib.pyplot as plt

data = json.load(open('loss_history.json'))
plt.plot(data['train_losses'], label='Train')
plt.plot(data['val_losses'], label='Val')
plt.legend()
plt.savefig('custom_curves.png')
EOF
```

### Find Best Checkpoint
```bash
# Find checkpoint with lowest validation loss
python << EOF
import json
import os

data = json.load(open('loss_history.json'))
val_losses = data['val_losses']
best_epoch = val_losses.index(min(val_losses))
print(f"Best epoch: {best_epoch}")
print(f"Best val loss: {val_losses[best_epoch]:.6f}")
print(f"Checkpoint: checkpoint_epoch_{best_epoch:04d}.pth")
EOF
```

## Troubleshooting

### OOM Error
```bash
# Reduce batch size
python train_gfno_2d_planes.py --batch-size 32  # or 16

# Check GPU memory
nvidia-smi

# Clear GPU cache (in Python)
python -c "import torch; torch.cuda.empty_cache()"
```

### Slow Training
```bash
# Check GPU utilization (should be >80%)
nvidia-smi

# Increase batch size if GPU not fully utilized
python train_gfno_2d_planes.py --batch-size 128

# Use more GPUs
python train_gfno_2d_planes.py --batch-size 256 --use-multi-gpu
```

### NaN Loss
```bash
# Reduce learning rate
python train_gfno_2d_planes.py --learning-rate 5e-4

# Increase gradient clipping
python train_gfno_2d_planes.py --grad-clip-norm 0.5

# Check data for NaN values
python << EOF
import numpy as np
data = np.load('plane_00/input_data.npy')
print(f"NaN count: {np.isnan(data).sum()}")
EOF
```

### Checkpoint Load Error
```bash
# Check checkpoint file
python -c "import torch; ckpt = torch.load('checkpoint.pth'); print(ckpt.keys())"

# Verify model architecture matches
python test_gfno_training_setup.py

# Try loading on CPU
python -c "import torch; ckpt = torch.load('checkpoint.pth', map_location='cpu'); print('Success')"
```

## Performance Optimization

### Profile Training
```bash
# Add to Python script:
# import torch.profiler as profiler
# with profiler.profile(...) as prof:
#     train_one_epoch()
# prof.export_chrome_trace("trace.json")
```

### Monitor Performance
```bash
# Log GPU stats to file
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used \
    --format=csv -l 1 > gpu_stats.csv &

# Plot GPU utilization
python << EOF
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('gpu_stats.csv')
df['utilization.gpu [%]'].plot()
plt.savefig('gpu_utilization.png')
EOF
```

## File Locations

### Important Files
```
GW_SciML/
├── train_gfno_2d_planes.py              # Main training script
├── test_gfno_training_setup.py          # Testing script
├── hpc/train_gfno_2d_planes.pbs         # HPC job script
├── docs/GFNO_TRAINING_README.md         # Full documentation
├── docs/GFNO_IMPLEMENTATION_SUMMARY.md  # Implementation details
└── docs/GFNO_QUICK_REFERENCE.md         # This file

Results Structure:
results/
└── run_YYYYMMDD_HHMMSS/
    ├── checkpoints/
    │   ├── checkpoint_epoch_0010.pth
    │   ├── latest_checkpoint.pth
    │   ├── final_checkpoint.pth
    │   └── loss_history.json
    ├── training_curves.png
    └── loss_history.json
```

## Contact & Support

- **Author**: Arpit Kapoor (z5370003@unsw.edu.au)
- **Documentation**: See `docs/GFNO_TRAINING_README.md`
- **Testing**: Run `test_gfno_training_setup.py`
- **Issues**: Check troubleshooting section above

## Useful One-Liners

```bash
# Count total sequences
find /path/to/2d_plane_sequences -name "*.npy" -path "*/input_geom.npy" | wc -l

# Get data size
du -sh /path/to/2d_plane_sequences

# Get results size
du -sh /path/to/results

# Count checkpoints
ls /path/to/results/*/checkpoints/*.pth | wc -l

# Latest checkpoint
ls -t /path/to/results/*/checkpoints/*.pth | head -1

# Training time estimation
python -c "epochs=100; time_per_epoch=5; print(f'Est. time: {epochs*time_per_epoch/60:.1f} hours')"

# GPU memory requirement
python -c "batch=64; mem_per_sample=9; print(f'Est. memory: {batch*mem_per_sample/1024:.1f} GB')"
```

---

**Last Updated**: November 3, 2025
