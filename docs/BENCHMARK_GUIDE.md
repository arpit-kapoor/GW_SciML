# Dataset Benchmark Guide

## Overview
The `benchmark_dataset_classes.py` script comprehensively compares the performance of `GWPlaneDataset` (in-memory) vs `GWPlaneDatasetFromFiles` (on-demand loading).

## Quick Start

### Basic Usage
```bash
python benchmark_dataset_classes.py --data-dir /Users/arpitkapoor/data/GW/2d_plane_sequences
```

### Quick Test (Few Planes)
```bash
python benchmark_dataset_classes.py \
    --data-dir /Users/arpitkapoor/data/GW/2d_plane_sequences \
    --max-planes 5 \
    --num-samples 50 \
    --num-batches 20
```

### Full Benchmark
```bash
python benchmark_dataset_classes.py \
    --data-dir /Users/arpitkapoor/data/GW/2d_plane_sequences \
    --batch-size 64 \
    --num-samples 200 \
    --num-batches 100
```

## What Gets Measured

### 1. Initialization Time
- **In-Memory**: Time to load all data into RAM
- **From-Files**: Time to scan directories and build index
- **Winner**: Usually From-Files (much faster)

### 2. Memory Usage
- **In-Memory**: Total RAM used by loaded data
- **From-Files**: RAM for index only (minimal)
- **Winner**: Always From-Files

### 3. Single Sample Access
- **Test**: Random access to 100 samples
- **Measures**: Average, min, max, std deviation
- **Winner**: Usually In-Memory (no disk I/O)

### 4. DataLoader Iteration
- **Test**: Batch iteration with PatchBatchSampler
- **Measures**: Batch time, throughput (samples/sec)
- **Winner**: Usually In-Memory (faster access)

### 5. Random Access Pattern
- **Test**: 500 completely random accesses (worst case)
- **Measures**: Throughput and average time
- **Winner**: Usually In-Memory (cache misses for files)

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | Required | Path to 2d_plane_sequences directory |
| `--val-ratio` | 0.2 | Validation set ratio |
| `--batch-size` | 32 | Batch size for iteration test |
| `--max-planes` | None | Limit planes (for quick testing) |
| `--num-samples` | 100 | Samples for single access test |
| `--num-batches` | 50 | Batches for iteration test |

## Example Output

```
================================================================================
BENCHMARK 1: Initialization Time
================================================================================

[GWPlaneDataset - In-Memory]
  Initialization time: 15.234s
  Memory usage: 2456.3 MB
  Dataset size: 1600 sequences

[GWPlaneDatasetFromFiles - On-Demand]
  Initialization time: 0.342s
  Memory usage: 12.5 MB
  Dataset size: 1600 sequences

Metric                    In-Memory            From-Files           Winner         
--------------------------------------------------------------------------------
Initialization Time           15.234s                 0.342s          From-Files     
                                                   (44.5x faster)
Memory Usage                2456.3 MB                12.5 MB          From-Files     
                                                   (196.5x less)

================================================================================
BENCHMARK 2: Single Sample Access Time
================================================================================

[In Memory]
  Samples tested: 100
  Average time: 0.125 ms
  Std dev: 0.032 ms
  Min time: 0.098 ms
  Max time: 0.234 ms

[From Files]
  Samples tested: 100
  Average time: 2.456 ms
  Std dev: 0.543 ms
  Min time: 1.234 ms
  Max time: 5.678 ms

Metric                    In-Memory            From-Files           Speedup        
--------------------------------------------------------------------------------
Average Access Time           0.125 ms                2.456 ms          19.65x
Min Access Time               0.098 ms                1.234 ms
Max Access Time               0.234 ms                5.678 ms

...

================================================================================
FINAL SUMMARY
================================================================================

┌─────────────────────────────┬──────────────┬──────────────┬─────────────┐
│ Benchmark                   │  In-Memory   │  From-Files  │   Winner    │
├─────────────────────────────┼──────────────┼──────────────┼─────────────┤
│ Initialization Time         │    15.234s   │     0.342s   │ From-Files  │
│ Memory Usage                │   2456.3 MB  │      12.5 MB │ From-Files  │
│ Single Access Time          │     0.125 ms │     2.456 ms │ In-Memory   │
│ Batch Processing Time       │     3.456 ms │     8.234 ms │ In-Memory   │
│ Random Access Time          │     0.134 ms │     2.678 ms │ In-Memory   │
└─────────────────────────────┴──────────────┴──────────────┴─────────────┘

================================================================================
RECOMMENDATIONS
================================================================================

📊 Use GWPlaneDataset (In-Memory) when:
  ✓ Dataset fits comfortably in RAM
  ✓ Need maximum training speed
  ✓ Can afford longer initialization time
  ✓ Have sufficient memory (needs ~2456 MB for this dataset)

💾 Use GWPlaneDatasetFromFiles (On-Demand) when:
  ✓ Dataset is too large for RAM
  ✓ Want fast startup time
  ✓ Have fast storage (SSD/NVMe)
  ✓ Can accept 2.4x slower access time

🚀 Training Speedup: In-Memory is ~2.38x faster for batch iteration
⚡ Initialization Speedup: From-Files is ~44.5x faster to start
💾 Memory Savings: From-Files uses 196.5x less memory
```

## Interpreting Results

### When In-Memory Wins
- **Single access**: No disk I/O overhead
- **Batch iteration**: Sequential memory access is fast
- **Random access**: RAM is always faster than disk

### When From-Files Wins
- **Initialization**: Only scans directories, doesn't load data
- **Memory usage**: Only stores index in RAM

### Real-World Considerations

1. **Storage Speed**: SSDs/NVMe reduce From-Files disadvantage
2. **OS Caching**: File system caches frequently accessed files
3. **Training Workflow**: Batch iteration is most important metric
4. **Multi-GPU**: DataLoader workers can parallelize file loading

## Tips for Best Performance

### For In-Memory Dataset
- Ensure sufficient RAM (dataset size + model + gradients)
- Use when dataset is small-medium (<10GB)
- Pre-load during setup phase

### For From-Files Dataset
- Use fast storage (NVMe SSD recommended)
- Consider DataLoader num_workers > 0 for parallel loading
- Enable OS file caching
- Monitor disk I/O with `iostat` or `iotop`

## Troubleshooting

### Benchmark crashes with OOM
```bash
# Use fewer planes for testing
python benchmark_dataset_classes.py \
    --data-dir /path/to/data \
    --max-planes 3
```

### Want faster benchmark
```bash
# Reduce test iterations
python benchmark_dataset_classes.py \
    --data-dir /path/to/data \
    --num-samples 20 \
    --num-batches 10
```

### Import errors
```bash
# Ensure you're in the project root
cd /Users/arpitkapoor/Library/CloudStorage/OneDrive-UNSW/Shared/Projects/10_Katana/04_groundwater/GW_SciML
python benchmark_dataset_classes.py --data-dir /Users/arpitkapoor/data/GW/2d_plane_sequences
```
