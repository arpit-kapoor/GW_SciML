"""
Benchmark script to compare performance of GWPlaneDataset vs GWPlaneDatasetFromFiles.

This script measures:
1. Initialization time
2. Single sample access time
3. Full dataset iteration time
4. Memory usage
5. Random access patterns

Usage:
    python benchmark_dataset_classes.py --data-dir /path/to/2d_plane_sequences
"""

import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import psutil
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.plane_dataset import GWPlaneDataset, GWPlaneDatasetFromFiles
from src.data.batch_sampler import PatchBatchSampler


def get_memory_usage():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB


def load_data_into_memory(data_dir, max_planes=None):
    """
    Load all data into memory for GWPlaneDataset.
    
    Args:
        data_dir: Path to data directory
        max_planes: Maximum number of planes to load (for testing)
    
    Returns:
        Tuple of (input_sequences, output_sequences) dictionaries
    """
    print("Loading data into memory for GWPlaneDataset...")
    start_time = time.time()
    
    input_sequences = {}
    output_sequences = {}
    
    plane_dirs = sorted([d for d in os.listdir(data_dir) if d.startswith('plane_')])
    if max_planes:
        plane_dirs = plane_dirs[:max_planes]
    
    for plane_dir in plane_dirs:
        plane_path = os.path.join(data_dir, plane_dir)
        plane_id = int(plane_dir.split('_')[1])
        
        # Load all arrays
        input_sequences[plane_id] = {
            'input_geom': np.load(os.path.join(plane_path, 'input_geom.npy')),
            'input_data': np.load(os.path.join(plane_path, 'input_data.npy')),
            'latent_geom': np.load(os.path.join(plane_path, 'latent_geom.npy')),
            'latent_features': np.load(os.path.join(plane_path, 'latent_features.npy')),
        }
        
        output_sequences[plane_id] = {
            'latent_geom': np.load(os.path.join(plane_path, 'output_latent_geom.npy')),
            'latent_features': np.load(os.path.join(plane_path, 'output_latent_features.npy')),
        }
    
    load_time = time.time() - start_time
    print(f"✓ Data loaded in {load_time:.2f}s")
    
    return input_sequences, output_sequences


def benchmark_initialization(data_dir, val_ratio=0.2, max_planes=None):
    """Benchmark initialization time for both classes."""
    print("\n" + "="*80)
    print("BENCHMARK 1: Initialization Time")
    print("="*80)
    
    results = {}
    
    # Benchmark GWPlaneDataset
    print("\n[GWPlaneDataset - In-Memory]")
    mem_before = get_memory_usage()
    start_time = time.time()
    
    input_seqs, output_seqs = load_data_into_memory(data_dir, max_planes)
    dataset1 = GWPlaneDataset(
        input_sequences=input_seqs,
        output_sequences=output_seqs,
        dataset='train',
        val_ratio=val_ratio
    )
    
    init_time1 = time.time() - start_time
    mem_after = get_memory_usage()
    mem_usage1 = mem_after - mem_before
    
    print(f"  Initialization time: {init_time1:.3f}s")
    print(f"  Memory usage: {mem_usage1:.1f} MB")
    print(f"  Dataset size: {len(dataset1)} sequences")
    
    results['in_memory'] = {
        'init_time': init_time1,
        'memory_mb': mem_usage1,
        'dataset_size': len(dataset1),
        'dataset': dataset1
    }
    
    # Benchmark GWPlaneDatasetFromFiles
    print("\n[GWPlaneDatasetFromFiles - On-Demand]")
    mem_before = get_memory_usage()
    start_time = time.time()
    
    dataset2 = GWPlaneDatasetFromFiles(
        data_dir=data_dir,
        dataset='train',
        val_ratio=val_ratio
    )
    
    init_time2 = time.time() - start_time
    mem_after = get_memory_usage()
    mem_usage2 = mem_after - mem_before
    
    print(f"  Initialization time: {init_time2:.3f}s")
    print(f"  Memory usage: {mem_usage2:.1f} MB")
    print(f"  Dataset size: {len(dataset2)} sequences")
    
    results['from_files'] = {
        'init_time': init_time2,
        'memory_mb': mem_usage2,
        'dataset_size': len(dataset2),
        'dataset': dataset2
    }
    
    # Summary
    print(f"\n{'Metric':<25} {'In-Memory':<20} {'From-Files':<20} {'Winner':<15}")
    print("-"*80)
    
    speedup_init = init_time1 / init_time2
    winner_init = "From-Files" if speedup_init > 1 else "In-Memory"
    print(f"{'Initialization Time':<25} {init_time1:>8.3f}s{'':<11} {init_time2:>8.3f}s{'':<11} {winner_init:<15}")
    print(f"{'                     ':<25} {'':<20} ({speedup_init:.1f}x faster)")
    
    mem_ratio = mem_usage1 / max(mem_usage2, 0.1)
    winner_mem = "From-Files" if mem_ratio > 1 else "In-Memory"
    print(f"{'Memory Usage':<25} {mem_usage1:>8.1f} MB{'':<10} {mem_usage2:>8.1f} MB{'':<10} {winner_mem:<15}")
    print(f"{'                     ':<25} {'':<20} ({mem_ratio:.1f}x less)")
    
    return results


def benchmark_single_access(datasets, num_samples=100):
    """Benchmark single sample access time."""
    print("\n" + "="*80)
    print("BENCHMARK 2: Single Sample Access Time")
    print("="*80)
    
    results = {}
    
    for name, data in datasets.items():
        dataset = data['dataset']
        print(f"\n[{name.replace('_', ' ').title()}]")
        
        # Random indices
        indices = np.random.randint(0, len(dataset), size=num_samples)
        
        # Warm-up (important for file caching)
        _ = dataset[0]
        
        # Benchmark
        times = []
        for idx in indices:
            start_time = time.perf_counter()
            sample = dataset[int(idx)]
            times.append(time.perf_counter() - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        print(f"  Samples tested: {num_samples}")
        print(f"  Average time: {avg_time*1000:.3f} ms")
        print(f"  Std dev: {std_time*1000:.3f} ms")
        print(f"  Min time: {min_time*1000:.3f} ms")
        print(f"  Max time: {max_time*1000:.3f} ms")
        
        results[name] = {
            'avg_time_ms': avg_time * 1000,
            'std_time_ms': std_time * 1000,
            'min_time_ms': min_time * 1000,
            'max_time_ms': max_time * 1000
        }
    
    # Summary
    print(f"\n{'Metric':<25} {'In-Memory':<20} {'From-Files':<20} {'Speedup':<15}")
    print("-"*80)
    
    avg1 = results['in_memory']['avg_time_ms']
    avg2 = results['from_files']['avg_time_ms']
    speedup = avg2 / avg1
    
    print(f"{'Average Access Time':<25} {avg1:>8.3f} ms{'':<10} {avg2:>8.3f} ms{'':<10} {speedup:.2f}x")
    print(f"{'Min Access Time':<25} {results['in_memory']['min_time_ms']:>8.3f} ms{'':<10} {results['from_files']['min_time_ms']:>8.3f} ms")
    print(f"{'Max Access Time':<25} {results['in_memory']['max_time_ms']:>8.3f} ms{'':<10} {results['from_files']['max_time_ms']:>8.3f} ms")
    
    return results


def benchmark_iteration(datasets, batch_size=32, num_batches=50):
    """Benchmark full iteration through dataset with DataLoader."""
    print("\n" + "="*80)
    print("BENCHMARK 3: DataLoader Iteration Speed")
    print("="*80)
    print(f"Batch size: {batch_size}, Number of batches: {num_batches}")
    
    results = {}
    
    for name, data in datasets.items():
        dataset = data['dataset']
        print(f"\n[{name.replace('_', ' ').title()}]")
        
        # Create dataloader with batch sampler
        batch_sampler = PatchBatchSampler(dataset, batch_size=batch_size)
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=4,  # Important for fair comparison
            pin_memory=True
        )
        
        # Benchmark iteration
        start_time = time.time()
        batch_times = []
        
        for i, batch in enumerate(dataloader):
            batch_start = time.perf_counter()
            
            # Simulate minimal processing (just access the data)
            _ = batch['input_geom']
            _ = batch['latent_features']
            
            batch_times.append(time.perf_counter() - batch_start)
            
            if i >= num_batches - 1:
                break
        
        total_time = time.time() - start_time
        avg_batch_time = np.mean(batch_times)
        throughput = batch_size / avg_batch_time  # samples per second
        
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Average batch time: {avg_batch_time*1000:.3f} ms")
        print(f"  Throughput: {throughput:.1f} samples/sec")
        print(f"  Batches processed: {len(batch_times)}")
        
        results[name] = {
            'total_time': total_time,
            'avg_batch_time_ms': avg_batch_time * 1000,
            'throughput': throughput,
            'num_batches': len(batch_times)
        }
    
    # Summary
    print(f"\n{'Metric':<25} {'In-Memory':<20} {'From-Files':<20} {'Speedup':<15}")
    print("-"*80)
    
    time1 = results['in_memory']['avg_batch_time_ms']
    time2 = results['from_files']['avg_batch_time_ms']
    speedup = time2 / time1
    
    print(f"{'Avg Batch Time':<25} {time1:>8.3f} ms{'':<10} {time2:>8.3f} ms{'':<10} {speedup:.2f}x")
    
    tput1 = results['in_memory']['throughput']
    tput2 = results['from_files']['throughput']
    tput_ratio = tput1 / tput2
    
    print(f"{'Throughput':<25} {tput1:>8.1f} samp/s{'':<8} {tput2:>8.1f} samp/s{'':<8} {tput_ratio:.2f}x")
    
    return results


def benchmark_random_access_pattern(datasets, num_accesses=500):
    """Benchmark random access patterns (worst case for file-based)."""
    print("\n" + "="*80)
    print("BENCHMARK 4: Random Access Pattern (Worst Case)")
    print("="*80)
    
    results = {}
    
    for name, data in datasets.items():
        dataset = data['dataset']
        print(f"\n[{name.replace('_', ' ').title()}]")
        
        # Create highly random access pattern
        indices = np.random.randint(0, len(dataset), size=num_accesses)
        
        # Warm-up
        _ = dataset[0]
        
        # Benchmark
        start_time = time.time()
        for idx in indices:
            _ = dataset[int(idx)]
        total_time = time.time() - start_time
        
        avg_time = total_time / num_accesses
        throughput = num_accesses / total_time
        
        print(f"  Total accesses: {num_accesses}")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Average time per access: {avg_time*1000:.3f} ms")
        print(f"  Throughput: {throughput:.1f} accesses/sec")
        
        results[name] = {
            'total_time': total_time,
            'avg_time_ms': avg_time * 1000,
            'throughput': throughput
        }
    
    # Summary
    print(f"\n{'Metric':<25} {'In-Memory':<20} {'From-Files':<20} {'Speedup':<15}")
    print("-"*80)
    
    time1 = results['in_memory']['avg_time_ms']
    time2 = results['from_files']['avg_time_ms']
    speedup = time2 / time1
    
    print(f"{'Avg Access Time':<25} {time1:>8.3f} ms{'':<10} {time2:>8.3f} ms{'':<10} {speedup:.2f}x")
    
    tput1 = results['in_memory']['throughput']
    tput2 = results['from_files']['throughput']
    tput_ratio = tput1 / tput2
    
    print(f"{'Throughput':<25} {tput1:>8.1f} acc/s{'':<9} {tput2:>8.1f} acc/s{'':<9} {tput_ratio:.2f}x")
    
    return results


def print_final_summary(all_results):
    """Print comprehensive final summary."""
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    print("\n┌─────────────────────────────┬──────────────┬──────────────┬─────────────┐")
    print("│ Benchmark                   │  In-Memory   │  From-Files  │   Winner    │")
    print("├─────────────────────────────┼──────────────┼──────────────┼─────────────┤")
    
    # Initialization
    init1 = all_results['initialization']['in_memory']['init_time']
    init2 = all_results['initialization']['from_files']['init_time']
    winner = "From-Files" if init2 < init1 else "In-Memory"
    print(f"│ Initialization Time         │ {init1:>9.3f}s   │ {init2:>9.3f}s   │ {winner:<11} │")
    
    # Memory
    mem1 = all_results['initialization']['in_memory']['memory_mb']
    mem2 = all_results['initialization']['from_files']['memory_mb']
    winner = "From-Files" if mem2 < mem1 else "In-Memory"
    print(f"│ Memory Usage                │ {mem1:>9.1f} MB │ {mem2:>9.1f} MB │ {winner:<11} │")
    
    # Single access
    acc1 = all_results['single_access']['in_memory']['avg_time_ms']
    acc2 = all_results['single_access']['from_files']['avg_time_ms']
    winner = "In-Memory" if acc1 < acc2 else "From-Files"
    print(f"│ Single Access Time          │ {acc1:>9.3f} ms │ {acc2:>9.3f} ms │ {winner:<11} │")
    
    # Batch iteration
    batch1 = all_results['iteration']['in_memory']['avg_batch_time_ms']
    batch2 = all_results['iteration']['from_files']['avg_batch_time_ms']
    winner = "In-Memory" if batch1 < batch2 else "From-Files"
    print(f"│ Batch Processing Time       │ {batch1:>9.3f} ms │ {batch2:>9.3f} ms │ {winner:<11} │")
    
    # Random access
    rand1 = all_results['random_access']['in_memory']['avg_time_ms']
    rand2 = all_results['random_access']['from_files']['avg_time_ms']
    winner = "In-Memory" if rand1 < rand2 else "From-Files"
    print(f"│ Random Access Time          │ {rand1:>9.3f} ms │ {rand2:>9.3f} ms │ {winner:<11} │")
    
    print("└─────────────────────────────┴──────────────┴──────────────┴─────────────┘")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    print("\n📊 Use GWPlaneDataset (In-Memory) when:")
    print("  ✓ Dataset fits comfortably in RAM")
    print("  ✓ Need maximum training speed")
    print("  ✓ Can afford longer initialization time")
    print(f"  ✓ Have sufficient memory (needs ~{mem1:.0f} MB for this dataset)")
    
    print("\n💾 Use GWPlaneDatasetFromFiles (On-Demand) when:")
    print("  ✓ Dataset is too large for RAM")
    print("  ✓ Want fast startup time")
    print("  ✓ Have fast storage (SSD/NVMe)")
    print(f"  ✓ Can accept {(acc2/acc1):.1f}x slower access time")
    
    # Calculate effective speedup during training
    training_speedup = batch1 / batch2
    print(f"\n🚀 Training Speedup: In-Memory is ~{training_speedup:.2f}x faster for batch iteration")
    print(f"⚡ Initialization Speedup: From-Files is ~{init1/init2:.2f}x faster to start")
    print(f"💾 Memory Savings: From-Files uses {mem_ratio:.1f}x less memory")


def main():
    parser = argparse.ArgumentParser(description='Benchmark GWPlane dataset classes')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to 2d_plane_sequences directory')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                       help='Validation ratio (default: 0.2)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for iteration benchmark (default: 32)')
    parser.add_argument('--max-planes', type=int, default=None,
                       help='Maximum number of planes to test (for quick testing)')
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of samples for single access benchmark (default: 100)')
    parser.add_argument('--num-batches', type=int, default=50,
                       help='Number of batches for iteration benchmark (default: 50)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("DATASET PERFORMANCE BENCHMARK")
    print("="*80)
    print(f"Data directory: {args.data_dir}")
    print(f"Validation ratio: {args.val_ratio}")
    print(f"Batch size: {args.batch_size}")
    if args.max_planes:
        print(f"Max planes: {args.max_planes}")
    print("="*80)
    
    # Store all results
    all_results = {}
    
    # Benchmark 1: Initialization
    init_results = benchmark_initialization(args.data_dir, args.val_ratio, args.max_planes)
    all_results['initialization'] = init_results
    
    # Benchmark 2: Single access
    access_results = benchmark_single_access(init_results, args.num_samples)
    all_results['single_access'] = access_results
    
    # Benchmark 3: DataLoader iteration
    iter_results = benchmark_iteration(init_results, args.batch_size, args.num_batches)
    all_results['iteration'] = iter_results
    
    # Benchmark 4: Random access pattern
    random_results = benchmark_random_access_pattern(init_results, num_accesses=500)
    all_results['random_access'] = random_results
    
    # Print final summary
    global mem_ratio
    mem_ratio = init_results['in_memory']['memory_mb'] / max(init_results['from_files']['memory_mb'], 0.1)
    print_final_summary(all_results)
    
    print("\n✓ Benchmark complete!")


if __name__ == '__main__':
    main()
