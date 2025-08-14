from torch.utils.data import Sampler
from collections import defaultdict
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

class PatchBatchSampler(Sampler):
    """Highly optimized batch sampler that groups samples by patch_id.
    
    Supports shuffling examples within batches and shuffling the order of patches.
    Can be reshuffled between epochs without recreating the sampler.
    
    Optimizations:
    - Cached patch groups to avoid dataset access
    - Pre-allocated batch structure
    - Lazy evaluation of shuffling
    - NumPy arrays for faster operations
    """
    
    def __init__(self, dataset, batch_size, shuffle_within_batches=True, shuffle_patches=True, seed=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle_within_batches = shuffle_within_batches
        self.shuffle_patches = shuffle_patches
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # Track epoch for logging (initialize before calling _reshuffle)
        self._epoch_count = 0
        
        # Build and cache patch groups once
        self._patch_groups, self._patch_ids = self._build_patch_groups_optimized()
        
        # Pre-allocate batch structure for faster reshuffling
        self._prebuild_batch_structure()
        
        # Initialize with first shuffle
        self._current_batches = None
        self._needs_reshuffle = True
    
    def _build_patch_groups_optimized(self):
        """Build groups of indices for each patch_id using vectorized approach."""
        print("Building patch groups (one-time operation)...")
        
        # Fast extraction using list comprehension and bulk operations
        dataset_size = len(self.dataset)
        
        # Check if dataset has a fast patch_id access method
        if hasattr(self.dataset, 'get_all_patch_ids'):
            # Use optimized dataset method if available
            patch_ids = self.dataset.get_all_patch_ids()
        else:
            # Fallback with parallel processing for large datasets
            if dataset_size > 1000:
                print(f"  Using parallel processing for large dataset ({dataset_size} samples)")
                patch_ids = self._extract_patch_ids_parallel(dataset_size)
            else:
                # Standard method with progress indicator for smaller datasets
                patch_ids = []
                print_every = max(1, dataset_size // 10)  # Print progress every 10%
                
                for idx in range(dataset_size):
                    if idx % print_every == 0:
                        print(f"  Progress: {idx}/{dataset_size} ({100*idx//dataset_size}%)")
                    patch_ids.append(self.dataset[idx]['patch_id'])
                
                patch_ids = np.array(patch_ids, dtype=np.int32)
        
        # Vectorized grouping using numpy
        unique_patches, inverse_indices, counts = np.unique(patch_ids, return_inverse=True, return_counts=True)
        
        # Build groups efficiently
        patch_groups = {}
        indices_array = np.arange(dataset_size)
        
        for i, patch_id in enumerate(unique_patches):
            mask = inverse_indices == i
            patch_groups[int(patch_id)] = indices_array[mask].tolist()
            
        print(f"Found {len(unique_patches)} patches with {dataset_size} total samples")
        print(f"Patch sizes: min={counts.min()}, max={counts.max()}, avg={counts.mean():.1f}")
        return patch_groups, patch_ids
    
    def _extract_patch_ids_parallel(self, dataset_size):
        """Extract patch_ids using parallel processing for large datasets."""
        def extract_chunk(start_idx, end_idx):
            return [self.dataset[idx]['patch_id'] for idx in range(start_idx, end_idx)]
        
        # Determine number of workers
        num_workers = min(mp.cpu_count(), 4)  # Cap at 4 to avoid overhead
        chunk_size = dataset_size // num_workers
        
        print(f"  Using {num_workers} workers with chunk size {chunk_size}")
        
        # Create chunks
        chunks = []
        for i in range(num_workers):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < num_workers - 1 else dataset_size
            chunks.append((start_idx, end_idx))
        
        # Process chunks in parallel
        patch_ids = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(extract_chunk, start, end) for start, end in chunks]
            
            for i, future in enumerate(futures):
                chunk_result = future.result()
                patch_ids.extend(chunk_result)
                print(f"  Completed chunk {i+1}/{num_workers}")
        
        return np.array(patch_ids, dtype=np.int32)
    
    def _prebuild_batch_structure(self):
        """Pre-build the batch structure to avoid reconstruction each epoch."""
        self._base_batches = []
        self._batch_patch_map = []  # Track which patch each batch belongs to
        
        for patch_id, patch_indices in self._patch_groups.items():
            # Split patch indices into batches of batch_size
            for i in range(0, len(patch_indices), self.batch_size):
                batch_indices = patch_indices[i:i + self.batch_size]
                self._base_batches.append(np.array(batch_indices, dtype=np.int32))
                self._batch_patch_map.append(patch_id)
        
        self._num_batches = len(self._base_batches)
        print(f"Pre-built {self._num_batches} batches")
    
    def _reshuffle(self):
        """Efficient reshuffling using pre-built structure."""
        if not hasattr(self, '_epoch_count'):
            self._epoch_count = 0
        
        self._epoch_count += 1
        
        # Use pre-built batches as base
        if self.shuffle_patches or self.shuffle_within_batches:
            # Create batch order indices
            batch_order = np.arange(self._num_batches)
            
            # Shuffle batch order if needed
            if self.shuffle_patches:
                np.random.shuffle(batch_order)
            
            # Prepare final batches
            self._current_batches = []
            for batch_idx in batch_order:
                batch = self._base_batches[batch_idx].copy()  # Fast numpy copy
                
                # Shuffle within batch if needed
                if self.shuffle_within_batches:
                    np.random.shuffle(batch)
                
                self._current_batches.append(batch.tolist())
        else:
            # No shuffling needed, use base batches directly
            self._current_batches = [batch.tolist() for batch in self._base_batches]
        
        self._needs_reshuffle = False
        
        # Log reshuffling if enabled (only for non-initial epochs to avoid spam)
        if (self.shuffle_patches or self.shuffle_within_batches) and self._epoch_count > 1:
            print(f"PatchBatchSampler reshuffled for epoch {self._epoch_count} "
                  f"(patches: {self.shuffle_patches}, within: {self.shuffle_within_batches})")
    
    def reshuffle(self):
        """Manually trigger reshuffling. Useful for custom epoch boundaries."""
        self._needs_reshuffle = True
    
    def get_epoch_count(self):
        """Get the current epoch count."""
        return getattr(self, '_epoch_count', 0)
    
    def __iter__(self):
        """Iterate over batches. Lazy reshuffling for optimal performance."""
        # Only reshuffle if needed (lazy evaluation)
        if self._needs_reshuffle or self._current_batches is None:
            self._reshuffle()
        
        for batch in self._current_batches:
            yield batch
        
        # Mark for reshuffling on next iteration
        self._needs_reshuffle = True
    
    def __len__(self):
        """Return number of batches."""
        return self._num_batches
