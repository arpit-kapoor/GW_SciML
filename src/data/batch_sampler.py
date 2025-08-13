from torch.utils.data import Sampler
from collections import defaultdict

class PatchBatchSampler(Sampler):
    """Batch sampler that groups samples by patch_id for efficient processing."""
    
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self._build_patch_groups()
    
    def _build_patch_groups(self):
        """Build groups of indices for each patch_id."""
        self.patch_groups = defaultdict(list)
        
        for idx in range(len(self.dataset)):
            patch_id = self.dataset[idx]['patch_id']
            self.patch_groups[patch_id].append(idx)
        
        # Convert to list of batches by splitting patch groups into batch_size chunks
        self.batches = []
        for patch_indices in self.patch_groups.values():
            # Split patch indices into batches of batch_size
            for i in range(0, len(patch_indices), self.batch_size):
                batch = patch_indices[i:i + self.batch_size]
                self.batches.append(batch)
    
    def __iter__(self):
        """Iterate over batches."""
        for batch in self.batches:
            yield batch
    
    def __len__(self):
        """Return number of batches."""
        return len(self.batches)
