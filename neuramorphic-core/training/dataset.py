"""
Synthetic dataset for neuromorphic training
"""

import torch
from torch.utils.data import Dataset


class SyntheticNeuromorphicDataset(Dataset):
    """Synthetic dataset for neuromorphic testing and training"""
    
    def __init__(self, vocab_size: int, seq_length: int, num_samples: int = 100):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        sequence = torch.randint(0, self.vocab_size, (self.seq_length,))
        
        # add simple patterns for better training
        if idx % 10 == 0:
            pattern = torch.randint(0, 100, (5,))
            sequence[:5] = pattern
            sequence[10:15] = pattern
            
        return {'input_ids': sequence, 'labels': sequence.clone()}
