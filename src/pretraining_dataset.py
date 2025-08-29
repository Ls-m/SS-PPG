import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import random
from typing import Tuple, List, Optional


class PPGPretrainingDataset(Dataset):
    """
    Dataset for PPG signal pretraining using masked signal reconstruction.
    
    This dataset loads PPG signals from BIDMC dataset and applies masking
    for self-supervised learning tasks.
    """
    
    def __init__(
        self, 
        data_dir: str, 
        sequence_length: int = 1000,
        mask_ratio: float = 0.15,
        mask_strategy: str = 'random',  # 'random', 'continuous', 'structured'
        overlap: float = 0.5,
        normalize: bool = True,
        augment: bool = True
    ):
        """
        Args:
            data_dir: Directory containing bidmc_*_Signals.csv files
            sequence_length: Length of each sequence for training
            mask_ratio: Ratio of signal to mask
            mask_strategy: How to mask the signal
            overlap: Overlap between consecutive sequences
            normalize: Whether to normalize the signals
            augment: Whether to apply data augmentation
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.mask_ratio = mask_ratio
        self.mask_strategy = mask_strategy
        self.overlap = overlap
        self.normalize = normalize
        self.augment = augment
        
        # Load all signal files
        self.data = self._load_data()
        
        # Create sequence indices
        self.sequences = self._create_sequences()
        
        print(f"Loaded {len(self.sequences)} sequences for pretraining")
    
    def _load_data(self) -> List[np.ndarray]:
        """Load PPG signals from all CSV files"""
        data = []
        signal_files = [f for f in os.listdir(self.data_dir) if f.endswith('_Signals.csv')]
        signal_files.sort()
        
        scalers = []
        
        for file in signal_files:
            file_path = os.path.join(self.data_dir, file)
            df = pd.read_csv(file_path)
            
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            
            # Extract PPG signal (PLETH column)
            ppg_signal = df['PLETH'].values
            
            # Remove any NaN values
            ppg_signal = ppg_signal[~np.isnan(ppg_signal)]
            
            if self.normalize:
                scaler = StandardScaler()
                ppg_signal = scaler.fit_transform(ppg_signal.reshape(-1, 1)).flatten()
                scalers.append(scaler)
            
            data.append(ppg_signal)
            
        self.scalers = scalers if self.normalize else None
        return data
    
    def _create_sequences(self) -> List[Tuple[int, int]]:
        """Create overlapping sequences from the data"""
        sequences = []
        step_size = int(self.sequence_length * (1 - self.overlap))
        
        for file_idx, signal in enumerate(self.data):
            for start_idx in range(0, len(signal) - self.sequence_length + 1, step_size):
                sequences.append((file_idx, start_idx))
        
        return sequences
    
    def _apply_augmentation(self, signal: np.ndarray) -> np.ndarray:
        """Apply data augmentation to the signal"""
        if not self.augment:
            return signal
        
        # Random noise
        if random.random() < 0.3:
            noise_std = 0.01 * np.std(signal)
            signal = signal + np.random.normal(0, noise_std, signal.shape)
        
        # Random scaling
        if random.random() < 0.3:
            scale_factor = np.random.uniform(0.8, 1.2)
            signal = signal * scale_factor
        
        # Random time shift (small)
        if random.random() < 0.2:
            shift = random.randint(-5, 5)
            if shift != 0:
                signal = np.roll(signal, shift)
        
        return signal
    
    def _create_mask(self, length: int) -> np.ndarray:
        """Create mask for the signal based on the chosen strategy"""
        mask = np.ones(length, dtype=bool)
        num_mask = int(length * self.mask_ratio)
        
        if self.mask_strategy == 'random':
            mask_indices = np.random.choice(length, num_mask, replace=False)
            mask[mask_indices] = False
            
        elif self.mask_strategy == 'continuous':
            # Create continuous masked segments
            num_segments = random.randint(1, 4)
            segment_length = num_mask // num_segments
            
            for _ in range(num_segments):
                start_idx = random.randint(0, length - segment_length)
                mask[start_idx:start_idx + segment_length] = False
                
        elif self.mask_strategy == 'structured':
            # Mask structured patterns (every k-th sample)
            k = int(1 / self.mask_ratio)
            mask[::k] = False
        
        return mask
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx) -> dict:
        file_idx, start_idx = self.sequences[idx]
        
        # Extract sequence
        sequence = self.data[file_idx][start_idx:start_idx + self.sequence_length].copy()
        
        # Apply augmentation
        sequence = self._apply_augmentation(sequence)
        
        # Create mask
        mask = self._create_mask(len(sequence))
        
        # Create masked sequence
        masked_sequence = sequence.copy()
        masked_sequence[~mask] = 0  # Mask with zeros
        
        return {
            'original': torch.FloatTensor(sequence).unsqueeze(0),  # Add channel dimension
            'masked': torch.FloatTensor(masked_sequence).unsqueeze(0),
            'mask': torch.BoolTensor(mask),
            'file_idx': file_idx,
            'start_idx': start_idx
        }


def create_pretraining_dataloader(
    data_dir: str,
    batch_size: int = 32,
    sequence_length: int = 1000,
    mask_ratio: float = 0.15,
    mask_strategy: str = 'random',
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """Create DataLoader for pretraining"""
    
    dataset = PPGPretrainingDataset(
        data_dir=data_dir,
        sequence_length=sequence_length,
        mask_ratio=mask_ratio,
        mask_strategy=mask_strategy,
        **kwargs
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )


if __name__ == "__main__":
    # Test the dataset
    data_dir = "/Users/eli/VscodeProjects/SS-PPG/data/bidmc_data/bidmc_csv"
    
    dataset = PPGPretrainingDataset(
        data_dir=data_dir,
        sequence_length=1000,
        mask_ratio=0.15,
        mask_strategy='continuous'
    )
    
    # Test loading
    sample = dataset[0]
    print("Sample shapes:")
    print(f"Original: {sample['original'].shape}")
    print(f"Masked: {sample['masked'].shape}")
    print(f"Mask: {sample['mask'].shape}")
    print(f"Mask ratio: {(~sample['mask']).sum().item() / len(sample['mask']):.3f}")
