import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler
import random
from typing import Tuple, List, Optional, Dict
import re
from scipy import signal

class PPGSubjectDataset(Dataset):
    """
    Dataset for PPG signal pretraining with subject-wise data management.
    
    This dataset loads PPG signals from BIDMC dataset and applies masking
    for self-supervised learning tasks, with proper subject-wise splitting.
    """
    
    def __init__(
        self, 
        data_dir: str,
        subjects: List[str],
        sequence_length: int = 1000,
        mask_ratio: float = 0.15,
        mask_strategy: str = 'random',  # 'random', 'continuous', 'structured'
        overlap: float = 0.5,
        normalize: bool = True,
        augment: bool = True,
        model_type: str = 'masked_reconstruction'
    ):
        """
        Args:
            data_dir: Directory containing bidmc_*_Signals.csv files
            subjects: List of subject IDs to include (e.g., ['01', '02', '03'])
            sequence_length: Length of each sequence for training
            mask_ratio: Ratio of signal to mask
            mask_strategy: How to mask the signal
            overlap: Overlap between consecutive sequences
            normalize: Whether to normalize the signals
            augment: Whether to apply data augmentation
            model_type: Type of pretraining task
        """
        self.data_dir = data_dir
        self.subjects = subjects
        self.sequence_length = sequence_length
        self.mask_ratio = mask_ratio
        self.mask_strategy = mask_strategy
        self.overlap = overlap
        self.normalize = normalize
        self.augment = augment
        self.model_type = model_type
        
        # Load data for specified subjects
        self.data = self._load_subject_data()
        
        # Create sequence indices
        self.sequences = self._create_sequences()
        
        print(f"Loaded {len(self.sequences)} sequences from {len(self.subjects)} subjects")
    def downsample_signal(self, signal_data: np.ndarray, 
                         original_rate: int, target_rate: int) -> np.ndarray:
        """Downsample the signal to target sampling rate."""
        if original_rate == target_rate:
            return signal_data
            
        downsample_factor = original_rate // target_rate
        downsampled = signal.decimate(signal_data, downsample_factor, ftype='fir')
        
        return downsampled
    def _load_subject_data(self) -> List[Dict]:
        """Load PPG signals from specified subject files"""
        data = []
        
        for subject_id in self.subjects:
            # Look for signal file for this subject
            signal_file = f"bidmc_{subject_id:02d}_Signals.csv" if isinstance(subject_id, int) else f"bidmc_{subject_id}_Signals.csv"
            file_path = os.path.join(self.data_dir, signal_file)
            
            if not os.path.exists(file_path):
                print(f"Warning: Signal file not found for subject {subject_id}: {file_path}")
                continue
            
            df = pd.read_csv(file_path)
            
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            
            # Extract PPG signal (PLETH column)
            if 'PLETH' not in df.columns:
                print(f"Warning: PLETH column not found in {signal_file}")
                continue
            
            ppg_signal = df['PLETH'].values
            
            # Remove any NaN values
            ppg_signal = ppg_signal[~np.isnan(ppg_signal)]
            
            if len(ppg_signal) < self.sequence_length:
                print(f"Warning: Signal too short for subject {subject_id}: {len(ppg_signal)} < {self.sequence_length}")
                continue
            # ppg_signal = self.downsample_signal(ppg_signal, 125, 25)

            # Normalize per subject
            if self.normalize:
                scaler = StandardScaler()
                ppg_signal = scaler.fit_transform(ppg_signal.reshape(-1, 1)).flatten()
            
            data.append({
                'subject_id': subject_id,
                'signal': ppg_signal,
                'file_path': file_path
            })
        
        return data
    
    def _create_sequences(self) -> List[Tuple[int, int]]:
        """Create overlapping sequences from the subject data"""
        sequences = []
        step_size = int(self.sequence_length * (1 - self.overlap))
        
        for subject_idx, subject_data in enumerate(self.data):
            signal = subject_data['signal']
            for start_idx in range(0, len(signal) - self.sequence_length + 1, step_size):
                sequences.append((subject_idx, start_idx))
        
        return sequences
    
    def _apply_augmentation(self, signal: np.ndarray) -> np.ndarray:
        """Apply data augmentation to the signal"""
        if not self.augment:
            return signal
        
        # Random scaling (amplitude variation)
        if random.random() < 0.3:
            scale_factor = random.uniform(0.8, 1.2)
            signal = signal * scale_factor
        
        # Add small amount of noise
        if random.random() < 0.3:
            noise_std = 0.01 * np.std(signal)
            noise = np.random.normal(0, noise_std, signal.shape)
            signal = signal + noise
        
        # Time shifting (circular shift)
        if random.random() < 0.3:
            shift = random.randint(-int(0.1 * len(signal)), int(0.1 * len(signal)))
            signal = np.roll(signal, shift)
        
        return signal
    
    def _create_mask(self, sequence_length: int) -> np.ndarray:
        """Create a mask for the signal based on the masking strategy"""
        mask = np.ones(sequence_length, dtype=bool)  # True = keep, False = mask
        n_mask = int(sequence_length * self.mask_ratio)
        
        if self.mask_strategy == 'random':
            mask_indices = np.random.choice(sequence_length, n_mask, replace=False)
            mask[mask_indices] = False
        
        elif self.mask_strategy == 'continuous':
            # Create continuous masked regions
            n_regions = random.randint(1, 3)
            region_size = n_mask // n_regions
            
            for _ in range(n_regions):
                start_idx = random.randint(0, sequence_length - region_size)
                end_idx = min(start_idx + region_size, sequence_length)
                mask[start_idx:end_idx] = False
        
        elif self.mask_strategy == 'structured':
            # Mask every nth sample
            step = int(1 / self.mask_ratio)
            mask[::step] = False
        
        return mask
    
    def _add_noise(self, signal: np.ndarray, noise_level: float = 0.4) -> np.ndarray:
        """Add noise to signal for denoising task"""
        noise = np.random.normal(0, noise_level * np.std(signal), signal.shape)
        return signal + noise
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """Get a training sample based on model type"""
        subject_idx, start_idx = self.sequences[idx]
        subject_data = self.data[subject_idx]
        signal = subject_data['signal']
        
        # Extract sequence
        sequence = signal[start_idx:start_idx + self.sequence_length].copy()
        
        # Apply augmentation
        sequence = self._apply_augmentation(sequence)
        
        # Convert to tensor
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)  # Add channel dimension
        
        # Return data based on model type
        if self.model_type == 'masked_reconstruction':
            # Create mask
            mask = self._create_mask(self.sequence_length)
            mask_tensor = torch.BoolTensor(mask).unsqueeze(0)
            
            # Create masked sequence
            masked_sequence = sequence_tensor.clone()
            masked_sequence[:, ~mask] = 0  # Zero out masked positions
            
            return sequence_tensor, masked_sequence, mask_tensor
        
        elif self.model_type == 'denoising':
            # Create noisy version
            noisy_sequence = self._add_noise(sequence)
            noisy_tensor = torch.FloatTensor(noisy_sequence).unsqueeze(0)
            
            return noisy_tensor, sequence_tensor
        
        elif self.model_type == 'contrastive':
            # Create two augmented versions
            aug1 = self._apply_augmentation(sequence)
            aug2 = self._apply_augmentation(sequence)
            
            aug1_tensor = torch.FloatTensor(aug1).unsqueeze(0)
            aug2_tensor = torch.FloatTensor(aug2).unsqueeze(0)
            
            return aug1_tensor, aug2_tensor
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")


class PPGLightningDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for PPG pretraining with subject-wise splits"""
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 16,
        sequence_length: int = 1000,
        mask_ratio: float = 0.15,
        mask_strategy: str = 'continuous',
        overlap: float = 0.5,
        num_workers: int = 4,
        train_subjects: Optional[List[str]] = None,
        val_subjects: Optional[List[str]] = None,
        test_subjects: Optional[List[str]] = None,
        model_type: str = 'masked_reconstruction',
        normalize: bool = True,
        augment: bool = True,
    ):
        super().__init__()
        # Don't save hyperparameters to avoid conflicts with Lightning module
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.mask_ratio = mask_ratio
        self.mask_strategy = mask_strategy
        self.overlap = overlap
        self.num_workers = num_workers
        self.model_type = model_type
        self.normalize = normalize
        self.augment = augment
        
        # Determine subject splits
        self.all_subjects = self._get_available_subjects()
        self.train_subjects = train_subjects or self._default_train_split()
        self.val_subjects = val_subjects or self._default_val_split()
        self.test_subjects = test_subjects or self._default_test_split()
        
        print(f"Subject splits:")
        print(f"  Train: {self.train_subjects}")
        print(f"  Val: {self.val_subjects}")
        print(f"  Test: {self.test_subjects}")
    
    def _get_available_subjects(self) -> List[str]:
        """Get list of available subjects from data directory"""
        subjects = []
        
        # Look for signal files
        for filename in os.listdir(self.data_dir):
            if filename.endswith('_Signals.csv'):
                # Extract subject ID from filename
                match = re.match(r'bidmc_(\d+)_Signals\.csv', filename)
                if match:
                    subjects.append(match.group(1))
        
        subjects.sort()
        return subjects
    
    def _default_train_split(self) -> List[str]:
        """Default training split (first 60% of subjects)"""
        n_subjects = len(self.all_subjects)
        n_train = int(0.7 * n_subjects)
        return self.all_subjects[:n_train]
    
    def _default_val_split(self) -> List[str]:
        """Default validation split (next 20% of subjects)"""
        n_subjects = len(self.all_subjects)
        n_train = int(0.7 * n_subjects)
        n_val = int(0.2 * n_subjects)
        return self.all_subjects[n_train:n_train + n_val]
    
    def _default_test_split(self) -> List[str]:
        """Default test split (last 20% of subjects)"""
        n_subjects = len(self.all_subjects)
        n_train = int(0.7 * n_subjects)
        n_val = int(0.1 * n_subjects)
        return self.all_subjects[n_train + n_val:]
    
    def setup(self, stage: str = None):
        """Setup datasets for each stage"""
        
        if stage == 'fit' or stage is None:
            # Training dataset
            self.train_dataset = PPGSubjectDataset(
                data_dir=self.data_dir,
                subjects=self.train_subjects,
                sequence_length=self.sequence_length,
                mask_ratio=self.mask_ratio,
                mask_strategy=self.mask_strategy,
                overlap=self.overlap,
                normalize=self.normalize,
                augment=self.augment,
                model_type=self.model_type
            )
            
            # Validation dataset
            self.val_dataset = PPGSubjectDataset(
                data_dir=self.data_dir,
                subjects=self.val_subjects,
                sequence_length=self.sequence_length,
                mask_ratio=self.mask_ratio,
                mask_strategy=self.mask_strategy,
                overlap=self.overlap,
                normalize=self.normalize,
                augment=False,  # No augmentation for validation
                model_type=self.model_type
            )
        
        if stage == 'test' or stage is None:
            # Test dataset
            self.test_dataset = PPGSubjectDataset(
                data_dir=self.data_dir,
                subjects=self.test_subjects,
                sequence_length=self.sequence_length,
                mask_ratio=self.mask_ratio,
                mask_strategy=self.mask_strategy,
                overlap=self.overlap,
                normalize=self.normalize,
                augment=False,  # No augmentation for testing
                model_type=self.model_type
            )
    
    def train_dataloader(self):
        """Return training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        """Return validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        """Return test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )