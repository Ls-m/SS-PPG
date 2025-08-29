import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import os
from datetime import datetime
import json
from tqdm import tqdm

from .pretraining_dataset import create_pretraining_dataloader
from .pretraining_models import create_pretraining_model


class PretrainingLoss:
    """Collection of loss functions for different pretraining tasks"""
    
    @staticmethod
    def masked_reconstruction_loss(
        predicted: torch.Tensor, 
        target: torch.Tensor, 
        mask: torch.Tensor,
        loss_type: str = 'mse'
    ) -> torch.Tensor:
        """
        Loss for masked signal reconstruction
        
        Args:
            predicted: Predicted signal (batch_size, seq_len)
            target: Original signal (batch_size, seq_len)
            mask: Boolean mask (batch_size, seq_len) - True for unmasked positions
            loss_type: Type of loss ('mse', 'mae', 'huber')
        """
        # Only compute loss on masked positions
        masked_positions = ~mask  # Invert mask: True for masked positions
        
        if masked_positions.sum() == 0:
            return torch.tensor(0.0, device=predicted.device, requires_grad=True)
        
        pred_masked = predicted[masked_positions]
        target_masked = target[masked_positions]
        
        if loss_type == 'mse':
            return nn.MSELoss()(pred_masked, target_masked)
        elif loss_type == 'mae':
            return nn.L1Loss()(pred_masked, target_masked)
        elif loss_type == 'huber':
            return nn.SmoothL1Loss()(pred_masked, target_masked)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    @staticmethod
    def contrastive_loss(embeddings: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        """
        SimCLR-style contrastive loss
        Assumes embeddings come in pairs (augmented versions of same signal)
        """
        batch_size = embeddings.shape[0] // 2
        
        # Compute similarity matrix
        sim_matrix = torch.mm(embeddings, embeddings.t()) / temperature
        
        # Create positive pairs mask
        positive_mask = torch.zeros(2 * batch_size, 2 * batch_size, dtype=torch.bool)
        for i in range(batch_size):
            positive_mask[i, i + batch_size] = True
            positive_mask[i + batch_size, i] = True
        
        # Compute loss
        exp_sim = torch.exp(sim_matrix)
        sum_exp_sim = exp_sim.sum(dim=1, keepdim=True)
        log_prob = sim_matrix - torch.log(sum_exp_sim)
        
        pos_log_prob = log_prob[positive_mask]
        loss = -pos_log_prob.mean()
        
        return loss
    
    @staticmethod
    def denoising_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Simple MSE loss for denoising"""
        return nn.MSELoss()(predicted, target)


class PPGPretrainer:
    """Main trainer class for PPG signal pretraining"""
    
    def __init__(
        self,
        model_type: str = 'masked_reconstruction',
        hidden_size: int = 256,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        device: str = 'auto'
    ):
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = create_pretraining_model(
            model_type=model_type,
            hidden_size=hidden_size
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Initialize loss function
        self.loss_fn = PretrainingLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            # Move to device
            original = batch['original'].to(self.device)  # (batch, 1, seq_len)
            masked = batch['masked'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.model_type == 'masked_reconstruction':
                predicted = self.model(masked, mask)  # (batch, seq_len)
                target = original.squeeze(1)  # (batch, seq_len)
                loss = self.loss_fn.masked_reconstruction_loss(predicted, target, mask)
                
            elif self.model_type == 'denoising':
                # Add noise to original signal
                noise_std = 0.05 * torch.std(original)
                noisy = original + torch.randn_like(original) * noise_std
                
                predicted = self.model(noisy)  # (batch, 1, seq_len)
                loss = self.loss_fn.denoising_loss(predicted, original)
                
            elif self.model_type == 'contrastive':
                # For contrastive learning, we need two views of the same data
                # This is simplified - in practice you'd create augmented pairs
                embeddings = self.model(original)
                # Create augmented version by adding small noise
                noise_std = 0.02 * torch.std(original)
                augmented = original + torch.randn_like(original) * noise_std
                embeddings_aug = self.model(augmented)
                
                # Combine embeddings
                combined_embeddings = torch.cat([embeddings, embeddings_aug], dim=0)
                loss = self.loss_fn.contrastive_loss(combined_embeddings)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update parameters
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move to device
                original = batch['original'].to(self.device)
                masked = batch['masked'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                # Forward pass
                if self.model_type == 'masked_reconstruction':
                    predicted = self.model(masked, mask)
                    target = original.squeeze(1)
                    loss = self.loss_fn.masked_reconstruction_loss(predicted, target, mask)
                    
                elif self.model_type == 'denoising':
                    noise_std = 0.05 * torch.std(original)
                    noisy = original + torch.randn_like(original) * noise_std
                    predicted = self.model(noisy)
                    loss = self.loss_fn.denoising_loss(predicted, original)
                    
                elif self.model_type == 'contrastive':
                    embeddings = self.model(original)
                    noise_std = 0.02 * torch.std(original)
                    augmented = original + torch.randn_like(original) * noise_std
                    embeddings_aug = self.model(augmented)
                    combined_embeddings = torch.cat([embeddings, embeddings_aug], dim=0)
                    loss = self.loss_fn.contrastive_loss(combined_embeddings)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        save_dir: str = './checkpoints',
        save_interval: int = 10
    ):
        """Main training loop"""
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            
            print(f"Train Loss: {train_loss:.6f}")
            
            # Validate
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)
                print(f"Val Loss: {val_loss:.6f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(os.path.join(save_dir, 'best_model.pth'))
                    print(f"New best validation loss: {best_val_loss:.6f}")
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
                self.save_model(checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")
            
            # Update learning rate (simple decay)
            if (epoch + 1) % 30 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.5
                print(f"Learning rate reduced to: {param_group['lr']}")
            
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
        
        # Save final model
        self.save_model(os.path.join(save_dir, 'final_model.pth'))
        
        # Save training history
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print("Training completed!")
    
    def save_model(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_type': self.model_type,
            'hidden_size': self.hidden_size,
            'history': self.history
        }, path)
    
    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        print(f"Model loaded from {path}")
    
    def extract_input_conv_layers(self) -> nn.ModuleList:
        """Extract the pretrained input convolution layers"""
        return self.model.input_convs.input_convs  # Return the conv layers only
    
    def save_input_conv_layers(self, path: str):
        """Save only the input convolution layers for transfer learning"""
        conv_state_dict = {}
        for i, conv_layer in enumerate(self.model.input_convs.input_convs):
            conv_state_dict[f'input_conv_{i}'] = conv_layer.state_dict()
        
        # Also save batch norm layers
        for i, bn_layer in enumerate(self.model.input_convs.batch_norms):
            conv_state_dict[f'batch_norm_{i}'] = bn_layer.state_dict()
        
        torch.save({
            'conv_layers': conv_state_dict,
            'hidden_size': self.hidden_size,
            'model_type': self.model_type
        }, path)
        
        print(f"Input convolution layers saved to {path}")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(self.history['train_loss'], label='Train Loss', alpha=0.8)
        if self.history['val_loss']:
            axes[0].plot(self.history['val_loss'], label='Val Loss', alpha=0.8)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Learning rate plot
        axes[1].plot(self.history['learning_rate'], alpha=0.8)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Example usage
    data_dir = "/Users/eli/VscodeProjects/SS-PPG/data/bidmc_data/bidmc_csv"
    
    # Create data loaders
    train_loader = create_pretraining_dataloader(
        data_dir=data_dir,
        batch_size=16,
        sequence_length=1000,
        mask_ratio=0.15,
        mask_strategy='continuous'
    )
    
    # Initialize trainer
    trainer = PPGPretrainer(
        model_type='masked_reconstruction',
        hidden_size=256,
        learning_rate=1e-4
    )
    
    print("Starting pretraining...")
    
    # Train (you can add validation loader if you split your data)
    trainer.train(
        train_loader=train_loader,
        epochs=50,
        save_dir='./pretraining_checkpoints'
    )
