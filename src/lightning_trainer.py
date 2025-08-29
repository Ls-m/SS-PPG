import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Any
import os
from datetime import datetime
import json

from .lightning_dataset import PPGLightningDataModule
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
        """Loss for masked signal reconstruction"""
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
    def denoising_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Loss for denoising autoencoder"""
        return nn.MSELoss()(predicted, target)
    
    @staticmethod
    def contrastive_loss(
        z1: torch.Tensor, 
        z2: torch.Tensor, 
        temperature: float = 0.1
    ) -> torch.Tensor:
        """InfoNCE loss for contrastive learning"""
        batch_size = z1.size(0)
        device = z1.device
        
        # Normalize representations
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(z1, z2.T) / temperature
        
        # Labels for positive pairs (diagonal)
        labels = torch.arange(batch_size, device=device)
        
        # InfoNCE loss
        loss = nn.CrossEntropyLoss()(sim_matrix, labels)
        return loss


class PPGLightningModule(pl.LightningModule):
    """PyTorch Lightning module for PPG pretraining"""
    
    def __init__(
        self,
        model_type: str = 'masked_reconstruction',
        hidden_size: int = 256,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        loss_type: str = 'mse',
        temperature: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize model
        self.model = create_pretraining_model(
            model_type=model_type,
            hidden_size=hidden_size,
            **kwargs
        )
        
        # Loss function
        self.loss_fn = PretrainingLoss()
        
        # Metrics storage
        self.training_step_outputs = []
        self.validation_step_outputs = []
    
    def forward(self, x, **kwargs):
        """Forward pass"""
        return self.model(x, **kwargs)
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        loss, logs = self._compute_loss(batch, 'train')
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        for key, value in logs.items():
            self.log(f'train_{key}', value, on_step=False, on_epoch=True)
        
        self.training_step_outputs.append(loss.detach())
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        loss, logs = self._compute_loss(batch, 'val')
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        for key, value in logs.items():
            self.log(f'val_{key}', value, on_step=False, on_epoch=True)
        
        self.validation_step_outputs.append(loss.detach())
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step"""
        loss, logs = self._compute_loss(batch, 'test')
        
        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        for key, value in logs.items():
            self.log(f'test_{key}', value, on_step=False, on_epoch=True)
        
        return loss
    
    def _compute_loss(self, batch, stage):
        """Compute loss based on model type"""
        logs = {}
        
        if self.hparams.model_type == 'masked_reconstruction':
            signals, masked_signals, masks = batch
            predictions = self.model(masked_signals)
            
            # Handle tensor dimensions - remove channel dimension if present
            if signals.dim() == 3 and signals.size(1) == 1:
                signals = signals.squeeze(1)  # (batch, seq_len)
            if masks.dim() == 3 and masks.size(1) == 1:
                masks = masks.squeeze(1)  # (batch, seq_len)
            
            loss = self.loss_fn.masked_reconstruction_loss(
                predictions, signals, masks, self.hparams.loss_type
            )
            
            # Compute reconstruction accuracy on masked positions
            with torch.no_grad():
                masked_positions = ~masks
                if masked_positions.sum() > 0:
                    pred_masked = predictions[masked_positions]
                    target_masked = signals[masked_positions]
                    mse = nn.MSELoss()(pred_masked, target_masked)
                    mae = nn.L1Loss()(pred_masked, target_masked)
                    logs.update({'mse': mse, 'mae': mae})
        
        elif self.hparams.model_type == 'denoising':
            # For denoising, we get the same batch structure as masked reconstruction
            # but we use different elements
            if len(batch) == 3:
                # This is coming from masked reconstruction dataset by mistake
                # Let's use the first two elements as noisy and clean
                noisy_signals, clean_signals = batch[0], batch[1]
            else:
                noisy_signals, clean_signals = batch
            
            # Handle tensor dimensions
            if clean_signals.dim() == 3 and clean_signals.size(1) == 1:
                clean_signals = clean_signals.squeeze(1)
            
            predictions = self.model(noisy_signals)
            if predictions.dim() == 3 and predictions.size(1) == 1:
                predictions = predictions.squeeze(1)
                
            loss = self.loss_fn.denoising_loss(predictions, clean_signals)
            
            # Compute denoising metrics
            with torch.no_grad():
                mse = nn.MSELoss()(predictions, clean_signals)
                mae = nn.L1Loss()(predictions, clean_signals)
                logs.update({'mse': mse, 'mae': mae})
        
        elif self.hparams.model_type == 'contrastive':
            # For contrastive, we might get 3 elements but need only 2
            if len(batch) == 3:
                aug1, aug2 = batch[0], batch[1]
            else:
                aug1, aug2 = batch
                
            z1 = self.model(aug1)
            z2 = self.model(aug2)
            loss = self.loss_fn.contrastive_loss(z1, z2, self.hparams.temperature)
            
            # Compute representation similarity
            with torch.no_grad():
                z1_norm = nn.functional.normalize(z1, dim=1)
                z2_norm = nn.functional.normalize(z2, dim=1)
                similarity = torch.mean(torch.sum(z1_norm * z2_norm, dim=1))
                logs.update({'similarity': similarity})
        
        else:
            raise ValueError(f"Unknown model type: {self.hparams.model_type}")
        
        return loss, logs
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=0.5,
            patience=10
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1,
            }
        }
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch"""
        if self.training_step_outputs:
            avg_loss = torch.stack(self.training_step_outputs).mean()
            self.log('train_epoch_avg_loss', avg_loss)
            self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch"""
        if self.validation_step_outputs:
            avg_loss = torch.stack(self.validation_step_outputs).mean()
            self.log('val_epoch_avg_loss', avg_loss)
            self.validation_step_outputs.clear()
    
    def save_input_conv_layers(self, save_path: str):
        """Save only the input convolution layers for transfer learning"""
        if hasattr(self.model, 'input_convs'):
            input_convs_state = self.model.input_convs.state_dict()
        else:
            raise AttributeError(f"Model {type(self.model)} does not have 'input_convs' attribute")
        
        save_dict = {
            'input_convs': input_convs_state,
            'model_type': self.hparams.model_type,
            'hidden_size': self.hparams.hidden_size,
            'save_timestamp': datetime.now().isoformat()
        }
        
        torch.save(save_dict, save_path)
        print(f"Input convolution layers saved to {save_path}")


class PPGLightningTrainer:
    """High-level trainer class for PPG pretraining with Lightning"""
    
    def __init__(
        self,
        model_type: str = 'masked_reconstruction',
        hidden_size: int = 256,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        **model_kwargs
    ):
        self.model_type = model_type
        self.model_kwargs = {
            'model_type': model_type,
            'hidden_size': hidden_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            **model_kwargs
        }
    
    def train(
        self,
        data_dir: str,
        epochs: int = 50,
        batch_size: int = 16,
        sequence_length: int = 1000,
        mask_ratio: float = 0.15,
        mask_strategy: str = 'continuous',
        overlap: float = 0.5,
        num_workers: int = 4,
        save_dir: str = 'lightning_checkpoints',
        save_interval: int = 10,
        train_subjects: Optional[List[str]] = None,
        val_subjects: Optional[List[str]] = None,
        test_subjects: Optional[List[str]] = None,
        **trainer_kwargs
    ):
        """Train the model using PyTorch Lightning"""
        
        # Create data module with subject-wise splits
        data_module = PPGLightningDataModule(
            data_dir=data_dir,
            batch_size=batch_size,
            sequence_length=sequence_length,
            mask_ratio=mask_ratio,
            mask_strategy=mask_strategy,
            overlap=overlap,
            num_workers=num_workers,
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            test_subjects=test_subjects
        )
        
        # Create Lightning module
        lightning_module = PPGLightningModule(**self.model_kwargs)
        
        # Setup callbacks
        callbacks = [
            pl.callbacks.ModelCheckpoint(
                dirpath=save_dir,
                filename='checkpoint-{epoch:02d}',
                every_n_epochs=save_interval,
                save_top_k=-1,  # Save all checkpoints
                save_last=True
            ),
            pl.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                mode='min'
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
        ]
        
        # Setup logger
        logger = pl.loggers.TensorBoardLogger(
            save_dir=save_dir,
            name='tensorboard_logs',
            version=f"{self.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=epochs,
            callbacks=callbacks,
            logger=logger,
            accelerator='auto',
            devices='auto',
            **trainer_kwargs
        )
        
        # Train the model
        trainer.fit(lightning_module, data_module)
        
        # Test the model
        if test_subjects is not None:
            trainer.test(lightning_module, data_module)
        
        # Save input conv layers for transfer learning
        conv_save_path = os.path.join(save_dir, 'pretrained_input_convs.pth')
        lightning_module.save_input_conv_layers(conv_save_path)
        
        return trainer, lightning_module