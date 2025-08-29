"""
PPG Signal Pretraining Pipeline

This package provides tools for pretraining transformer input convolution layers
on PPG (photoplethysmogram) signals using self-supervised learning tasks.

Main Components:
- pretraining_dataset.py: Dataset loader for PPG signals with masking
- pretraining_models.py: Models for different pretraining tasks
- pretraining_trainer.py: Training pipeline
- transfer_learning.py: Tools for transferring pretrained weights

Usage:
1. Run pretraining: python pretraining_trainer.py
2. Transfer to your model: Use functions in transfer_learning.py
"""

from .pretraining_models import (
    InputConvLayers,
    MaskedSignalReconstructionModel,
    ContrastivePPGModel,
    DenoisingAutoencoderModel,
    create_pretraining_model
)
from .transfer_learning import (
    load_pretrained_input_convs,
    initialize_transformer_with_pretrained_convs,
    PretrainedConvTransferHelper
)

__version__ = "1.0.0"
__all__ = [
    "PPGPretrainingDataset",
    "create_pretraining_dataloader",
    "InputConvLayers",
    "MaskedSignalReconstructionModel",
    "ContrastivePPGModel", 
    "DenoisingAutoencoderModel",
    "create_pretraining_model",
    "load_pretrained_input_convs",
    "initialize_transformer_with_pretrained_convs",
    "PretrainedConvTransferHelper"
]
