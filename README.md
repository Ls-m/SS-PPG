# PPG Signal Pretraining Pipeline

This project implements a self-supervised pretraining pipeline for PPG (photoplethysmogram) signal processing, specifically designed to pretrain the input convolution layers of transformer models before fine-tuning on downstream tasks like respiratory signal estimation.

## Overview

The pipeline includes several pretraining tasks:

1. **Masked Signal Reconstruction** (Recommended): Masks portions of the PPG signal and trains the model to reconstruct the original signal
2. **Denoising Autoencoder**: Adds noise to the signal and trains the model to recover the clean version
3. **Contrastive Learning**: Learns representations by contrasting different augmented views of the same signal

## Project Structure

```
src/
├── pretraining_dataset.py    # Dataset loader with masking strategies
├── pretraining_models.py     # Model architectures for pretraining
├── pretraining_trainer.py    # Training pipeline and loss functions
├── transfer_learning.py      # Tools for transferring pretrained weights
└── __init__.py              # Package initialization
```

## Data Format

The system expects BIDMC dataset CSV files in the following format:
- Files: `bidmc_XX_Signals.csv` where XX is the subject number
- Columns: `Time [s], RESP, PLETH, V, AVR, II`
- PPG signal is extracted from the `PLETH` column

## Quick Start

### 1. Pretraining

```python
from src.pretraining_trainer import PPGPretrainer
from src.pretraining_dataset import create_pretraining_dataloader

# Create data loader
data_dir = "data/bidmc_data/bidmc_csv"
train_loader = create_pretraining_dataloader(
    data_dir=data_dir,
    batch_size=16,
    sequence_length=1000,
    mask_ratio=0.15,
    mask_strategy='continuous'  # or 'random', 'structured'
)

# Initialize trainer
trainer = PPGPretrainer(
    model_type='masked_reconstruction',  # or 'denoising', 'contrastive'
    hidden_size=256,
    learning_rate=1e-4
)

# Train
trainer.train(
    train_loader=train_loader,
    epochs=50,
    save_dir='./pretraining_checkpoints'
)

# Save only the input conv layers for transfer
trainer.save_input_conv_layers('./pretrained_input_convs.pth')
```

### 2. Transfer Learning

```python
from src.transfer_learning import PretrainedConvTransferHelper
import torch.nn as nn

# Load pretrained convolution layers
helper = PretrainedConvTransferHelper('./pretrained_input_convs.pth')

# Create your transformer model with pretrained convs
class YourTransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Use pretrained input convolutions
        self.input_convs = helper.create_compatible_input_convs()
        
        # Your transformer layers here...
        # ...

# Optionally freeze pretrained layers during initial fine-tuning
model = YourTransformerModel()
for conv in model.input_convs:
    for param in conv.parameters():
        param.requires_grad = False  # Freeze pretrained weights
```

## Pretraining Tasks

### Masked Signal Reconstruction (Recommended)
- **Task**: Predict masked portions of PPG signal
- **Benefits**: Learns temporal dependencies and signal patterns
- **Masking strategies**: 
  - `random`: Random time points
  - `continuous`: Continuous segments 
  - `structured`: Regular patterns

### Denoising Autoencoder
- **Task**: Remove noise from corrupted PPG signals
- **Benefits**: Learns robust feature representations
- **Noise types**: Gaussian noise, scaling, time shifts

### Contrastive Learning
- **Task**: Learn representations by contrasting augmented views
- **Benefits**: Learns invariant features
- **Augmentations**: Noise, scaling, time shifts

## Model Architecture

The input convolution layers use multi-scale convolutions:
```python
self.input_convs = nn.ModuleList([
    nn.Conv1d(1, hidden_size // 4, kernel_size=k, padding=k//2)
    for k in [3, 7, 15, 31]  # Multi-scale kernels
])
```

This captures features at different temporal scales:
- 3: Fine-grained features (high-frequency)
- 7: Medium-scale patterns 
- 15: Broader temporal patterns
- 31: Long-range dependencies

## Training Tips

1. **Sequence Length**: Use 1000-2000 samples (8-16 seconds at 125Hz)
2. **Mask Ratio**: Start with 15% for masked reconstruction
3. **Learning Rate**: Use 1e-4 with AdamW optimizer
4. **Batch Size**: 16-32 works well depending on GPU memory
5. **Data Augmentation**: Light augmentation helps generalization

## Hyperparameters

```python
# Recommended settings for masked reconstruction
{
    'sequence_length': 1000,
    'mask_ratio': 0.15,
    'mask_strategy': 'continuous',
    'hidden_size': 256,
    'learning_rate': 1e-4,
    'batch_size': 16,
    'epochs': 50-100
}
```

## Expected Results

After pretraining, you should see:
1. Convergent loss curves
2. Meaningful signal reconstructions
3. Improved performance when transferred to downstream tasks
4. Better feature representations in the early layers

## Integration with Your Main Project

1. Pretrain the input convolution layers using this pipeline
2. Save the pretrained weights
3. Initialize your transformer model with pretrained convs
4. Fine-tune on your specific respiratory estimation task
5. Optionally unfreeze pretrained layers for end-to-end training

## Requirements

- PyTorch >= 1.9
- NumPy
- Pandas  
- Scikit-learn
- Matplotlib
- tqdm

## Example Training Command

```bash
cd src
python pretraining_trainer.py
```

This will start pretraining with default parameters and save checkpoints to `./pretraining_checkpoints/`.

## Next Steps

1. Run pretraining on your BIDMC data
2. Transfer the pretrained weights to your respiratory estimation model
3. Compare performance with and without pretraining
4. Experiment with different pretraining tasks and hyperparameters

The pretrained input convolution layers should provide better initial feature representations for your PPG-to-respiratory signal estimation task.
