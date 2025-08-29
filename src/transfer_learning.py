import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import os


def load_pretrained_input_convs(
    checkpoint_path: str, 
    target_hidden_size: int = 256
) -> Dict[str, Any]:
    """
    Load pretrained input convolution layers from checkpoint.
    
    Args:
        checkpoint_path: Path to the saved checkpoint
        target_hidden_size: Expected hidden size for compatibility check
        
    Returns:
        Dictionary containing the state dict and metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'conv_layers' in checkpoint:
        # This is a saved input conv layers file
        conv_layers = checkpoint['conv_layers']
        hidden_size = checkpoint['hidden_size']
    else:
        # This is a full model checkpoint - extract input conv layers
        model_state = checkpoint['model_state_dict']
        hidden_size = checkpoint.get('hidden_size', 256)
        
        conv_layers = {}
        for key, value in model_state.items():
            if key.startswith('input_convs.'):
                # Remove the 'input_convs.' prefix
                new_key = key[12:]  # len('input_convs.') = 12
                conv_layers[new_key] = value
    
    # Verify compatibility
    if hidden_size != target_hidden_size:
        print(f"Warning: Pretrained model hidden size ({hidden_size}) != target ({target_hidden_size})")
    
    return {
        'state_dict': conv_layers,
        'hidden_size': hidden_size,
        'model_type': checkpoint.get('model_type', 'unknown')
    }


def initialize_transformer_with_pretrained_convs(
    transformer_model: nn.Module,
    pretrained_conv_path: str,
    freeze_convs: bool = False
) -> nn.Module:
    """
    Initialize a transformer model with pretrained input convolution layers.
    
    Args:
        transformer_model: Your main transformer model
        pretrained_conv_path: Path to pretrained conv layers
        freeze_convs: Whether to freeze the pretrained layers
        
    Returns:
        Updated transformer model
    """
    # Load pretrained weights
    pretrained_data = load_pretrained_input_convs(pretrained_conv_path)
    pretrained_state = pretrained_data['state_dict']
    
    # Get the current model state
    current_state = transformer_model.state_dict()
    
    # Update the input conv layers
    updated_count = 0
    for key, value in pretrained_state.items():
        full_key = f'input_convs.{key}'
        if full_key in current_state:
            if current_state[full_key].shape == value.shape:
                current_state[full_key] = value
                updated_count += 1
                print(f"Updated: {full_key}")
            else:
                print(f"Shape mismatch for {full_key}: {current_state[full_key].shape} vs {value.shape}")
        else:
            print(f"Key not found in target model: {full_key}")
    
    # Load the updated state dict
    transformer_model.load_state_dict(current_state)
    
    # Optionally freeze the conv layers
    if freeze_convs:
        for name, param in transformer_model.named_parameters():
            if name.startswith('input_convs.'):
                param.requires_grad = False
                print(f"Frozen: {name}")
    
    print(f"Successfully updated {updated_count} parameters from pretrained model")
    print(f"Pretrained model info: {pretrained_data['model_type']}, hidden_size: {pretrained_data['hidden_size']}")
    
    return transformer_model


class PretrainedConvTransferHelper:
    """Helper class for transferring pretrained conv layers"""
    
    def __init__(self, pretrained_path: str):
        self.pretrained_path = pretrained_path
        self.pretrained_data = load_pretrained_input_convs(pretrained_path)
    
    def create_compatible_input_convs(self) -> nn.ModuleList:
        """Create input conv layers compatible with the pretrained weights"""
        hidden_size = self.pretrained_data['hidden_size']
        
        input_convs = nn.ModuleList([
            nn.Conv1d(1, hidden_size // 4, kernel_size=k, padding=k//2)
            for k in [3, 7, 15, 31]
        ])
        
        # Load pretrained weights
        state_dict = self.pretrained_data['state_dict']
        
        for i, conv in enumerate(input_convs):
            conv_key = f'input_convs.{i}.weight'
            bias_key = f'input_convs.{i}.bias'
            
            if conv_key in state_dict:
                conv.weight.data = state_dict[conv_key]
            if bias_key in state_dict:
                conv.bias.data = state_dict[bias_key]
        
        return input_convs
    
    def get_hidden_size(self) -> int:
        """Get the hidden size from pretrained model"""
        return self.pretrained_data['hidden_size']
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the pretrained model"""
        return {
            'model_type': self.pretrained_data['model_type'],
            'hidden_size': self.pretrained_data['hidden_size'],
            'checkpoint_path': self.pretrained_path
        }


# Example usage functions
def create_your_transformer_model_with_pretrained_convs(
    pretrained_conv_path: str,
    num_layers: int = 6,
    num_heads: int = 8,
    sequence_length: int = 1000,
    output_size: int = 1,
    freeze_convs: bool = False
) -> nn.Module:
    """
    Example function showing how to create your transformer model
    with pretrained input convolutions.
    
    Replace this with your actual transformer architecture.
    """
    helper = PretrainedConvTransferHelper(pretrained_conv_path)
    hidden_size = helper.get_hidden_size()
    
    class YourTransformerModel(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Use pretrained input convolutions
            self.input_convs = helper.create_compatible_input_convs()
            
            # Batch normalization (if you use it)
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(hidden_size // 4) for _ in range(4)
            ])
            
            # Your transformer layers
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            
            # Output layers for your specific task
            self.output_projection = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, output_size)
            )
            
        def forward(self, x):
            # Apply input convolutions (pretrained)
            conv_outputs = []
            for conv, bn in zip(self.input_convs, self.batch_norms):
                out = conv(x)
                out = bn(out)
                out = torch.relu(out)
                conv_outputs.append(out)
            
            # Concatenate features
            features = torch.cat(conv_outputs, dim=1)  # (batch, hidden_size, seq_len)
            
            # Transpose for transformer
            features = features.transpose(1, 2)  # (batch, seq_len, hidden_size)
            
            # Apply transformer
            encoded = self.transformer(features)
            
            # Global average pooling and output
            pooled = encoded.mean(dim=1)  # (batch, hidden_size)
            output = self.output_projection(pooled)
            
            return output
    
    model = YourTransformerModel()
    
    # Freeze conv layers if requested
    if freeze_convs:
        for conv in model.input_convs:
            for param in conv.parameters():
                param.requires_grad = False
    
    return model


if __name__ == "__main__":
    # Example usage
    pretrained_path = "./pretraining_checkpoints/best_model.pth"
    
    if os.path.exists(pretrained_path):
        # Method 1: Create model with pretrained convs from scratch
        model = create_your_transformer_model_with_pretrained_convs(
            pretrained_conv_path=pretrained_path,
            freeze_convs=False  # Set to True if you want to freeze pretrained layers
        )
        
        print("Created transformer model with pretrained input convolutions")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
        
        # Method 2: Use helper class
        helper = PretrainedConvTransferHelper(pretrained_path)
        print("Pretrained model info:", helper.get_model_info())
        
    else:
        print(f"Pretrained model not found at {pretrained_path}")
        print("Please run pretraining first!")
