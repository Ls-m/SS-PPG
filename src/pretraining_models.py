import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class InputConvLayers(nn.Module):
    """
    Input convolution layers for PPG signal processing.
    These are the layers that will be pretrained and then transferred.
    """
    
    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Multi-scale convolutions (same as your transformer model)
        self.input_convs = nn.ModuleList([
            nn.Conv1d(1, hidden_size // 4, kernel_size=k, padding=k//2)
            for k in [3, 7, 15, 31]
        ])
        
        # Batch normalization for each conv layer
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_size // 4) for _ in range(4)
        ])
        
        # Activation
        self.activation = nn.GELU()
        
        # Optional dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through input convolutions
        
        Args:
            x: Input tensor of shape (batch_size, 1, sequence_length)
            
        Returns:
            Features of shape (batch_size, hidden_size, sequence_length)
        """
        # Apply multi-scale convolutions
        conv_outputs = []
        for conv, bn in zip(self.input_convs, self.batch_norms):
            out = conv(x)
            out = bn(out)
            out = self.activation(out)
            conv_outputs.append(out)
        
        # Concatenate along channel dimension
        features = torch.cat(conv_outputs, dim=1)  # (batch, hidden_size, seq_len)
        
        return self.dropout(features)


class MaskedSignalReconstructionModel(nn.Module):
    """
    Model for masked signal reconstruction pretraining task.
    
    This model uses the InputConvLayers to extract features and then
    reconstructs the original signal from masked inputs.
    """
    
    def __init__(
        self, 
        hidden_size: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Input convolution layers (the ones we want to pretrain)
        self.input_convs = InputConvLayers(hidden_size)
        
        # Transformer-like encoder for processing features
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Reconstruction head
        self.reconstruction_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, masked_signal: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for reconstruction
        
        Args:
            masked_signal: Masked input signal (batch_size, 1, seq_len)
            mask: Optional boolean mask (batch_size, seq_len) - True for unmasked positions
            
        Returns:
            Reconstructed signal (batch_size, seq_len)
        """
        batch_size, _, seq_len = masked_signal.shape
        
        # Extract features using input convolutions
        features = self.input_convs(masked_signal)  # (batch, hidden_size, seq_len)
        
        # Transpose for transformer: (batch, seq_len, hidden_size)
        features = features.transpose(1, 2)
        
        # Apply layer norm
        features = self.layer_norm(features)
        
        # Create attention mask for transformer (True for positions to attend to)
        # We'll allow attention to all positions (both masked and unmasked)
        attn_mask = None
        
        # Process through transformer encoder
        encoded = self.encoder(features, src_key_padding_mask=None)
        
        # Reconstruct signal
        reconstructed = self.reconstruction_head(encoded)  # (batch, seq_len, 1)
        reconstructed = reconstructed.squeeze(-1)  # (batch, seq_len)
        
        return reconstructed


class ContrastivePPGModel(nn.Module):
    """
    Alternative model for contrastive learning pretraining.
    """
    
    def __init__(self, hidden_size: int = 256, projection_dim: int = 128):
        super().__init__()
        
        # Input convolution layers (the ones we want to pretrain)
        self.input_convs = InputConvLayers(hidden_size)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, projection_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for contrastive learning
        
        Args:
            x: Input signal (batch_size, 1, seq_len)
            
        Returns:
            Projection features (batch_size, projection_dim)
        """
        # Extract features
        features = self.input_convs(x)  # (batch, hidden_size, seq_len)
        
        # Global pooling
        pooled = self.global_pool(features).squeeze(-1)  # (batch, hidden_size)
        
        # Project to contrastive space
        projected = self.projection_head(pooled)
        
        return F.normalize(projected, dim=1)


class DenoisingAutoencoderModel(nn.Module):
    """
    Denoising autoencoder for PPG signal pretraining.
    """
    
    def __init__(self, hidden_size: int = 256):
        super().__init__()
        
        # Input convolution layers (the ones we want to pretrain)
        self.input_convs = InputConvLayers(hidden_size)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size // 2),
            nn.GELU(),
            nn.Conv1d(hidden_size // 2, hidden_size // 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size // 4),
            nn.GELU(),
            nn.Conv1d(hidden_size // 4, 1, kernel_size=3, padding=1)
        )
        
    def forward(self, noisy_signal: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for denoising
        
        Args:
            noisy_signal: Noisy input signal (batch_size, 1, seq_len)
            
        Returns:
            Denoised signal (batch_size, 1, seq_len)
        """
        # Encode
        encoded = self.input_convs(noisy_signal)
        
        # Decode
        decoded = self.decoder(encoded)
        
        return decoded


def create_pretraining_model(
    model_type: str = 'masked_reconstruction',
    hidden_size: int = 256,
    **kwargs
) -> nn.Module:
    """
    Factory function to create pretraining models
    
    Args:
        model_type: Type of model ('masked_reconstruction', 'contrastive', 'denoising')
        hidden_size: Hidden dimension size
        **kwargs: Additional arguments for specific models
        
    Returns:
        Pretraining model
    """
    if model_type == 'masked_reconstruction':
        return MaskedSignalReconstructionModel(hidden_size=hidden_size, **kwargs)
    elif model_type == 'contrastive':
        return ContrastivePPGModel(hidden_size=hidden_size, **kwargs)
    elif model_type == 'denoising':
        return DenoisingAutoencoderModel(hidden_size=hidden_size, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test the models
    batch_size, seq_len = 8, 1000
    
    # Test input
    x = torch.randn(batch_size, 1, seq_len)
    mask = torch.rand(batch_size, seq_len) > 0.15  # 15% masking
    
    # Test masked reconstruction model
    model = MaskedSignalReconstructionModel(hidden_size=256)
    output = model(x, mask)
    print(f"Masked reconstruction output shape: {output.shape}")
    
    # Test contrastive model
    contrastive_model = ContrastivePPGModel(hidden_size=256)
    contrastive_output = contrastive_model(x)
    print(f"Contrastive output shape: {contrastive_output.shape}")
    
    # Test denoising model
    denoising_model = DenoisingAutoencoderModel(hidden_size=256)
    denoising_output = denoising_model(x)
    print(f"Denoising output shape: {denoising_output.shape}")
