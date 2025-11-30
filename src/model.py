"""
Model Architecture Module for Raman Spectroscopy Classification

This module defines the 1D CNN architecture for classifying Raman spectra.
Components are modular and reusable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class Conv1DBlock(nn.Module):
    """
    Reusable 1D convolutional block with optional batch normalization and dropout.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[int] = None,
        use_batch_norm: bool = True,
        dropout: float = 0.0,
        activation: str = "relu"
    ):
        super(Conv1DBlock, self).__init__()
        
        if padding is None:
            padding = kernel_size // 2
        
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        
        self.batch_norm = nn.BatchNorm1d(out_channels) if use_batch_norm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2)
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class Raman1DCNN(nn.Module):
    """
    1D CNN for Raman spectroscopy classification.
    
    Architecture:
    - Backbone: Multiple 1D convolutional layers with pooling
    - Classification head: Fully connected layers with softmax output
    - Optional ordinal/regression head for quality metrics
    """
    
    def __init__(
        self,
        input_length: int,
        n_classes: int,
        n_channels: List[int] = [32, 64, 128, 256],
        kernel_sizes: List[int] = [7, 5, 5, 3],
        pool_sizes: List[int] = [2, 2, 2, 2],
        use_batch_norm: bool = True,
        dropout: float = 0.3,
        fc_hidden: List[int] = [128, 64],
        use_ordinal_head: bool = False,
        ordinal_classes: Optional[int] = None
    ):
        super(Raman1DCNN, self).__init__()
        
        self.input_length = input_length
        self.n_classes = n_classes
        self.use_ordinal_head = use_ordinal_head
        
        # Build backbone (convolutional layers)
        backbone_layers = []
        in_channels = 1  # Input is single-channel spectrum
        
        for i, (out_channels, kernel_size, pool_size) in enumerate(
            zip(n_channels, kernel_sizes, pool_sizes)
        ):
            # Convolutional block
            backbone_layers.append(
                Conv1DBlock(
                    in_channels, out_channels,
                    kernel_size=kernel_size,
                    use_batch_norm=use_batch_norm,
                    dropout=dropout if i < len(n_channels) - 1 else 0.0
                )
            )
            
            # Pooling
            backbone_layers.append(nn.MaxPool1d(kernel_size=pool_size, stride=pool_size))
            
            in_channels = out_channels
        
        self.backbone = nn.Sequential(*backbone_layers)
        
        # Calculate flattened size after backbone
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_length)
            dummy_output = self.backbone(dummy_input)
            self.flattened_size = dummy_output.numel()
        
        # Build classification head
        fc_layers = []
        prev_size = self.flattened_size
        
        for hidden_size in fc_hidden:
            fc_layers.append(nn.Linear(prev_size, hidden_size))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        fc_layers.append(nn.Linear(prev_size, n_classes))
        self.classification_head = nn.Sequential(*fc_layers)
        
        # Optional ordinal head for quality/defect density prediction
        if use_ordinal_head:
            if ordinal_classes is None:
                ordinal_classes = n_classes
            ordinal_layers = []
            prev_size = self.flattened_size
            
            for hidden_size in fc_hidden:
                ordinal_layers.append(nn.Linear(prev_size, hidden_size))
                ordinal_layers.append(nn.ReLU())
                ordinal_layers.append(nn.Dropout(dropout))
                prev_size = hidden_size
            
            ordinal_layers.append(nn.Linear(prev_size, ordinal_classes))
            self.ordinal_head = nn.Sequential(*ordinal_layers)
        else:
            self.ordinal_head = None
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 1, spectrum_length)
            return_features: Whether to return backbone features
        
        Returns:
            Tuple of (classification_logits, ordinal_logits)
            If return_features=True, also returns backbone features
        """
        # Backbone
        features = self.backbone(x)
        
        # Flatten
        flattened = features.view(features.size(0), -1)
        
        # Classification head
        class_logits = self.classification_head(flattened)
        
        # Ordinal head (if enabled)
        ordinal_logits = None
        if self.ordinal_head is not None:
            ordinal_logits = self.ordinal_head(flattened)
        
        if return_features:
            return class_logits, ordinal_logits, flattened
        return class_logits, ordinal_logits
    
    def predict(
        self,
        x: torch.Tensor,
        return_probs: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with softmax probabilities.
        
        Args:
            x: Input tensor
            return_probs: Whether to return probabilities or logits
        
        Returns:
            Tuple of (predicted_classes, probabilities_or_logits)
        """
        self.eval()
        with torch.no_grad():
            class_logits, ordinal_logits = self.forward(x)
            
            if return_probs:
                class_probs = F.softmax(class_logits, dim=1)
            else:
                class_probs = class_logits
            
            predicted_classes = torch.argmax(class_probs, dim=1)
            
            return predicted_classes, class_probs


def create_model(
    input_length: int,
    n_classes: int,
    config: Optional[dict] = None
) -> Raman1DCNN:
    """
    Factory function to create a Raman1DCNN model with custom or default config.
    
    Args:
        input_length: Length of input spectrum
        n_classes: Number of classification classes
        config: Optional dictionary with model configuration:
            - n_channels: List of channel sizes
            - kernel_sizes: List of kernel sizes
            - pool_sizes: List of pooling sizes
            - use_batch_norm: Whether to use batch normalization
            - dropout: Dropout rate
            - fc_hidden: List of FC layer sizes
            - use_ordinal_head: Whether to use ordinal head
            - ordinal_classes: Number of ordinal classes
    
    Returns:
        Raman1DCNN model instance
    """
    if config is None:
        config = {}
    
    return Raman1DCNN(
        input_length=input_length,
        n_classes=n_classes,
        n_channels=config.get('n_channels', [32, 64, 128, 256]),
        kernel_sizes=config.get('kernel_sizes', [7, 5, 5, 3]),
        pool_sizes=config.get('pool_sizes', [2, 2, 2, 2]),
        use_batch_norm=config.get('use_batch_norm', True),
        dropout=config.get('dropout', 0.3),
        fc_hidden=config.get('fc_hidden', [128, 64]),
        use_ordinal_head=config.get('use_ordinal_head', False),
        ordinal_classes=config.get('ordinal_classes', None)
    )


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(model: nn.Module, method: str = "xavier") -> None:
    """
    Initialize model weights.
    
    Args:
        model: PyTorch model
        method: Initialization method ('xavier', 'kaiming', 'normal')
    """
    for module in model.modules():
        if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
            if method == "xavier":
                nn.init.xavier_uniform_(module.weight)
            elif method == "kaiming":
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            elif method == "normal":
                nn.init.normal_(module.weight, mean=0, std=0.02)
            
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

