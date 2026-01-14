"""
Model architecture for Visual Memory AI.
Implements ResNet50-based memorability predictor.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Tuple


class MemorabilityPredictor(nn.Module):
    """
    CNN-based model for predicting image memorability.
    Uses pretrained ResNet50 as backbone.
    """
    
    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        dropout: float = 0.5,
        hidden_dim: int = 512
    ):
        """
        Initialize model.
        
        Args:
            backbone: Backbone architecture ('resnet50' or 'resnet101')
            pretrained: Whether to use pretrained weights
            dropout: Dropout probability
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.backbone_name = backbone
        
        # Load pretrained backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            backbone_dim = 2048
        elif backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove original classification head
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images [batch_size, 3, 224, 224]
            
        Returns:
            Memorability scores [batch_size, 1]
        """
        features = self.backbone(x)
        scores = self.regressor(features)
        return scores.squeeze(-1)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature embeddings (for similarity search).
        
        Args:
            x: Input images [batch_size, 3, 224, 224]
            
        Returns:
            Feature vectors [batch_size, feature_dim]
        """
        with torch.no_grad():
            features = self.backbone(x)
            features = features.view(features.size(0), -1)
        return features


class VisionTransformerPredictor(nn.Module):
    """
    Vision Transformer-based memorability predictor.
    Alternative to CNN-based model.
    """
    
    def __init__(
        self,
        model_name: str = 'vit_b_16',
        pretrained: bool = True,
        dropout: float = 0.5,
        hidden_dim: int = 512
    ):
        """
        Initialize ViT model.
        
        Args:
            model_name: ViT variant ('vit_b_16', 'vit_b_32', 'vit_l_16')
            pretrained: Whether to use pretrained weights
            dropout: Dropout probability
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        # Load pretrained ViT
        if model_name == 'vit_b_16':
            self.backbone = models.vit_b_16(pretrained=pretrained)
            backbone_dim = 768
        elif model_name == 'vit_b_32':
            self.backbone = models.vit_b_32(pretrained=pretrained)
            backbone_dim = 768
        elif model_name == 'vit_l_16':
            self.backbone = models.vit_l_16(pretrained=pretrained)
            backbone_dim = 1024
        else:
            raise ValueError(f"Unsupported ViT model: {model_name}")
        
        # Replace classification head
        self.backbone.heads = nn.Identity()
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(backbone_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images [batch_size, 3, 224, 224]
            
        Returns:
            Memorability scores [batch_size, 1]
        """
        features = self.backbone(x)
        scores = self.regressor(features)
        return scores.squeeze(-1)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings."""
        with torch.no_grad():
            features = self.backbone(x)
        return features


def create_model(
    architecture: str = 'resnet50',
    pretrained: bool = True,
    device: Optional[str] = None
) -> nn.Module:
    """
    Factory function to create memorability prediction model.
    
    Args:
        architecture: Model architecture
        pretrained: Whether to use pretrained weights
        device: Device to move model to
        
    Returns:
        Model instance
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if architecture in ['resnet50', 'resnet101']:
        model = MemorabilityPredictor(
            backbone=architecture,
            pretrained=pretrained
        )
    elif architecture.startswith('vit'):
        model = VisionTransformerPredictor(
            model_name=architecture,
            pretrained=pretrained
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    model = model.to(device)
    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    model = create_model('resnet50', device='cpu')
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test feature extraction
    features = model.get_features(dummy_input)
    print(f"Feature shape: {features.shape}")