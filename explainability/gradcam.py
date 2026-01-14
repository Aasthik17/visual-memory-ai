"""
Grad-CAM implementation for Visual Memory AI.
Provides explainability through class activation mapping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import cv2
from PIL import Image


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    Generates heatmaps showing which regions influence predictions.
    """
    
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Model to explain
            target_layer: Layer to compute gradients from (default: last conv layer)
        """
        self.model = model
        self.model.eval()
        
        # Automatically find target layer if not specified
        if target_layer is None:
            self.target_layer = self._find_target_layer()
        else:
            self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _find_target_layer(self) -> nn.Module:
        """Find the last convolutional layer in the model."""
        # For ResNet backbone
        if hasattr(self.model, 'backbone'):
            for module in reversed(list(self.model.backbone.modules())):
                if isinstance(module, nn.Conv2d):
                    return module
        
        # Fallback: search entire model
        for module in reversed(list(self.model.modules())):
            if isinstance(module, nn.Conv2d):
                return module
        
        raise ValueError("Could not find convolutional layer in model")
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_score: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Generate Class Activation Map.
        
        Args:
            input_tensor: Input image [1, 3, H, W]
            target_score: Optional target score for computing gradients
            
        Returns:
            Heatmap as numpy array [H, W]
        """
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Use predicted score if no target provided
        if target_score is None:
            target_score = output
        
        # Backward pass
        self.model.zero_grad()
        target_score.backward(retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients  # [1, C, H, W]
        activations = self.activations  # [1, C, H, W]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1).squeeze()  # [H, W]
        
        # ReLU to keep only positive influences
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def overlay_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlay heatmap on original image.
        
        Args:
            image: Original image [H, W, 3] in RGB format
            heatmap: Heatmap [H, W]
            alpha: Transparency of heatmap
            colormap: OpenCV colormap to use
            
        Returns:
            Overlayed image [H, W, 3]
        """
        # Resize heatmap to match image
        h, w = image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        
        # Convert heatmap to RGB
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8),
            colormap
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Overlay
        overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlayed


class GradCAMPlusPlus(GradCAM):
    """
    Improved Grad-CAM++ for better localization.
    Uses weighted combination of positive gradients.
    """
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_score: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """Generate Grad-CAM++ heatmap."""
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_score is None:
            target_score = output
        
        # Backward pass
        self.model.zero_grad()
        target_score.backward(retain_graph=True)
        
        gradients = self.gradients  # [1, C, H, W]
        activations = self.activations  # [1, C, H, W]
        
        # Compute weights using Grad-CAM++ formulation
        grad_2 = gradients.pow(2)
        grad_3 = gradients.pow(3)
        
        # Avoid division by zero
        denom = 2 * grad_2 + (grad_3 * activations).sum(dim=(2, 3), keepdim=True) + 1e-8
        alpha = grad_2 / denom
        
        # Positive gradients only
        relu_grad = F.relu(target_score * gradients)
        weights = (alpha * relu_grad).sum(dim=(2, 3), keepdim=True)
        
        # Weighted combination
        cam = torch.sum(weights * activations, dim=1).squeeze()
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


def visualize_memorability(
    model: nn.Module,
    image_path: str,
    device: str = 'cpu',
    save_path: Optional[str] = None
) -> Tuple[float, np.ndarray]:
    """
    Complete pipeline: predict memorability and generate explanation.
    
    Args:
        model: Trained memorability model
        image_path: Path to input image
        device: Device to run inference on
        save_path: Optional path to save visualization
        
    Returns:
        Tuple of (memorability_score, overlayed_image)
    """
    from torchvision import transforms
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict memorability
    model.eval()
    with torch.no_grad():
        score = model(input_tensor).item()
    
    # Generate Grad-CAM
    gradcam = GradCAM(model)
    heatmap = gradcam.generate_cam(input_tensor)
    
    # Resize original image for overlay
    image_resized = cv2.resize(image_np, (224, 224))
    overlayed = gradcam.overlay_heatmap(image_resized, heatmap)
    
    # Save if requested
    if save_path:
        Image.fromarray(overlayed).save(save_path)
    
    return score, overlayed


if __name__ == "__main__":
    from models.model import create_model
    
    # Create dummy model
    model = create_model('resnet50', pretrained=True, device='cpu')
    
    # Test Grad-CAM
    dummy_input = torch.randn(1, 3, 224, 224)
    
    gradcam = GradCAM(model)
    heatmap = gradcam.generate_cam(dummy_input)
    
    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")