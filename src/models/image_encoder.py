import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class FrozenImageEncoder(nn.Module):
    
    """
    Outputs 512 Embeddings.
    """
    
    def __init__(self):
        super().__init__()
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Remove final classification layer
        self.features = nn.Sequential(*list(model.children())[:-1])  # up to avgpool

        # Freeze everything
        for p in self.features.parameters():
            p.requires_grad = False
            
    def forward(self, x):
        x = self.features(x)        # (B, 512, 1, 1)
        x = torch.flatten(x, 1)     # (B, 512)
        return x

        