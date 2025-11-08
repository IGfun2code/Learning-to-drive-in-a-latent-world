# encoder.py
import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNetEncoder(nn.Module):
    """
    Simple vision encoder using a pretrained ResNet-18.
    Outputs a flat feature vector.
    """
    def __init__(self, pretrained: bool = True, out_dim: int = 512, freeze: bool = True):
        super().__init__()
        base = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        # remove classifier head
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # (B, 512, 1, 1)
        
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # optional projection
        self.proj = nn.Linear(512, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        feats = self.backbone(x)              # (B, 512, 1, 1)
        feats = feats.view(feats.size(0), -1) # (B, 512)
        feats = self.proj(feats)              # (B, out_dim)
        return feats
