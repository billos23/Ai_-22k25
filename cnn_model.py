# cnn_model.py
import torch
import torch.nn as nn
from torchvision import models


class FashionCNN(nn.Module):
  
    def __init__(self, num_classes=10, freeze_backbone=False):
        super().__init__()
        
        
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
       
        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        with torch.no_grad():
            self.backbone.conv1.weight = nn.Parameter(
                original_conv.weight.mean(dim=1, keepdim=True)
            )
        
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            for param in self.backbone.conv1.parameters():
                param.requires_grad = True
        
        
        num_features = self.backbone.fc.in_features  
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
