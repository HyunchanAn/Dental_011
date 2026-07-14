import torch
import torch.nn as nn
import torchvision.models as models

class AgeRegressionModel(nn.Module):
    def __init__(self, pretrained=True):
        super(AgeRegressionModel, self).__init__()
        # Use ResNet18 as a lightweight backbone
        # models.resnet18(weights=models.ResNet18_Weights.DEFAULT) -> Not available in older torchvision, use pretrained=True
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Modify the last FC layer for 1 output (regression)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, 1)
        )
        
    def forward(self, x):
        # outputs a single age value per batch item
        return self.backbone(x).squeeze(-1)
