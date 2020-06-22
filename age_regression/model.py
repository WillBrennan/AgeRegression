import torch
import torch.nn as nn
from torchvision import models


class AgeRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnext101_32x8d(pretrained=True)
        self.model.fc = nn.Linear(512 * 4, 1)

    def forward(self, x: torch.Tensor):
        x_age = self.model(x)
        return x_age