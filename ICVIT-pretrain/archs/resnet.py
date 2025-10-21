import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def build_resnet(channel=3, n_classes=1, pretrained=False):
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)
    model.conv1 = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(512*4, n_classes)

    return model
