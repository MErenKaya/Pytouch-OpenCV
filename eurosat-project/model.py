import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def get_model(num_classes=4):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model