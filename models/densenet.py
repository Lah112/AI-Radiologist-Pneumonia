import torch.nn as nn
import torchvision.models as models

def get_densenet(num_classes=2, pretrained=True):
    model = models.densenet121(pretrained=pretrained)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    return model
