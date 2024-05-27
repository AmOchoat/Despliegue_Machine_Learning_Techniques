import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import pandas as pd

# Arquitectura
class DenseNet121_Classifier(nn.Module):
    def __init__(self, num_classes=6):
        super(DenseNet121_Classifier, self).__init__()
        self.densenet = models.densenet121(weights='DEFAULT')
        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.densenet(x)
        return x

model = DenseNet121_Classifier(num_classes=6)
model.load_state_dict(torch.load('model\densenet121_hemorrhage_classifier.pth', map_location=torch.device('cpu')))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Summarize
print(model)