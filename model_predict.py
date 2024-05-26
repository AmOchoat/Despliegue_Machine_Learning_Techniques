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

# Transformaciones para las imágenes
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset personalizado para cargar las imágenes
class ImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_folder, image_file)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, image_file

# Modelo entrenado
#model = DenseNet121_Classifier(num_classes=6)
#model.load_state_dict(torch.load('save_models/densenet121_hemorrhage_classifier.pth'))
#model.eval()
#
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device)
#
## Carpeta con las imágenes a predecir
#image_folder = 'test_png'
#output_file = 'predictions.csv'
#
## Crear el DataLoader (para predecir imagenes de forma paralela)
#dataset = ImageDataset(image_folder, transform=transform)
#data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
#
## Realizar predicciones
#predictions = []
#
#with torch.no_grad():
#    for images, image_files in data_loader:
#        images = images.to(device)
#        # Predicción
#        outputs = model(images)
#        # Softmax para que estén entre 0 y 1
#        outputs = torch.softmax(outputs, dim=1).cpu().numpy()
#        for image_file, output in zip(image_files, outputs):
#            predictions.append([image_file] + output.tolist())
#
## Guardar las predicciones en un archivo CSV
#columns = ['image'] + [f'class_{i}' for i in range(1, 7)]
#df = pd.DataFrame(predictions, columns=columns)
#df.to_csv(output_file, index=False)
#print(f'Predicciones guardadas en {output_file}')
