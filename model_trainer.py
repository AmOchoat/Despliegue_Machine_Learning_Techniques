import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import os
import sys
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt

# Redirigir stdout a un archivo log.txt
log_file_path = 'log.txt'
sys.stdout = open(log_file_path, 'w')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HemorrhageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels.iloc[idx, 0] + '.png')
        image = Image.open(img_name).convert('RGB')
        labels = self.labels.iloc[idx, 1:].values.astype('float')
        labels = torch.tensor(labels, dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, labels

# Transformaciones para las imágenes
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Instanciar el dataset
csv_file = '../data/sampled_data_train.csv'
root_dir = '../data/png_train/images'
dataset = HemorrhageDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)

# Dividir el dataset en entrenamiento y validación
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

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

# Instanciar el modelo, definir la pérdida y el optimizador
model = DenseNet121_Classifier(num_classes=6).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00002)

# Callbacks
class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model = model.state_dict()
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model = model.state_dict()
            self.counter = 0

early_stopping = EarlyStopping(patience=8, verbose=True)

# Función para calcular métricas
def calculate_metrics(outputs, labels):
    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    preds = outputs > 0.5
    precision = precision_score(labels, preds, average='micro', zero_division=0)
    recall = recall_score(labels, preds, average='micro', zero_division=0)
    auc = roc_auc_score(labels, outputs, average='micro')
    return precision, recall, auc

# Entrenamiento del Modelo
num_epochs = 65
train_losses, val_losses = [], []
train_precisions, val_precisions = [], []
train_recalls, val_recalls = [], []
train_aucs, val_aucs = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    all_train_outputs = []
    all_train_labels = []
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        all_train_outputs.append(outputs)
        all_train_labels.append(labels)
        
    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    
    all_train_outputs = torch.cat(all_train_outputs)
    all_train_labels = torch.cat(all_train_labels)
    
    train_precision, train_recall, train_auc = calculate_metrics(all_train_outputs, all_train_labels)
    train_precisions.append(train_precision)
    train_recalls.append(train_recall)
    train_aucs.append(train_auc)
    
    model.eval()
    val_loss = 0.0
    
    all_val_outputs = []
    all_val_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            
            all_val_outputs.append(outputs)
            all_val_labels.append(labels)
    
    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)
    
    all_val_outputs = torch.cat(all_val_outputs)
    all_val_labels = torch.cat(all_val_labels)
    
    val_precision, val_recall, val_auc = calculate_metrics(all_val_outputs, all_val_labels)
    val_precisions.append(val_precision)
    val_recalls.append(val_recall)
    val_aucs.append(val_auc)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, '
          f'Train Precision: {train_precision:.4f}, Val Precision: {val_precision:.4f}, '
          f'Train Recall: {train_recall:.4f}, Val Recall: {val_recall:.4f}, '
          f'Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}')

    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        model.load_state_dict(early_stopping.best_model)
        break

print('Entrenamiento completado')

save_dir = 'save_models'
os.makedirs(save_dir, exist_ok=True)
model_save_path = os.path.join(save_dir, 'densenet121_hemorrhage_classifier.pth')
torch.save(model.state_dict(), model_save_path)
print(f'Modelo guardado en {model_save_path}')

# Graficar las métricas
def plot_metrics(train_values, val_values, metric_name):
    plt.figure(figsize=(10, 5))
    plt.plot(train_values, label=f'Train {metric_name}')
    plt.plot(val_values, label=f'Val {metric_name}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig(f'training_process/{metric_name}.png')
    plt.close()

os.makedirs('training_process', exist_ok=True)
plot_metrics(train_losses, val_losses, 'Loss')
plot_metrics(train_precisions, val_precisions, 'Precision')
plot_metrics(train_recalls, val_recalls, 'Recall')
plot_metrics(train_aucs, val_aucs, 'AUC')

# Cerrar el archivo de log y restaurar stdout
sys.stdout.close()
sys.stdout = sys.__stdout__