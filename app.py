import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
import streamlit as st
import zipfile
from pathlib import Path
import pandas as pd
from glob import glob
from prepare_data import prepare_images
from model_predict import DenseNet121_Classifier, transform, ImageDataset

# Variables
zip_path = Path.cwd() / Path('data\dcm')\

if not zip_path.exists():
    zip_path.mkdir(parents=True)

zip_name = zip_path / 'data.zip'
png_folder = Path.cwd() / Path('data/png/')
if not png_folder.exists():
    png_folder.mkdir(parents=True)
output_file = 'predictions.csv'

# Título
st.image('src/Banner.png', use_column_width=True)
st.title('Detección y clasificación automáticas de hemorragias intracraneales agudas')

st.write('Este es un prototipo de una aplicación que permite subir un archivo ZIP con imágenes DICOM de un paciente, procesarlas y predecir si el paciente tiene o no hemorragias intracraneales agudas.')

zip_file = st.file_uploader('Sube un archivo ZIP de un paciente', type='zip')

if zip_file:
    # Guardar el archivo en el sistema    
    with open(zip_name, 'wb') as f:
        f.write(zip_file.read())

    # Extraer el archivo ZIP
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall(zip_path)
    
    # Delete the ZIP file
    zip_name.unlink()
    st.write('Archivo subido correctamente')

    # Si el directorio no está vacío, mostrar un botón para procesar las imágenes
    if any(zip_path.iterdir()):

        if st.button('Procesar imágenes'):
            st.write('Procesando imágenes...')            
            prepare_images(glob(str(zip_path)+"/*"), str(png_folder)+'/')            
            st.write('Imágenes procesadas')


    # Mostrar botón para predecir si el directorio de imágenes no está vacío
    if any(png_folder.iterdir()):
        if st.button('Predecir'):
            st.write('Prediciendo imágenes...')            

            model = DenseNet121_Classifier(num_classes=6)
            model.load_state_dict(torch.load('model\densenet121_hemorrhage_classifier.pth', map_location=torch.device('cpu')))
            model.eval()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            # Crear el DataLoader (para predecir imagenes de forma paralela)
            dataset = ImageDataset(png_folder, transform=transform)
            data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

            # Realizar predicciones
            predictions = []

            with torch.no_grad():
                for images, image_files in data_loader:
                    images = images.to(device)
                    # Predicción
                    outputs = model(images)
                    # Softmax para que estén entre 0 y 1
                    outputs = torch.softmax(outputs, dim=1).cpu().numpy()
                    for image_file, output in zip(image_files, outputs):
                        predictions.append([image_file] + output.tolist())

            # Guardar las predicciones en un archivo CSV
            columns = ['image'] + [f'class_{i}' for i in range(1, 7)]
            df = pd.DataFrame(predictions, columns=columns)
            df.to_csv(output_file, index=False)
            print(f'Predicciones guardadas en {output_file}')

            st.write('Predicciones realizadas')
            # Botón para descargar el archivo
            st.write('Descarga el archivo con las predicciones')
            with open(output_file, 'rb') as f:
                st.download_button('Descargar archivo', f, file_name='predictions.csv', mime='text/csv')
