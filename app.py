import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
import streamlit as st
import zipfile
from pathlib import Path
import pandas as pd
import PIL
import pydicom
import time
from glob import glob
from prepare_data import prepare_images
from model_predict import DenseNet121_Classifier, transform, ImageDataset
import streamlit as st

# Variables
# Variables
zip_path = st.session_state.get('zip_path', Path.cwd() / Path('data\dcm'))
png_folder = st.session_state.get('png_folder', Path.cwd() / Path('data/png/'))
zip_name = st.session_state.get('zip_name', zip_path / 'data.zip')
output_file = output_file = 'predictions.csv'

if "process" not in st.session_state:
    st.session_state["process"] = False
if "predictions" not in st.session_state:
    st.session_state["predictions"] = False
if "df" not in st.session_state:
    st.session_state["df"] = None

if not zip_path.exists():
    zip_path.mkdir(parents=True)

if not png_folder.exists():
    png_folder.mkdir(parents=True)
  
# Título
st.image('src/Banner.png', use_column_width=True)
st.title('Detección y clasificación automática de Hemorragia Intracerebral')

st.write('Este es un prototipo de una aplicación que permite subir un archivo ZIP con imágenes DICOM de un paciente, procesarlas y predecir si el paciente tiene o no hemorragias intracraneales agudas.')

tab1, tab2 = st.tabs(["Subir datos y realizar predicciones", "Ver resultados"])


def change_zip():
    for file in png_folder.iterdir():
        file.unlink()
    for file in zip_path.iterdir():
        file.unlink()
    st.session_state["process"] = False
    st.session_state["predictions"] = False
    st.session_state["df"] = None

zip_file = tab1.file_uploader('Sube un archivo ZIP de un paciente', type='zip', on_change=change_zip)

if zip_file:
    # Guardar el archivo en el sistema    
    with open(zip_name, 'wb') as f:
        f.write(zip_file.read())

    # Extraer el archivo ZIP
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall(zip_path)
    
    # Delete the ZIP file
    zip_name.unlink()
    tab1.write('Archivo subido correctamente')

    # Si el directorio no está vacío, mostrar un botón para procesar las imágenes
    if any(zip_path.iterdir()) and not st.session_state["process"]:
        button_process = tab1.button('Procesar imágenes')
        if button_process:
            tab1.write('Procesando imágenes...')            
            prepare_images(glob(str(zip_path)+"/*"), str(png_folder)+'/')            
            tab1.write('Imágenes procesadas')
            st.session_state["process"] = True

    # Mostrar botón para predecir si el directorio de imágenes no está vacío
    if any(png_folder.iterdir()) and st.session_state["process"]:
        button_predict = tab1.button('Predecir')
        if button_predict:
            tab1.write('Prediciendo imágenes...')            

            model = DenseNet121_Classifier(num_classes=6)
            model.load_state_dict(torch.load('densenet121_hemorrhage_classifier.pth', map_location=torch.device('cpu')))
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
                    outputs = torch.sigmoid(outputs).cpu().numpy()
                    for image_file, output in zip(image_files, outputs):
                        predictions.append([image_file] + output.tolist())

            # Guardar las predicciones en un archivo CSV
            # columns are image, epidural,intraparenchymal,intraventricular,subarachnoid,subdural,any
            columns = ['image', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']
            df = pd.DataFrame(predictions, columns=columns)
            df.to_csv(output_file, index=False)
            st.session_state["df"] = df
                      
            tab1.success('Predicciones realizadas')
            st.session_state["predictions"] = True
            # Botón para descargar el archivo
            # Make to columns layout
            col1, col2 = tab1.columns(2)
            if st.session_state["predictions"]:
                col1.write('Descarga el archivo con las predicciones')
                with open(output_file, 'rb') as f:
                    col2.download_button('Descargar archivo', f, 'predictions.csv', 'text/csv')

# Make a picker for the images

if st.session_state["df"] is not None:
    df = st.session_state["df"]
    tab2_1, tab2_2 = tab2.tabs(['Tabla de predicciones', 'Imágenes y predicciones'])
    tab2_1.write(df)
    
    tab2_2.write('Seleccione una imagen para ver la predicción')
    img_name = tab2_2.selectbox('Imágenes', [img.name for img in zip_path.iterdir()])
    
    img_path = zip_path / img_name

    img_dicom = pydicom.read_file(img_path)
    img_id = str(img_dicom.SOPInstanceUID)
    tab2_2.write(f'ID de la imagen: {img_id}')

    tab2_2.write('Predicciones:')
    row = df[df['image'] == img_id + '.png'].values[0][1:]
    columns = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']
    tab2_2.write(pd.DataFrame([row], columns=columns))    
    img = PIL.Image.open(str(png_folder / (img_id + '.png')))
    tab2_2.image(img, caption='Imagen original')
    
else:
    tab2.write('No hay resultados para mostrar')

