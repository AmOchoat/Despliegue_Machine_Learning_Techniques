### Download Data

-   Training Data: [Download data_train.zip](https://drive.google.com/file/d/1CrWEE5tPOS_TIS8j0oz-FlxLeFrcip6u/view?usp=sharing)
-   Prediction Data: [Download data_predict.zip](https://drive.google.com/file/d/1KfeCf_UwwJxIKS4vfC8erXed-Yi3--Bq/view?usp=sharing)

### Steps

1.  Convert DCM to PNG

    `python3 prepare_data.py -dcm_path train_images -png_path train_png`

2.  Train the Model

    `python3 model_trainer.py`

    In the `model_trainer.py` file, update the following lines to match your directories:

    -   `save_dir`
    -   `model_save_path`
    -   `csv_file`
    -   `root_dir`

    Ensure these paths point to where you wish to save the models and where the training files and labels are located.

3.  Predict

    We have created a Streamlit app to make predictions. You can access the app using the following link: [Streamlit App](https://machine-learning-techniques-proyecto.streamlit.app/)
