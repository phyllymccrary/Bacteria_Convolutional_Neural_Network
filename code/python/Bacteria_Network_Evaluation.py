import numpy as np
import os
from os.path import join
from PIL import Image

from keras.models import load_model

import json

import matplotlib.pyplot as plt


# Loads single image
def load_image(img_path):
    if not os.path.exists(img_path):
        print("Image doesn't exist: {}".format(img_path))
        return

    img = Image.open(img_path)
    img = img.resize((28, 28), Image.ANTIALIAS)
    img = img.convert('L')
    img = np.expand_dims(img, axis=3)
    img = np.expand_dims(img, axis=0)
    return img


# Saves the model and the weights
def get_model(data_column, models_dir, history_dir):
    model_name = data_column + "_model" + ".h5"
    model_file = join(models_dir, model_name)

    model = load_model(model_file)

    history_file = join(history_dir, "{}_history.json".format(data_column))
    history_json = open(history_file, 'r')
    history = json.loads(history_json.read())

    return model, history


# Plots the history of the model
def plot_history(history):
    # summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


# Get a dataset from the Excel sheet
def get_dataset(base_dataset, column_name):
    return base_dataset[["strain_id", "strain_image", column_name]]


save_dir = "Enter a directory to retrieve you models and history"  # Directory for saving things
models_dir = join(save_dir, "models")  # models directory
history_dir = join(save_dir, "history")  # history directory

data_column = "Choose a column to analyze"  # e.g. "toby.max.od" or "carb.lag" or "env"

model, history = get_model(data_column, models_dir, history_dir)

plot_history(history)

img_path = "Enter an image you want to analyze"
img = load_image(img_path)

predict_image = model.predict(img)

print(predict_image)
