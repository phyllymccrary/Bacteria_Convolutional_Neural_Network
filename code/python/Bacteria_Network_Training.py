import os
from os.path import join

import math

import json

import pandas as pd
import numpy as np

from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential

from skimage.io import imread

import matplotlib.pyplot as plt

os.environ['KERAS_BACKEND'] = 'tensorflow'


# Get a dataset from the Excel sheet
def get_dataset(base_dataset, column_name):
    return base_dataset[["strain_id", "strain_image", column_name]]


# Splits the data 70%-train, 30%-test
def train_test_split(dataset):
    msk = np.random.rand(len(dataset)) < 0.7
    train_dataset = dataset[msk].sample(frac=1)
    test_dataset = dataset[~msk].sample(frac=1)
    return train_dataset, test_dataset


# Binarize / One Hot Encoding for categorical data
def encode_category(dataset, data_column):
    one_hot = pd.get_dummies(dataset[data_column])
    # Join the encoded df
    return dataset.join(one_hot)


# Generator for loading batches of images
def generate_images(dataframe, y_column, batch_size=128):
    if y_column == 'env':
        classes = dataframe[y_column].unique()

    while True:

        batch_input = []
        batch_output = []

        sub_dataframe = dataframe.sample(n=batch_size, replace=True)  # Get a subset of the dataframe

        for index, row in sub_dataframe.iterrows():  # iterate through each row

            x = row["strain_image"]
            img = imread(x)
            img = np.expand_dims(img, axis=3)
            batch_input.append(img)

            if y_column != 'env':
                y = row[y_column]
            else:
                y = row[classes]

            batch_output.append(y)

        batch_input = np.array(batch_input)
        batch_output = np.array(batch_output)

        yield (batch_input, batch_output)


# Trains the model
def train_model(dataset, validate_dataset, y_column):
    # ----------------------------------- Metadata -------------------------------------------------
    img_x, img_y = 28, 28
    input_shape = (img_x, img_y, 1)

    batch_size = 128
    steps_per_epoch = int(math.ceil(dataset.shape[0] / batch_size))

    num_epoch = 100
    val_steps = 64

    # ----------------------------------- CNN Architecture -----------------------------------------

    model = Sequential()

    # Convolution Layer w/ ReLU Activation Function
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu',
                     input_shape=input_shape))

    # Max Pooling Layer
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Convolution Layer w/ ReLU Activation Function
    model.add(Conv2D(64, (5, 5), activation='relu'))

    # Max Pooling Layer
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Input Layer: connects the convolution and dense layers
    model.add(Flatten())

    # Hidden Layers
    model.add(Dense(1000, kernel_initializer='normal', activation='sigmoid'))

    # Changes the final layer to accommodate for a regression column or categorical
    if y_column != 'env':
        # Single node output layer with linear activation function
        model.add(Dense(1, activation='linear'))

        # Uses Mean Squared Error Loss Function and Optimizer
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

    else:
        # Gets the amount of classes for the column
        class_amount = dataset[y_column].unique().size

        # Adds #[class_amount]-nodes to final layer
        model.add(Dense(class_amount, activation='softmax'))

        # Uses cross entropy for categories and NADAM optimizer
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    model.summary()

    train_gen = generate_images(dataset, y_column, batch_size)
    validate_gen = generate_images(validate_dataset, y_column, batch_size)

    history = model.fit_generator(train_gen, validation_data=validate_gen,
                                  steps_per_epoch=steps_per_epoch, nb_epoch=num_epoch, verbose=1,
                                  validation_steps=val_steps)

    return model, history


def plot_history(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


# Saves the model and the weights
def save_model(model, history, data_column, models_dir, history_dir):
    # Save Model in H5 file
    model_weights = data_column + "_model" + ".h5"
    model_file = join(models_dir, model_weights)

    model.save(model_file)
    print("Saved {} model to disk".format(data_column))

    # Save Model's History to JSON file
    history_file = join(history_dir, "{}_history.json".format(data_column))
    history_dict = history.history
    json.dump(history_dict, open(history_file, 'w'))
    print("Saved {} model's history to disk".format(data_column))


# Helper function for creating directory if doesn't exist
def get_directory(destination, dir_name):
    path = join(save_dir, dir_name)
    if not os.path.exists(path):
        print("Woops, directory doesn't exist. But we gotcha. It exists now!")
        os.makedirs(path)

    return path


base_directory = "Enter path location for directory where the images are stored"
bacteria_workbook = "Enter path location for excel file"

save_dir = "Enter a directory to retrieve you models and history"  # Directory for saving things
models_dir = get_directory(save_dir, "models")  # must have a models directory
history_dir = get_directory(save_dir, "history")  # must have a models directory

cnn_data = pd.read_excel(bacteria_workbook, dtype=object, sheet_name="CNN Dataset")

# # Use if want to train model for one column


data_column = "Choose a column to analyze"  # e.g. "toby.max.od" or "carb.lag" or "env"
dataset = get_dataset(cnn_data, data_column)

if data_column == 'env':
    # Encodes categories to binary
    dataset = encode_category(dataset, data_column)

# Gets training and testing set
train_dataset, test_dataset = train_test_split(dataset)

# Trains model and returns history object
model, history = train_model(dataset=train_dataset, validate_dataset=test_dataset, y_column=data_column)

# Save model
save_model(model, history, data_column, models_dir, history_dir)

plot_history(history)

# # Use if want to train a model for each column in the dataset


# columns = cnn_data.columns.values

# for column in columns:
#     data_column = column # e.g. "toby.max.od" or "carb.lag" or "env"
#     dataset = get_dataset(cnn_data, data_column)

#     # Make a CNN model for every column
#     if column == 'env':
#         # Encodes categories to binary
#         dataset = encode_category(dataset, data_column)

#     # Gets training and testing set
#     train_dataset, test_dataset = train_test_split(dataset)

#     # Trains model
#     model, history = train_model(dataset=train_dataset, validate_dataset=test_dataset, y_column=data_column)

#     # Save model
#     save_model(model, history, data_column, models_dir, history_dir)
