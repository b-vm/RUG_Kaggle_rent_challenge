from pathlib import Path
import math
from data_loader import load_dataset
import tensorflow as tf
import requests
import shutil
from tqdm import tqdm

from cv2 import cv2

import pandas as pd
from sklearn.metrics import mean_absolute_error

from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras import Model

# import the necessary packages
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np

from logger import log

def get_images(image_urls):
    filenames = []
    log.info("Downloading images")
    for idx, image_url in tqdm(enumerate(image_urls[0:100])):
        filename = f"./data/images/{idx}.png"
        filenames.append(filename)
        if Path(filename).exists():
            log.debug(f"{filename} already exists, skipping...")
            continue

        res = requests.get(image_url, stream = True)
        # Save image to image folder
        # first check if image already exists
        if res.status_code == 200:
            res.raw.decode_content = True
            with open(filename, 'wb') as f:
                shutil.copyfileobj(res.raw, f)
        else:
            log.warning(f"Failed to download {filename}")
            filenames.pop()
            filenames.append("Not Found")
    return filenames

def load_images(image_filenames):
    images = []
    for image_path in image_filenames:
        if image_path == "Not Found":
            images.append(np.nan)
            continue
        image = cv2.imread(image_path)
        image = cv2.resize(image, (32,32))
        images.append(image)
    return np.array(images)

def create_cnn(width, height, depth, filters=(16, 32, 64)):
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    inputShape = (height, width, depth)
    chanDim = -1
    # define the model input
    inputs = Input(shape=inputShape)
    # loop over the number of filters
    x = None
    for (i, f) in enumerate(filters):
        # if this is the first CONV layer then set the input
        # appropriately
        if i == 0:
            x = inputs
        # CONV => RELU => BN => POOL
        x = Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)
    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = Dense(4)(x)
    x = Activation("relu")(x)
    # check to see if the regression node should be added
    x = Dense(1, activation="linear")(x)

    # construct the CNN
    model = Model(inputs, x)
    # return the CNN
    return model

def encode_image(image_urls, rent):
    image_filenames = get_images(image_urls)
    images = load_images(image_filenames)
    images = images / 255.0
    images = images.astype('float32')
    rent = np.asarray(rent[0:100]).astype('float32')

    not_nan_locations = []
    for idx, image in enumerate(images):
        if type(image) != float:
            not_nan_locations.append(idx)

    train_data, rem_data, train_labels, rem_labels = train_test_split(images[not_nan_locations], rent[not_nan_locations], test_size=0.3, random_state=1001)
    valid_data, test_data, valid_labels, test_labels = train_test_split(rem_data, rem_labels, test_size=0.5, random_state=1001)

    max_price = max(train_labels) if max(train_labels) > max(test_labels) else max(test_labels)
    train_labels /= max_price
    test_labels /= max_price

    model = create_cnn(32, 32, 3)
    model.compile(loss="mean_absolute_error", optimizer=Adam(learning_rate=1e-3))
    # model.summary()
    model.fit(x=train_data, y=test_data,
        validation_data=(valid_data, valid_labels),
        epochs=200, batch_size=8)

    y_pred = model.predict(test_data)

    average_score = mean_absolute_error(test_labels, y_pred)

    print(f"Mean Absolute Error: {average_score}")
    # TODO: CNN Regressor







if __name__=="__main__":
    df = load_dataset("./data/train.csv")
    encode_image(df['coverImageUrl'], df['rent'])
