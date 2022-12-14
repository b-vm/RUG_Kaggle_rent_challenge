from pathlib import Path
import math
from data_loader import load_dataset
import tensorflow as tf
import requests
from scipy import stats
import shutil
from tqdm import tqdm

from cv2 import cv2

import pandas as pd
from sklearn.metrics import mean_absolute_error

from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras import Model
from tensorflow.keras.losses import MeanSquaredError

# import the necessary packages
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np

from logger import log

def get_test_images(image_urls, base_folder="./data/test_images/"):
    filenames = []
    log.info("Downloading images")
    for idx, image_url in tqdm(enumerate(image_urls)):
        filename = f"{base_folder}{idx}.png"
        filenames.append(filename)
        if not Path(filename).exists():
            log.debug(f"{filename} already exists, skipping download...")
            filenames.pop()
            filenames.append("Not Found")
            continue

        # res = requests.get(image_url, stream = True)
        # # Save image to image folder
        # # first check if image already exists
        # if res.status_code == 200:
        #     res.raw.decode_content = True
        #     with open(filename, 'wb') as f:
        #         shutil.copyfileobj(res.raw, f)
        # else:
        #     log.warning(f"Failed to download {filename}, with status code {res.status_code}")
        #     filenames.pop()
        #     filenames.append("Not Found")
    return filenames

def get_images(image_urls, base_folder="./data/images/"):
    filenames = []
    log.info("Downloading images")
    for idx, image_url in tqdm(enumerate(image_urls)):
        filename = f"{base_folder}{idx}.png"
        filenames.append(filename)
        if not Path(filename).exists():
            log.debug(f"{filename} already exists, skipping download...")
            filenames.pop()
            filenames.append("Not Found")
            continue

        # res = requests.get(image_url, stream = True)
        # # Save image to image folder
        # # first check if image already exists
        # if res.status_code == 200:
        #     res.raw.decode_content = True
        #     with open(filename, 'wb') as f:
        #         shutil.copyfileobj(res.raw, f)
        # else:
        #     log.warning(f"Failed to download {filename}, with status code {res.status_code}")
        #     filenames.pop()
        #     filenames.append("Not Found")
    return filenames

def load_images(image_filenames):
    images = []
    not_nan_locations = []
    for idx, image_path in tqdm(enumerate(image_filenames)):
        if image_path == "Not Found":
            continue
        not_nan_locations.append(idx)
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        image = cv2.resize(image, (32,32))
        images.append(image)
    return np.array(images), not_nan_locations

def create_cnn(width, height, depth, filters=[32]):
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
    # x = Dense(4)(x)
    # x = Activation("relu")(x)
    # check to see if the regression node should be added
    x = Dense(1, activation="linear")(x)

    # construct the CNN
    model = Model(inputs, x)
    # return the CNN
    return model

def encode_image(image_urls, test_image_urls, rent):
    image_filenames = get_images(image_urls)
    images, not_nan_locations = load_images(image_filenames)


    print(images.shape)
    print(rent.shape)

    orig_rent = rent

    # images = images[not_nan_locations]
    rent = rent.iloc[not_nan_locations]

    print(images.shape)
    print(rent.shape)

    train_data, rem_data, train_labels, rem_labels = train_test_split(images, rent, test_size=0.2, random_state=1001)
    valid_data, test_data, valid_labels, test_labels = train_test_split(rem_data, rem_labels, test_size=0.5, random_state=1001)

    max_price = max(train_labels) if max(train_labels) > max(test_labels) else max(test_labels)
    train_labels /= max_price
    test_labels /= max_price

    model = create_cnn(32, 32, 3)
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=1e-3))
    earlyStoppingCallback = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    # model.summary()
    model.fit(x=train_data, y=train_labels,
              validation_data=(valid_data, valid_labels),
              callbacks=[earlyStoppingCallback],
              epochs=10, batch_size=8, shuffle=True)

    y_pred = model.predict(test_data)

    average_score = mean_absolute_error(test_labels, y_pred) * max_price

    print(f"Mean Absolute Error: {average_score}")

    predicted_rent = model.predict(images)
    average_score = mean_absolute_error(predicted_rent * max_price, rent)

    print(f"Mean Absolute Error: {average_score}")


    # Add rent data to original array
    y_df = pd.DataFrame(image_urls)
    y_df["imageBasedRent"] = np.nan
    y_df["rent"] = orig_rent
    y_df.iloc[not_nan_locations, y_df.columns.get_loc("imageBasedRent")] = predicted_rent * max_price
    print(y_df.head())

    y_df.to_csv("./imageBasedRent.csv")

    test_image_filenames = get_test_images(test_image_urls, base_folder = "./data/test_images/")
    test_images, test_not_nan_locations = load_images(test_image_filenames)
    test_predict = model.predict(test_images)
    test_df = pd.DataFrame(test_image_urls)
    test_df["imageBasedRent"] = np.nan
    test_df.iloc[test_not_nan_locations, test_df.columns.get_loc("imageBasedRent")] = test_predict * max_price
    test_df.to_csv("./test_imageBasedRent.csv")

if __name__=="__main__":
    train_df = load_dataset("./data/train.csv")
    test_df = load_dataset("./data/test.csv")

    train_df = train_df[["coverImageUrl", "rent"]]
    train_df = train_df[(np.abs(stats.zscore(train_df['rent'])) < 2)]

    # train_df[(np.abs(stats.zscore(train_df)) < 3).all(axis=1)]
    encode_image(train_df['coverImageUrl'], test_df['coverImageUrl'], train_df['rent'])
