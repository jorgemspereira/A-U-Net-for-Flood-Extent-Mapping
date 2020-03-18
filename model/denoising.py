import os

import cv2
import numpy as np
import tifffile as tiff
from keras import backend as K, losses
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Conv2D, Flatten
from keras.layers import Dense, Input, Dropout
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.optimizers import Adam
from skimage.draw import random_shapes
from sklearn.model_selection import train_test_split

np.random.seed(12345)
BATCH_SIZE = 40
EPOCHS = 200
WEIGHTS_PATH_DENOISER = "./weights_unet/weights_autoencoder.hdf5"

base_path = '/tmp'
path_train_masks_template = "{}/flood-data/devset_0{}_segmentation_masks/"


def get_files(path_images, min_range=1, max_range=7):
    files_input = []

    for index in range(min_range, max_range):
        complete_path = path_images.format(base_path, index)
        files_input.extend([complete_path + x for x in os.listdir(complete_path) if not x.startswith("._")
                            and x.endswith(".tif")])

    files_input.sort(key=lambda x: x.split("/")[-1])

    return files_input


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def create_auto_encoder():
    inputs = Input(shape=(320, 320, 1), name='encoder_input')
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=2, activation='relu', padding='same')(inputs)
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(x)

    x = Conv2D(filters=32, kernel_size=(3, 3), strides=2, activation='relu', padding='same')(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(x)

    x = Conv2D(filters=32, kernel_size=(3, 3), strides=2, activation='relu', padding='same')(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(x)

    x = Conv2D(filters=32, kernel_size=(3, 3), strides=2, activation='relu', padding='same')(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=2, activation='relu', padding='same')(x)
    x = Dropout(0.5)(x)

    shape = K.int_shape(x)

    latent = Dense(512, name='latent_vector')(Flatten()(x))
    encoder = Model(inputs, latent, name='encoder')

    latent_inputs = Input(shape=(512,), name='decoder_input')
    x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)

    x = Conv2DTranspose(filters=16, kernel_size=(3, 3), strides=2, activation='relu', padding='same')(x)
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(x)

    x = Conv2DTranspose(filters=16, kernel_size=(3, 3), strides=2, activation='relu', padding='same')(x)
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(x)

    x = Conv2DTranspose(filters=16, kernel_size=(3, 3), strides=2, activation='relu', padding='same')(x)
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(x)

    x = Conv2DTranspose(filters=16, kernel_size=(3, 3), strides=2, activation='relu', padding='same')(x)
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(x)

    x = Conv2DTranspose(filters=16, kernel_size=(3, 3), strides=2, activation='relu', padding='same')(x)
    outputs = Conv2D(filters=1, kernel_size=(3, 3), strides=1, activation='sigmoid', padding='same')(x)
    decoder = Model(latent_inputs, outputs, name='decoder')

    auto_encoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
    auto_encoder.compile(loss=dice_coef_loss, optimizer=Adam(1e-3))

    return auto_encoder


def add_random_noise(img):

    # Add random shapes
    transformation = np.random.randint(0, 11)

    if transformation == 0:
        shape = random_shapes((320, 320), min_shapes=1, max_shapes=6, shape='circle', multichannel=False)[0]
        shape = np.where(shape == 255, 0, 1)
        img = np.where((img + shape) >= 1, 1, 0).astype(np.uint8)

    if transformation == 1:
        shape = random_shapes((320, 320), min_shapes=1, max_shapes=6, shape='rectangle', multichannel=False)[0]
        shape = np.where(shape == 255, 0, 1)
        img = np.where((img + shape) >= 1, 1, 0).astype(np.uint8)

    if transformation == 2:
        shape = random_shapes((320, 320), min_shapes=1, max_shapes=6, shape='triangle', multichannel=False)[0]
        shape = np.where(shape == 255, 0, 1)
        img = np.where((img + shape) >= 1, 1, 0).astype(np.uint8)

    if transformation == 3:
        shape = random_shapes((320, 320), min_shapes=1, max_shapes=6, shape='circle', multichannel=False)[0]
        shape = np.where(shape == 255, 0, 1)
        img = np.where((img - shape) <= 0, 0, 1).astype(np.uint8)

    if transformation == 4:
        shape = random_shapes((320, 320), min_shapes=1, max_shapes=6, shape='rectangle', multichannel=False)[0]
        shape = np.where(shape == 255, 0, 1)
        img = np.where((img - shape) <= 0, 0, 1).astype(np.uint8)

    if transformation == 5:
        shape = random_shapes((320, 320), min_shapes=1, max_shapes=6, shape='triangle', multichannel=False)[0]
        shape = np.where(shape == 255, 0, 1)
        img = np.where((img - shape) <= 0, 0, 1).astype(np.uint8)

    # Random morphological operations
    transformation = np.random.randint(0, 7)

    if transformation == 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)

    if transformation == 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)

    if transformation == 2:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)

    if transformation == 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)

    return img


def auto_encoder_generator(path_input, batch_size=5, shuffle=True, random_noise=True):
    ids_file_all = path_input[:]
    while True:
        if len(ids_file_all) < batch_size:
            ids_file_all = path_input[:]
        x, y = list(), list()
        total_patches = 0
        while total_patches < batch_size:
            index = 0
            if shuffle: index = np.random.randint(1, len(ids_file_all) - 1) if len(ids_file_all) < 1 else 0
            img_id = ids_file_all.pop(index)
            img = tiff.imread(img_id)
            x.append(img.reshape((320, 320, 1)))
            if random_noise: img = add_random_noise(img)
            y.append(img.reshape((320, 320, 1)))
            total_patches += 1
        yield (np.array(x), np.array(y))


def get_callbacks():
    return [
        ModelCheckpoint(WEIGHTS_PATH_DENOISER, verbose=1, monitor='val_loss', save_best_only=True),
        ReduceLROnPlateau(factor=0.1, verbose=1, monitor='val_loss', patience=5, min_lr=1e-5),
        EarlyStopping(patience=10, verbose=1, monitor='val_loss')
    ]


def train_denoising():

    y = get_files(path_train_masks_template)
    x_train, x_val = train_test_split(y, test_size=0.15)

    train_gen = auto_encoder_generator(x_train, batch_size=BATCH_SIZE)
    val_gen = auto_encoder_generator(x_val, batch_size=1, random_noise=False)

    auto_encoder = create_auto_encoder()
    auto_encoder.fit_generator(train_gen,
                               steps_per_epoch=(len(x_train) // BATCH_SIZE),
                               validation_data=val_gen,
                               validation_steps=len(x_val),
                               epochs=EPOCHS,
                               callbacks=get_callbacks())
