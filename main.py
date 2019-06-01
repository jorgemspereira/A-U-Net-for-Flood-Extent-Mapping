import argparse
import os.path

import imageio
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.engine.saving import load_model
from keras.optimizers import Adam
from keras.preprocessing import image
from keras import backend as K, losses
from sklearn.metrics import jaccard_similarity_score
from sklearn.model_selection import train_test_split

from arguments.arguments import Mode
from generator.generator import image_generator
from losses.lovasz_losses_tf import keras_lovasz_hinge
from model.unet_model import unet_model
from callbacks.clr_callback import CyclicLR

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

base_path = '/home/jpereira/A-U-Net-Model-Leveraging-Multiple-Remote-Sensing-Data-Sources-for-Flood-Extent-Mapping'

path_train_images_template = '{}/dataset/devset_0{}_satellite_images/'
path_test_images_template = '{}/dataset/testset_0{}_satellite_images/'

path_train_masks_template = "{}/flood-data/devset_0{}_segmentation_masks/"
path_test_masks_template = "{}/flood-data/testset_0{}_segmentation_masks/"


# for binary classification
N_CLASSES = 1

N_BANDS = 4
N_EPOCHS = 100
SEED = 20

PATCH_SZ = 320
BATCH_SIZE = 16

WEIGHTS_PATH = 'weights_unet'

if not os.path.exists(WEIGHTS_PATH):
    os.makedirs(WEIGHTS_PATH)
WEIGHTS_PATH += '/weights.hdf5'


def get_files(path_images, path_masks):
    files_input, masks_input = [], []

    for index in range(1, 7):
        complete_path = path_images.format(base_path, index)
        complete_path_masks = path_masks.format(base_path, index)

        files_input.extend([complete_path + x for x in os.listdir(complete_path) if not x.startswith("._")])
        masks_input.extend([complete_path_masks + x for x in os.listdir(complete_path_masks) if not x.startswith("._")
                            and x.endswith(".png")])

    masks_input.sort(key=lambda x: x.split("_")[-1])
    files_input.sort(key=lambda x: x.split("/")[-1])

    return files_input, masks_input


def get_callbacks(steps):
    return [
        EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=10, mode='auto'),
        ModelCheckpoint(WEIGHTS_PATH, monitor='val_loss', save_best_only=True, mode='auto'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, mode='auto')
        # CyclicLR(base_lr=1e-4, max_lr=1e-3, step_size=steps * 4, mode='triangular2')
    ]


def custom_loss(y_true, y_pred):
    return keras_lovasz_hinge(y_true, y_pred) + losses.binary_crossentropy(y_true, y_pred)


def train_net():
    x, y = get_files(path_train_images_template, path_train_masks_template)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=SEED)

    train_steps = len(x_train) // BATCH_SIZE
    validation_steps = len(x_val) // BATCH_SIZE

    train_gen = image_generator(x_train, y_train, batch_size=BATCH_SIZE, random_transformation=True)
    val_gen = image_generator(x_val, y_val, batch_size=BATCH_SIZE, shuffle=False, random_transformation=False)

    model = unet_model(N_CLASSES, PATCH_SZ, n_channels=N_BANDS)
    model.compile(optimizer=Adam(lr=1e-3), loss=custom_loss, metrics=["accuracy"])

    model.fit_generator(train_gen,
                        steps_per_epoch=train_steps,
                        epochs=N_EPOCHS,
                        validation_data=val_gen,
                        validation_steps=validation_steps,
                        verbose=1, shuffle=True,
                        callbacks=get_callbacks(train_steps))
    return model


def get_model(args):
    if args['mode'] == Mode.train: return train_net()
    return load_model(WEIGHTS_PATH, {'custom_loss': custom_loss})


def picture_from_mask(mask):
    colors = {
        0: [255, 255, 255],
        1: [0, 0, 255]
    }
    pict = np.empty(shape=(3, mask.shape[0], mask.shape[1]))
    for cl in range(len(colors)):
        for ch in range(3):
            pict[ch, :, :] = np.where(mask == cl, colors[cl][ch], pict[ch, :, :])
    return np.moveaxis(pict, 0, -1)


def calculate_results(model):
    print("Calculating results...")

    x, y = get_files(path_test_images_template, path_test_masks_template)
    test_generator = image_generator(x, y, batch_size=1, shuffle=False, random_transformation=False)
    predictions = model.predict_generator(test_generator, steps=len(x), verbose=1)

    results = []
    for index, pred in enumerate(predictions):
        result = np.where(pred.reshape((320, 320)) < 0.5, 0, 1)
        original = imageio.imread(y[index])

        name_image = y[index].split("_")[-1].split(".")[0]
        score = jaccard_similarity_score(original, result)

        img = image.array_to_img(picture_from_mask(original))
        img.save('./results/{}_original.jpg'.format(name_image))

        img = image.array_to_img(picture_from_mask(result))
        img.save('./results/{}_predicted.jpg'.format(name_image))

        # print("{} -> {}".format(y[index], score))
        results.append(score)

    return np.mean(results)


# noinspection PyTypeChecker
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", dest='mode', choices=list(Mode), type=Mode.from_string, default=Mode.train)
    return vars(parser.parse_args())


def main():
    args = parse_args()
    model = get_model(args)
    print(calculate_results(model))


if __name__ == '__main__':
    main()
