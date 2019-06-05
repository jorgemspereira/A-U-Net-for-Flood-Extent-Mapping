import argparse
import os.path

import imageio
import numpy as np
import pydensecrf.densecrf as dcrf
import tensorflow as tf
import tifffile as tiff
from keras import losses
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.engine.saving import load_model
from keras.optimizers import Adam
from keras.preprocessing import image
from numpy.random import seed
from segmentation_models.metrics import iou_score
from sklearn.metrics import jaccard_similarity_score
from sklearn.model_selection import train_test_split
from tensorflow import set_random_seed

from arguments.arguments import Mode
from generator.generator import image_generator
from losses.lovasz_losses_tf import keras_lovasz_hinge
from model.unet_model import unet_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# for binary classification
N_CLASSES = 1

N_BANDS = 9
N_EPOCHS = 100
SEED = 1234

PATCH_SZ = 320
BATCH_SIZE = 16

WEIGHTS_PATH = 'weights_unet'

if not os.path.exists(WEIGHTS_PATH):
    os.makedirs(WEIGHTS_PATH)
WEIGHTS_PATH += '/weights.hdf5'

seed(SEED)
set_random_seed(SEED)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

base_path = '/home/jpereira/A-U-Net-Model-Leveraging-Multiple-Remote-Sensing-Data-Sources-for-Flood-Extent-Mapping'

path_train_images_template = '{}/dataset/devset_0{}_satellite_images/'
path_test_images_template = '{}/dataset/testset_0{}_satellite_images/'
path_original_test_images_template = '{}/flood-data/testset_0{}_satellite_images/'

path_train_masks_template = "{}/flood-data/devset_0{}_segmentation_masks/"
path_test_masks_template = "{}/flood-data/testset_0{}_segmentation_masks/"


def get_files(path_images, path_masks, min_range=1, max_range=7):
    files_input, masks_input = [], []

    for index in range(min_range, max_range):
        complete_path = path_images.format(base_path, index)
        complete_path_masks = path_masks.format(base_path, index)

        files_input.extend([complete_path + x for x in os.listdir(complete_path) if not x.startswith("._")])
        masks_input.extend([complete_path_masks + x for x in os.listdir(complete_path_masks) if not x.startswith("._")
                            and x.endswith(".png")])

    masks_input.sort(key=lambda x: x.split("_")[-1])
    files_input.sort(key=lambda x: x.split("/")[-1])

    return files_input, masks_input


def get_original_images(path, min_range=1, max_range=7):
    files = []

    for index in range(min_range, max_range):
        complete_path = path.format(base_path, index)
        files.extend([complete_path + x for x in os.listdir(complete_path) if not x.startswith("._")])

    files.sort(key=lambda x: x.split("/")[-1])

    return files


def get_callbacks(steps):
    return [
        EarlyStopping(monitor='val_loss', restore_best_weights=True, verbose=1, patience=10, mode='auto'),
        ModelCheckpoint(WEIGHTS_PATH, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, mode='auto')
        # CyclicLR(base_lr=1e-5, max_lr=1e-4, step_size=steps * 8, mode='triangular2')
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

    model = unet_model(n_classes=N_CLASSES, init_seed=SEED, im_sz=PATCH_SZ, n_channels=N_BANDS)
    model.compile(optimizer=Adam(lr=1e-3), loss=custom_loss, metrics=[iou_score, "accuracy"])
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
    return load_model(WEIGHTS_PATH, {'iou_score': iou_score, 'custom_loss': custom_loss})


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


def post_processing(img, output_probs):
    h = output_probs.shape[0]
    w = output_probs.shape[1]

    output_probs = np.expand_dims(output_probs, 0)
    output_probs = np.append(1 - output_probs, output_probs, axis=0)

    d = dcrf.DenseCRF2D(w, h, 2)
    U = -np.log(output_probs)
    U = U.reshape((2, -1))
    U = np.ascontiguousarray(U)
    img = np.ascontiguousarray(img)

    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=2, compat=3)
    d.addPairwiseBilateral(sxy=10, srgb=60, rgbim=img.astype(np.uint8), compat=10)

    Q = d.inference(5)
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w))

    return Q


def calculate_results_aux(model, x, y, original_images, dcrf=False):
    test_generator = image_generator(x, y, batch_size=1, shuffle=False, random_transformation=False)
    predictions = model.predict_generator(test_generator, steps=len(x), verbose=1)

    results = []
    for index, pred in enumerate(predictions):
        original = imageio.imread(y[index])

        if dcrf:
            original_image = tiff.imread(original_images[index])
            original_image = original_image[:, :, 0:3]
            result = post_processing(original_image, pred)
        else:
            result = np.where(pred.reshape((PATCH_SZ, PATCH_SZ)) < 0.5, 0, 1)

        name_image = y[index].split("_")[-1].split(".")[0]
        score = jaccard_similarity_score(original, result)

        img = image.array_to_img(picture_from_mask(original))
        img.save('./results/{}_original.jpg'.format(name_image))

        img = image.array_to_img(picture_from_mask(result))
        img.save('./results/{}_predicted.jpg'.format(name_image))

        # print("{} -> {}".format(y[index], score))
        results.append(score)

    return np.mean(results)


def calculate_results(model):
    print("Calculating results...")

    original_images = get_original_images(path_original_test_images_template)
    x, y = get_files(path_test_images_template, path_test_masks_template)
    print("Same locations --------> {}".format(calculate_results_aux(model, x, y, original_images, dcrf=True)))

    original_images = get_original_images(path_original_test_images_template, min_range=7, max_range=8)
    x, y = get_files(path_test_images_template, path_test_masks_template, min_range=7, max_range=8)
    print("Different locations ---> {}".format(calculate_results_aux(model, x, y, original_images, dcrf=True)))


# noinspection PyTypeChecker
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", dest='mode', choices=list(Mode), type=Mode.from_string, default=Mode.train)
    return vars(parser.parse_args())


def main():
    args = parse_args()
    model = get_model(args)
    calculate_results(model)


if __name__ == '__main__':
    main()
