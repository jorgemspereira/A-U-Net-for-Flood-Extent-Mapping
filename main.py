import argparse
import os.path
from itertools import product

import imageio
import numpy as np
import tensorflow as tf
import tifffile as tiff
from keras import losses
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.engine.saving import load_model
from keras.optimizers import Adam
from keras.preprocessing import image
from numpy.random import seed
from patchify import patchify
from segmentation_models.metrics import iou_score
from sklearn.metrics import jaccard_similarity_score, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow import set_random_seed
from tqdm import tqdm

from arguments.arguments import Mode
from generator.generator import image_generator
from losses.lovasz_losses_tf import keras_lovasz_hinge
from model.unet_model import unet_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# for binary classification
N_CLASSES = 1

N_BANDS = 4
N_EPOCHS = 100
SEED = 1234

ORIGI_SZ = 320
PATCH_SZ = 128
STEP_SZ = 16
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

path_train_images_template = '{}/dataset/devset_0{}_satellite_images_patches/'
path_train_masks_template = "{}/dataset/devset_0{}_segmentation_masks_patches/"

path_test_images_template = '{}/dataset/testset_0{}_satellite_images/'
path_test_masks_template = "{}/flood-data/testset_0{}_segmentation_masks/"


def get_files(path_images, path_masks, min_range=1, max_range=7):
    files_input, masks_input = [], []

    for index in range(min_range, max_range):
        complete_path = path_images.format(base_path, index)
        complete_path_masks = path_masks.format(base_path, index)

        files_input.extend([complete_path + x for x in os.listdir(complete_path) if not x.startswith("._")])
        masks_input.extend([complete_path_masks + x for x in os.listdir(complete_path_masks) if not x.startswith("._")
                            and x.endswith(".png")])

    masks_input.sort(key=lambda x: x.split("/")[-1])
    files_input.sort(key=lambda x: x.split("/")[-1])

    return files_input, masks_input


def get_callbacks():
    return [
        EarlyStopping(monitor='val_loss', restore_best_weights=True, verbose=1, patience=5, mode='auto'),
        ModelCheckpoint(WEIGHTS_PATH, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, mode='auto')
    ]


def custom_loss(y_true, y_pred):
    return keras_lovasz_hinge(y_true, y_pred) + losses.binary_crossentropy(y_true, y_pred)


def calculate_weights(x_train):
    result = []
    for index in range(1, 7):
        result.append(sum([1 for el in x_train if "devset_0{}".format(index) in el]))

    result = [((x * 100) / sum(result)) for x in result]
    result = [(100 - x) / 100 for x in result]

    return result


def train_net():
    x, y = get_files(path_train_images_template, path_train_masks_template)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=SEED)

    train_steps = len(x_train) // BATCH_SIZE
    validation_steps = len(x_val) // BATCH_SIZE
    weights = calculate_weights(x_train)

    train_gen = image_generator(x_train, y_train, PATCH_SZ, weights, batch_size=BATCH_SIZE, random_transformation=True)
    val_gen = image_generator(x_val, y_val, PATCH_SZ, batch_size=BATCH_SIZE, shuffle=False, random_transformation=False)

    model = unet_model(n_classes=N_CLASSES, init_seed=SEED, im_sz=PATCH_SZ, n_channels=N_BANDS)
    model.compile(optimizer=Adam(lr=1e-4), loss=custom_loss, metrics=[iou_score, "accuracy"])
    model.fit_generator(train_gen,
                        steps_per_epoch=train_steps,
                        epochs=N_EPOCHS,
                        validation_data=val_gen,
                        validation_steps=validation_steps,
                        verbose=1, shuffle=True,
                        callbacks=get_callbacks())
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


def reconstruct_patches(patches, image_size, step):
    i_h, i_w = image_size[:2]
    p_h, p_w = patches.shape[1:3]

    img = np.zeros(image_size)
    patch_count = np.zeros(image_size)

    n_h = int((i_h - p_h) / step + 1)
    n_w = int((i_w - p_w) / step + 1)
    for p, (i, j) in zip(patches, product(range(n_h), range(n_w))):
        img[i * step:i * step + p_h, j * step:j * step + p_w] += p
        patch_count[i * step:i * step + p_h, j * step:j * step + p_w] += 1

    return img / patch_count


def predict_aux(x, model):
    dim_x, dim_y, dim = x.shape
    patches = patchify(x, (PATCH_SZ, PATCH_SZ, N_BANDS), step=STEP_SZ)
    width_window, height_window, z, width_x, height_y, num_channel = patches.shape
    patches = np.reshape(patches, (width_window * height_window,  width_x, height_y, num_channel))
    predictions = model.predict(patches, batch_size=20)
    return reconstruct_patches(predictions, (dim_x, dim_y, N_CLASSES), STEP_SZ)


def calculate_results_aux(model, x, y):
    results_iou, results_acc = [], []
    for index in tqdm(range(0, len(x))):
        pred = predict_aux(tiff.imread(x[index]), model)
        result = np.where(pred.reshape((ORIGI_SZ, ORIGI_SZ)) < 0.5, 0, 1)

        mask = imageio.imread(y[index])
        name_image = y[index].split("_")[-1].split(".")[0]
        score_iou = jaccard_similarity_score(mask, result)
        score_acc = accuracy_score(mask.ravel(), result.ravel())

        img = image.array_to_img(picture_from_mask(mask))
        img.save('./results/{}_original.jpg'.format(name_image))

        img = image.array_to_img(picture_from_mask(result))
        img.save('./results/{}_predicted.jpg'.format(name_image))

        # location = y[index].find("testset_0")
        # location = int(y[index][location + 9:location + 10])

        # try:
        #     results_iou[location] += [score_iou]
        # except KeyError:
        #     results_iou[location] = [score_iou]
        #
        # try:
        #     results_acc[location] += [score_acc]
        # except KeyError:
        #     results_acc[location] = [score_acc]

        results_iou.append(score_iou)
        results_acc.append(score_acc)

    # results_acc = [np.mean(value) for value in results_acc.values()]
    # results_iou = [np.mean(value) for value in results_iou.values()]

    return np.mean(results_acc), np.mean(results_iou)


def calculate_results(model):
    print("Calculating results...")

    x, y = get_files(path_test_images_template, path_test_masks_template)
    accuracy, iou = calculate_results_aux(model, x, y)
    print("Same locations --------> Acc: {} | IOU: {}".format(accuracy, iou))

    x, y = get_files(path_test_images_template, path_test_masks_template, min_range=7, max_range=8)
    accuracy, iou = calculate_results_aux(model, x, y)
    print("Different locations ---> Acc: {} | IOU: {}".format(accuracy, iou))


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
