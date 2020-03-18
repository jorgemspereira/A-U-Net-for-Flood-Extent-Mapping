import argparse
import os.path
from itertools import product

import imageio
import numpy as np
import tensorflow as tf
import tifffile as tiff
import pydensecrf.densecrf as dcrf
import keras
from keras import Model
from keras import backend as K
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.engine.saving import load_model
from keras.optimizers import Adam, SGD
from keras.preprocessing import image
from numpy.random import seed
from patchify import patchify
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import set_random_seed
from tqdm import tqdm
from arguments.arguments import Mode, NumberChannels
from generator.generator import image_generator
from helpers.gen_patches import gen_patches
from helpers.pre_process import pre_process
from model.unet_model import unet_model
from model.mish import Mish, mish, Swish, swish
from model.adamod import AdaMod
from model.swa import SWA

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# for binary classification
N_CLASSES = 1

N_EPOCHS = 200
SEED = 12345

ORIGI_SZ = 320

PATCH_SZ = 128 # Consider changing in prediction/training to 128/256
STEP_SZ = 16 # Consider changing in for prediction/training to 16/4
BATCH_SIZE = 12

WEIGHTS_PATH = 'weights_unet'

if not os.path.exists(WEIGHTS_PATH): os.makedirs(WEIGHTS_PATH)

seed(SEED)
set_random_seed(SEED)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

base_path = '/tmp'

path_train_images_template = '{}/dataset/devset_0{}_satellite_images_patches/'
path_train_masks_template = "{}/dataset/devset_0{}_segmentation_masks_patches/"

path_test_images_template = '{}/dataset/testset_0{}_satellite_images/'
path_test_masks_template = "{}/flood-data/testset_0{}_segmentation_masks/"

def update_weights_path(args):
    global WEIGHTS_PATH
    WEIGHTS_PATH += '/weights_complete_{}_augmented_channels.hdf5'.format(args['channels'].value)

def get_files(path_images, path_masks, min_range=1, max_range=7):
    files_input, masks_input = [], []
    for index in range(min_range, max_range):
        complete_path = path_images.format(base_path, index)
        complete_path_masks = path_masks.format(base_path, index)
        files_input.extend([complete_path + x for x in os.listdir(complete_path) if not x.startswith("._")])
        masks_input.extend([complete_path_masks + x for x in os.listdir(complete_path_masks) if not x.startswith("._") and x.endswith(".tif")])
    masks_input.sort(key=lambda x: x.split("/")[-1])
    files_input.sort(key=lambda x: x.split("/")[-1])
    return files_input, masks_input

def get_callbacks():
    return [
         #SWA(start_epoch=100, batch_size=BATCH_SIZE),
         ModelCheckpoint(WEIGHTS_PATH, verbose=1, monitor='val_loss', save_best_only=True),
         ReduceLROnPlateau(factor=0.1, verbose=1, monitor='val_loss', patience=5, min_lr=1e-5),
         EarlyStopping(patience=10, verbose=1, monitor='val_loss')
    ]

def calculate_weights(x_train):
    result = []
    for index in range(1, 7): result.append(sum([1 for el in x_train if "devset_0{}".format(index) in el]))
    result = [((x * 100) / sum(result)) for x in result]
    result = [(100 - x) / 100 for x in result]
    result = result + (1 - np.min(result))
    return result

def generate_stratified_validation(x, y, validation_size=0.15):
    masks = [tiff.imread(file)[:, :, 0] for file in y]
    coverage = [np.sum(mask) / pow(ORIGI_SZ, 2) for mask in masks]
    def cov_to_class(val):
        for i in range(0, 11):
            if val * 10 <= i:
                return i
    coverage_class = [cov_to_class(val) for val in coverage]
    return train_test_split(x, y, test_size=validation_size, stratify=coverage_class, random_state=SEED)

def dice_coefficient(y_true, y_pred, smooth=1e-9):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def custom_loss(y_true, y_pred):
    #dice_coefficient_loss = 1.0 - dice_coefficient(y_true, y_pred)
    weight = K.reshape(y_pred[:, :, :, 1], (BATCH_SIZE, PATCH_SZ, PATCH_SZ, 1))
    y_pred = K.reshape(y_pred[:, :, :, 0], (BATCH_SIZE, PATCH_SZ, PATCH_SZ, 1))
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    y_pred_logit = K.log(y_pred / (1.0 - y_pred))
    loss = (1.0 - y_true) * y_pred_logit + (1.0 + (weight - 1.) * y_true) * (K.log(1.0 + K.exp(-K.abs(y_pred_logit))) + K.maximum(-y_pred_logit, 0.0))
    return (K.sum(loss) / K.sum(weight)) #+ (0.1 * dice_coefficient_loss)

def train_net(args):
    x, y = get_files(path_train_images_template, path_train_masks_template)
    x_train, x_val, y_train, y_val = generate_stratified_validation(x, y, validation_size=0.15)
    train_steps = len(x_train) // BATCH_SIZE
    validation_steps = len(x_val) // BATCH_SIZE
    weights = calculate_weights(x_train)
    train_gen = image_generator(x_train, y_train, PATCH_SZ, weights, batch_size=BATCH_SIZE, random_transformation=True)
    val_gen = image_generator(x_val, y_val, PATCH_SZ, batch_size=BATCH_SIZE, shuffle=False, random_transformation=False)
    train_gen2 = image_generator(x_train + x_val, y_train + y_val, PATCH_SZ, weights, batch_size=BATCH_SIZE, random_transformation=True)
    val_gen2 = image_generator(x_train + x_val, y_train + y_val, PATCH_SZ, batch_size=BATCH_SIZE, shuffle=False, random_transformation=True)
    model = unet_model(n_classes=N_CLASSES, init_seed=SEED, im_sz=PATCH_SZ, n_channels=args['channels'].value)
    model.compile(optimizer=Adam(lr=1e-3), loss=custom_loss)
    model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=N_EPOCHS, validation_data=val_gen, validation_steps=validation_steps, verbose=1, shuffle=True, callbacks=get_callbacks())
    return model

def get_model(args):
    if args['mode'] == Mode.train: train_net(args)
    old_model = load_model(WEIGHTS_PATH, custom_objects={"dice_coefficient": dice_coefficient, "custom_loss": custom_loss, 'Mish': Mish(mish), 'Swish': Swish(), 'AdaMod': AdaMod })
    model = unet_model(n_classes=N_CLASSES, init_seed=SEED, im_sz=PATCH_SZ, n_channels=args['channels'].value)
    model.compile(optimizer=Adam(lr=1e-3), loss=custom_loss)
    for layer in model.layers: layer.set_weights(old_model.get_layer(name=layer.name).get_weights())
    input_layer = model.get_layer("input_layer")
    output_layer = model.get_layer("output_layer")
    return Model(inputs=[input_layer.input], outputs=[output_layer.output])

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

def reconstruct_patches(patches, image_size, step, weighted_average=False):
    i_h, i_w = image_size[:2]
    p_h, p_w = patches.shape[1:3]
    img = np.zeros(image_size)
    patch_count = np.zeros(image_size)
    n_h = int((i_h - p_h) / step + 1)
    n_w = int((i_w - p_w) / step + 1)
    for p, (i, j) in zip(patches, product(range(n_h), range(n_w))):
        if weighted_average:
           x, y = np.meshgrid(np.linspace(-1.0,1.0,p_w), np.linspace(-1.0,1.0,p_h))
           d = np.sqrt(x*x+y*y)
           sigma, mu = 1.0, 0.0
           gaussianmask = np.expand_dims( 1.0 - np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) ), 2)
           img[i * step:i * step + p_h, j * step:j * step + p_w] += p * gaussianmask
           patch_count[i * step:i * step + p_h, j * step:j * step + p_w] += gaussianmask
        else:
           img[i * step:i * step + p_h, j * step:j * step + p_w] += p
           patch_count[i * step:i * step + p_h, j * step:j * step + p_w] += 1.0
    return img / patch_count

def predict_aux(args, x, model, cubed=False, weighted=False):
    dim_x, dim_y, dim = x.shape
    patches = patchify(x, (PATCH_SZ, PATCH_SZ, args['channels'].value), step=STEP_SZ)
    width_window, height_window, z, width_x, height_y, num_channel = patches.shape
    patches = np.reshape(patches, (width_window * height_window,  width_x, height_y, num_channel))
    predictions = model.predict(patches, batch_size=1)
    if cubed:
       for n, p in enumerate(patches):
          predictions[n] = np.power(predictions[n], 0.5)
          predictions[n] = predictions[n] / ( predictions[n] + np.power(1.0 - predictions[n], 0.5) )
    return reconstruct_patches(predictions, (dim_x, dim_y, N_CLASSES), STEP_SZ, weighted)

def post_processing(img, output_probs):
    h = output_probs.shape[0]
    w = output_probs.shape[1]
    output_probs = np.expand_dims(output_probs, 0)
    output_probs = np.append(1 - output_probs, output_probs, axis=0)
    d = dcrf.DenseCRF2D(w, h, 2)
    U = -np.log(output_probs + 1e-8)
    U = U.reshape((2, -1))
    U = np.ascontiguousarray(U)
    img = np.ascontiguousarray(img) * 256.0
    d.setUnaryEnergy(U.astype(np.float32))
    d.addPairwiseGaussian(sxy=1e8, compat=np.array([1e-8,1e-8]).astype(np.float32))
    d.addPairwiseBilateral(sxy=150, srgb=1e8, rgbim=img.astype(np.uint8), compat=np.array([1e-8,1e-8]).astype(np.float32))
    Q = d.inference(10)
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w))
    return Q
     
def calculate_results_aux(args, model, x, y, dcrf=False, cubed=False, weighted=False):
    tps, tns, fns, fps = 0, 0, 0, 0
    for index in tqdm(range(0, len(x))):
        pred = predict_aux(args, tiff.imread(x[index]), model, cubed, weighted)
        name_image = y[index].split("_")[-1].split(".")[0]
        if dcrf: result = post_processing(tiff.imread(x[index])[:, :, 0:3], pred)
        else: result = np.where(pred.reshape((ORIGI_SZ, ORIGI_SZ)) < 0.5, 0, 1)
        mask = imageio.imread(y[index])
        mask = np.where(mask == 255, 1, 0) if np.any(mask == 255) else mask
        tn, fp, fn, tp = confusion_matrix(mask.ravel(), result.ravel(), labels=[0, 1]).ravel()
        fps += fp; fns += fn; tps += tp; tns += tn
        img = image.array_to_img(picture_from_mask(mask))
        img.save('./results/{}_original.jpg'.format(name_image))
        img = image.array_to_img(picture_from_mask(result))
        img.save('./results/{}_predicted.jpg'.format(name_image))
    return (tps + tns) / (tps + tns + fps + fns), tps / (tps + fns + fps)

def calculate_results(args, model):
    print("Calculating results...")
    x, y = get_files(path_test_images_template, path_test_masks_template)
    accuracy, iou = calculate_results_aux(args, model, x, y)
    print("Same locations --------> Acc: {} | IOU: {}".format(accuracy, iou))
    accuracy, iou = calculate_results_aux(args, model, x, y, cubed=True)
    print("Same locations with cubed probability estimates --------> Acc: {} | IOU: {}".format(accuracy, iou))
    accuracy, iou = calculate_results_aux(args, model, x, y, cubed=True, weighted=True)
    print("Same locations with cubed and weighted probability estimates --------> Acc: {} | IOU: {}".format(accuracy, iou))    
    accuracy, iou = calculate_results_aux(args, model, x, y, dcrf=True, cubed=True, weighted=True)
    print("Same locations with CRF post-processing --------> Acc: {} | IOU: {}".format(accuracy, iou))
    x, y = get_files(path_test_images_template, path_test_masks_template, min_range=7, max_range=8)
    accuracy, iou = calculate_results_aux(args, model, x, y)
    print("Different locations ---> Acc: {} | IOU: {}".format(accuracy, iou))
    accuracy, iou = calculate_results_aux(args, model, x, y, cubed=True)
    print("Different locations with cubed probability estimates ---> Acc: {} | IOU: {}".format(accuracy, iou))
    accuracy, iou = calculate_results_aux(args, model, x, y, cubed=True, weighted=True)
    print("Different locations with cubed and weighted probability estimates ---> Acc: {} | IOU: {}".format(accuracy, iou))
    accuracy, iou = calculate_results_aux(args, model, x, y, dcrf=True, cubed=True, weighted=True)
    print("Different locations with CRF post-processing ---> Acc: {} | IOU: {}".format(accuracy, iou))

# noinspection PyTypeChecker
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", dest='mode', choices=list(Mode), type=Mode.from_string, default=Mode.train)
    parser.add_argument("--channels", dest='channels', choices=list(NumberChannels), type=NumberChannels.from_string, default=NumberChannels.four)
    return vars(parser.parse_args())

def main():
    args = parse_args()
    #pre_process(args, base_path)
    #gen_patches(args, PATCH_SZ, STEP_SZ, base_path)
    update_weights_path(args)
    calculate_results(args, get_model(args))

if __name__ == '__main__':
    main()