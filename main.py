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
from sklearn.isotonic import IsotonicRegression
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
from model.se import AttentiveNormalization
from model.evonorm import EvoNormS0

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# for binary classification
N_CLASSES = 1

N_EPOCHS = 200
SEED = 12345

ORIGI_SZ = 320

PATCH_SZ = 128 # Consider changing in prediction/training to 128/256
STEP_SZ = 16 # Consider changing in for prediction/training to 16/1
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
#    WEIGHTS_PATH += '/weights_complete_{}_augmented_channels.hdf5'.format(args['channels'].value)
    WEIGHTS_PATH += '/weights_complete_{}_channels.hdf5'.format(args['channels'].value)
    
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

def tversky_loss(y_true, y_pred, beta=0.5):
  numerator = tf.reduce_sum(y_true * y_pred, axis=-1)
  denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)
  return 1.0 - (numerator + 1) / (tf.reduce_sum(denominator, axis=-1) + 1)

def custom_loss(y_true, y_pred):
    dice_coefficient_loss = tversky_loss(y_true, y_pred)
    weight = K.reshape(y_pred[:, :, :, 1], (BATCH_SIZE, K.shape(y_pred)[1], K.shape(y_pred)[2], 1))
    y_pred = K.reshape(y_pred[:, :, :, 0], (BATCH_SIZE, K.shape(y_pred)[1], K.shape(y_pred)[2], 1))
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    y_pred_logit = K.log(y_pred / (1.0 - y_pred))
    loss = (1.0 - y_true) * y_pred_logit + (1.0 + (weight - 1.) * y_true) * (K.log(1.0 + K.exp(-K.abs(y_pred_logit))) + K.maximum(-y_pred_logit, 0.0))
    return (K.sum(loss) / K.sum(weight)) #+ (0.25 * dice_coefficient_loss)

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
    if args['mode'] == Mode.train: 
      model = train_net(args)
    else: 
      old_model = load_model(WEIGHTS_PATH, custom_objects={'EvoNormS0': EvoNormS0, 'custom_loss': custom_loss, 'Mish': Mish(mish), 'Swish': Swish(), 'AdaMod': AdaMod, 'AttentiveNormalization': AttentiveNormalization })
      model = unet_model(n_classes=N_CLASSES, init_seed=SEED, im_sz=PATCH_SZ, n_channels=args['channels'].value)
      model.compile(optimizer=Adam(lr=1e-3), loss=custom_loss)
      for layer in model.layers: layer.set_weights(old_model.get_layer(name=layer.name).get_weights())
    input_layer = model.get_layer("input_layer")
    output_layer = model.get_layer("output_layer")
    model = Model(inputs=[input_layer.input], outputs=[output_layer.output])
    input_layer = old_model.get_layer("input_layer")
    output_layer = old_model.get_layer("output_layer")    
    old_model = Model(inputs=[input_layer.input], outputs=[output_layer.output])
    # Train calibration model    
    #x, y = get_files(path_train_images_template, path_train_masks_template)
    #x_train, x_val, y_train, y_val = generate_stratified_validation(x, y, validation_size=0.9999)
    #val_gen = image_generator(x_val, y_val, PATCH_SZ, batch_size=1, shuffle=False, random_transformation=False)
    #act = []
    #for i, j in enumerate(val_gen):
    #  act.append(j[1])
    #  if i == len(x_val) - 1: break
    #val_gen = image_generator(x_val, y_val, PATCH_SZ, batch_size=1, shuffle=False, random_transformation=False,  include_seg_mask=False)
    #calibration = IsotonicRegression()
    #calibration.fit(old_model.predict_generator(val_gen, len(x_val)).flatten(), np.array(act).flatten())
    calibration = None
    return model, calibration

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

def reconstruct_patches(patches_list, image_size, step, weighted_average=False):
    i_h, i_w = image_size[:2]
    p_h, p_w = patches_list[0].shape[1:3]
    img = np.zeros(image_size)
    patch_count = np.zeros(image_size)
    n_h = int((i_h - p_h) / step + 1)
    n_w = int((i_w - p_w) / step + 1)
    for patches in patches_list:
      for p, (i, j) in zip(patches, product(range(n_h), range(n_w))):
        if weighted_average:
           x, y = np.meshgrid(np.linspace(-1.0,1.0,p_w), np.linspace(-1.0,1.0,p_h))
           d = np.sqrt(x*x+y*y)
           sigma, mu = 0.5, 0.0
           gaussianmask = np.expand_dims( 1.0 - np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) ), 2)
           img[i * step:i * step + p_h, j * step:j * step + p_w] += p * gaussianmask
           patch_count[i * step:i * step + p_h, j * step:j * step + p_w] += gaussianmask
        else:
           img[i * step:i * step + p_h, j * step:j * step + p_w] += p
           patch_count[i * step:i * step + p_h, j * step:j * step + p_w] += 1.0
    return img / patch_count

def predict_aux(args, x, model, calibrate=None):
    dim_x, dim_y, dim = x.shape
    patches = patchify(x, (PATCH_SZ, PATCH_SZ, args['channels'].value), step=STEP_SZ)
    width_window, height_window, z, width_x, height_y, num_channel = patches.shape
    patches = np.reshape(patches, (width_window * height_window,  width_x, height_y, num_channel))
    predictions = model.predict(patches, batch_size=1)    
    rec1 = reconstruct_patches( [predictions] , (dim_x, dim_y, N_CLASSES), STEP_SZ, False)    
    for n, p in enumerate(patches):
       if not(calibrate is None): predictions[n] = np.reshape( calibrate.predict( predictions[n].flatten() ) , predictions[n].shape )
       else:
         predictions[n] = np.power(predictions[n], 0.5)
         predictions[n] = predictions[n] / ( predictions[n] + np.power(1.0 - predictions[n], 0.5) )
    rec2 = reconstruct_patches( [predictions] , (dim_x, dim_y, N_CLASSES), STEP_SZ, False)    
    newpatches = patches
    newpredictions = predictions
    for n, p in enumerate(newpatches): newpatches[n] = np.transpose(p, axes=(1, 0, 2))
    newpredictions = model.predict(newpatches, batch_size=1)
    for n, p in enumerate(newpatches):
       if not(calibrate is None): newpredictions[n] = np.reshape( calibrate.predict( newpredictions[n].flatten() ) , newpredictions[n].shape )
       else:
         newpredictions[n] = np.power(newpredictions[n], 0.5)
         newpredictions[n] = newpredictions[n] / ( newpredictions[n] + np.power(1.0 - newpredictions[n], 0.5) )
       newpredictions[n] = np.transpose(newpredictions[n], axes=(1, 0, 2))
    rec3 = reconstruct_patches( [ predictions, newpredictions ] , (dim_x, dim_y, N_CLASSES), STEP_SZ, False)
    
    #newpredictions2 = predictions
    #for n, p in enumerate(newpatches): newpatches[n] = p[::-1, :, :]
    #newpredictions2 = model.predict(newpatches, batch_size=1)
    #for n, p in enumerate(newpatches):
    #   newpredictions2[n] = np.power(newpredictions2[n], 0.5)
    #   newpredictions2[n] = newpredictions2[n] / ( newpredictions2[n] + np.power(1.0 - newpredictions2[n], 0.5) )
    #   newpredictions2[n] = newpredictions2[n][::-1, :, :]
    #newpredictions3 = predictions   
    #for n, p in enumerate(newpatches): newpatches[n] = p[:, ::-1, :]
    #newpredictions3 = model.predict(newpatches, batch_size=1)
    #for n, p in enumerate(newpatches):
    #   newpredictions3[n] = np.power(newpredictions3[n], 0.5)
    #   newpredictions3[n] = newpredictions3[n] / ( newpredictions3[n] + np.power(1.0 - newpredictions3[n], 0.5) )
    #   newpredictions3[n] = newpredictions3[n][:, ::-1, :]    
    #rec4 = reconstruct_patches( [ predictions, newpredictions, newpredictions2, newpredictions3 ] , (dim_x, dim_y, N_CLASSES), STEP_SZ, False)
    return rec1, rec2, rec3

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
    d.addPairwiseGaussian(sxy=50, compat=np.array([1e-8,1e-8]).astype(np.float32))
    d.addPairwiseBilateral(sxy=50, srgb=50, rgbim=img.astype(np.uint8), compat=np.array([1e-8,1e-8]).astype(np.float32))
    Q = d.inference(10)
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w))
    return Q

def calculate_results_aux(args, model, calibrate, x, y):
    tps1, tns1, fns1, fps1 = 0, 0, 0, 0
    tps2, tns2, fns2, fps2 = 0, 0, 0, 0
    tps3, tns3, fns3, fps3 = 0, 0, 0, 0
    for index in tqdm(range(0, len(x))):
        img = tiff.imread(x[index])
        name_image = y[index].split("_")[-1].split(".")[0]        
        pred1, pred2, pred3 = predict_aux(args, img, model, calibrate)
        #result1 = post_processing(img[:, :, 0:3], pred1)
        result1 = np.where(pred1.reshape((ORIGI_SZ, ORIGI_SZ)) < 0.5, 0, 1)
        result2 = np.where(pred2.reshape((ORIGI_SZ, ORIGI_SZ)) < 0.5, 0, 1)
        result3 = np.where(pred3.reshape((ORIGI_SZ, ORIGI_SZ)) < 0.5, 0, 1)
        mask = imageio.imread(y[index])
        mask = np.where(mask == 255, 1, 0) if np.any(mask == 255) else mask
        tn1, fp1, fn1, tp1 = confusion_matrix(mask.ravel(), result1.ravel(), labels=[0, 1]).ravel()
        fps1 += fp1; fns1 += fn1; tps1 += tp1; tns1 += tn1
        tn2, fp2, fn2, tp2 = confusion_matrix(mask.ravel(), result2.ravel(), labels=[0, 1]).ravel()
        fps2 += fp2; fns2 += fn2; tps2 += tp2; tns2 += tn2
        tn3, fp3, fn3, tp3 = confusion_matrix(mask.ravel(), result3.ravel(), labels=[0, 1]).ravel()
        fps3 += fp3; fns3 += fn3; tps3 += tp3; tns3 += tn3
        img = image.array_to_img(picture_from_mask(mask))
        img.save('./results/{}_original.jpg'.format(name_image))
        img = image.array_to_img(picture_from_mask(result1))
        img.save('./results/{}_predicted.jpg'.format(name_image))
    acc1 = (tps1 + tns1) / (tps1 + tns1 + fps1 + fns1)
    iou1 = tps1 / (tps1 + fns1 + fps1)
    acc2 = (tps2 + tns2) / (tps2 + tns2 + fps2 + fns2)
    iou2 = tps2 / (tps2 + fns2 + fps2)
    acc3 = (tps3 + tns3) / (tps3 + tns3 + fps3 + fns3)
    iou3 = tps3 / (tps3 + fns3 + fps3)
    return acc1, iou1, acc2, iou2, acc3, iou3

def calculate_results(args, model, calibrate=None):
    print("Calculating results...")
    x, y = get_files(path_test_images_template, path_test_masks_template)
    accuracy1, iou1, accuracy2, iou2, accuracy3, iou3 = calculate_results_aux(args, model, calibrate, x, y)
    print("Same locations --------> Acc: {} | IOU: {}".format(accuracy1, iou1))
    print("Same locations with cubed probability estimates --------> Acc: {} | IOU: {}".format(accuracy2, iou2))
    print("Same locations with cubed probability estimates and augmentations --------> Acc: {} | IOU: {}".format(accuracy3, iou3))
    x, y = get_files(path_test_images_template, path_test_masks_template, min_range=7, max_range=8)
    accuracy1, iou1, accuracy2, iou2, accuracy3, iou3 = calculate_results_aux(args, model, calibrate, x, y)
    print("Different locations ---> Acc: {} | IOU: {}".format(accuracy1, iou1))
    print("Different locations with cubed probability estimates ---> Acc: {} | IOU: {}".format(accuracy2, iou2))
    print("Different locations with cubed probability estimates and augmentations ---> Acc: {} | IOU: {}".format(accuracy3, iou3))
    for i in range(1,7):
      x, y = get_files(path_test_images_template, path_test_masks_template, min_range=i, max_range=i+1)
      accuracy1, iou1, accuracy2, iou2, accuracy3, iou3 = calculate_results_aux(args, model, calibrate, x, y)
      print("Location " + repr(i) + " ---> Acc: {} | IOU: {}".format(accuracy1, iou1))
      print("Location " + repr(i) + " with cubed probability estimates ---> Acc: {} | IOU: {}".format(accuracy2, iou2))
      print("Location " + repr(i) + " with cubed probability estimates and augmentations ---> Acc: {} | IOU: {}".format(accuracy3, iou3))

# noinspection PyTypeChecker
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", dest='mode', choices=list(Mode), type=Mode.from_string, default=Mode.train)
    parser.add_argument("--channels", dest='channels', choices=list(NumberChannels), type=NumberChannels.from_string, default=NumberChannels.four)
    return vars(parser.parse_args())

def main():
    args = parse_args()
    #pre_process(args, base_path, testAndTrain=True)
    #gen_patches(args, PATCH_SZ, STEP_SZ, base_path)
    update_weights_path(args)
    model, calibrate = get_model(args)
    calibrate = None
    calculate_results(args, model, calibrate)

if __name__ == '__main__':
    main()