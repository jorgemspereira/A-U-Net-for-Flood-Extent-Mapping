from unet_model import *
from wnet_model import *
from gen_patches import *
from generator import *
from clr_callback import *
import os.path
import numpy as np
import tifffile as tiff
import glob
import gc
import tensorflow as tf

from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.Session(config=config)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

N_BANDS = 3
N_CLASSES = 6  # imp surface, car, building, background, low veg, tree
N_EPOCHS = 50

#train input
DATASET = 'vaihingen' #'vaihingen'
MODEL = 'W'#'W'
ID = '7'
#UNET_WEIGHTS = 'weights_unet2/unet_weights.hdf5'

if DATASET == 'potsdam':
    TRAIN_IDS = ['2_10','2_11','3_10','3_11','4_10','4_11','5_10','5_11','6_7','6_8','6_9','6_10','6_11','7_7','7_8','7_9','7_10','7_11']
    VAL_IDS = ['2_12','3_12','4_12','5_12','6_12','7_12']
    path_img = '/home/mdias/deep-wnet/datasets/potsdam/Images_lab/top_potsdam_{}_RGB.tif'
    path_mask = '/home/mdias/deep-wnet/datasets/potsdam/Masks/top_potsdam_{}_label.tif'
    PATCH_SZ = 320   # should divide by 16
    VALIDATION_STEPS = 2400
    if MODEL == 'U':
        STEPS_PER_EPOCH = 8000
        BATCH_SIZE = 32
        MAX_QUEUE = 30
    elif MODEL == 'W':
        STEPS_PER_EPOCH = 10000
        BATCH_SIZE = 12
        MAX_QUEUE = 10
elif DATASET == 'vaihingen':
    TRAIN_IDS = ['1', '3', '11', '13', '15', '17', '21', '26', '28', '30', '32', '34']
    VAL_IDS = ['5', '7', '23', '37']
    path_img = '/tmp/vaihingen/Images_lab/top_mosaic_09cm_area{}.tif'
    #path_mask = '/tmp/vaihingen/Masks/top_mosaic_09cm_area{}.tif'
    path_mask = './datasets/vaihingen/y_true/top_mosaic_09cm_area{}.tif'
    PATCH_SZ = 320   # should divide by 16
    BATCH_SIZE = 12
    STEPS_PER_EPOCH = 4000
    VALIDATION_STEPS = 1000
    MAX_QUEUE = 10

def get_model():
    if MODEL == 'U':
        model = unet_model(N_CLASSES, PATCH_SZ, n_channels=N_BANDS)
    elif MODEL == 'W':
        model = wnet_model(DATASET, N_CLASSES, PATCH_SZ, n_channels=N_BANDS)
    return model

weights_path = 'weights_' + MODEL + '_' + DATASET + '_' + ID
if not os.path.exists(weights_path):
    os.makedirs(weights_path)
weights_path += '/weights.hdf5'

if __name__ == '__main__':
    def train_net():
            print("start train net")
            model = get_model()
            #if MODEL == 'W':
            #    model.load_weights(UNET_WEIGHTS, by_name = True)
            if os.path.isfile(weights_path):
                model.load_weights(weights_path)
            early_stopping = EarlyStopping(monitor='val_loss', restore_best_weights = True, patience = 5, mode ='min')
            model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True, mode = 'min')
            csv_logger = CSVLogger('log_unet.csv', append=True, separator=';')
            #tensorboard = TensorBoard(log_dir='./tensorboard_unet/', write_graph=True, write_images=True)
            step_size = (STEPS_PER_EPOCH//BATCH_SIZE)*8
            clr = CyclicLR(base_lr = 10e-5, max_lr = 10e-4, step_size = step_size, mode='triangular2')

            train_gen = image_generator(TRAIN_IDS, path_img, path_mask, batch_size = BATCH_SIZE, patch_size = PATCH_SZ)
            val_gen = image_generator(VAL_IDS, path_img, path_mask, batch_size = BATCH_SIZE, patch_size = PATCH_SZ)

            model.fit_generator(train_gen,
               steps_per_epoch=STEPS_PER_EPOCH,
               nb_epoch=N_EPOCHS,
               validation_data=val_gen,
               validation_steps=VALIDATION_STEPS,
               verbose=1, shuffle=True, max_queue_size=MAX_QUEUE,
               callbacks=[model_checkpoint, csv_logger, early_stopping, clr]
               #callbacks=[model_checkpoint, csv_logger, tensorboard, early_stopping, clr]
            )
            return model
    train_net()
