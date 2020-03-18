import os
import imageio
from scipy import ndimage
from tqdm import tqdm
from patchify import patchify
import numpy as np
import tifffile as tiff

path_train_images = '{}/dataset/devset_0{}_satellite_images/'
path_train_masks = '{}/flood-data/devset_0{}_segmentation_masks/'

new_path_train_images = '{}/dataset/devset_0{}_satellite_images_patches/'
new_path_train_masks = '{}/dataset/devset_0{}_segmentation_masks_patches/'

def verify_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def get_distance(f):
    f = f != 1.0
    dist_func = ndimage.distance_transform_edt
    distance = np.where(f, dist_func(f), -(dist_func(1-f)))
    return distance

def gen_patches(args, patch_size, step_size, base_path):
    print("Generating patches...")
    for index in range(1, 7):
        file_path = path_train_images.format(base_path, index)
        file_new_path = new_path_train_images.format(base_path, index)
        mask_file_path = path_train_masks.format(base_path, index)
        mask_file_new_path = new_path_train_masks.format(base_path, index)
        verify_folder(file_new_path)
        verify_folder(mask_file_new_path)
        files = [x for x in os.listdir(file_path) if not x.startswith("._")]
        masks = [x for x in os.listdir(mask_file_path) if not x.startswith("._") and x.endswith(".png")]
        files.sort(key=lambda x: x.split("/")[-1])
        masks.sort(key=lambda x: x.split("/")[-1])
        for idx in tqdm(range(0, len(files))):
            img_id = files[idx].split(".")[0]
            img = tiff.imread(file_path + files[idx])
            mask = imageio.imread(mask_file_path + masks[idx])
            img_patches = patchify(img, (patch_size, patch_size, args["channels"].value), step=step_size)
            mask_patches = patchify(mask, (patch_size, patch_size), step=step_size)
            counter = 0
            for x in img_patches:
                for y in x:
                    file_name = file_new_path + img_id + "_" + str(counter).zfill(2) + ".tif"
                    tiff.imwrite(file_name, y.reshape((patch_size, patch_size, args["channels"].value)), compress=9)
                    counter += 1
            counter = 0
            for x in mask_patches:
                for y in x:
                    y = np.where(y == 255, 1, 0) if np.any(y == 255) else y
                    result_distance = np.ones_like(y)
                    if 0 < np.count_nonzero(y) < y.size:
                        result_distance = np.abs(get_distance(y))
                        result_distance = 1.0 - (result_distance / np.amax(result_distance))
                        result_distance = np.power(0.1 + np.maximum(0.9, result_distance), 2)
                    res = np.dstack((y, result_distance))
                    mask_name = mask_file_new_path + img_id + "_" + str(counter).zfill(2) + ".tif"
                    tiff.imwrite(mask_name, res.reshape((patch_size, patch_size, 2)), compress=9)
                    counter += 1
    print("Done.")
