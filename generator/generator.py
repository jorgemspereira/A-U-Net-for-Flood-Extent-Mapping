from random import randint

import imageio
import numpy as np
import tifffile as tiff


def get_input(path):
    return tiff.imread(path)


def get_mask(path):
    return imageio.imread(path)


def get_rand_patch(img, mask):
    patch_img, patch_mask = img, mask
    random_transformation = np.random.randint(1, 4)

    # reverse first dimension
    if random_transformation == 1:
        patch_img = img[::-1, :, :]
        patch_mask = mask[::-1, :]

    # reverse second dimension
    elif random_transformation == 2:
        patch_img = img[:, ::-1, :]
        patch_mask = mask[:, ::-1]

    return patch_img, patch_mask


def image_generator(path_input, path_mask, batch_size=5, random_transformation=False, shuffle=True):
    ids_file_all = path_input[:]
    ids_mask_all = path_mask[:]
    while True:
        if len(ids_file_all) < batch_size:
            ids_file_all = path_input[:]
            ids_mask_all = path_mask[:]
        x, y = list(), list()
        total_patches = 0
        while total_patches < batch_size:
            index = 0
            if shuffle: index = randint(1, len(ids_file_all) - 1) if len(ids_file_all) != 1 else 0
            img = get_input(ids_file_all.pop(index))
            mask = get_mask(ids_mask_all.pop(index))
            if random_transformation: img, mask = get_rand_patch(img, mask)
            mask = np.where(mask == 255, 1, 0) if np.any(mask == 255) else mask
            x.append(img)
            y.append(mask.reshape((320, 320, 1)))
            total_patches += 1
        yield (np.array(x), np.array(y))
