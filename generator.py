import numpy as np
from gen_patches import *
import tifffile as tiff

def get_input(path):
    #image = rasterio.open(path).read().transpose([1,2,0])
    image = tiff.imread(path)
    return image

def get_mask(path):
    mask = tiff.imread(path)
    return mask

def image_generator(ids_file, path_image, path_mask, batch_size = 5, patch_size = 160):
    ids_file_all = ids_file[:]
    while True:
        if not ids_file:
            ids_file = ids_file_all[:]
        id = ids_file.pop(np.random.choice(len(ids_file)))
        image = get_input(path_image.format(id))
        mask = get_mask(path_mask.format(id))
        total_patches = 0
        x = list()
        y = list()
        while total_patches < batch_size:
            img_patch, mask_patch = get_rand_patch(image, mask, patch_size)
            x.append(img_patch)
            y.append(mask_patch)
            total_patches += 1

        batch_x = np.array( x )
        batch_y = np.array( y )
        yield ( batch_x, [batch_y , batch_x])
        #yield ( batch_x, batch_y )
