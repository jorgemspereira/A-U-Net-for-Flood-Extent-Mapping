import imageio
import numpy as np
import tifffile as tiff
from scipy.interpolate import UnivariateSpline


def get_input(path):
    return tiff.imread(path)


def get_mask(path):
    return imageio.imread(path)


def create_lut(x, y):
    spl = UnivariateSpline(x, y)
    return spl(range(65535))


def rescale(value):
    return round((value * 65535) / 255)


def apply_lut(content, table, patch_sz):
    res = []
    for el in np.nditer(content):
        res.append(table[int(round(el * 65535))])
    return (np.array(res) / 65535).reshape((patch_sz, patch_sz))


def change_temperature(img, mode, patch_size):

    incr_ch_lut = create_lut([0, rescale(64), rescale(128), rescale(192), rescale(256)],
                             [0, rescale(70), rescale(140), rescale(210), rescale(256)])

    decr_ch_lut = create_lut([0, rescale(64), rescale(128), rescale(192), rescale(256)],
                             [0, rescale(30), rescale(80), rescale(120), rescale(192)])

    c_r, c_g, c_b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    if mode == "warm":
        c_r = apply_lut(c_r, incr_ch_lut, patch_size)
        c_b = apply_lut(c_b, decr_ch_lut, patch_size)

    elif mode == "cold":
        c_r = apply_lut(c_r, decr_ch_lut, patch_size)
        c_b = apply_lut(c_b, incr_ch_lut, patch_size)

    else:
        raise ValueError("Mode should be equal to warm or cold.")

    return np.dstack((c_r, c_g, c_b, img[:, :, 3:]))


def get_random_transformation(img, mask, patch_size):
    patch_img, patch_mask = img, mask
    random_transformation = np.random.randint(1, 8)

    # reverse first dimension
    if random_transformation == 1:
        patch_img = img[::-1, :, :]
        patch_mask = mask[::-1, :]

    # reverse second dimension
    elif random_transformation == 2:
        patch_img = img[:, ::-1, :]
        patch_mask = mask[:, ::-1]

    elif random_transformation == 3:
        patch_img = change_temperature(img, "warm", patch_size)

    elif random_transformation == 4:
        patch_img = change_temperature(img, "cold", patch_size)

    return patch_img, patch_mask


def calculate_weight(img_id, weights):
    if weights is not None:
        index = img_id.find("devset_0")
        value = weights[int(img_id[index + 8:index + 9]) - 1]
        return value
    return 1


def image_generator(path_input, path_mask, patch_size, weights=None, batch_size=5,
                    random_transformation=False, shuffle=True):
    ids_file_all = path_input[:]
    ids_mask_all = path_mask[:]
    while True:
        if len(ids_file_all) < batch_size:
            ids_file_all = path_input[:]
            ids_mask_all = path_mask[:]
        x, y, weights_res = list(), list(), list()
        total_patches = 0
        while total_patches < batch_size:
            index = 0
            if shuffle: index = np.random.randint(1, len(ids_file_all) - 1) if len(ids_file_all) != 1 else 0
            img_id, mask_id = ids_file_all.pop(index), ids_mask_all.pop(index)
            img, mask = get_input(img_id), get_mask(mask_id)
            if random_transformation: img, mask = get_random_transformation(img, mask, patch_size)
            mask = np.where(mask == 255, 1, 0) if np.any(mask == 255) else mask
            x.append(img)
            y.append(mask.reshape((patch_size, patch_size, 1)))
            weights_res.append(calculate_weight(img_id, weights))
            total_patches += 1
        yield (np.array(x), np.array(y), np.array(weights_res))
