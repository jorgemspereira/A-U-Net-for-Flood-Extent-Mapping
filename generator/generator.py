import cv2
import random
import numpy as np
import tifffile as tiff
from scipy.interpolate import UnivariateSpline

def get_input(path):
    return tiff.imread(path)

def get_mask(path):
    return tiff.imread(path)

def create_lut(x, y):
    spl = UnivariateSpline(x, y)
    return spl(range(65535))

def rescale(value):
    return round((value * 65535) / 255)

def apply_lut(content, table, patch_sz):
    prev_content, res = (content * 65535), []
    for el in np.nditer(prev_content): res.append(table[int(el)])
    return (np.array(res) / 65535).reshape((patch_sz, patch_sz))

def get_incr_lut(pos):
    global result_incr_lut
    try: return dict(zip(list(range(len(result_incr_lut[pos]))), result_incr_lut[pos]))
    except NameError:
        result_incr_lut = [ [] for i in range(0,11) ]
        for i in range(4,11):
            result_incr_lut[i] = create_lut([0, rescale(64), rescale(128), rescale(192), rescale(256)], [0, rescale(64+((70-64) * (i / 10.0))), rescale(128+((140-128) * (i / 10.0))), rescale(192+((210-192) * (i / 10.0))), rescale(256)])
        return dict(zip(list(range(len(result_incr_lut[pos]))), result_incr_lut[pos]))

def get_decr_lut(pos):
    global result_decr_lut
    try: return dict(zip(list(range(len(result_decr_lut[pos]))), result_decr_lut[pos]))
    except NameError:
        result_decr_lut = [ [] for i in range(0,11) ]
        for i in range(4,11):
            result_decr_lut[i] = create_lut([0, rescale(64), rescale(128), rescale(192), rescale(256)], [0, rescale(64+((30-64) * (i / 10.0))), rescale(128+((80-128) * (i / 10.0))), rescale(192+((120-192) * (i / 10.0))), rescale(192)])
        return dict(zip(list(range(len(result_decr_lut[pos]))), result_decr_lut[pos]))

def change_temperature(img, mode, patch_size):
    pos = np.random.randint(5, 11)
    c_r, c_g, c_b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    if img.shape[2] > 3: c_nir = img[:, :, 3]
    if mode == "warm":
        c_r   = apply_lut(c_r, get_incr_lut(pos), patch_size)
        c_b   = apply_lut(c_b, get_decr_lut(pos), patch_size)
        if img.shape[2] > 3: c_nir = apply_lut(c_nir, get_incr_lut(pos), patch_size)
    elif mode == "cold":
        c_r   = apply_lut(c_r, get_decr_lut(pos), patch_size)
        c_b   = apply_lut(c_b, get_incr_lut(pos), patch_size)
        if img.shape[2] > 3: c_nir = apply_lut(c_nir, get_decr_lut(pos), patch_size)
    else:
        raise ValueError("Mode should be equal to warm or cold.")
    if img.shape[2]: return np.dstack((c_r, c_g, c_b, c_nir, img[:, :, 4:]))
    return np.dstack((c_r, c_g, c_b, img[:, :, 3:]))

def _maybe_process_in_chunks(process_fn, **kwargs):
    def __process_fn(img):
        num_channels = img.shape[2]
        if num_channels > 4:
            chunks = []
            for index in range(0, num_channels, 4):
                chunk = img[:, :, index : index + 4]
                chunk = process_fn(chunk, **kwargs)
                chunks.append(chunk)
            img = np.dstack(chunks)
        else: img = process_fn(img, **kwargs)
        return img
    return __process_fn

def get_random_transformation(img, mask, patch_size):
    patch_img, patch_mask = img, mask
    # reverse first dimension
    random_transformation = np.random.randint(1, 10)
    if random_transformation == 1:
        patch_img = patch_img[::-1, :, :]
        patch_mask = patch_mask[::-1, :, :]
    # reverse second dimension
    random_transformation = np.random.randint(1, 10)
    if random_transformation == 1:
        patch_img = patch_img[:, ::-1, :]
        patch_mask = patch_mask[:, ::-1, :]
    # rotate image
    random_transformation = np.random.randint(1, 10)
    if random_transformation == 1:
        times = np.random.randint(1, 3)
        patch_img = np.rot90(patch_img, k=times, axes=(0, 1))
        patch_mask = np.rot90(patch_mask, k=times, axes=(0, 1))
    # shift+scale+rotatee
    random_transformation = np.random.randint(1, 10)
    if random_transformation == 1:
        angle = random.uniform(1.0, 22.5)
        scale = random.uniform(0.0, 0.01)
        dx = random.uniform(0.0, 0.015625)
        dy = random.uniform(0.0, 0.015625)
        height, width = patch_img.shape[:2]
        center = (width / 2, height / 2)
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        matrix[0, 2] += dx * width
        matrix[1, 2] += dy * height
        warp_affine_fn = _maybe_process_in_chunks( cv2.warpAffine, M=matrix, dsize=(width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )
        patch_img = warp_affine_fn(patch_img)
        warp_affine_fn = _maybe_process_in_chunks( cv2.warpAffine, M=matrix, dsize=(width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101 )
        patch_mask = warp_affine_fn(patch_mask)
    # transpose image
    random_transformation = np.random.randint(1, 10)
    if random_transformation == 1:
        patch_img = np.transpose(patch_img, axes=(1, 0, 2))
        patch_mask = np.transpose(patch_mask, axes=(1, 0, 2))
    # change colour temperature
    random_transformation = np.random.randint(1, 10)
    if random_transformation == 1:
        temp=random.choice(['warm','cold'])
        patch_img = change_temperature(patch_img, temp, patch_size)
    return patch_img, patch_mask

def calculate_weight(img_id, weights):
    if weights is not None:
        index = img_id.find("devset_0")
        value = weights[int(img_id[index + 8:index + 9]) - 1]
        return value
    return 1

def image_generator(path_input, path_mask, patch_size, weights=None, batch_size=5, random_transformation=False, shuffle=True, include_seg_mask=True):
    ids_file_all = path_input[:]
    ids_mask_all = path_mask[:]
    while True:
        if len(ids_file_all) < batch_size:
            ids_file_all = path_input[:]
            ids_mask_all = path_mask[:]
        x, y, weights_res, weights_map = list(), list(), list(), list()
        total_patches = 0
        while total_patches < batch_size:
            index = 0
            if shuffle: index = np.random.randint(1, len(ids_file_all) - 1) if len(ids_file_all) != 1 else 0
            img_id, mask_id = ids_file_all.pop(index), ids_mask_all.pop(index)
            img, mask = get_input(img_id), get_mask(mask_id)
            if random_transformation: img, mask = get_random_transformation(img, mask, patch_size)
            x.append(img)
            y.append(mask[:, :, 0].reshape((patch_size, patch_size, 1)))
            weights_map.append(mask[:, :, 1].reshape((patch_size, patch_size, 1)))
            weights_res.append(calculate_weight(img_id, weights))
            total_patches += 1
        if include_seg_mask: yield ([np.array(x), np.array(weights_map)], np.array(y), np.array(weights_res))
        else: yield np.array(x)