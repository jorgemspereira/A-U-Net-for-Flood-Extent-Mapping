import os
import shutil
import subprocess
import gdal
import numpy as np
from tqdm import tqdm

path_train_images = '{}/flood-data/devset_0{}_satellite_images/'
path_train_masks = '{}/flood-data/devset_0{}_segmentation_masks/'

new_path_train_images = '{}/dataset/devset_0{}_augmented_satellite_images/'
new_path_train_masks = '{}/dataset/devset_0{}_augmented_segmentation_masks/'

def verify_folder(folder):
    if not os.path.exists(folder): os.makedirs(folder)

def valid_shape(shape):
    if len(shape) == 3: return shape[1] == 320 and shape[2] == 320
    return shape[0] == 320 and shape[1] == 320

def split_merged_aux(src_name, dst_name, width, height, tile_size, i, j, nodata="0"):
    w = min(i + tile_size, width) - i
    h = min(j + tile_size, height) - j
    gdaltranString = "gdal_translate -a_nodata " + nodata + " -of GTIFF -srcwin " + str(i) + ", " + str(j) + ", " + str(w) + ", " + str(h) + " " + src_name + " " + dst_name + str(i) + "_" + str(j) + ".tif"
    subprocess.check_output(gdaltranString, shell=True)
    result = gdal.Open(dst_name + str(i) + "_" + str(j) + ".tif")
    array = result.ReadAsArray()
    nan_array = array.astype(np.float)
    nan_array[array == int(nodata)] = np.nan
    if np.isnan(nan_array).any() or not valid_shape(nan_array.shape):
        os.remove(dst_name + str(i) + "_" + str(j) + ".tif")

def split_merged(src_name, dst_name, nodata="0"):
    dset = gdal.Open(src_name)
    width = dset.RasterXSize
    height = dset.RasterYSize
    tile_size = 320
    n_iter = len(list(range(0, width, tile_size))) + len(list(range(tile_size // 2, width - tile_size // 2, tile_size)))
    with tqdm(total=n_iter) as pbar:
        for i in range(0, width, tile_size):
            for j in range(0, height, tile_size):
                split_merged_aux(src_name, dst_name, width, height, tile_size, i, j, nodata)
            for j in range(tile_size // 2, height - tile_size // 2, tile_size):
                split_merged_aux(src_name, dst_name, width, height, tile_size, i, j, nodata)
            pbar.update(1)
        for i in range(tile_size // 2, width - tile_size // 2, tile_size):
            for j in range(0, height, tile_size):
                split_merged_aux(src_name, dst_name, width, height, tile_size, i, j, nodata)
            for j in range(tile_size // 2, height - tile_size // 2, tile_size):
                split_merged_aux(src_name, dst_name, width, height, tile_size, i, j, nodata)
            pbar.update(1)

def merge_and_split(base_path, new_base_path, files, nodata="0"):
    files = [base_path + name for name in files]
    merge_command = ["gdal_merge.py", "-a_nodata", nodata, "-o", "merged.tif", "-of", "gtiff"] + files
    merge_command = " ".join(merge_command)
    subprocess.check_output(merge_command, shell=True)
    split_merged("merged.tif", new_base_path, nodata)
    os.remove("merged.tif")

def gen_images(base_path):
    print("Generating images...")
    if os.path.exists(base_path + "/dataset"): shutil.rmtree(base_path + "/dataset")
    for index in range(1, 7):
        file_path = path_train_images.format(base_path, index)
        file_new_path = new_path_train_images.format(base_path, index)
        mask_file_path = path_train_masks.format(base_path, index)
        mask_file_new_path = new_path_train_masks.format(base_path, index)
        verify_folder(file_new_path)
        verify_folder(mask_file_new_path)
        files = [x for x in os.listdir(file_path) if not x.startswith("._")]
        masks = [x for x in os.listdir(mask_file_path) if not x.startswith("._") and x.endswith(".tif")]
        merge_and_split(file_path, file_new_path, files)
        merge_and_split(mask_file_path, mask_file_new_path, masks, nodata="255")
    print("Done.")
