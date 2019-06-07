import os

import imageio
from tqdm import tqdm
from patchify import patchify

import tifffile as tiff

base_path = '/home/jpereira/A-U-Net-Model-Leveraging-Multiple-Remote-Sensing-Data-Sources-for-Flood-Extent-Mapping'

path_train_images = '{}/dataset/devset_0{}_satellite_images/'
path_train_masks = '{}/flood-data/devset_0{}_segmentation_masks/'

new_path_train_images = '{}/dataset/devset_0{}_satellite_images_patches/'
new_path_train_masks = '{}/dataset/devset_0{}_segmentation_masks_patches/'

PATCH_SZ = 64
BANDS_SZ = 4
STEP_SZ = 16


def verify_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def create_patches(file_path_original, file_path_goal, mask_path_original, mask_path_goal):
    for index in range(1, 7):

        file_path = file_path_original.format(base_path, index)
        file_new_path = file_path_goal.format(base_path, index)

        mask_file_path = mask_path_original.format(base_path, index)
        mask_file_new_path = mask_path_goal.format(base_path, index)

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

            img_patches = patchify(img, (PATCH_SZ, PATCH_SZ, BANDS_SZ), step=STEP_SZ)
            mask_patches = patchify(mask, (PATCH_SZ, PATCH_SZ), step=STEP_SZ)

            counter = 0
            for x in img_patches:
                for y in x:
                    file_name = file_new_path + img_id + "_" + str(counter).zfill(2) + ".tif"
                    tiff.imwrite(file_name, y.reshape((PATCH_SZ, PATCH_SZ, BANDS_SZ)))
                    counter += 1

            counter = 0
            for x in mask_patches:
                for y in x:
                    mask_name = mask_file_new_path + img_id + "_" + str(counter).zfill(2) + ".png"
                    imageio.imwrite(mask_name, y.reshape((PATCH_SZ, PATCH_SZ)))
                    counter += 1


def main():
    create_patches(path_train_images, new_path_train_images, path_train_masks, new_path_train_masks)


if __name__ == '__main__':
    main()
