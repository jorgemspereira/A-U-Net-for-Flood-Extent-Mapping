import os

import tifffile as tiff
import numpy as np
from skimage.color import rgb2lab
from tqdm import tqdm

path_train_images_template = '/home/jsilva/flood-data/devset_0{}_satellite_images/'
new_path_train_images = '/home/jpereira/A-U-Net-Model-Leveraging-Multiple-Remote-' \
                        'Sensing-Data-Sources-for-Flood-Extent-Mapping/dataset/devset_0{}_satellite_images/'

path_test_images_template = '/home/jsilva/flood-data/testset_0{}_satellite_images/'
new_path_test_images = '/home/jpereira/A-U-Net-Model-Leveraging-Multiple-Remote-' \
                       'Sensing-Data-Sources-for-Flood-Extent-Mapping/dataset/testset_0{}_satellite_images/'


def convert(path_original, path_goal, max_range):
    for index in range(1, max_range):
        path = path_original.format(index)
        new_path = path_goal.format(index)

        if not os.path.exists(new_path):
            os.makedirs(new_path)

        files = [x for x in os.listdir(path) if not x.startswith("._")]

        for f in tqdm(files):
            img = tiff.imread(path + f)
            new_img = rgb2lab(img[:, :, 0:3])

            new_img[:, :, 0] = new_img[:, :, 0] / 100
            new_img[:, :, 1] = np.interp(new_img[:, :, 1], (-128, 128), (0, 1))
            new_img[:, :, 2] = np.interp(new_img[:, :, 2], (-128, 128), (0, 1))

            tiff.imsave(new_path + f, new_img)


def main():
    convert(path_train_images_template, new_path_train_images, max_range=7)
    convert(path_test_images_template, new_path_test_images, max_range=8)


if __name__ == '__main__':
    main()
