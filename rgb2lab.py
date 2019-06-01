import os

import tifffile as tiff
import numpy as np
from skimage import exposure, img_as_float
from skimage.color import rgb2lab
from tqdm import tqdm

base_path = '/home/jpereira/A-U-Net-Model-Leveraging-Multiple-Remote-Sensing-Data-Sources-for-Flood-Extent-Mapping'


new_path_train_images = '{}/dataset/devset_0{}_satellite_images/'
new_path_test_images = '{}/dataset/testset_0{}_satellite_images/'

path_train_images_template = '{}/flood-data/devset_0{}_satellite_images/'
path_test_images_template = '{}/flood-data/testset_0{}_satellite_images/'


def convert(path_original, path_goal, max_range):
    for index in range(1, max_range):
        path = path_original.format(base_path, index)
        new_path = path_goal.format(base_path, index)
        if not os.path.exists(new_path): os.makedirs(new_path)
        files = [x for x in os.listdir(path) if not x.startswith("._")]

        for f in tqdm(files):
            img = tiff.imread(path + f)
            new_img = img[:, :, 0:3]
            p2, p98 = np.percentile(img, (2, 98))
            new_img = exposure.rescale_intensity(new_img, in_range=(p2, p98))
            new_img = rgb2lab(new_img[:, :, 0:3])
            new_img[:, :, 0] = new_img[:, :, 0] / 100.0
            new_img[:, :, 1] = np.interp(new_img[:, :, 1], (-128.0, 128.0), (0.0, 1.0))
            new_img[:, :, 2] = np.interp(new_img[:, :, 2], (-128.0, 128.0), (0.0, 1.0))
            new_img = np.dstack((new_img , img_as_float(img[:, :, 3])))
            p2, p98 = np.percentile(new_img[:, :, 3], (2, 98))

            new_img[:, :, 3] = (1.0 + exposure.rescale_intensity(new_img[:, :, 3], in_range=(p2, p98))) / 2.0
            tiff.imsave(new_path + f, new_img)


def main():
    convert(path_train_images_template, new_path_train_images, max_range=7)
    convert(path_test_images_template, new_path_test_images, max_range=8)


if __name__ == '__main__':
    main()
