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

path_elevation_and_slope_train_template = '{}/flood-data/devset_0{}_elevation_and_slope/'
path_elevation_and_slope_test_template = '{}/flood-data/testset_0{}_elevation_and_slope/'

path_gdacs_water_and_magnitude_train_template = '{}/flood-data/devset_0{}_gdacs_water_and_magnitude/'
path_gdacs_water_and_magnitude_test_template = '{}/flood-data/testset_0{}_gdacs_water_and_magnitude/'

path_imperviousness_train_template = '{}/flood-data/devset_0{}_imperviousness/'
path_imperviousness_test_template = '{}/flood-data/testset_0{}_imperviousness/'

path_nvdi_train_template = '{}/flood-data/devset_0{}_NDVI/'
path_nvdi_test_template = '{}/flood-data/testset_0{}_NDVI/'


def get_other_paths(dataset_type, index):
    if dataset_type == "train":
        return [path_elevation_and_slope_train_template.format(base_path, index),
                path_gdacs_water_and_magnitude_train_template.format(base_path, index),
                path_imperviousness_train_template.format(base_path, index),
                path_nvdi_train_template.format(base_path, index)]
    elif dataset_type == "test":
        return [path_elevation_and_slope_test_template.format(base_path, index),
                path_gdacs_water_and_magnitude_test_template.format(base_path, index),
                path_imperviousness_test_template.format(base_path, index),
                path_nvdi_test_template.format(base_path, index)]
    else:
        raise ValueError("Invalid Type.")


def calculate_extremes(files):
    max_value_final, min_value_final = 0, 99999

    for file in files:
        content = tiff.imread(file)
        max_value = np.amax(content)

        # Replace NO DATA VALUE for another value to get minimum
        content[content == -9999] = 1000000
        min_value = np.amin(content)

        if max_value > max_value_final:
            max_value_final = max_value

        if min_value < min_value_final:
            min_value_final = min_value

    return min_value_final, max_value_final


def get_all_paths_files(train_path, test_path, alias=""):
    files = []
    for index in range(1, 7):
        path = train_path.format(base_path, index)
        files.extend([path + x for x in os.listdir(path) if x.startswith(alias)])

    for index in range(1, 7):
        path = test_path.format(base_path, index)
        files.extend([path + x for x in os.listdir(path) if x.startswith(alias)])

    return files


def get_elevation_extremes():
    global elevation_min, elevation_max
    try:
        return elevation_min, elevation_max
    except NameError:
        files = get_all_paths_files(path_elevation_and_slope_train_template,
                                    path_elevation_and_slope_test_template,
                                    alias="elevation_")
        result = calculate_extremes(files)
        elevation_min, elevation_max = result[0], result[1]
        return elevation_min, elevation_max


def get_slope_extremes():
    global slope_min, slope_max
    try:
        return slope_min, slope_max
    except NameError:
        files = get_all_paths_files(path_elevation_and_slope_train_template,
                                    path_elevation_and_slope_test_template,
                                    alias="slope_")
        result = calculate_extremes(files)
        slope_min, slope_max = result[0], result[1]
        return slope_min, slope_max


def get_magnitude_extremes():
    global magnitude_min, magnitude_max
    try:
        return magnitude_min, magnitude_max
    except NameError:
        files = get_all_paths_files(path_gdacs_water_and_magnitude_train_template,
                                    path_gdacs_water_and_magnitude_test_template,
                                    alias="mag_")
        result = calculate_extremes(files)
        magnitude_min, magnitude_max = result[0], result[1]
        return magnitude_min, magnitude_max


def get_signal_extremes():
    global signal_min, signal_max
    try:
        return signal_min, signal_max
    except NameError:
        files = get_all_paths_files(path_gdacs_water_and_magnitude_train_template,
                                    path_gdacs_water_and_magnitude_test_template,
                                    alias="signal_")
        result = calculate_extremes(files)
        signal_min, signal_max = result[0], result[1]
        return signal_min, signal_max


def get_imperviousness_extremes():
    global imperviousness_min, imperviousness_max
    try:
        return imperviousness_min, imperviousness_max
    except NameError:
        files = get_all_paths_files(path_imperviousness_train_template,
                                    path_imperviousness_test_template)
        result = calculate_extremes(files)
        imperviousness_min, imperviousness_max = result[0], result[1]
        return imperviousness_min, imperviousness_max


def get_nvdi_extremes():
    global nvdi_min, nvdi_max
    try:
        return nvdi_min, nvdi_max
    except NameError:
        files = get_all_paths_files(path_nvdi_train_template,
                                    path_nvdi_test_template)
        result = calculate_extremes(files)
        nvdi_min, nvdi_max = result[0], result[1]
        return nvdi_min, nvdi_max


def scale(extremes, features):
    features = np.array(features, dtype=np.float64)
    features[features == -9999] = np.nan
    features = (features - extremes[0]) / (extremes[1] - extremes[0])
    features[np.isnan(features)] = -1
    return features


def get_other_relevant_files_for_id(id, dataset_type, index):
    paths = get_other_paths(dataset_type, index)

    elevation_extremes = get_elevation_extremes()
    elevation = tiff.imread([paths[0] + x for x in os.listdir(paths[0]) if x.startswith("elevation_" + id)][0])
    elevation = scale(elevation_extremes, elevation)

    slope_extremes = get_slope_extremes()
    slope = tiff.imread([paths[0] + x for x in os.listdir(paths[0]) if x.startswith("slope_" + id)][0])
    slope = scale(slope_extremes, slope)

    magnitude_extremes = get_magnitude_extremes()
    magnitude = tiff.imread([paths[1] + x for x in os.listdir(paths[1]) if x.startswith("mag_" + id)][0])
    magnitude = scale(magnitude_extremes, magnitude)

    signal_extremes = get_signal_extremes()
    signal = tiff.imread([paths[1] + x for x in os.listdir(paths[1]) if x.startswith("signal_" + id)][0])
    signal = scale(signal_extremes, signal)

    # imperviousness_extremes = get_imperviousness_extremes()
    # imperviousness = tiff.imread([paths[2] + x for x in os.listdir(paths[2]) if x.startswith(id)][0])
    # imperviousness = scale(imperviousness_extremes, imperviousness)

    nvdi_extremes = get_nvdi_extremes()
    nvdi = tiff.imread([paths[3] + x for x in os.listdir(paths[3]) if x.startswith(id)][0])
    nvdi = scale(nvdi_extremes, nvdi)

    return np.dstack((elevation, slope, magnitude, signal, nvdi))


def convert(path_original, path_goal, dataset_type, max_range):
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
            new_img = np.dstack((new_img, img_as_float(img[:, :, 3])))
            p2, p98 = np.percentile(new_img[:, :, 3], (2, 98))
            new_img[:, :, 3] = (1.0 + exposure.rescale_intensity(new_img[:, :, 3], in_range=(p2, p98))) / 2.0
            new_img = np.dstack((new_img, get_other_relevant_files_for_id(f.split(".")[0], dataset_type, index)))
            tiff.imsave(new_path + f, new_img)


def main():
    convert(path_train_images_template, new_path_train_images, "train", max_range=7)
    convert(path_test_images_template, new_path_test_images, "test", max_range=8)


if __name__ == '__main__':
    main()
