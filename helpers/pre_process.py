import os
import shutil
import numpy as np
import tifffile as tiff
from tqdm import tqdm
from arguments.arguments import NumberChannels

new_path_train_images = '{}/dataset/devset_0{}_satellite_images/'
new_path_test_images = '{}/dataset/testset_0{}_satellite_images/'

path_train_images_template = '{}/flood-data/devset_0{}_satellite_images/'
path_test_images_template = '{}/flood-data/testset_0{}_satellite_images/'

path_elevation_and_slope_train_template = '{}/flood-data/devset_0{}_elevation_and_slope/'
path_elevation_and_slope_test_template = '{}/flood-data/testset_0{}_elevation_and_slope/'

path_imperviousness_train_template = '{}/flood-data/devset_0{}_imperviousness/'
path_imperviousness_test_template = '{}/flood-data/testset_0{}_imperviousness/'

path_ndvi_train_template = '{}/flood-data/devset_0{}_NDVI/'
path_ndvi_test_template = '{}/flood-data/testset_0{}_NDVI/'

path_ndwi_train_template = '{}/flood-data/devset_0{}_NDWI/'
path_ndwi_test_template = '{}/flood-data/testset_0{}_NDWI/'

def get_other_paths(dataset_type, index):
    if dataset_type == "train":
        return [path_elevation_and_slope_train_template.format(base_path, index),
                path_imperviousness_train_template.format(base_path, index),
                path_ndvi_train_template.format(base_path, index),
                path_ndwi_train_template.format(base_path, index)]
    elif dataset_type == "test":
        return [path_elevation_and_slope_test_template.format(base_path, index),
                path_imperviousness_test_template.format(base_path, index),
                path_ndvi_test_template.format(base_path, index),
                path_ndwi_test_template.format(base_path, index)]
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
        if max_value > max_value_final: max_value_final = max_value
        if min_value < min_value_final: min_value_final = min_value
    return min_value_final, max_value_final

def get_all_paths_files(train_path, test_path, alias=""):
    files = []
    for index in range(1, 7):
        path = train_path.format(base_path, index)
        files.extend([path + x for x in os.listdir(path) if x.startswith(alias) and "._" not in x])
    for index in range(1, 8):
        path = test_path.format(base_path, index)
        files.extend([path + x for x in os.listdir(path) if x.startswith(alias) and "._" not in x])
    return files

def get_elevation_extremes():
    global elevation_min, elevation_max
    try: return elevation_min, elevation_max
    except NameError:
        files = get_all_paths_files(path_elevation_and_slope_train_template,
                                    path_elevation_and_slope_test_template,
                                    alias="elevation_")
        result = calculate_extremes(files)
        elevation_min, elevation_max = result[0], result[1]
        return elevation_min, elevation_max

def get_imperviousness_extremes():
    global imperviousness_min, imperviousness_max
    try: return imperviousness_min, imperviousness_max
    except NameError:
        files = get_all_paths_files(path_imperviousness_train_template,
                                    path_imperviousness_test_template)
        result = calculate_extremes(files)
        imperviousness_min, imperviousness_max = result[0], result[1]
        return imperviousness_min, imperviousness_max

def get_ndvi_extremes():
    global ndvi_min, ndvi_max
    try: return ndvi_min, ndvi_max
    except NameError:
        files = get_all_paths_files(path_ndvi_train_template,
                                    path_ndvi_test_template)
        result = calculate_extremes(files)
        ndvi_min, ndvi_max = result[0], result[1]
        return ndvi_min, ndvi_max

def get_ndwi_extremes():
    global ndwi_min, ndwi_max
    try: return ndwi_min, ndwi_max
    except NameError:
        files = get_all_paths_files(path_ndwi_train_template,
                                    path_ndwi_test_template)
        result = calculate_extremes(files)
        ndwi_min, ndwi_max = result[0], result[1]
        return ndwi_min, ndwi_max

def scale(extremes, features):
    features = np.array(features, dtype=np.float64)
    features[features == -9999] = np.nan
    features = (features - extremes[0]) / (extremes[1] - extremes[0])
    features[np.isnan(features)] = -1
    return features

def get_other_relevant_files_for_id(args, id, dataset_type, index):
    paths, result = get_other_paths(dataset_type, index), []
    if args['channels'] in [NumberChannels.six, NumberChannels.seven, NumberChannels.eight]:
        ndvi_extremes = get_ndvi_extremes()
        ndvi = tiff.imread([paths[2] + x for x in os.listdir(paths[2]) if x.startswith(id)][0])
        ndvi = scale(ndvi_extremes, ndvi)
        result.append(ndvi)
        ndwi_extremes = get_ndwi_extremes()
        ndwi = tiff.imread([paths[3] + x for x in os.listdir(paths[3]) if x.startswith(id)][0])
        ndwi = scale(ndwi_extremes, ndwi)
        result.append(ndwi)
    if args['channels'] in [NumberChannels.seven, NumberChannels.eight]:
        elevation_extremes = get_elevation_extremes()
        elevation = tiff.imread([paths[0] + x for x in os.listdir(paths[0]) if x.startswith("elevation_" + id)][0])
        elevation = scale(elevation_extremes, elevation)
        result.append(elevation)
    if args['channels'] in [NumberChannels.eight]:
        imperviousness_extremes = get_imperviousness_extremes()
        imperviousness = tiff.imread([paths[1] + x for x in os.listdir(paths[1]) if x.startswith(id)][0])
        imperviousness = scale(imperviousness_extremes, imperviousness)
        result.append(imperviousness)
    return np.dstack(result)

def convert(args, path_original, path_goal, dataset_type, max_range):
    for index in range(1, max_range):
        path = path_original.format(base_path, index)
        new_path = path_goal.format(base_path, index)
        if not os.path.exists(new_path): os.makedirs(new_path)
        files = [x for x in os.listdir(path) if not x.startswith("._")]
        for f in tqdm(files):
            img = tiff.imread(path + f)
            new_img = img / 65535
            if args['channels'] not in [NumberChannels.four, NumberChannels.three]:
                name = f.split(".")[0]
                new_img = np.dstack((new_img, get_other_relevant_files_for_id(args, name, dataset_type, index)))
            if args['channels'] in [NumberChannels.three]:
                new_img = new_img[:, :, 0:3]
            tiff.imsave(new_path + f, new_img, compress=9)

def pre_process(args, path, testAndTrain=True):
    global base_path
    base_path = path
    print("Pre-processing input images...")
    if os.path.exists(base_path + "/dataset"): shutil.rmtree(base_path + "/dataset")
    if testAndTrain: convert(args, path_train_images_template, new_path_train_images, "train", max_range=7)
    convert(args, path_test_images_template, new_path_test_images, "test", max_range=8)
    print("Done.")
