from os import listdir
from os.path import isfile, join
import tifffile as tiff
import numpy as np
from skimage.color import rgb2lab, lab2rgb


dataset = input('Potsdam (p) or Vaihingen (v) dataset? ')
while True:
    if dataset == 'p':
        path_img = './potsdam/Images/'
        new_path_img = './potsdam/Images_lab/'
        break
    elif dataset == 'v':
        path_img = './vaihingen/Images/'
        new_path_img = './vaihingen/Images_lab/'
        break
    else:
        dataset = input('p or v?')

files = [f for f in listdir(path_img) if isfile(join(path_img, f))]
print(len(files))
i=0
for f in files:
    print(i)
    i=i+1
    if '.tif' in f:
        img = tiff.imread(path_img+f)
        new_img = rgb2lab(img)
        new_img[:,:,0] = new_img[:,:,0]/100
        new_img[:,:,1] = np.interp(new_img[:,:,1], (-128, 128), (0, 1))
        new_img[:,:,2] = np.interp(new_img[:,:,2], (-128, 128), (0, 1))
        tiff.imsave(new_path_img+f, new_img)
