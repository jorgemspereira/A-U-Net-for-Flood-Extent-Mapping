import cv2
import numpy as np
from keras.preprocessing import image

def picture_from_mask(mask):
    colors = {
        0: [255, 255, 255],
        1: [255, 0, 0]
    }
    pict = np.empty(shape=(3, mask.shape[0], mask.shape[1]))
    for cl in range(len(colors)):
        for ch in range(3):
            pict[ch, :, :] = np.where(mask == cl, colors[cl][ch], pict[ch, :, :])
    return np.moveaxis(pict, 0, -1)

def main():
    id_img = "060504"
    fst_path = "./results/{}_original.jpg".format(id_img)
    scn_path = "./results/{}_predicted.jpg".format(id_img)
    fst_img = cv2.imread(fst_path, cv2.IMREAD_GRAYSCALE)
    scn_img = cv2.imread(scn_path, cv2.IMREAD_GRAYSCALE)
    fst_img = cv2.threshold(fst_img, 100, 255, cv2.THRESH_BINARY)[1]
    scn_img = cv2.threshold(scn_img, 100, 255, cv2.THRESH_BINARY)[1]
    fst_img = np.where(fst_img == 0, 0, 1)
    scn_img = np.where(scn_img == 0, 0, 1)
    differences = np.abs(np.subtract(fst_img, scn_img))
    img = image.array_to_img(picture_from_mask(differences))
    img.save('./diffs_{}.jpg'.format(id_img))

if __name__ == '__main__':
    main()
