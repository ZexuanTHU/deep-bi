import cv2
import numpy as np


class CLAHE(object):
    def __init__(self, clipLimit=2.0, tileGridSize=(8, 8)):
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize

    def __call__(self, imgs):
        assert (len(imgs.shape) == 4)  # 4D arrays
        assert (imgs.shape[1] == 1)  # check the channel is 1
        # create a CLAHE object (Arguments are optional).
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        imgs_equalized = np.empty(imgs.shape)
        for i in range(imgs.shape[0]):
            imgs_equalized[i, 0] = clahe.apply(
                np.array(imgs[i, 0], dtype=np.uint8))
        return imgs_equalized


class AdjustGamma(object):
    def __init__(self, gamma=1.0):
        self.gamma = gamma

    def __call__(self, imgs):
        assert (len(imgs.shape) == 4)  # 4D arrays
        assert (imgs.shape[1] == 1)  # check the channel is 1
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / self.gamma
        table = np.array([((i / 255.0) ** invGamma) *
                          255 for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        new_imgs = np.empty(imgs.shape)
        for i in range(imgs.shape[0]):
            new_imgs[i, 0] = cv2.LUT(
                np.array(imgs[i, 0], dtype=np.uint8), table)
        return new_imgs
