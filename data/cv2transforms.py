import cv2
import numpy as np
import torch
import torchvision.transforms as transforms


def composed_transforms():
    # assert len(img.shape) == 3
    compose = transforms.Compose([
        transforms.Grayscale(3),
        Norm(),
        CLAHE(),
        AdjustGamma(),
        BiValue(),
        ToTensor()
    ])
    return compose


class Norm(object):
    def __init__(self):
        pass

    def __call__(self, img):
        img = np.array(img)
        assert (len(img.shape) == 3)
        assert(img.shape[2] == 3)
        img_norm = np.empty(img.shape)
        img_std = np.std(img)
        img_mean = np.mean(img)
        img_norm = (img - img_mean) / img_std
        img_norm = ((img_norm - np.min(img_norm)) /
                    np.max(img_norm) - np.min(img_norm)) * 255

        return img_norm


class CLAHE(object):
    def __init__(self, clipLimit=2.0, tileGridSize=(8, 8)):
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize

    def __call__(self, img):
        img = np.array(img)
        img = img[:, :, 0]
        img = np.reshape(img, (img.shape[0], img.shape[1], 1))
        assert (len(img.shape) == 3)  # 3D arrays
        assert (img.shape[2] == 1)  # check the channel is 1
        # create a CLAHE object (Arguments are optional).
        clahe = cv2.createCLAHE(self.clipLimit, self.tileGridSize)
        return clahe.apply(np.array(img, dtype=np.uint8))


class AdjustGamma(object):
    def __init__(self, gamma=1.0):
        self.gamma = gamma

    def __call__(self, img):
        img = np.reshape(img, (img.shape[0], img.shape[1], 1))
        assert (len(img.shape) == 3)  # 4D arrays
        assert (img.shape[2] == 1)  # check the channel is 1
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / self.gamma
        table = np.array([((i / 255.0) ** invGamma) *
                          255 for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        new_img = cv2.LUT(img, table)
        return new_img


class BiValue(object):
    def __call__(self, img):
        return np.array(img) / 25
#         return 1*(np.array(img) < 128)


class ToTensor(object):
    def __call__(self, img):
        #         img = img.transpose((2, 0, 1))
        return torch.from_numpy(img)

        # class RandomExtract(object):
        #     def __init__(self, patch_h, patch_w, N_patches, inside=True):
        #         self.patch_h = patch_h
        #         self.pathc_w = patch_w
        #         self.N_patches = N_patches
        #         self.inside = inside

        #     def __call__(self, full_img, full_mask):
        #         if(self.N_patches % full_img.shape[])

        # def is_patch_inside_FOV(x, y, img_w, img_h, patch_h):
        #     x_ = x - int(img_w/2)  # origin (0,0) shifted to image center
        #     y_ = y - int(img_h/2)  # origin (0,0) shifted to image center
        #     # radius is 270 (from DRIVE db docs), minus the patch diagonal (assumed it is a square #this is the limit to contain the full patch in the FOV
        #     R_inside = 270 - int(patch_h * np.sqrt(2.0) / 2.0)
        #     radius = np.sqrt((x_*x_)+(y_*y_))
        #     if radius < R_inside:
        #         return True
        #     else:
        #         return False
