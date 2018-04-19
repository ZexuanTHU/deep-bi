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


class RandomExtract(object):
    def __init__(self, patch_h, patch_w, N_patches, inside=True):
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.N_patches = N_patches
        self.inside = inside

    def __call__(self, full_imgs, full_masks):
        if (self.N_patches % full_imgs.shape[0] != 0):
            print("N_patches: plase enter a multiple of 20")
            exit()
        assert (len(full_imgs.shape) == 4 and len(
            full_masks.shape) == 4)  # 4D arrays
        # check the channel is 1 or 3
        assert (full_imgs.shape[1] == 1 or full_imgs.shape[1] == 3)
        assert (full_masks.shape[1] == 1)  # masks only black and white
        assert (full_imgs.shape[2] == full_masks.shape[2]
                and full_imgs.shape[3] == full_masks.shape[3])
        patches = np.empty(
            (self.N_patches, full_imgs.shape[1], self.patch_h, self.patch_w))
        patches_masks = np.empty(
            (self.N_patches, full_masks.shape[1], self.patch_h, self.patch_w))
        img_h = full_imgs.shape[2]  # height of the full image
        img_w = full_imgs.shape[3]  # width of the full image
        # (0,0) in the center of the image
        # N_patches equally divided in the full images
        patch_per_img = int(self.N_patches/full_imgs.shape[0])
        print("patches per full image: " + str(patch_per_img))
        iter_tot = 0  # iter over the total numbe rof patches (N_patches)
        for i in range(full_imgs.shape[0]):  # loop over the full images
            k = 0
            while k < patch_per_img:
                x_center = np.random.randint(
                    0+int(self.patch_w/2), img_w-int(self.patch_w/2))
                # print "x_center " +str(x_center)
                y_center = np.random.randint(
                    0+int(self.patch_h/2), img_h-int(self.patch_h/2))
                # print "y_center " +str(y_center)
                # check whether the patch is fully contained in the FOV
                if self.inside == True:
                    if is_patch_inside_FOV(x_center, y_center, img_w, img_h, self.patch_h) == False:
                        continue
                patch = full_imgs[i, :, y_center-int(self.patch_h/2):y_center+int(
                    self.patch_h/2), x_center-int(self.patch_w/2):x_center+int(self.patch_w/2)]
                patch_mask = full_masks[i, :, y_center-int(self.patch_h/2):y_center+int(
                    self.patch_h/2), x_center-int(self.patch_w/2):x_center+int(self.patch_w/2)]
                patches[iter_tot] = patch
                patches_masks[iter_tot] = patch_mask
                iter_tot += 1  # total
                k += 1  # per full_img
        return patches, patches_masks


def is_patch_inside_FOV(x, y, img_w, img_h, patch_h):
    x_ = x - int(img_w/2)  # origin (0,0) shifted to image center
    y_ = y - int(img_h/2)  # origin (0,0) shifted to image center
    # radius is 270 (from DRIVE db docs), minus the patch diagonal (assumed it is a square #this is the limit to contain the full patch in the FOV
    R_inside = 270 - int(patch_h * np.sqrt(2.0) / 2.0)
    radius = np.sqrt((x_*x_)+(y_*y_))
    if radius < R_inside:
        return True
    else:
        return False


