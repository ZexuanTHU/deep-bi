import torch
import torchvision
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import sys
import numpy as np
sys.path.append('./data/')
from dataset import FruitFlyNeuronDataset
from cv2transforms import composed_transforms, RandomExtract


ROOT_DIR = 'DRIVE/training/'
training_dir = {_: os.path.join(ROOT_DIR, _)
                for _ in ['images', '1st_manual', 'mask']}


def show_img(item):
    for i in item:
        assert (len(np.array(item[i]).shape) == 3 or 2)
        plt.figure()
        plt.imshow(item[i])


def show_patches(patches, patches_masks, N_show=10):
    patches = np.reshape(patches, (190000, 48, 48))
    patches_masks = np.reshape(patches_masks, (190000, 48, 48))
    for i in range(N_show):
        plt.figure()        
        ax1 = plt.subplot(1,2,1)
        ax1.set_title('patch #{}'.format(i))
        plt.imshow(patches[i])
        ax2 = plt.subplot(1,2,2)
        ax2.set_title('mask patch #{}'.format(i))
        plt.imshow(patches_masks[i])
    plt.show()


compose = composed_transforms()

# imgs_dataset = FruitFlyNeuronDataset(root_dir=training_dir, transforms=compose)
# comp_imgs_dataset = FruitFlyNeuronDataset(root_dir=training_dir)

# for i in range(len(imgs_dataset)):
#     show_img(imgs_dataset[i])
#     show_img(comp_imgs_dataset[i])
#     if i == 3:
#         plt.show()
#         break

test = FruitFlyNeuronDataset(root_dir=training_dir, transforms=compose)
full_imgs = np.empty((20, 584, 565))
full_masks = np.empty((20, 584, 565))
for i in range(len(test)):
    full_imgs[i] = test[i]['images']
    full_masks[i] = test[i]['mask']
full_imgs = np.reshape(full_imgs, (20, 584, 565, 1)).transpose((0, 3, 1, 2))
full_masks = np.reshape(full_masks, (20, 584, 565, 1)).transpose((0, 3, 1, 2))
rx = RandomExtract(patch_h=48, patch_w=48, N_patches=190000)
patches, patches_masks = rx(full_imgs=full_imgs, full_masks=full_masks)
show_patches(patches, patches_masks)