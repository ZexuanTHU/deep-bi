import torch
import torchvision
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import sys
import numpy as np
sys.path.append('./data/')
from dataset import FruitFlyNeuronDataset
from cv2transforms import composed_transforms


ROOT_DIR = 'DRIVE/training/'
training_dir = {_: os.path.join(ROOT_DIR, _)
                for _ in ['images', '1st_manual', 'mask']}


def show_img(item):
    for i in item:
        assert (len(np.array(item[i]).shape) == 3 or 2)
        plt.figure()
        plt.imshow(item[i])


compose = composed_transforms()

imgs_dataset = FruitFlyNeuronDataset(root_dir=training_dir, transforms=compose)
for i in range(len(imgs_dataset)):
    show_img(imgs_dataset[i])
    if i == 3:
        plt.show()
        break
