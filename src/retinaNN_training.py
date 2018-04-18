import torch
import torchvision
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import sys
sys.path.append('./data/')
from dataset import FruitFlyNeuronDataset
from cv2transforms import CLAHE

transforms.Compose([
    transforms.GrayScale(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    CLAHE()
    # AdjustGamma()
])
ROOT_DIR = 'DRIVE/training/'
training_dir = {_: os.path.join(ROOT_DIR, _)
                for _ in ['images', '1st_manual', 'mask']}


def show_img(item):
    for i in item:
        assert len(item[i].shape) == 3
        plt.figure()
        plt.imshow(item[i])

imgs_dataset = FruitFlyNeuronDataset(root_dir=training_dir)
for i in range(len(imgs_dataset)):
    show_img(imgs_dataset[i])
    if i == 3:
        plt.show()
        break
