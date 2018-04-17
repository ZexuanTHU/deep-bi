'FruitFlyNeronDataset Module'

__author__ = 'Shian Yip @ THU'

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
from PIL import Image
from skimage import io

# original_imgs_train = "../DRIVE/training/images/"
# groundTruth_imgs_train = "../DRIVE/training/1st_manual/"
# borderMasks_imgs_train = "../DRIVE/training/mask/"


class FruitFlyNeuronDataset(Dataset):
    '''Fruit Fly Neuron Dataset'''

    def __init__(self, root_dir, transforms=None):
        # self.original_imgs_dir = original_imgs_dir
        # self.ground_truth_dir = ground_truth_dir
        # self.border_masks_dir = border_masks_dir
        self.root_dir = root_dir
        self.transforms = transforms

    def __len__(self):
        return len(os.listdir(self.root_dir).items()[0])

    def __getitem__(self, idx):
        item_name = {d: os.path.join(self.root_dir[d], sorted(
            os.listdir(self.root_dir[d]))[idx]) for d in self.root_dir}
        item = {_: io.imread(item_name[_]) for _ in item_name}

        if self.transforms:
            item = self.transforms(item)

        return item


if __name__ == '__main__':
    pass
