import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn
import os
import matplotlib.pyplot as plt
import sys
import numpy as np
sys.path.append('./')
from args import opt
sys.path.append('./lib/')
from dataset import FruitFlyNeuronDataset
from cv2transforms import composed_transforms, RandomExtract
from utils import show_img, show_patches, initialize_weights


patch_h = int(opt.patch_height)
patch_w = int(opt.patch_width)
N_patches = int(opt.N_subimgs)
root_dir = opt.root_dir

training_dir = {_: os.path.join(root_dir, _)
                for _ in ['images', '1st_manual', 'mask']}

compose = composed_transforms()

# imgs_dataset = FruitFlyNeuronDataset(root_dir=training_dir, transforms=compose)
# comp_imgs_dataset = FruitFlyNeuronDataset(root_dir=training_dir)

# for i in range(len(imgs_dataset)):
#     show_img(imgs_dataset[i])
#     show_img(comp_imgs_dataset[i])
#     if i == 3:
#         plt.show()
#         break

training_dataset = FruitFlyNeuronDataset(
    root_dir=training_dir, transforms=compose)
full_imgs = np.empty((20, 584, 565))
full_masks = np.empty((20, 584, 565))
for i in range(len(training_dataset)):
    full_imgs[i] = training_dataset[i]['images']
    full_masks[i] = training_dataset[i]['mask']
full_imgs = np.reshape(full_imgs, (20, 584, 565, 1)).transpose((0, 3, 1, 2))
full_masks = np.reshape(full_masks, (20, 584, 565, 1)).transpose((0, 3, 1, 2))
rx = RandomExtract(patch_h=patch_h, patch_w=patch_w, N_patches=N_patches)
patches, patches_masks = rx(full_imgs=full_imgs, full_masks=full_masks)
show_patches(patches, patches_masks)


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels,
                               kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)

# U-Net model


class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.enc1 = _EncoderBlock(3, 64)
        self.enc2 = _EncoderBlock(64, 128)
        self.enc3 = _EncoderBlock(128, 256)
        self.enc4 = _EncoderBlock(256, 512, dropout=True)
        self.center = _DecoderBlock(512, 1024, 512)
        self.dec4 = _DecoderBlock(1024, 512, 256)
        self.dec3 = _DecoderBlock(512, 256, 128)
        self.dec2 = _DecoderBlock(256, 128, 64)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(enc4)
        dec4 = self.dec4(
            torch.cat([center, F.upsample(enc4, center.size()[2:], mode='bilinear')], 1))
        dec3 = self.dec3(
            torch.cat([dec4, F.upsample(enc3, dec4.size()[2:], mode='bilinear')], 1))
        dec2 = self.dec2(
            torch.cat([dec3, F.upsample(enc2, dec3.size()[2:], mode='bilinear')], 1))
        dec1 = self.dec1(
            torch.cat([dec2, F.upsample(enc1, dec2.size()[2:], mode='bilinear')], 1))
        final = self.final(dec1)
        return F.upsample(final, x.size()[2:], mode='bilinear')


net = UNet(10)
print(net)
