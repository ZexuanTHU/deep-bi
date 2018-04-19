import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn
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
        ax1 = plt.subplot(1, 2, 1)
        ax1.set_title('patch #{}'.format(i))
        plt.imshow(patches[i])
        ax2 = plt.subplot(1, 2, 2)
        ax2.set_title('mask patch #{}'.format(i))
        plt.imshow(patches_masks[i])
    plt.show()


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


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


from ..utils import initialize_weights


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