import matplotlib.pyplot as plt
import numpy as np
from torch import nn


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
