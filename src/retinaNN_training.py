import torch
import torchvision
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
# transforms.Compose([
#     transforms.GrayScale(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     CLAHE(),
#     AdjustGamma()
# ])

import sys
sys.path.append('./data/')
from dataset import FruitFlyNeuronDataset

ROOT_DIR = 'DRIVE/training/'
training_dir = {_: os.path.join(ROOT_DIR, _)
                for _ in ['images', '1st_manual', 'mask']}
print(os.path, ROOT_DIR)
imgs_dataset = FruitFlyNeuronDataset(root_dir=training_dir)
item = imgs_dataset[0]
for i in item:
    plt.figure()
    plt.imshow(item[i])
plt.show()