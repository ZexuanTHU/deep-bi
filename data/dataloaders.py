import torch
import torchvision
import torchvision.transforms as transforms
from cv2transforms import CLAHE, AdjustGamma
import os
from dataset import FruitFlyNeuronDataset
import matplotlib.pyplot as plt
# transforms.Compose([
#     transforms.GrayScale(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     CLAHE(),
#     AdjustGamma()
# ])

ROOT_DIR = 'DRIVE/training/'
training_dir = {_: os.path.join(ROOT_DIR, _)
                for _ in ['images', '1st_manual', 'mask']}

imgs_dataset = FruitFlyNeuronDataset(root_dir=training_dir)
item = imgs_dataset[0]
for i in item:
    plt.figure()
    plt.imshow(item[i])
plt.show()