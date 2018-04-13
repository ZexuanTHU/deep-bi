import torch
import torchvision
import torchvision.transforms as transforms
from cv2transforms import CLAHE, AdjustGamma

transforms.Compose([
    transforms.GrayScale(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    CLAHE(),
    AdjustGamma()
])

