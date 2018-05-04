import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_x=None, transforms_lr=None, mode='train'):
        self.transform_x = transforms.Compose(transforms_x)
        self.transform_lr = transforms.Compose(transforms_lr)

        self.files = sorted(glob.glob('%s/*.*' % root))

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])

        x = self.transform_x(img)
        x_lr = self.transform_lr(img)

        return {'x': x, 'x_lr': x_lr}

    def __len__(self):
        return len(self.files)
