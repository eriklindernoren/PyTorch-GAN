import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)

        self.files = sorted(glob.glob(os.path.join(root, '%s' % mode) + '/*.*'))

    def __getitem__(self, index):

        img_pair = self.transform(Image.open(self.files[index % len(self.files)]))
        _, h, w = img_pair.shape
        half_w = int(w/2)

        item_A = img_pair[:, :, :half_w]
        item_B = img_pair[:, :, half_w:]

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return len(self.files)
