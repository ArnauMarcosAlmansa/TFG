import pickle

import cv2
import matplotlib.pyplot as plt
import rasterio
import os
import numpy as np
import torch
from src.config import device


def pkl_save(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def pkl_load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


class SyntheticEODataset:
    def __init__(self, path, transform=None):
        self.transform = transform
        self.images = []

        png_filenames = sorted([filename for filename in os.listdir(path) if filename.endswith(".png")])
        for filename in png_filenames:
            full_filename = path + filename
            im = cv2.imread(full_filename)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255

            data = pkl_load(".".join(full_filename.split(".")[0:-1]) + ".pkl")
            self.images.append((torch.from_numpy(im).to(device), data))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        if not self.transform:
            return self.images[item]

        return self.transform(self.images[item])
