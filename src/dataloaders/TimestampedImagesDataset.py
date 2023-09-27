import os

import cv2
import numpy as np
import rasterio


class TimestampedImagesDataset:
    def __init__(self, path, transform=None):
        self.transform = transform
        self.timestamped_images = []

        tif_filenames = [filename for filename in os.listdir(path) if filename.endswith(".tif")]
        for month, filename in enumerate(sorted(tif_filenames)):
            im = rasterio.open(path + filename).read().transpose((1, 2, 0)).astype(np.float32)[:, :, 0:3] / 1000
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            self.timestamped_images.append({'timestamp': month / 12, 'image': im})

    def __len__(self):
        return len(self.timestamped_images)

    def __getitem__(self, item):
        if not self.transform:
            return self.timestamped_images[item]

        return self.transform(self.timestamped_images[item])


class ToList:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        return [data[key] for key in self.keys]


class ToTensor:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        timestamp, image = data
        return [data[key] for key in self.keys]