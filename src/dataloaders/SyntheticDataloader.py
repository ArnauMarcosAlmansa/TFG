import pickle

import cv2
import matplotlib.pyplot as plt
import rasterio
import os
import numpy as np
import torch
from src.config import device
from src.volume_render.cameras.Camera import Camera
from src.volume_render.cameras.PinholeCamera import PinholeCamera


def pkl_save(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def pkl_load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


class SyntheticEODataset:
    def __init__(self, path, transform=None):
        self.transform = transform
        self.pixels = []

        png_filenames = sorted([filename for filename in os.listdir(path) if filename.endswith(".png")])
        for num, filename in enumerate(png_filenames):
            full_filename = path + filename
            im = cv2.imread(full_filename)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255

            data = pkl_load(".".join(full_filename.split(".")[0:-1]) + ".pkl")

            camera = PinholeCamera(im.shape[1], im.shape[0], 50, torch.from_numpy(data['camera_pose']))
            rays_o, rays_d = camera.get_rays()

            sunpose = torch.from_numpy(data['sun_pose'].astype(np.float32)).squeeze()[0:3, 0:3]
            x_forward = torch.tensor([1.0, 0.0, 0.0])
            sun_dir = torch.matmul(sunpose, x_forward)

            data['j'] = num

            for i in range(im.shape[0]):
                for j in range(im.shape[1]):
                    self.pixels.append((torch.from_numpy(im[i, j]).to(device), rays_o[i, j], rays_d[i, j], sun_dir, data))

            print(f"Loaded {num + 1}/{len(png_filenames)}")

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, item):
        if not self.transform:
            return self.pixels[item]

        return self.transform(self.pixels[item])
