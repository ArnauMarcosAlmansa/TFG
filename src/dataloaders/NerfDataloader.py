import json
import os.path
import random

import cv2
import numpy as np
import torch

from src.config import device
from src.volume_render.cameras.PinholeCamera import PinholeCamera


class NerfDataset:
    def __init__(self, json_path, transform=None, size=800):
        self.images = []
        self.width = 0
        self.height = 0
        self.transform = transform

        self.focal = 0
        self.pose = None

        self.early = True

        dirname = os.path.dirname(json_path)
        with open(json_path, "r") as f:
            doc = json.load(f)
            frames = doc["frames"]
            camera_angle_x = float(doc['camera_angle_x'])
            for index, frame in enumerate(frames, 1):
                im = cv2.imread(dirname + "/" + frame["file_path"] + ".png", cv2.IMREAD_UNCHANGED)
                im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA).astype(np.float32) / 255
                im = cv2.resize(im, (size, size))

                self.width = im.shape[1]
                self.height = im.shape[0]
                self.focal = .5 * self.width / np.tan(.5 * camera_angle_x)

                pose = np.zeros((4, 4,), dtype=np.float32)
                for i in range(4):
                    pose[i, :] = frame["transform_matrix"][i]

                self.pose = torch.from_numpy(pose)

                camera = PinholeCamera(im.shape[1], im.shape[0], self.focal, torch.from_numpy(pose).to(device), 4, 6)
                rays_o, rays_d = camera.get_rays()

                self.images.append((im, rays_o, rays_d))

                print(f"LOADED {index} / {len(frames)}")


    def compute_idex(self, item):
        image_index = item // (self.width * self.height)
        pixel_index = item % (self.width * self.height)
        x, y = pixel_index % self.height, pixel_index // self.height
        return image_index, x, y

    def __len__(self):
        return len(self.images) * self.width * self.height

    def __getitem__(self, item):
        image_index, x, y = self.compute_idex(item)

        if not self.early:
            return self.images[image_index][0][x, y][:3], self.images[image_index][1][x, y], self.images[image_index][2][x, y]

        while self.images[image_index][0][x, y][3] == 0:
            image_index, x, y = self.compute_idex(random.randint(0, len(self) - 1))

        return self.images[image_index][0][x, y][:3], self.images[image_index][1][x, y], self.images[image_index][2][x, y]
