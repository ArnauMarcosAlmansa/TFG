import json
import os

import cv2
import numpy as np
import torch

from src.config import device
from src.volume_render.cameras.ComplexPinholeCamera import ComplexPinholeCamera


BAND_NAMES = [
    "BP850-27",
    "BP635-27",
    "BP590-27",
    "BP525-27",
    "BP505-27",
    "BP470-27",
    "BP324-27",
    "BP550-27",
]

class MultispectralRealDataset:
    def __init__(self, json_path, transform=None, size=800):
        self.images = []
        self.poses = []
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

            for index, frame in enumerate(frames, 1):

                cx, cy = float(frame['cx']) / 8, float(frame['cy']) / 8
                cax, cay = float(frame['camera_angle_x']), float(frame['camera_angle_y'])

                h, w = int(frame['h']) // 8, int(frame['w']) // 8

                self.width = w
                self.height = h

                fx = .5 * w / np.tan(.5 * cax)
                fy = .5 * h / np.tan(.5 * cay)

                bands = []
                altura, angle, _ = frame["file_path"].split("/")[-1].split(".")[0].split("-")
                dir = frame["file_path"].split("/")[1]

                for i, name in enumerate(BAND_NAMES, 1):
                    im = cv2.imread(dirname + "/" + dir + "/" + altura + "/" + angle + "/" + f"{i}-{name}.tiff", cv2.IMREAD_GRAYSCALE) / 255
                    im = im.astype(np.float32)
                    im = cv2.resize(im, (w, h))
                    bands.append(im)

                im = np.stack(bands, -1)

                self.width = im.shape[1]
                self.width = im.shape[1]
                self.height = im.shape[0]

                pose = np.zeros((4, 4,), dtype=np.float32)
                for i in range(4):
                    pose[i, :] = frame["transform_matrix"][i]

                self.pose = torch.from_numpy(pose)
                self.poses.append(self.pose)

                camera = ComplexPinholeCamera(w, h, fx, fy, cx, cy, torch.from_numpy(pose).to(device), 0, 65000)
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

        return self.images[image_index][0][x, y], self.images[image_index][1][x, y], self.images[image_index][2][x, y]
