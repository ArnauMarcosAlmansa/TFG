import json
import os.path

import cv2
import numpy as np
import torch

from src.volume_render.cameras.PinholeCamera import PinholeCamera


class NerfDataset:
    def __init__(self, json_path, transform=None):
        self.images = []
        self.width = 0
        self.height = 0
        self.transform = transform

        self.focal = 0
        self.pose = None

        dirname = os.path.dirname(json_path)
        with open(json_path, "r") as f:
            doc = json.load(f)
            frames = doc["frames"]
            camera_angle_x = float(doc['camera_angle_x'])
            for index, frame in enumerate(frames, 1):
                im = cv2.imread(dirname + "/" + frame["file_path"] + ".png")
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255

                self.width = im.shape[1]
                self.height = im.shape[0]
                self.focal = .5 * self.width / np.tan(.5 * camera_angle_x)

                pose = np.zeros((4, 4,), dtype=np.float32)
                for i in range(4):
                    pose[i, :] = frame["transform_matrix"][i]

                self.pose = torch.from_numpy(pose)

                camera = PinholeCamera(im.shape[1], im.shape[0], self.focal, torch.from_numpy(pose), 4, 6)
                rays_o, rays_d = camera.get_rays()

                self.images.append((im, rays_o, rays_d))

                print(f"LOADED {index} / {len(frames)}")

    def __len__(self):
        return len(self.images) * self.width * self.height

    def __getitem__(self, item):
        image_index = item // (self.width * self.height)
        pixel_index = item % (self.width * self.height)
        x, y = pixel_index % self.height, pixel_index // self.height
        return self.images[image_index][0][x, y], self.images[image_index][1][x, y], self.images[image_index][2][x, y]
