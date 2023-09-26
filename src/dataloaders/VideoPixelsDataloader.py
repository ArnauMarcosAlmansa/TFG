import cv2
import numpy as np
import torch

from src.config import device


class VideoPixelsDataset:
    def __init__(self, path, transform=None, decimate=1, every_n=1):
        self.transform = transform
        self.points = []
        video = cv2.VideoCapture(path)

        if not video.isOpened():
            raise IOError("Could not open video.")

        t = 0
        while video.isOpened():
            ret, im = video.read()
            if not ret:
                print("break")
                break

            if t % every_n != 0:
                print("continue")
                t += 1
                continue

            print(f"Reading frame {t}")

            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            for y in range(0, im.shape[0], decimate):
                for x in range(0, im.shape[1], decimate):
                    color = im[y, x]
                    self.points.append({'x': x, 'y': y, 't': t, 'color': color})
            t += 1

        # normalizar

        min_x = min(p['x'] for p in self.points)
        min_y = min(p['y'] for p in self.points)

        max_x = max(p['x'] for p in self.points)
        max_y = max(p['y'] for p in self.points)
        max_t = max(p['t'] for p in self.points)

        for p in self.points:
            p['x'] = ((p['x'] - min_x) / max_x - 0.5) * 2
            p['y'] = ((p['y'] - min_y) / max_y - 0.5) * 2
            p['t'] = (p['t'] / max_t) * 2 * np.pi

    def __len__(self):
        return len(self.points)

    def __getitem__(self, item):
        if not self.transform:
            return self.points[item]

        return self.transform(self.points[item])


class ToTensor:
    def __call__(self, data):

        data = torch.tensor([data['x'], data['y'], data['t']], device=device), torch.tensor(data['color'], device=device)

        return data
