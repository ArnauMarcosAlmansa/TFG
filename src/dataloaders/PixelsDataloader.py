import cv2
import numpy as np
import torch

from src.config import device


class PixelsDataset:
    def __init__(self, path, transform=None):
        self.transform = transform
        self.points = []
        im = cv2.imread(path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255

        for y in range(im.shape[0]):
            for x in range(im.shape[1]):
                color = im[y, x]
                self.points.append({'x': x, 'y': y, 'color': color})

        # normalizar

        min_x = min(p['x'] for p in self.points)
        min_y = min(p['y'] for p in self.points)

        max_x = max(p['x'] for p in self.points)
        max_y = max(p['y'] for p in self.points)

        for p in self.points:
            p['x'] = ((p['x'] - min_x) / max_x - 0.5) * 2
            p['y'] = ((p['y'] - min_y) / max_y - 0.5) * 2

    def __len__(self):
        return len(self.points)

    def __getitem__(self, item):
        if not self.transform:
            return self.points[item]

        return self.transform(self.points[item])


class ToTensor:
    def __call__(self, data):

        data = torch.tensor([data['x'], data['y']], device=device), torch.tensor(data['color'], device=device)

        return data
