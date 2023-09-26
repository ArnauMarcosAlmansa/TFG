import rasterio
import os
import numpy as np
import torch
from src.config import device


class SateliteDataset:
    def __init__(self, path, transform=None):
        self.transform = transform
        self.points = []
        tif_filenames = [filename for filename in os.listdir(path) if filename.endswith(".jpg")]
        for month, filename in enumerate(sorted(tif_filenames)):
            im = rasterio.open(path + filename).read()
            for y in range(im.shape[0]):
                for x in range(im.shape[1]):
                    bands = im[:, y, x]
                    self.points.append({'x': x, 'y': y, 'month': month, 'bands': bands})

        # normalizar

        min_x = min(p['x'] for p in self.points)
        min_y = min(p['y'] for p in self.points)

        max_x = max(p['x'] for p in self.points)
        max_y = max(p['y'] for p in self.points)

        max_band0 = max(p['bands'][0] for p in self.points)
        max_band1 = max(p['bands'][1] for p in self.points)
        max_band2 = max(p['bands'][2] for p in self.points)
        max_color = max(max_band0, max_band1, max_band2)

        for p in self.points:
            p['x'] = ((p['x'] - min_x) / max_x - 0.5) * 2
            p['y'] = ((p['y'] - min_y) / max_y - 0.5) * 2
            p['month'] = p['month'] / 12
            p['bands'] = p['bands'].astype(np.float32) / max_color

    def __len__(self):
        return len(self.points)

    def __getitem__(self, item):
        if not self.transform:
            return self.points[item]

        return self.transform(self.points[item])


class OnlyOneBand:
    def __call__(self, data):
        inputs, labels = data
        labels = labels[0]
        return inputs, labels


class OnlyColorBands:
    def __call__(self, data):
        inputs, labels = data
        labels = torch.tensor([labels[2], labels[1], labels[0]])
        return inputs, labels


class StandardizeDict:
    def __init__(self, dict):
        self.dict = dict

    def __call__(self, item):
        for key, (mean, std) in self.dict.items():
            item[key] = (item[key] - mean) / std

        return item


class NormalizeDict:
    def __init__(self, dict):
        self.dict = dict

    def __call__(self, item):
        for key, (min, max) in self.dict.items():
            item[key] = (item[key] - min) / max

        return item


class ToTensor:
    def __call__(self, item):
        return torch.tensor([float(item['x']), float(item['y']), float(item['month'])]), torch.tensor(
            item['bands'].astype(np.float32))


class PositionalEncode:
    def __init__(self, L):
        self.L = L

    def __call__(self, data):
        inputs, labels = data
        result = self.do_positional_encoding(inputs)
        return result, labels

    def do_positional_encoding(self, inputs):
        result = torch.zeros(inputs.shape[0] * self.L * 2, device=device)
        for i in range(inputs.shape[0]):
            for l in range(self.L):
                result[i * self.L * 2 + l * 2] = torch.sin(2 ** l * np.pi * inputs[i])
                result[i * self.L * 2 + l * 2 + 1] = torch.cos(2 ** l * np.pi * inputs[i])

        return result
