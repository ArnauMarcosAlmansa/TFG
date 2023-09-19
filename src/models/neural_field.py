import statistics

import numpy as np
import pandas
import torch.nn
from matplotlib import pyplot as plt

from trainer import Trainer, CheckPoint, Validation
import os
import rasterio
from rasterio.plot import show
from torchvision import transforms
from statistics import mean, stdev
import seaborn as sns


class SateliteDataset:
    def __init__(self, path, transform=None):
        self.transform = transform
        self.points = []
        tif_filenames = [filename for filename in os.listdir(path) if filename.endswith(".tif")]
        for month, filename in enumerate(sorted(tif_filenames)):
            im = rasterio.open(path + filename).read()
            for y in range(im.shape[0]):
                for x in range(im.shape[1]):
                    bands = im[:, y, x]
                    self.points.append({'x': x, 'y': y, 'month': month, 'bands': bands})

        # normalizar

        min_x = min(p['x'] for p in self.points)
        min_y = min(p['y'] for p in self.points)
        min_month = min(p['month'] for p in self.points)

        max_x = max(p['x'] for p in self.points)
        max_y = max(p['y'] for p in self.points)
        max_month = max(p['month'] for p in self.points)

        for p in self.points:
            p['x'] = (p['x'] - min_x) / max_x
            p['y'] = (p['y'] - min_y) / max_y
            p['month'] = (p['month'] - min_month) / max_month

        mean_x = mean(p['x'] for p in self.points)
        mean_y = mean(p['y'] for p in self.points)
        mean_month = mean(p['month'] for p in self.points)

        std_x = stdev(p['x'] for p in self.points)
        std_y = stdev(p['y'] for p in self.points)
        std_month = stdev(p['month'] for p in self.points)

        for p in self.points:
            p['x'] = (p['x'] - mean_x) / std_x
            p['y'] = (p['y'] - mean_y) / std_y
            p['month'] = (p['month'] - mean_month) / std_month
            p['bands'] = p['bands'].astype(np.float32) / 255

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


class SpatialEncode:
    def __init__(self, L):
        self.L = L

    def __call__(self, data):
        inputs, labels = data
        result = torch.zeros(inputs.shape[0] * self.L * 2)
        for i in range(inputs.shape[0]):
            for l in range(self.L):
                result[i * self.L * 2 + l * 2] = torch.sin(inputs[i] + 2 * np.pi ** l)
                result[i * self.L * 2 + l * 2 + 1] = torch.cos(inputs[i] + 2 * np.pi ** l)

        return result, labels


class SateliteModel(torch.nn.Module):
    def __init__(self, inputs=3, depth=8, width=100):
        super().__init__()
        # añadir spatial encoding
        self.il = torch.nn.Linear(in_features=inputs, out_features=width)
        self.fcs = [torch.nn.Linear(in_features=width, out_features=width) for _ in range(depth)]
        self.ol = torch.nn.Linear(in_features=width, out_features=1)

    def forward(self, x):
        x = torch.sin(self.il(x))
        for fc in self.fcs:
            x = torch.sin(fc(x))
        x = self.ol(x)
        return x


def stack_bands(data):
    stack = np.zeros((len(data), 12))
    for i, d in enumerate(data):
        stack[i] = d['bands']

    return stack


def compute_bands_correlations():
    data = SateliteDataset("../../data/s2/")
    bands_stack = stack_bands(data)
    df = pandas.DataFrame(bands_stack)
    cm = df.corr()
    sns.heatmap(cm, annot=True)
    plt.show()


def train():
    data = SateliteDataset("../../data/s2/", transform=transforms.Compose([
        ToTensor(),
        OnlyOneBand(),
        SpatialEncode(10)
    ]))

    train_data, test_data = torch.utils.data.random_split(data, [0.8, 0.2])

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=5, shuffle=True,
                                               num_workers=4)

    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=4, shuffle=True,
                                              num_workers=4)

    # for im in train_loader:
    #     print(im)

    print(len(train_loader))
    print(len(test_loader))

    model = SateliteModel(inputs=60, width=200, depth=12)
    optim = torch.optim.Adam(params=model.parameters(), lr=0.00001)
    loss = torch.nn.MSELoss()
    trainer = Trainer(model=model, optimizer=optim, loss=loss, train_loader=train_loader, test_loader=test_loader,
                      checkpoint=CheckPoint("./checkpoints/"), validation=Validation())

    trainer.train(100)


if __name__ == '__main__':
    # compute_bands_correlations()
    train()
