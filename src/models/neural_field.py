import statistics

import numpy as np
import pandas
import torch.nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from trainer import Trainer, CheckPoint, Validation
import os
import rasterio
from torchvision import transforms
import seaborn as sns


def column(it, name):
    for d in it:
        yield d[name]


class SateliteDataset:
    def __init__(self, path, transform=None):
        self.transform = transform
        self.points = []
        tif_filenames = [filename for filename in os.listdir(path) if filename.endswith(".tif")]
        for month, filename in enumerate(sorted(tif_filenames)):
            im = rasterio.open(path + filename).read()
            plt.imshow(im[0])
            plt.show()
            for y in range(im.shape[0]):
                for x in range(im.shape[1]):
                    bands = im[:, y, x]
                    self.points.append({'x': x, 'y': y, 'month': month, 'bands': bands})

        # normalizar

        min_x = min(p['x'] for p in self.points)
        min_y = min(p['y'] for p in self.points)

        max_x = max(p['x'] for p in self.points)
        max_y = max(p['y'] for p in self.points)

        for p in self.points:
            p['x'] = ((p['x'] - min_x) / max_x - 0.5) * 2
            p['y'] = ((p['y'] - min_y) / max_y - 0.5) * 2
            p['month'] = p['month'] / 12
            p['bands'] = p['bands'].astype(np.float32) / 1000

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


class SpatialEncode:
    def __init__(self, L):
        self.L = L

    def __call__(self, data):
        inputs, labels = data
        result = self.do_spatial_encoding(inputs)
        return result, labels

    def do_spatial_encoding(self, inputs):
        result = torch.zeros(inputs.shape[0] * self.L * 2)
        for i in range(inputs.shape[0]):
            for l in range(self.L):
                result[i * self.L * 2 + l * 2] = torch.sin(2 ** l * np.pi * inputs[i])
                result[i * self.L * 2 + l * 2 + 1] = torch.cos(2 ** l * np.pi * inputs[i])

        return result


class SateliteModel(torch.nn.Module):
    def __init__(self, inputs=3, depth=8, width=100):
        super().__init__()
        # añadir spatial encoding
        self.il = torch.nn.Linear(in_features=inputs, out_features=width)
        self.fcs = [torch.nn.Linear(in_features=width, out_features=width) for _ in range(depth)]
        self.ol = torch.nn.Linear(in_features=width, out_features=1)

    def forward(self, x):
        x = F.relu(self.il(x))
        for fc in self.fcs:
            x = F.relu(fc(x))
        x = F.sigmoid(self.ol(x))
        return x


class MyNerf(torch.nn.Module):
    def __init__(self):
        super().__init__()
        width = 256
        inputs = 60

        self.fc1 = torch.nn.Linear(in_features=inputs, out_features=width)
        self.fc2 = torch.nn.Linear(in_features=width, out_features=width)
        self.fc3 = torch.nn.Linear(in_features=width, out_features=width)
        self.fc4 = torch.nn.Linear(in_features=width, out_features=width)
        self.fc5 = torch.nn.Linear(in_features=width, out_features=width)

        self.fc6 = torch.nn.Linear(in_features=width + inputs, out_features=width)
        self.fc7 = torch.nn.Linear(in_features=width, out_features=width)
        self.fc8 = torch.nn.Linear(in_features=width, out_features=width)
        self.fc8 = torch.nn.Linear(in_features=width, out_features=128)

        self.fc9 = torch.nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        initial_x = x
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        # x = torch.cat([x, initial_x], 1)
        # x = F.relu(self.fc6(x))
        # x = F.relu(self.fc7(x))
        # x = self.fc8(x)
        # x = F.sigmoid(self.fc9(x))

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        x = F.leaky_relu(self.fc5(x))
        x = torch.cat([x, initial_x], -1)
        x = F.leaky_relu(self.fc6(x))
        x = F.leaky_relu(self.fc7(x))
        x = self.fc8(x)
        x = F.sigmoid(self.fc9(x))

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


def view_data():
    data = SateliteDataset("../../data/s2/")

    loader = torch.utils.data.DataLoader(data,
                                batch_size=5, shuffle=True,
                                num_workers=4)

    for m in column(loader, 'bands'):
        print(m)


def train():
    data = SateliteDataset("../../data/s2/", transform=transforms.Compose([
        ToTensor(),
        OnlyOneBand(),
        SpatialEncode(10)
    ]))

    train_data, test_data = torch.utils.data.random_split(data, [0.8, 0.2])

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=2, shuffle=True,
                                               num_workers=4)

    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=2, shuffle=True,
                                              num_workers=4)

    # for im in train_loader:
    #     print(im)

    print(len(train_loader))
    print(len(test_loader))

    # model = SateliteModel(inputs=60, width=256, depth=8)
    model = MyNerf()
    optim = torch.optim.Adam(params=model.parameters(), lr=0.0001)
    loss = torch.nn.MSELoss()
    trainer = Trainer(model=model, optimizer=optim, loss=loss, train_loader=train_loader, test_loader=test_loader,
                      checkpoint=CheckPoint("./checkpoints/"), validation=Validation())

    trainer.train(100)

    return model


def query(model):
    im = np.zeros((100, 100))
    se = SpatialEncode(10)
    for i, y in enumerate(np.arange(-1, 1, 0.02)):
        for j, x in enumerate(np.arange(-1, 1, 0.02)):
            q = torch.tensor([x, y, 0.5])
            im[i, j] = model(se.do_spatial_encoding(q))

    return im

if __name__ == '__main__':
    # compute_bands_correlations()
    # view_data()
    # exit()
    model = train()
    im = query(model)
    plt.imshow(im, cmap='gray')
    plt.show()
