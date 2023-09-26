import statistics

import numpy as np
import pandas
import torch.nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from models.trainer import Trainer, CheckPoint, Validation
import os
import rasterio
from torchvision import transforms
import seaborn as sns

from src.dataloaders.SateliteDataLoader import SateliteDataset, ToTensor, OnlyColorBands, PositionalEncode
from src.models.MyNerf import MyNerf


def column(it, name):
    for d in it:
        yield d[name]


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


def view_bands():
    im = rasterio.open("/home/arnau/workspace/UAB/TFG/data/s2/2018-01.tif").read()

    for i in range(12):
        plt.imshow(im[i])
        plt.title(f"Band {i}")
        plt.show()


def train():
    data = SateliteDataset("../data/sfakeone/", transform=transforms.Compose([
        ToTensor(),
        OnlyColorBands(),
        PositionalEncode(10)
    ]))

    # train_data, test_data = torch.utils.data.random_split(data, [0.8, 0.2])
    #
    # train_loader = torch.utils.data.DataLoader(train_data,
    #                                            batch_size=5, shuffle=True,
    #                                            num_workers=4)
    #
    # test_loader = torch.utils.data.DataLoader(test_data,
    #                                           batch_size=5, shuffle=True,
    #                                           num_workers=4)

    train_loader = torch.utils.data.DataLoader(data,
                                               batch_size=512, shuffle=True,
                                               num_workers=4)

    # for im in train_loader:
    #     print(im)

    # model = SateliteModel(inputs=60, width=256, depth=8)
    model = MyNerf()
    # load_checkpoint(model, "/home/arnau/workspace/UAB/TFG/src/models/checkpoints/siren_checkpoint_epoch_99.ckpt")
    optim = torch.optim.Adam(params=model.parameters(), lr=0.001)
    loss = torch.nn.MSELoss()
    trainer = Trainer(model=model, optimizer=optim, loss=loss, train_loader=train_loader, test_loader=None,
                      checkpoint=CheckPoint("./checkpoints/"), validation=None)

    trainer.train(100)

    return model


def query(model):
    model.eval()
    resolution = 5
    im = np.zeros((resolution, resolution, 3))
    se = PositionalEncode(10)
    step = 2 / resolution
    for i, y in enumerate(np.arange(-1, 1, step)):
        for j, x in enumerate(np.arange(-1, 1, step)):
            q = torch.tensor([x, y, 1])
            im[i, j, :] = (model(se.do_positional_encoding(q))).detach().numpy()

    return im


def load_checkpoint(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])


if __name__ == '__main__':
    # compute_bands_correlations()
    # view_data()
    # exit()
    # view_bands()
    # exit()

    # view_bands()
    model = train()
    # model = MyNerf()
    # load_checkpoint(model, "/home/arnau/workspace/UAB/TFG/src/models/checkpoints/siren_checkpoint_epoch_99.ckpt")
    im = query(model)
    plt.imshow(im)
    plt.show()
