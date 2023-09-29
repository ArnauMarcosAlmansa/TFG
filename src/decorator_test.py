import statistics

import cv2
import imageio
import numpy as np
import pandas
import torch.nn
from matplotlib import pyplot as plt

from src.dataloaders.TimestampedImagesDataset import TimestampedImagesDataset, ToList
from src.models.trainer import Trainer, CheckPoint, Validation, RenderLossTrainer, Checkpoint
import rasterio
from torchvision import transforms
import seaborn as sns

from src.dataloaders.SateliteDataLoader import SateliteDataset, OnlyColorBands, PositionalEncode
from src.models.MyNerf import MyNerf

from src.config import device, fix_cuda


def train():
    data = TimestampedImagesDataset("../data/s2/", transform=transforms.Compose([
        ToList(['timestamp', 'image']),
    ]))

    train_loader = torch.utils.data.DataLoader(data,
                                               batch_size=1, shuffle=True,
                                               num_workers=0)
    model = MyNerf(inputs=60, outputs=3).to(device)
    optim = torch.optim.Adam(params=model.parameters(), lr=0.001)
    loss = torch.nn.MSELoss()

    trainer = RenderLossTrainer(
        model=model,
        optimizer=optim,
        loss=loss,
        train_loader=train_loader,
        name='TEST',
        height=60,
        width=60
    )
    trainer = Checkpoint(trainer, "./checkpoints_renderloss/")

    trainer.train(1000)

    return model


def query(model, when):
    model.eval()
    resolution = 60
    im = np.zeros((resolution, resolution, 3))
    se = PositionalEncode(10)
    step = 2 / resolution
    for i, y in enumerate(np.arange(-1, 1, step)):
        for j, x in enumerate(np.arange(-1, 1, step)):
            q = torch.tensor([x, y, when], device=device)
            im[i, j, :] = (model(se.do_positional_encoding(q))).cpu().detach().numpy()

    return im


def load_checkpoint(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])


if __name__ == '__main__':
    fix_cuda()
    # model = train()
    model = MyNerf(inputs=60, outputs=3).to(device)
    load_checkpoint(model, "./checkpoints_renderloss/TEST/checkpoint_epoch_0000000619.ckpt")

    images = []
    for t in np.arange(0, 1, 1 / 365):
        print(f"{t}/1")
        im = query(model, t)
        images.append(im)

    imageio.mimsave('video.gif', images, fps=20)
