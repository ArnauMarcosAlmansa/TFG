import copy
import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch as t
import torch.utils.data

from src.config import device, fix_cuda
from src.dataloaders.NerfDataloader import NerfDataset
from src.dataloaders.SyntheticDataloader import SyntheticEODataset
from src.models.layers.PositionalEncode import PositionalEncode, Mapping
from src.training.StaticRenderTrainer import StaticRenderTrainer
from src.training.decorators.Checkpoint import Checkpoint
from src.training.decorators.Tensorboard import Tensorboard
from src.training.decorators.Validation import Validation
from src.volume_render.cameras.PinholeCamera import PinholeCamera
from src.volume_render.SimpleRenderer import SimpleRenderer


class Test(t.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        width = 256
        self.encode = Mapping(10, 3)
        self.block1 = t.nn.Sequential(
            t.nn.Linear(60, width),
            t.nn.ReLU(),
            t.nn.Linear(width, width),
            t.nn.ReLU(),
            t.nn.Linear(width, width),
            t.nn.ReLU(),
            t.nn.Linear(width, width),
            t.nn.ReLU(),
            t.nn.Linear(width, width),
            t.nn.ReLU(),
            t.nn.Linear(width, width),
            t.nn.ReLU(),
            t.nn.Linear(width, width),
            t.nn.ReLU(),
        )

        self.block2 = t.nn.Sequential(
            t.nn.Linear(width + 60, width),
            t.nn.ReLU(),
            t.nn.Linear(width, width),
            t.nn.ReLU(),
            t.nn.Linear(width, width // 2),
            t.nn.ReLU(),
            t.nn.Linear(width // 2, 4),
        )

    def forward(self, x):
        x = self.encode(x)
        y = self.block1(x)
        y = torch.cat([y, x], -1)
        y = self.block2(y)
        return y[:, :3], y[:, 3],


if __name__ == '__main__':
    fix_cuda()

    # data = SyntheticEODataset("/home/amarcos/workspace/TFG/scripts/generated_eo_test_data/")
    train_data = NerfDataset("/DATA/nerf_synthetic/lego/transforms_train.json")
    val_data = NerfDataset("/DATA/nerf_synthetic/lego/transforms_val.json")
    # test_data = NerfDataset("/DATA/nerf_synthetic/lego/transforms_test.json")

    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=1024)
    val_loader = torch.utils.data.DataLoader(val_data, shuffle=True, batch_size=1024)
#     test_loader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=1024 * 8)

    c = PinholeCamera(800, 800, train_data.focal, t.eye(4), 4, 6)
    model = Test().to(device)
    loss = t.nn.MSELoss()
    optim = t.optim.Adam(params=model.parameters(), lr=0.01, betas=(0.9, 0.999))
    r = SimpleRenderer(c, model, 100)

    trainer = StaticRenderTrainer(model, optim, loss, train_loader, 'NERF_LEGO_TEST_4', renderer=r)
    trainer = Checkpoint(trainer, "./checkpoints_nerf/")
    # trainer = Validation(trainer, val_loader)
    trainer = Tensorboard(trainer)

    trainer.train(1000)

    torch.no_grad()
    model.eval()

    _, _, _ = next(iter(val_loader))

    c.pose = train_data.pose
    images = []

    for o in np.arange(0, 1, 1):
        print(o)
        c.pose = copy.deepcopy(train_data.pose)
        c.pose[0, 3] += o
        im = r.render()
        # im = (im - im.min()) / (im.max() - im.min())
        im = (im.detach().cpu().numpy() * 255).astype(np.uint8)
        images.append(im)
        plt.imshow(images[-1])
        plt.show()

    os.environ['IMAGEIO_FFMPEG_EXE'] = '/usr/bin/ffmpeg'

    writer = imageio.get_writer('test.mp4', fps=10)

    for im in images:
        writer.append_data(im)
    writer.close()
