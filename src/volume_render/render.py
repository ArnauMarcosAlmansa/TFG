import copy
import os
import random

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
from src.training.Trainer import TrainLoopWithCheckpoints
from src.training.decorators.Checkpoint import Checkpoint
from src.training.decorators.Tensorboard import Tensorboard
from src.training.decorators.Validation import Validation
from src.training.decorators.VisualValidation import VisualValidation
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

# https://keras.io/examples/vision/nerf/
if __name__ == '__main__':
    fix_cuda()

    # data = SyntheticEODataset("/home/amarcos/workspace/TFG/scripts/generated_eo_test_data/")
    train_data = NerfDataset("/home/arnau-marcos-almansa/Downloads/nerf_synthetic/lego/transforms_train.json", size=80)
    val_data = NerfDataset("/home/arnau-marcos-almansa/Downloads/nerf_synthetic/lego/transforms_val.json", size=80)
    # test_data = NerfDataset("/DATA/nerf_synthetic/lego/transforms_test.json")

    k = 1

    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=4096 * k, generator=torch.Generator(device='cuda'), num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_data, shuffle=True, batch_size=4096 * k, generator=torch.Generator(device='cuda'), num_workers=4)
#     test_loader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=1024 * 8)

    c = PinholeCamera(80, 80, train_data.focal, train_data.pose, 4, 6)
    model = Test().to(device)
    loss = t.nn.MSELoss()
    optim = t.optim.RAdam(params=model.parameters(), lr=0.0005 * k, betas=(0.9, 0.999))
    r = SimpleRenderer(c, model, 128)

    # r.render_arbitrary_rays(torch.tensor([[0, 0, 0]], device=device), torch.tensor([[1, 0, 0]], device=device))

    trainer = StaticRenderTrainer(model, optim, loss, train_loader, 'NERF_LEGO_TEST_16', renderer=r)
    trainer = Validation(trainer, r, val_loader)
    trainer = VisualValidation(trainer, r, val_data)
    # trainer = Validation(trainer, val_loader)
    trainer = Tensorboard(trainer)

    loop = TrainLoopWithCheckpoints(trainer, "./checkpoints_nerf/")

    loop.train(1000)

    torch.no_grad()
    model.eval()

    _, _, _ = next(iter(val_loader))

    c.pose = val_data.pose
    images = []

    posei = random.randint(0, len(val_data.poses) - 1)

    c.pose = copy.deepcopy(val_data.poses[posei]).to(device)
    rgb, disp, acc, weights, depth = r.render()
    # im = (im - im.min()) / (im.max() - im.min())
    im = (rgb.detach().cpu().numpy() * 255).astype(np.uint8)
    images.append(im)
    plt.imshow(images[-1])
    plt.show()
    plt.imshow(val_data.images[posei][0])
    plt.show()
    plt.imshow(depth.detach().cpu().numpy())
    plt.show()


    # for o in np.arange(0, 1, 1):
    #     print(o)
    #     c.pose = copy.deepcopy(train_data.pose).to(device)
    #     c.pose[0, 3] += o
    #     im = r.render()
    #     # im = (im - im.min()) / (im.max() - im.min())
    #     im = (im.detach().cpu().numpy() * 255).astype(np.uint8)
    #     images.append(im)
    #     plt.imshow(images[-1])
    #     plt.show()

    os.environ['IMAGEIO_FFMPEG_EXE'] = '/usr/bin/ffmpeg'

    writer = imageio.get_writer('NERF_LEGO_TEST_16.mp4', fps=10)

    for im in images:
        writer.append_data(im)
    writer.close()
