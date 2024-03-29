import copy
import os
import random

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch as t
import torch.utils.data
from tqdm import trange

from src.config import device, fix_cuda
from src.dataloaders.MultispectralSyntheticEODataloader import MultispectralSyntheticEODataset
from src.dataloaders.NerfDataloader import NerfDataset, MultiSpectralNerfDataset
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


BAND_NAMES = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B09",
    "B11",
    "B12",
    "B8A",
]

class Test2(t.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        width = 256
        self.encode_points = Mapping(10, 3)
        self.encode_viewdirs = Mapping(4, 3)
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
            t.nn.ReLU()
        )

        self.block2 = t.nn.Sequential(
            t.nn.Linear(width + 60, width),
            t.nn.ReLU(),
            t.nn.Linear(width, width),
            t.nn.ReLU(),
            t.nn.Linear(width, width),
            t.nn.Linear(width, width + 1),
        )

        self.block3 = t.nn.Sequential(
            t.nn.Linear(width + 24, width // 2),
            t.nn.ReLU(),
            t.nn.Linear(width // 2, 12),
        )

    def forward(self, x):
        points = x[:, 0:3]
        viewdirs = x[:, 3:6]
        points = self.encode_points(points)
        viewdirs = self.encode_viewdirs(viewdirs)
        y = self.block1(points)
        y = torch.cat([y, points], -1)
        y = self.block2(y)
        density = y[:, -1]
        y = torch.cat([y[:, :-1], viewdirs], -1)
        color = self.block3(y)
        return color, density,


class Mini(t.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        width = 128
        self.encode_points = Mapping(10, 3)
        self.encode_viewdirs = Mapping(4, 3)
        self.block1 = t.nn.Sequential(
            t.nn.Linear(60, width),
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
            t.nn.Linear(width, width + 1),
        )

        self.block3 = t.nn.Sequential(
            t.nn.Linear(width + 24, width // 2),
            t.nn.ReLU(),
            t.nn.Linear(width // 2, 12),
        )

    def forward(self, x):
        points = x[:, 0:3]
        viewdirs = x[:, 3:6]
        points = self.encode_points(points)
        viewdirs = self.encode_viewdirs(viewdirs)
        y = self.block1(points)
        y = torch.cat([y, points], -1)
        y = self.block2(y)
        density = y[:, -1]
        y = torch.cat([y[:, :-1], viewdirs], -1)
        color = self.block3(y)
        return color, density,


def rebuild_rgb(im4: torch.Tensor):
    sh4 = im4.shape
    rgb_im = np.zeros((sh4[0], sh4[1], 3))
    rgb_im[:, :, 0] = im4[:, :, 0]
    rgb_im[:, :, 2] = im4[:, :, 3]
    rgb_im[:, :, 1] = im4[:, :, 1] - 0.5 * im4[:, :, 0] + im4[:, :, 2] - 0.5 * im4[:, :, 3]

    return rgb_im

if __name__ == '__main__':
    fix_cuda()

    bands = [
        (1, 0, 0),
        (0.5, 0.5, 0),
        (0, 0.5, 0.5),
        (0, 0, 1),
    ]

    # data = SyntheticEODataset("/home/amarcos/workspace/TFG/scripts/generated_eo_test_data/")
    train_data = MultispectralSyntheticEODataset("/home/amarcos/workspace/TFG/scripts/generated_neo_multispectral_data/transforms_train.json", bands, size=800)
    val_data = MultispectralSyntheticEODataset("/home/amarcos/workspace/TFG/scripts/generated_neo_multispectral_data/transforms_val.json", bands, size=800)
    test_data = MultispectralSyntheticEODataset("/home/amarcos/workspace/TFG/scripts/generated_neo_multispectral_data/transforms_test.json", bands, size=800)
    # test_data = NerfDataset("/DATA/nerf_synthetic/lego/transforms_test.json")

    k = 1

    # train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=4096 * k, generator=torch.Generator(device='cuda'),num_workers=4)
    # val_loader = torch.utils.data.DataLoader(val_data, shuffle=True, batch_size=4096 * k, generator=torch.Generator(device='cuda'),num_workers=4)
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=512 * k, generator=torch.Generator(device='cuda'), num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_data, shuffle=True, batch_size=512 * k, generator=torch.Generator(device='cuda'), num_workers=4)
#     test_loader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=1024 * 8)

    c = PinholeCamera(800, 800, train_data.focal, train_data.pose, 3, 8)
    model = Test2().to(device)
    loss = t.nn.MSELoss()
    optim = t.optim.RAdam(params=model.parameters(), lr=0.0005 * k, betas=(0.9, 0.999))
    r = SimpleRenderer(c, model, 128, n_channels=12)

    # r.render_arbitrary_rays(torch.tensor([[0, 0, 0]], device=device), torch.tensor([[1, 0, 0]], device=device))

    trainer = StaticRenderTrainer(model, optim, loss, train_loader, 'TEST_3', renderer=r)
    trainer = Validation(trainer, r, val_loader)
    # trainer = VisualValidation(trainer, r, val_data)
    # trainer = Validation(trainer, val_loader)
    trainer = Tensorboard(trainer)

    loop = TrainLoopWithCheckpoints(trainer, "./checkpoints_multinerf_neo_cubes/")

    loop.train(0)

    torch.no_grad()
    model.eval()

    _, _, _ = next(iter(val_loader))

    c.pose = train_data.pose
    images = []

    # posei = random.randint(0, len(train_data.poses) - 1)
    # for o in np.arange(0, 1, 1):
    #     print(o)
    #     c.pose = copy.deepcopy(train_data.poses[posei]).to(device)
    #     rgb, disp, acc, weights, depth = r.render()
    #     # im = (im - im.min()) / (im.max() - im.min())
    #     im = rgb.detach().cpu().numpy()
    #     images.append(im)
    #     plt.imshow(images[-1][:, :, 0], cmap='gray')
    #     plt.show()
    #     plt.imshow(images[-1][:, :, 1], cmap='gray')
    #     plt.show()
    #     plt.imshow(images[-1][:, :, 2], cmap='gray')
    #     plt.show()
    #     plt.imshow(images[-1][:, :, 3], cmap='gray')
    #     plt.show()
    #     plt.imshow(cv2.cvtColor(images[-1][:, :, 1:4], cv2.COLOR_BGR2RGB))
    #     plt.show()
    #     plt.imshow(cv2.cvtColor(train_data.images[posei][0][:, :, 1:4], cv2.COLOR_BGR2RGB))
    #     plt.show()
    #     plt.imshow(depth.detach().cpu().numpy())
    #     plt.show()


    total_depth_mse = 0
    total_mse = 0
    for posei in trange(len(test_data.poses)):
        c.pose = copy.deepcopy(test_data.poses[posei]).to(device)
        rgb, weights, depth = r.render()
        # im = (im - im.min()) / (im.max() - im.min())
        im = rgb.detach().cpu().numpy()
        depth = depth.detach().cpu().numpy()

        gt = test_data.images[posei][0]
        depth_gt = test_data.depths[posei]

        for i, band in zip(range(12), BAND_NAMES):
            cv2.imwrite(f"reconstructed-{band}.png", (im[:, :, i] * 255).astype(np.uint8))
            cv2.imwrite(f"original-{band}.png", (gt[:, :, i] * 255).astype(np.uint8))

        plt.imshow(depth)
        plt.imsave(f"reconstructed-DEPTH.png")
        plt.imshow(depth_gt)
        plt.imsave(f"original-DEPTH.png")

        plt.imshow(depth)
        plt.show()
        plt.imshow(cv2.cvtColor(im[:, :, 1:4], cv2.COLOR_BGR2RGB))
        plt.show()
        plt.imshow(cv2.cvtColor(gt[:, :, 1:4], cv2.COLOR_BGR2RGB))
        plt.show()

        total_mse += np.mean(np.square(im - gt))
        total_depth_mse += np.mean(np.square(depth - depth_gt))


    print(f"val MSE = {total_mse / len(val_data.poses)}")
    print(f"val DEPTH MSE = {total_depth_mse / len(val_data.poses)}")


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

    writer = imageio.get_writer('TEST_3.mp4', fps=10)

    for im in images:
        writer.append_data(im)
    writer.close()
