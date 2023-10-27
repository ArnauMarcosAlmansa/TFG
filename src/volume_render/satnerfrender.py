import copy

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch as t
import torch.utils.data

from src.config import device, fix_cuda
from src.dataloaders.SyntheticDataloader import SyntheticEODataset
from src.models.MySatNerf import MySatNerf
from src.models.losses.SatNerfLoss import SatNerfLoss
from src.training.SatnerfRenderTrainer import SatnerfRenderTrainer
from src.training.decorators.Checkpoint import Checkpoint
from src.volume_render.MySatNerfRenderer import MySatNerfRenderer
from src.volume_render.cameras.PinholeCamera import PinholeCamera


if __name__ == '__main__':
    fix_cuda()

    data = SyntheticEODataset("/home/amarcos/workspace/TFG/scripts/generated_eo_data/")

    loader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=1024 * 3)

    torch.no_grad()

    c = PinholeCamera(120, 120, 50, t.eye(4))
    model = MySatNerf().to(device)
    loss = SatNerfLoss()
    optim = t.optim.Adam(params=model.parameters(), lr=0.001, betas=(0.9, 0.999))
    r = MySatNerfRenderer(c, model, 100)

    trainer = SatnerfRenderTrainer(model, optim, loss, loader, 'STATIC_RENDERED_PIXELS_NOBACKGROUND', renderer=r)
    trainer = Checkpoint(trainer, "./checkpoints_satnerfrender/")

    trainer.train(10000)

    torch.no_grad()
    model.eval()

    _, _, _, _, d = next(iter(loader))

    pose = d['camera_pose'].squeeze()
    c.pose = pose[0]
    images = []

    for o in np.arange(-0.001, 0.001, 0.0002):
        print(o)
        c.pose = copy.deepcopy(pose[0])
        c.pose[0, 3] += o
        im = r.render_depth()
        im = (im - im.min()) / (im.max() - im.min())
        im = (im.detach().cpu().numpy() * 255).astype(np.uint8)
        images.append(im)
        plt.imshow(images[-1])
        plt.show()

    imageio.mimsave('video_render_test.gif', images, duration=100)
