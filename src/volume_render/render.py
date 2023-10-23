import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch as t
import torch.utils.data

from src.config import device, fix_cuda
from src.dataloaders.SyntheticDataloader import SyntheticEODataset
from src.models.layers.PositionalEncode import PositionalEncode
from src.training.StaticRenderTrainer import StaticRenderTrainer
from src.training.decorators.Checkpoint import Checkpoint
from src.volume_render.cameras.PinholeCamera import PinholeCamera
from src.volume_render.Renderer import Renderer


class Test(t.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        width = 100
        self.encode = PositionalEncode(10)
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
        )

        self.block2 = t.nn.Sequential(
            t.nn.Linear(width + 60, width),
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

    data = SyntheticEODataset("/home/amarcos/workspace/TFG/scripts/generated_eo_data/")

    loader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=1024 * 10)

    torch.no_grad()

    c = PinholeCamera(240, 240, 50, t.eye(4))
    model = Test().to(device)
    loss = t.nn.MSELoss()
    optim = t.optim.Adam(params=model.parameters(), lr=0.001, betas=(0.9, 0.999))
    r = Renderer(c, model, 100)

    trainer = StaticRenderTrainer(model, optim, loss, loader, 'STATIC_RENDERED_PIXELS', renderer=r)
    trainer = Checkpoint(trainer, "./checkpoints_staticrender/")

    trainer.train(1000)

    torch.no_grad()
    model.eval()

    _, _, _, d = next(iter(loader))

    pose = d['camera_pose'].squeeze()
    c.pose = pose[0]
    images = []

    import gc

    for o in np.arange(-0.1, 0.1, 0.02):
        c.pose[0, 3] = o
        im = r.render()
        images.append((im.detach().cpu().numpy() * 255).astype(np.uint8))
        plt.imshow(images[-1])
        plt.show()

    imageio.mimsave('video_render_test.gif', images, duration=2000)
