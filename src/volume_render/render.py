import imageio
import numpy as np
import torch as t
import torch.utils.data

from src.config import device
from src.dataloaders.SyntheticDataloader import SyntheticEODataset
from src.models.layers.PositionalEncode import PositionalEncode
from src.training.StaticRenderTrainer import StaticRenderTrainer
from src.training.decorators.Checkpoint import Checkpoint
from src.volume_render.cameras.PinholeCamera import PinholeCamera
from src.volume_render.Renderer import Renderer


class Test(t.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        width = 256
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

    data = SyntheticEODataset("/home/amarcos/workspace/TFG/scripts/generated_eo_data/")

    loader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=1)

    c = PinholeCamera(80, 80, 50, t.eye(4))
    model = Test().to(device)
    loss = t.nn.MSELoss()
    optim = t.optim.Adam(params=model.parameters(), lr=0.001)
    r = Renderer(c, model, 100)

    trainer = StaticRenderTrainer(model, optim, loss, loader, 'STATIC_RENDERED', renderer=r)
    trainer = Checkpoint(trainer, "./checkpoints_staticrender/")

    trainer.train(1000)

    model.eval()

    im, d = next(loader)

    pose = d['camera_pose']
    c.pose = pose
    images = []
    for o in np.arange(-1, 1, 0.01):
        c.pose[0, 3] = o
        im = r.render()
        images.append(im)

    imageio.mimsave('video_render.gif', images, fps=20)
