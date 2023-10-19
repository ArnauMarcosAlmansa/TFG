import numpy as np

from src.volume_render.cameras.Camera import Camera
import torch as t
import torch.nn.functional as F

class OrthographicCamera(Camera):
    def __init__(self, w, h, f, pose):
        super().__init__(w, h, pose)
        self.f = f

    def get_rays(self):
        xs = t.linspace(0, self.w - 1, self.w)
        ys = t.linspace(0, self.h - 1, self.h)

        i, j = t.meshgrid(xs, ys)
        i = i.t()
        j = j.t()

        dirs = t.stack([(i - self.w / 2) / self.f, -(j - self.h / 2) / self.f, -t.ones_like(i)], -1)

        directions = F.normalize(t.sum(dirs[..., np.newaxis, :] * self.pose[:3, :3], -1), p=2.0, dim=-1)

        origins = self.pose[:3, -1].expand(directions.shape)

        return origins, directions