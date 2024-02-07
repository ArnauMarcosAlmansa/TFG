import numpy as np

from src.config import device
from src.volume_render.cameras.Camera import Camera
import torch
import torch.nn.functional as F


def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

class ComplexPinholeCamera(Camera):
    def __init__(self, w, h, fx, fy, cx, cy, pose, near, far):
        super().__init__(w, h, pose)
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.near = near
        self.far = far

    def get_rays(self):
        K = torch.tensor([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1], ])
        origins, directions = get_rays(self.h, self.w, K, self.pose)
        return origins.to(device), directions.to(device)
