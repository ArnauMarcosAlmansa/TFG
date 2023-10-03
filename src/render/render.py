import matplotlib.pyplot as plt
import numpy as np
import torch


class PinholeRenderer:
    def __init__(self, H, W, focal, model, near, far):
        self.H = H
        self.W = W
        self.focal = focal
        self.model = model
        self.near = near
        self.far = far
        self.randomize = True
        self.n_samples = 200



    def get_rays(self, c2w):
        # como se hace en el NeRF
        columns = torch.arange(0, self.W, dtype=torch.float32) - self.W /2
        rows = torch.arange(0, self.H, dtype=torch.float32) - self.H /2
        i, j = torch.meshgrid(columns, rows, indexing='xy')
        dirs = torch.stack([i / self.focal, -j / self.focal, -torch.ones_like(i)], -1)
        ends = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
        origins = torch.broadcast_to(c2w[:3, -1], ends.shape)

        return origins, ends

    def render_rays(self, origins, ends):
        depths = torch.linspace(self.near, self.far, self.n_samples)
        depths += torch.rand(depths.size()) * (self.far - self.near) / self.n_samples

        points: torch.Tensor = origins[...,None,:] + ends[...,None,:] * depths[...,:,None]

        flat_points = torch.reshape(points, (-1, 3))

        raw = self.model(flat_points)
        sigma = torch.relu(raw[..., 3])
        rgb = torch.relu(raw[..., :3])

        dists = torch.concat([depths[..., 1:] - depths[..., :-1], torch.broadcast_to(torch.tensor([1e10]), depths[..., :1].shape)], -1)
        alpha = 1. - torch.exp(-sigma * dists)
        weights = alpha * torch.cumprod(1. - alpha + 1e-10, -1)

        rgb_map = torch.sum(weights[..., None] * rgb, -2)
        depth_map = torch.sum(weights * depths, -1)
        acc_map = torch.sum(weights, -1)

        return rgb_map, depth_map, acc_map


if __name__ == '__main__':




    r = PinholeRenderer(60, 60, 0.2, model, 1, 10)
    origins, ends = r.get_rays(torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]))
    rgb_map, depth_map, acc_map = r.render_rays(origins, ends)
    rgb_map = torch.reshape(rgb_map, (60, 60, 3))

    plt.imshow(rgb_map)
    plt.show()






