from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import torch as t
import torch.nn.functional as F
import torch.utils.data

from src.config import device
from src.dataloaders.SyntheticDataloader import SyntheticEODataset


class Camera(ABC):
    def __init__(self, w: int, h: int, pose):
        self.w = w
        self.h = h
        self.pose = pose

    @abstractmethod
    def get_rays(self):
        pass


class PinholeCamera(Camera):
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

        directions = t.sum(dirs[..., np.newaxis, :] * self.pose[:3, :3], -1)
        directions = F.normalize(directions, p=2.0, dim=-1)

        origins = self.pose[:3, -1].expand(directions.shape)

        return origins, directions


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


class Renderer:
    def __init__(self, camera: Camera, sampler, n_samples):
        self.camera = camera
        self.sampler = sampler
        self.n_samples = n_samples
        self.perturb = True
        self.lindisp = True
        self.raw_noise_std = 0
        self.white_bkgd = True

    # def raw2outputs(self, raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False):
    #
    #     raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - t.exp(-act_fn(raw) * dists)
    #
    #     dists = z_vals[..., 1:] - z_vals[..., :-1]
    #     dists = t.cat([dists, t.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]
    #
    #     dists = dists * t.norm(rays_d[..., None, :], dim=-1)
    #
    #     rgb = t.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    #     noise = 0.
    #     if raw_noise_std > 0.:
    #         noise = t.randn(raw[..., 3].shape) * raw_noise_std
    #
    #     alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    #     # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    #     weights = alpha * t.cumprod(t.cat([t.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:,
    #                       :-1]
    #     rgb_map = t.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
    #
    #     depth_map = t.sum(weights * z_vals, -1)
    #     disp_map = 1. / t.max(1e-10 * t.ones_like(depth_map), depth_map / t.sum(weights, -1))
    #     acc_map = t.sum(weights, -1)
    #
    #     if white_bkgd:
    #         rgb_map = rgb_map + (1. - acc_map[..., None])
    #
    #     return rgb_map, disp_map, acc_map, weights, depth_map
    #
    # def batchify_rays(self, rays_flat, chunk=1024 * 32, **kwargs):
    #     """Render rays in smaller minibatches to avoid OOM.
    #     """
    #     all_ret = {}
    #     for i in range(0, rays_flat.shape[0], chunk):
    #         ret = self.render_rays(rays_flat[i:i + chunk], **kwargs)
    #         for k in ret:
    #             if k not in all_ret:
    #                 all_ret[k] = []
    #             all_ret[k].append(ret[k])
    #
    #     all_ret = {k: t.cat(all_ret[k], 0) for k in all_ret}
    #     return all_ret
    #
    # def render_rays_0(self, ray_batch):
    #     N_samples = self.n_samples
    #     N_rays = ray_batch.shape[0]
    #     rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    #     bounds = t.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    #     near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]
    #
    #     t_vals = t.linspace(0., 1., steps=N_samples)
    #     if not self.lindisp:
    #         z_vals = near * (1. - t_vals) + far * (t_vals)
    #     else:
    #         z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))
    #
    #     z_vals = z_vals.expand([N_rays, N_samples])
    #
    #     if self.perturb:
    #         # get intervals between samples
    #         mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    #         upper = t.cat([mids, z_vals[..., -1:]], -1)
    #         lower = t.cat([z_vals[..., :1], mids], -1)
    #         # stratified samples in those intervals
    #         t_rand = t.rand(z_vals.shape)
    #
    #         z_vals = lower + (upper - lower) * t_rand
    #
    #     pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]
    #
    #     raw = self.sampler(pts)
    #     rgb_map, disp_map, acc_map, weights, depth_map = self.raw2outputs(raw, z_vals, rays_d, self.raw_noise_std,
    #                                                                       self.white_bkgd)
    #
    #     ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
    #     for k in ret:
    #         if t.isnan(ret[k]).any() or t.isinf(ret[k]).any():
    #             print(f"! [Numerical Error] {k} contains nan or inf.")
    #
    #     return ret

    def render(self, near=0., far=1., **kwargs):
        rays_o, rays_d = self.camera.get_rays()

        rays_d = rays_d / self.n_samples

        sh = rays_d.shape  # [..., 3]
        # que el step sea un poco aleatorio para evitar overfitting
        step = (far - near) / self.n_samples
        density2alpha = lambda raw, dists, act_fn=F.relu: 1. - t.exp(-act_fn(raw) * step)
        rgb = t.zeros((80, 80, 3)).to(device)
        for i in range(self.n_samples - 1, -1, -1):
            points = rays_o + rays_d * step * i
            distance_slice = t.dist(points, rays_o)

            rgb_slice, density_slice = self.sampler(t.reshape(points, (sh[0] * sh[1], sh[2])))
            rgb_slice = t.reshape(t.sigmoid(rgb_slice), (80, 80, 3))
            density_slice = t.reshape(t.relu(density_slice), (80, 80, 1))

            alpha_slice = density2alpha(density_slice, distance_slice)
            rgb = (1 - alpha_slice) * rgb + alpha_slice * rgb_slice

        return rgb


class PositionalEncode(t.nn.Module):
    def __init__(self, L, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.L = L

    def forward(self, x):
        return self.do_positional_encoding(x)

    def do_positional_encoding(self, inputs):
        result = t.zeros(inputs.shape[0], inputs.shape[1] * self.L * 2)
        for i in range(inputs.shape[1]):
            for l in range(self.L):
                result[:, i * self.L * 2 + l * 2] = t.sin(2 ** l * np.pi * inputs[:, i])
                result[:, i * self.L * 2 + l * 2 + 1] = t.cos(2 ** l * np.pi * inputs[:, i])

        return result.to(device)


class Sin(t.nn.Module):
    def forward(self, x):
        return t.sin(x)


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
    model.train()
    for i in range(10000):
        running_loss = 0
        print(f"STARTING EPOCH {i + 1}")
        for data in loader:
            im, d = data
            c.pose = d['camera_pose'].squeeze()

            image = r.render()
            model.zero_grad()
            l = loss(image, im.squeeze())
            l.backward()
            optim.step()
            running_loss += l

        print(f"LOSS = {running_loss / len(loader):.5f}")
        print(f"ENDING EPOCH {i + 1}")

        if i % 10 == 9:
            plt.figure()
            plt.title(f"EPOCH {i + 1}")
            plt.imshow(image.cpu().detach())
            plt.show()
