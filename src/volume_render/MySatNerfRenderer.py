import gc

import torch

from src.config import device
from src.volume_render.cameras.Camera import Camera
import torch as t
import torch.nn.functional as F


def density2alpha(raw, dists, act_fn=F.relu):
    return 1. - t.exp(-act_fn(raw) * dists)


class MySatNerfRenderer:
    def __init__(self, camera: Camera, sampler, n_samples):
        self.camera = camera
        self.sampler = sampler
        self.n_samples = n_samples
        self.perturb = True
        self.lindisp = True
        self.raw_noise_std = 0
        self.white_bkgd = True

    def render_arbitrary_rays(self, rays_o, rays_d, sun_dirs, near=0., far=1.):
        rays_d = rays_d / self.n_samples

        sh = rays_d.shape  # [..., 3]
        # que el step sea un poco aleatorio para evitar overfitting
        step = (far - near) / self.n_samples

        rgb = t.zeros((rays_o.shape[0], 3)).to(device)
        for sample_index in range(self.n_samples - 1, -1, -1):
            points = rays_o + rays_d * step * sample_index

            density_slice, uncertainty_slice, albedo_slice, shading_slice, ambient_slice = self.sampler(points, sun_dirs)
            rgb_slice = albedo_slice
            density_slice = density_slice

            alpha_slice = density2alpha(density_slice, step).unsqueeze(-1)
            alpha_slice = torch.broadcast_to(alpha_slice, rgb_slice.shape)
            rgb = (1 - alpha_slice) * rgb + (alpha_slice * rgb_slice * shading_slice + ambient_slice)

        return rgb

    def render_matrix_rays(self, rays_o, rays_d, sun_dirs, near=0., far=1.):
        rgb = t.zeros((self.camera.h, self.camera.w, 3)).to(device)
        for i in range(rgb.shape[0]):
            rgb[i] = self.render_arbitrary_rays(rays_o[i], rays_d[i], sun_dirs, near, far).detach()

        return rgb

    def render(self, near=0., far=1.):
        rays_o, rays_d = self.camera.get_rays()
        return self.render_matrix_rays(rays_o, rays_d, near, far)

    def render_depth_arbitrary_rays(self, rays_o, rays_d, near=0., far=1.):
        rays_d = rays_d / self.n_samples

        # que el step sea un poco aleatorio para evitar overfitting
        step = (far - near) / self.n_samples

        depth = t.zeros((rays_o.shape[0], 1)).to(device)
        for sample_index in range(self.n_samples - 1, -1, -1):
            points = rays_o + rays_d * step * sample_index

            distance = step * sample_index

            density_slice, _, _, _, _ = self.sampler(points)
            density_slice = t.relu(density_slice)

            depth = depth + distance * density_slice.unsqueeze(-1)

        return depth

    def render_depth_matrix_rays(self, rays_o, rays_d, near=0., far=1.):
        rgb = t.zeros((self.camera.h, self.camera.w, 1)).to(device)
        for i in range(rgb.shape[0]):
            rgb[i] = self.render_depth_arbitrary_rays(rays_o[i], rays_d[i], near, far).detach()

        return rgb

    def render_depth(self, near=0., far=1.):
        rays_o, rays_d = self.camera.get_rays()
        return self.render_depth_matrix_rays(rays_o, rays_d, near, far)
