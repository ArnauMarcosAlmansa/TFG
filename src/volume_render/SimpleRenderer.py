import gc

import torch

from src.config import device
from src.volume_render.cameras.Camera import Camera
import torch as t
import torch.nn.functional as F


def density2alpha(raw, dists, act_fn=F.relu):
    return 1. - t.exp(-act_fn(raw) * dists)


class SimpleRenderer:
    def __init__(self, camera: Camera, sampler, n_samples, n_channels=3):
        self.camera = camera
        self.sampler = sampler
        self.n_samples = n_samples
        self.perturb = True
        self.lindisp = True
        self.raw_noise_std = 0
        self.white_bkgd = True
        self.n_channels = n_channels

    def render_arbitrary_rays_old(self, rays_o, rays_d):
        # cuanta distancia tiene que avanzar el rayo cada iteracion
        step = (self.camera.far - self.camera.near) / self.n_samples
        rays_steps = rays_d * step

        rgb = t.zeros((rays_o.shape[0], 3)).to(device)
        Ti = t.ones((rays_o.shape[0], 3)).to(device)
        distance = 0
        for sample_index in range(self.n_samples):
            points = rays_o + rays_steps * sample_index + self.camera.near * rays_d
            distance += step

            rgb_slice, density_slice = self.sampler(points)
            rgb_slice = t.sigmoid(rgb_slice)
            density_slice = t.relu(density_slice)

            alpha_slice = density2alpha(density_slice, distance).unsqueeze(-1)
            alpha_slice = torch.broadcast_to(alpha_slice, rgb_slice.shape)

            Ti = Ti * (1 - alpha_slice)
            w = Ti * alpha_slice

            rgb = rgb + w * rgb_slice

        return rgb

    def raw2outputs(self, color_slices, density_slices, z_vals, rays_directions):
        # calcular las distancias entre muestras para poder calcular el alpha
        distances_between_samples = z_vals[..., 1:] - z_vals[..., :-1]
        distances_between_samples = torch.cat([distances_between_samples, torch.Tensor([1e10]).to(device).expand(distances_between_samples[..., :1].shape)],
                          -1)  # [N_rays, N_samples]
        distances_between_samples = distances_between_samples * torch.norm(rays_directions[..., None, :], dim=-1)

        # activacion de las densidades
        densities = F.relu(density_slices)
        # calcular alpha
        alpha = 1. - torch.exp(-densities * distances_between_samples)
        # calcular pesos
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:, :-1]
        # calcular profundidad
        depths = torch.sum(weights * z_vals, -1)

        # activacion del color
        color = torch.sigmoid(color_slices)  # [N_rays, N_samples, N_channels]
        # integrar el color
        rgb_map = torch.sum(weights[..., None] * color, -2)  # [N_rays, N_channels]

        return rgb_map, weights, depths

    def render_arbitrary_rays(self, rays_o, rays_d):
        t_vals = torch.linspace(0., 1., steps=self.n_samples, device=device)
        z_vals = self.camera.near * (1. - t_vals) + self.camera.far * (t_vals)

        # añadir ruido a las muestras para evitar el overfitting
        if self.perturb:
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            t_rand = torch.rand(z_vals.shape)

            z_vals = lower + (upper - lower) * t_rand


        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        shape = pts.shape

        pts_flat = torch.reshape(pts, (shape[0] * shape[1], shape[2]))

        input_dirs = rays_d[:, None].expand(pts.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])

        pts = torch.cat([pts_flat, input_dirs_flat], -1)

        rgb_slices, density_slices = self.sampler(pts)
        rgb_slices = torch.reshape(rgb_slices, (shape[0], shape[1], self.n_channels))
        density_slices = torch.reshape(density_slices, (shape[0], shape[1]))

        rgb_map, weights, depth_map = self.raw2outputs(rgb_slices, density_slices, z_vals, rays_d)

        return rgb_map, weights, depth_map

    def render_matrix_rays(self, rays_o, rays_d):
        rgb = t.zeros((self.camera.h, self.camera.w, self.n_channels)).to(device)
        weights = t.zeros((self.camera.h, self.camera.w, self.n_samples)).to(device)
        depth = t.zeros((self.camera.h, self.camera.w)).to(device)
        for i in range(rgb.shape[0]):
            rgb_map, weights_map, depth_map = self.render_arbitrary_rays(rays_o[i], rays_d[i])
            rgb[i] = rgb_map.detach()
            weights[i] = weights_map.detach()
            depth[i] = depth_map.detach()

        return rgb, weights, depth

    def render(self):
        rays_o, rays_d = self.camera.get_rays()
        return self.render_matrix_rays(rays_o, rays_d)
