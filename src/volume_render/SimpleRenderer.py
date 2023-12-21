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

    def raw2outputs(self, rgb_slices, density_slices, z_vals, rays_d, raw_noise_std=0, white_bkgd=False):
        raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).to(device).expand(dists[..., :1].shape)],
                          -1)  # [N_rays, N_samples]

        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        rgb = torch.sigmoid(rgb_slices)  # [N_rays, N_samples, 3]
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn(density_slices.shape) * raw_noise_std

        alpha = raw2alpha(density_slices + noise, dists)  # [N_rays, N_samples]
        # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:,
                          :-1]
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
        acc_map = torch.sum(weights, -1)

        if white_bkgd:
            rgb_map = rgb_map + (1. - acc_map[..., None])

        return rgb_map, disp_map, acc_map, weights, depth_map

    def render_arbitrary_rays(self, rays_o, rays_d):
        # cuanta distancia tiene que avanzar el rayo cada iteracion
        step = (self.camera.far - self.camera.near) / self.n_samples
        rays_steps = rays_d * step

        t_vals = torch.linspace(0., 1., steps=self.n_samples, device=device)
        z_vals = self.camera.near * (1. - t_vals) + self.camera.far * (t_vals)

        if self.perturb:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
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

        rgb_map, disp_map, acc_map, weights, depth_map = self.raw2outputs(rgb_slices, density_slices, z_vals, rays_d)

        return rgb_map, disp_map, acc_map, weights, depth_map

    def render_matrix_rays(self, rays_o, rays_d):
        rgb = t.zeros((self.camera.h, self.camera.w, self.n_channels)).to(device)
        disp = t.zeros((self.camera.h, self.camera.w)).to(device)
        acc = t.zeros((self.camera.h, self.camera.w)).to(device)
        weights = t.zeros((self.camera.h, self.camera.w, self.n_samples)).to(device)
        depth = t.zeros((self.camera.h, self.camera.w)).to(device)
        for i in range(rgb.shape[0]):
            rgb_map, disp_map, acc_map, weights_map, depth_map = self.render_arbitrary_rays(rays_o[i], rays_d[i])
            rgb[i] = rgb_map.detach()
            disp[i] = disp_map.detach()
            acc[i] = acc_map.detach()
            weights[i] = weights_map.detach()
            depth[i] = depth_map.detach()

        return rgb, disp, acc, weights, depth

    def render(self):
        rays_o, rays_d = self.camera.get_rays()
        return self.render_matrix_rays(rays_o, rays_d)
