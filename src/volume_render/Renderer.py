from src.config import device
from src.volume_render.cameras.Camera import Camera
import torch as t
import torch.nn.functional as F

class Renderer:
    def __init__(self, camera: Camera, sampler, n_samples):
        self.camera = camera
        self.sampler = sampler
        self.n_samples = n_samples
        self.perturb = True
        self.lindisp = True
        self.raw_noise_std = 0
        self.white_bkgd = True

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
