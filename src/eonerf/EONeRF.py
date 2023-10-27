from dataclasses import dataclass

import torch.nn
import torch.nn.functional as F

from src.config import device, fix_cuda
from src.dataloaders.SyntheticDataloader import SyntheticEODataset
from src.models.layers.PositionalEncode import PositionalEncode
from src.training.Trainer import Trainer
from src.training.decorators.Checkpoint import Checkpoint
from src.volume_render.cameras.Camera import Camera
from src.volume_render.cameras.PinholeCamera import PinholeCamera


def make_mlp1(w: int) -> torch.nn.Module:
    mlp1 = torch.nn.Sequential()
    mlp1.append(torch.nn.Linear(60, w))
    mlp1.append(torch.nn.ReLU())
    for i in range(8):
        mlp1.append(torch.nn.Linear(w, w))
        mlp1.append(torch.nn.ReLU())

    mlp1.append(torch.nn.Linear(w, w + 1))
    mlp1.append(torch.nn.ReLU())

    return mlp1


def make_mlp2(w: int) -> torch.nn.Module:
    mlp2 = torch.nn.Sequential()
    for i in range(8):
        mlp2.append(torch.nn.Linear(w, w))
        mlp2.append(torch.nn.ReLU())

    mlp2.append(torch.nn.Linear(w, 3))
    mlp2.append(torch.nn.Sigmoid())

    return mlp2


def make_mlp3(w: int) -> torch.nn.Module:
    mlp3 = torch.nn.Sequential()
    for i in range(8):
        mlp3.append(torch.nn.Linear(w, w))
        mlp3.append(torch.nn.ReLU())

    mlp3.append(torch.nn.Linear(w, 2))
    mlp3.append(torch.nn.Sigmoid())

    return mlp3


def make_mlp4(w: int) -> torch.nn.Module:
    mlp4 = torch.nn.Sequential()
    mlp4.append(torch.nn.Linear(60, w))
    mlp4.append(torch.nn.ReLU())
    for i in range(8):
        mlp4.append(torch.nn.Linear(w, w))
        mlp4.append(torch.nn.ReLU())

    mlp4.append(torch.nn.Linear(w, 3))
    mlp4.append(torch.nn.Sigmoid())

    return mlp4


class EONeRF(torch.nn.Module):
    def __init__(self, *, n_images, w=256):
        super().__init__()
        self.encode_x = PositionalEncode(10)

        self.t_emb = torch.nn.Embedding(n_images, 4)
        self.a_emb = torch.nn.Embedding(n_images, 3)
        self.b_emb = torch.nn.Embedding(n_images, 3)

        self.mlp1 = make_mlp1(w)
        self.mlp2 = make_mlp2(w)
        self.mlp3 = make_mlp3(w + 4)
        self.mlp4 = make_mlp4(w)

    def forward(self, x, d, j):
        x = self.encode_x(x)
        d = self.encode_x(d)

        m = self.mlp1(x)
        m, density = m[:, :-1], m[:, -1]

        albedo = self.mlp2(m)
        m = torch.cat([m, self.t_emb(j)], -1)

        u = self.mlp3(m)
        transient, uncertainty = u[:, 0], u[:, 1]

        ambient = self.mlp4(d)

        return density, albedo, ambient, uncertainty, transient

    def get_ab(self, j):
        return self.a_emb(j), self.b_emb(j)


@dataclass
class RenderOutput:
    density: torch.Tensor
    albedo: torch.Tensor
    ambient: torch.Tensor
    uncertainty: torch.Tensor
    transient: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor


def density2alpha(raw, dists, act_fn=F.relu):
    return 1. - torch.exp(-act_fn(raw) * dists)


class Renderer:
    def __init__(self, camera: Camera, model: EONeRF):
        self.camera = camera
        self.model = model
        self.n_samples = 200

    def render_arbitrary_rays(self, rays_o, rays_d, sun_dirs, j, near=0., far=1.):
        rays_d = rays_d / self.n_samples

        sh = rays_d.shape  # [..., 3]
        # que el step sea un poco aleatorio para evitar overfitting
        step = (far - near) / self.n_samples

        rgb = torch.zeros((rays_o.shape[0], 3)).to(device)
        unc = torch.zeros(rays_o.shape[0]).to(device)
        trans = torch.zeros(rays_o.shape[0]).to(device)
        Ti = torch.ones((rays_o.shape[0], 3)).to(device)
        for sample_index in range(self.n_samples):
            points = rays_o + rays_d * step * sample_index + near

            density, albedo, ambient, uncertainty, \
                transient = self.model(points, sun_dirs, j)
            rgb_slice = torch.sigmoid(albedo)
            density_slice = torch.relu(density)

            alpha_slice = density2alpha(density_slice, step).unsqueeze(-1)
            alpha_slice = torch.broadcast_to(alpha_slice, rgb_slice.shape)

            Ti = Ti * (1 - alpha_slice)
            w = Ti * alpha_slice

            rgb = rgb + w * rgb_slice
            unc = unc + w[..., 0].squeeze() * uncertainty
            trans = trans + w[..., 0].squeeze() * transient

        a, b = self.model.get_ab(j)

        # TODO: renderizar las sombras
        # s(r) = sgeo(r)τ (r) = T (rsun(tN ))
        # N∑
        # i=1
        # Tiαiτ (xi)

        # INTENTO
        # depth = self.render_depth_arbitrary_rays(rays_o, rays_d)
        # potential_shadow_points = rays_o + rays_d * depth + near
        # self.model(potential_shadow_points, sun_dirs, j)

        # color = a * (l * rgb) + b
        color = a * (rgb) + b

        return color, unc

    def render_matrix_rays(self, rays_o, rays_d, near=0., far=1.):
        rgb = torch.zeros((self.camera.h, self.camera.w, 3)).to(device)
        uncertainty = torch.zeros((self.camera.h, self.camera.w, 1)).to(device)
        for i in range(rgb.shape[0]):
            color, unc = self.render_arbitrary_rays(rays_o[i], rays_d[i], near, far)
            rgb[i] = color
            uncertainty[i] = unc

        return uncertainty

    def render(self, near=0., far=1.):
        rays_o, rays_d = self.camera.get_rays()
        return self.render_matrix_rays(rays_o, rays_d, near, far)

    def render_depth_arbitrary_rays(self, rays_o, rays_d, near=0., far=1.):
        rays_d = rays_d / self.n_samples

        # que el step sea un poco aleatorio para evitar overfitting
        step = (far - near) / self.n_samples

        depth = torch.zeros((rays_o.shape[0], 1)).to(device)
        Ti = torch.ones((rays_o.shape[0], 1)).to(device)
        for sample_index in range(self.n_samples - 1, -1, -1):
            points = rays_o + rays_d * step * sample_index

            distance = step * sample_index

            density = self.model(points)[0]
            alpha_slice = density2alpha(density, step).unsqueeze(-1)

            Ti = Ti * (1 - alpha_slice)
            w = Ti * alpha_slice

            depth = depth + w * distance

        return depth

    def render_depth_matrix_rays(self, rays_o, rays_d, near=0., far=1.):
        rgb = torch.zeros((self.camera.h, self.camera.w, 1)).to(device)
        for i in range(rgb.shape[0]):
            rgb[i] = self.render_depth_arbitrary_rays(rays_o[i], rays_d[i], near, far).detach()

        return rgb

    def render_depth(self, near=0., far=1.):
        rays_o, rays_d = self.camera.get_rays()
        return self.render_depth_matrix_rays(rays_o, rays_d, near, far)


class EONeRFLoss:
    def __call__(self, color, uncertainty, ground_truth):
        b = uncertainty + 0.05
        first = torch.square(torch.norm(color - ground_truth, p=2, dim=-1)) / (2 * torch.square(b))
        second = (torch.log(b + 3)) / 2
        return torch.mean(first + second)


class EONeRFTrainer(Trainer):
    def __init__(self, model, optimizer, loss, train_loader, name, renderer):
        super().__init__(model, optimizer, loss, train_loader, name)
        self.renderer = renderer

    def train_one_epoch(self, epoch):
        running_loss = 0.
        last_loss = 0.
        l = len(self.train_loader)
        for i, data in enumerate(self.train_loader):
            print(f"Batch {i + 1}/{l}")

            # Every data instance is an input + label pair
            colors, rays_o, rays_d, sun_dirs, d = data

            # Zero your gradients for every batch!
            self._optimizer.zero_grad()

            # Make predictions for this batch
            color, uncertainty = self.renderer.render_arbitrary_rays(rays_o, rays_d, sun_dirs, d['j'].to(device))

            # Compute the loss and its gradients
            loss = self._loss(color, uncertainty, colors)
            loss.backward()

            # Adjust learning weights
            self._optimizer.step()

            # Gather data and report
            running_loss += loss.item()

        print(f"EPOCH {epoch}, train_loss = {running_loss / l:.5f}")

        return running_loss / l


if __name__ == '__main__':
    fix_cuda()

    data = SyntheticEODataset("/home/amarcos/workspace/TFG/scripts/generated_eo_data/")

    loader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=1024 + 512)

    torch.no_grad()

    c = PinholeCamera(120, 120, 50, torch.eye(4))
    model = EONeRF(n_images=17).to(device)
    loss = EONeRFLoss()
    optim = torch.optim.Adam(params=model.parameters(), lr=0.01, betas=(0.9, 0.999))
    r = Renderer(c, model)

    trainer = EONeRFTrainer(model, optim, loss, loader, 'STATIC_RENDERED_COMPOSITING_AAAA', renderer=r)
    trainer = Checkpoint(trainer, "./checkpoints_staticrender/")

    trainer.train(100)
