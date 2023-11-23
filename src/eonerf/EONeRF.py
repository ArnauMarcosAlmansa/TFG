from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch.nn
import torch.nn.functional as F

from src.config import device, fix_cuda
from src.dataloaders.SyntheticDataloader import SyntheticEODataset
from src.models.layers.PositionalEncode import PositionalEncode
from src.training.EpochSummary import EpochSummary
from src.training.Trainer import Trainer
from src.training.decorators.Checkpoint import Checkpoint
from src.training.decorators.Tensorboard import Tensorboard
from src.training.decorators.VisualValidation import VisualValidation
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
        self.encode_x = Mapping(10, 3, True)
        
        self.t_emb = torch.nn.Embedding(n_images, 4)
        self.a_emb = torch.nn.Embedding(n_images, 3)
        self.a_emb.weight.data.uniform_(0.9, 1.1)
        self.b_emb = torch.nn.Embedding(n_images, 3)
        self.b_emb.weight.data.uniform_(-0.1, 0.1)

        self.mlp1 = make_mlp1(w)
        self.mlp2 = make_mlp2(w)
        # self.mlp3 = make_mlp3(w + 4)
        self.mlp3 = make_mlp3(w)
        self.mlp4 = make_mlp4(w)

    def forward(self, x, d, j):
        x = self.encode_x(x)
        d = self.encode_x(d)

        m = self.mlp1(x)
        m, density = m[:, :-1], m[:, -1]

        albedo = self.mlp2(m)
        # m = torch.cat([m, self.t_emb(j)], -1)

        u = self.mlp3(m)
        transient, uncertainty = u[:, 0], u[:, 1]

        ambient = self.mlp4(d)

        return density, albedo, ambient, uncertainty, transient

    def get_ab(self, j):
        return self.a_emb(j), self.b_emb(j)


def density2alpha(density, dists):
    return 1. - torch.exp(-density * dists)


class Renderer:
    def __init__(self, camera: Camera, model: EONeRF):
        self.camera = camera
        self.model = model
        self.n_samples = 128
        self.raw_rgb = True

    def render_arbitrary_rays(self, rays_o, rays_d, sun_dirs, j, near=0., far=1.):
        rays_d = rays_d / self.n_samples

        sh = rays_d.shape  # [..., 3]
        # que el step sea un poco aleatorio para evitar overfitting
        step = (far - near) / self.n_samples

        rgb = torch.zeros((rays_o.shape[0], 3)).to(device)
        unc = torch.zeros(rays_o.shape[0]).to(device)
        depth = torch.zeros((rays_o.shape[0])).to(device)
        Ti = torch.ones((rays_o.shape[0], 3)).to(device)
        for sample_index in range(self.n_samples):
            points = rays_o + rays_d * step * sample_index + near

            density, albedo, ambient, uncertainty, \
                _transient = self.model(points, sun_dirs, j)
            rgb_slice = torch.sigmoid(albedo)
            density_slice = torch.relu(density)

            alpha_slice = density2alpha(density_slice, step).unsqueeze(-1)
            alpha_slice = torch.broadcast_to(alpha_slice, rgb_slice.shape)

            Ti = Ti * (1 - alpha_slice)
            w = Ti * alpha_slice

            rgb = rgb + w * rgb_slice
            unc = unc + w[..., 0].squeeze() * uncertainty
            distance = step * sample_index
            depth = depth + w[..., 0].squeeze() * distance

        a, b = self.model.get_ab(j)

        if self.raw_rgb:
            return rgb, unc, depth

        sun_rays_d = -sun_dirs.to(device)
        potential_shadow_points = rays_o + rays_d * depth.unsqueeze(-1) + near
        strans = torch.zeros(rays_o.shape[0]).to(device)
        for sample_index in range(self.n_samples):
            points = potential_shadow_points + sun_rays_d * step * sample_index + near

            density, albedo, ambient, uncertainty, \
                transient = self.model(points, sun_dirs, j)
            rgb_slice = torch.sigmoid(albedo)
            density_slice = torch.relu(density)

            alpha_slice = density2alpha(density_slice, step).unsqueeze(-1)
            alpha_slice = torch.broadcast_to(alpha_slice, rgb_slice.shape)

            Ti = Ti * (1 - alpha_slice)
            w = Ti * alpha_slice

            strans = strans + w[..., 0].squeeze() * transient

        s = Ti * torch.broadcast_to(strans.unsqueeze(-1), Ti.shape)

        def l(s, a):
            return s + (1 - s) * a

        color = a * (l(s, ambient) * rgb) + b
        # color = l(s, ambient) * rgb

        return color, unc, depth

    def render_matrix_rays(self, rays_o, rays_d, sun_dir, j, near=0., far=1.):
        J = torch.cuda.LongTensor([j], device=device)
        J = torch.broadcast_to(J, (rays_o.shape[0],))
        sun_dirs = torch.broadcast_to(sun_dir, rays_o.shape)
        rgb = torch.zeros((self.camera.h, self.camera.w, 3)).to(device)
        uncertainty = torch.zeros((self.camera.h, self.camera.w, 1)).to(device)
        depth = torch.zeros((self.camera.h, self.camera.w, 1)).to(device)

        for i in range(rgb.shape[0]):
            color, unc, dep = self.render_arbitrary_rays(rays_o[i], rays_d[i], sun_dirs[i], J, near, far)
            rgb[i] = color.detach()
            uncertainty[i] = unc.unsqueeze(-1).detach()
            depth[i] = dep.unsqueeze(-1).detach()

        return rgb, uncertainty, depth

    def render(self, sun_dir, j, near=0., far=1.):
        rays_o, rays_d = self.camera.get_rays()
        return self.render_matrix_rays(rays_o, rays_d, sun_dir, j, near, far)


class EONeRFLoss:
    def __call__(self, color, uncertainty, ground_truth):
        b = uncertainty + 0.05
        norm_squared = torch.square(torch.norm(color - ground_truth, p=2, dim=-1))
        doubled_b_squared = (2 * torch.square(b))
        first = norm_squared / doubled_b_squared
        second = (torch.log(b) + 3) / 2
        return torch.mean(first + second)


class EONeRFTrainer(Trainer):
    def __init__(self, model, optimizer, loss, train_loader, name, renderer):
        super().__init__(model, optimizer, loss, train_loader, name)
        self.renderer = renderer
        self.mse_loss = torch.nn.MSELoss()

    def train_one_epoch(self, epoch):
        self._model.train()
        running_loss = 0.
        last_loss = float('inf')
        l = len(self.train_loader)
        for i, data in enumerate(self.train_loader):
            # Every data instance is an input + label pair
            colors, rays_o, rays_d, sun_dirs, d = data

            # Zero your gradients for every batch!
            self._optimizer.zero_grad()

            # Make predictions for this batch

            # Compute the loss and its gradients
            if True:
                # print("USING MSE loss")
                color, uncertainty, depth = self.renderer.render_arbitrary_rays(rays_o, rays_d, sun_dirs,
                                                                                d['j'].to(device))
                loss = self.mse_loss(color, colors)
            else:
                print("USING MSE custom")
                color, uncertainty, depth = self.renderer.render_arbitrary_rays(rays_o, rays_d, sun_dirs,
                                                                                d['j'].to(device))
                loss = self._loss(color, uncertainty, colors)

            loss.backward()

            # Adjust learning weights
            self._optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            # print(f"Batch {i + 1}/{l} train_loss = {loss.item():.5f}")

        print(f"EPOCH {epoch}, train_loss = {running_loss / l:.5f}")
        last_loss = running_loss / l

        return EpochSummary(epoch).with_training_loss(running_loss / l)


if __name__ == '__main__':
    fix_cuda()

    data = SyntheticEODataset("/home/amarcos/workspace/TFG/scripts/generated_eo_test_data/")
    loader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=1024)

    torch.no_grad()

    c = PinholeCamera(120, 120, 50, torch.eye(4))
    model = EONeRF(n_images=data.n_images).to(device)
    loss = EONeRFLoss()
    optim = torch.optim.Adam(params=model.parameters(), lr=0.1, betas=(0.9, 0.999))
    r = Renderer(c, model)

    colors, rays_o, rays_d, sun_dirs, d = next(iter(data))

    trainer = EONeRFTrainer(model, optim, loss, loader, 'ESPEON', renderer=r)
    trainer = Checkpoint(trainer, "./checkpoints_debugtest/")
    trainer = VisualValidation(trainer, r, d['camera_pose'], sun_dirs, d['j'])
    trainer = Tensorboard(trainer)

    trainer.train(10000)

    colors, rays_o, rays_d, sun_dirs, d = next(iter(data))

    torch.no_grad()
    color, uncertainty, depth = r.render(sun_dirs.to(device), d['j'])
    plt.imshow(color.cpu().detach())
    plt.show()
