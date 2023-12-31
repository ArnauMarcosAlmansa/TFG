import torch.nn

from src.config import device
from src.training.Trainer import Trainer
from src.volume_render.SimpleRenderer import SimpleRenderer


class SatnerfRenderTrainer(Trainer):

    def __init__(self, model: torch.nn.Module, optimizer, loss, train_loader, name: str, renderer: SimpleRenderer):
        super().__init__(model, optimizer, loss, train_loader, name)
        self.renderer = renderer

    def train_one_epoch(self, epoch):
        running_loss = 0.0
        l = len(self.train_loader)
        for i, data in enumerate(self.train_loader):
            colors, rays_o, rays_d, sun_dirs, d = data

            # Zero your gradients for every batch!
            self._optimizer.zero_grad()

            print(f"Rendering {i + 1}/{l}")
            self.renderer.camera.pose = d['camera_pose'].squeeze()
            outputs, uncertainty = self.renderer.render_arbitrary_rays(rays_o, rays_d, sun_dirs)

            print(f"Computing loss {i + 1}/{l}")
            # Compute the loss and its gradients
            loss = self._loss(outputs, uncertainty, colors.squeeze().to(device))
            loss.backward()

            # Adjust learning weights
            self._optimizer.step()

            # Gather data and report
            running_loss += loss.item()

            print(f"Done {i + 1}/{l}")

        print(f"EPOCH {epoch}, train_loss = {running_loss / l:.5f}")

        return running_loss / l
