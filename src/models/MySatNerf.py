import torch
import torch.nn.functional as F

from src.models.layers.PositionalEncode import PositionalEncode
from src.models.layers.Sin import Sin


class MySatnerfLoss:
    def __call__(self, gt, predicted):
        pass


class MySatNerf(torch.nn.Module):
    def __init__(self, width=256):
        super().__init__()

        self.encode_x = PositionalEncode(10)
        self.encode_w = PositionalEncode(10)

        self.spatial = torch.nn.Sequential(
            torch.nn.Linear(60, width),
            Sin(),
            torch.nn.Linear(width, width),
            Sin(),
            torch.nn.Linear(width, width),
            Sin(),
            torch.nn.Linear(width, width),
            Sin(),
            torch.nn.Linear(width, width),
            Sin(),
            torch.nn.Linear(width, width),
            Sin(),
            torch.nn.Linear(width, width),
            Sin(),
            torch.nn.Linear(width, width),
        )

        self.middle = torch.nn.Sequential(
            Sin(),
            torch.nn.Linear(width - 1, width),
            Sin(),
        )

        self.ambient = torch.nn.Sequential(
            Sin(),
            torch.nn.Linear(60, width // 2),
            Sin(),
            torch.nn.Linear(width // 2, 3)
        )

        self.uncertainty = torch.nn.Sequential(
            # TODO: el embedding tiene que venir dado por cada imagen
            # torch.nn.Embedding(100, 10),
            Sin(),
            # TODO: conseguir poner a 10
            torch.nn.Linear(60, 1),
            torch.nn.Softplus()
        )

        self.albedo = torch.nn.Sequential(
            Sin(),
            torch.nn.Linear(width, 3),
            torch.nn.Sigmoid()
        )

        self.shading = torch.nn.Sequential(
            Sin(),
            torch.nn.Linear(width + 60, width // 2),
            Sin(),
            torch.nn.Linear(width // 2, width // 2),
            Sin(),
            torch.nn.Linear(width // 2, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x, w):
        initial_x = x
        initial_w = w

        m = self.encode_x(x)
        m = self.spatial(m)
        density = m[:, -1]

        m = self.middle(m[:, :-1])
        albedo = self.albedo(m)

        w = self.encode_w(initial_w)
        uncertainty = self.uncertainty(w)

        ambient = self.ambient(w)

        shading = self.shading(torch.cat([m, w], -1))

        return density, uncertainty, albedo, shading, ambient
