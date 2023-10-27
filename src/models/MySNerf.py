import torch
import torch.nn

from src.models.layers.PositionalEncode import PositionalEncode
from src.models.layers.Sin import Sin


class MySNerf(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encode_x = PositionalEncode(10)
        self.encode_w = PositionalEncode(10)

        self.spatial_head = torch.nn.Sequential(
            Sin(),
            torch.nn.Linear(60, 100),
            Sin(),
            torch.nn.Linear(100, 100),
            Sin(),
            torch.nn.Linear(100, 100),
            Sin(),
            torch.nn.Linear(100, 100),
            Sin(),
            torch.nn.Linear(100, 100),
            Sin(),
            torch.nn.Linear(100, 100),
            Sin(),
            torch.nn.Linear(100, 100),
            Sin(),
            torch.nn.Linear(100, 51),
        )

        self.albedo_head = torch.nn.Sequential(
            Sin(),
            torch.nn.Linear(50, 3),
            torch.nn.Sigmoid()
        )

        self.sun_head = torch.nn.Sequential(
            Sin(),
            torch.nn.Linear(50, 50),
            Sin(),
            torch.nn.Linear(50, 50),
            Sin(),
            torch.nn.Linear(50, 50),
            Sin(),
            torch.nn.Linear(50, 1),
            torch.nn.Sigmoid()
        )
