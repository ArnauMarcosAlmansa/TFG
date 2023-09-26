import numpy as np
import torch

from src.dataloaders.SateliteDataLoader import PositionalEncode


class TimedRenderLoss:
    def __init__(self, /, width, height, data):
        self.width = width
        self.height = height
        self.data = data
        self.mse = torch.nn.MSELoss()

    def __call__(self, model):
        total_mse = torch.tensor(0, dtype=torch.float32)
        for when, target in self.data:
            total_mse += self.mse(self.render(model, when), target.squeeze())

        return total_mse / len(self.data)

    def render(self, model, when):
        model.eval()
        im = torch.zeros((self.height, self.width, 12))
        se = PositionalEncode(10)
        v_step = 2 / self.height
        h_step = 2 / self.width
        for i, y in enumerate(np.arange(-1, 1, v_step)):
            for j, x in enumerate(np.arange(-1, 1, h_step)):
                q = torch.tensor([x, y, when])
                im[i, j, :] = (model(se.do_positional_encoding(q)))

        return im
