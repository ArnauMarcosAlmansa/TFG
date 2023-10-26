import numpy as np

from src.config import device
import torch as t


class PositionalEncode(t.nn.Module):
    def __init__(self, L, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.L = L

    def forward(self, x):
        return self.do_positional_encoding(x)

    def do_positional_encoding(self, inputs):
        result = t.zeros(inputs.shape[0], inputs.shape[1] * self.L * 2, device=device)
        for i in range(inputs.shape[1]):
            for l in range(self.L):
                result[:, i * self.L * 2 + l * 2] = t.sin(2 ** l * np.pi * inputs[:, i])
                result[:, i * self.L * 2 + l * 2 + 1] = t.cos(2 ** l * np.pi * inputs[:, i])

        return result
