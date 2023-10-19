import torch


class Sin(torch.nn.Module):
    def forward(self, x):
        return torch.sin(x)
