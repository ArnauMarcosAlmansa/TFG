import torch


class Siren(torch.nn.Module):
    def __init__(self, inputs=40, width=200, depth=8):
        super().__init__()
        self.first = torch.nn.Linear(in_features=inputs, out_features=width)
        self.fcs = torch.nn.ModuleList(torch.nn.Linear(in_features=width, out_features=width) for _ in range(0, depth))
        self.last = torch.nn.Linear(in_features=width, out_features=3)

    def forward(self, x):
        x = torch.sin(self.first(x))
        for fc in self.fcs:
            x = torch.sin(fc(x))
        x = torch.sin(self.last(x))
        return x
