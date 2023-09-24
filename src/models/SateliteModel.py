import torch
from torch.nn import functional as F


class SateliteModel(torch.nn.Module):
    def __init__(self, inputs=3, depth=8, width=100):
        super().__init__()
        # añadir spatial encoding
        self.il = torch.nn.Linear(in_features=inputs, out_features=width)
        self.fcs = [torch.nn.Linear(in_features=width, out_features=width) for _ in range(depth)]
        self.ol = torch.nn.Linear(in_features=width, out_features=1)

    def forward(self, x):
        x = F.relu(self.il(x))
        for fc in self.fcs:
            x = F.relu(fc(x))
        x = F.sigmoid(self.ol(x))
        return x
