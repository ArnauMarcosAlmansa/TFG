import torch
from torch.nn import functional as F


class MyNerf(torch.nn.Module):
    def __init__(self, inputs=60, width=256, outputs=3):
        super().__init__()

        self.fc1 = torch.nn.Linear(in_features=inputs, out_features=width)
        self.fc2 = torch.nn.Linear(in_features=width, out_features=width)
        self.fc3 = torch.nn.Linear(in_features=width, out_features=width)
        self.fc4 = torch.nn.Linear(in_features=width, out_features=width)
        self.fc5 = torch.nn.Linear(in_features=width, out_features=width)

        self.fc6 = torch.nn.Linear(in_features=width + inputs, out_features=width)
        self.fc7 = torch.nn.Linear(in_features=width, out_features=width)
        self.fc8 = torch.nn.Linear(in_features=width, out_features=width)
        self.fc8 = torch.nn.Linear(in_features=width, out_features=128)

        self.fc9 = torch.nn.Linear(in_features=128, out_features=outputs)

    def forward(self, x):
        initial_x = x

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = torch.cat([x, initial_x], -1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)
        x = F.sigmoid(self.fc9(x))

        return x
