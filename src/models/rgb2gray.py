import matplotlib.pyplot as plt
import numpy as np
import torch.nn
import torch.nn.functional as F
import cv2


class RGB2GRAY(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = torch.nn.Linear(in_features=3, out_features=1)

    def forward(self, x):
        return self.fc1(x)


class Styler(torch.nn.Module):
    def __init__(self, width=200):
        super().__init__()

        self.fc1 = torch.nn.Linear(in_features=5, out_features=width)
        self.fc2 = torch.nn.Linear(in_features=width, out_features=width)
        self.fc3 = torch.nn.Linear(in_features=width, out_features=width)
        self.fc4 = torch.nn.Linear(in_features=width, out_features=width)
        self.fc5 = torch.nn.Linear(in_features=width, out_features=width)
        self.fc6 = torch.nn.Linear(in_features=width, out_features=width)
        self.fc7 = torch.nn.Linear(in_features=width, out_features=width)
        self.fc8 = torch.nn.Linear(in_features=width, out_features=width)
        self.fc9 = torch.nn.Linear(in_features=width, out_features=3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        return x


def rgb2gray(R, G, B):
    return 0.2126 * R + 0.7152 * G + 0.0722 * B


def generate_data():
    x = torch.rand((1000, 3))
    y = torch.zeros(1000)

    for i in range(1000):
        y[i] = rgb2gray(x[i, 0], x[i, 1], x[i, 2])

    return x, y


class Trainer:
    def __init__(self, model, optimizer, loss):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss

    def train(self, x, y, epochs):
        for epoch in range(epochs):
            self.model.train()

            predictions: torch.Tensor = self.model(x).squeeze()

            loss = self.loss(predictions, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print(f"Epoch: {epoch + 1} | Loss: {loss:.10f}")


def load_monalisa():
    monalisa = cv2.imread("../../monalisa.jpg")
    starry_monalisa = cv2.imread("../../starry_monalisa.jpg")

    h, w, d = monalisa.shape

    monalisa = cv2.cvtColor(monalisa, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    starry_monalisa = cv2.cvtColor(starry_monalisa, cv2.COLOR_BGR2RGB).astype(np.float32) / 255

    gray = cv2.cvtColor(monalisa, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=7, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=7, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

    monalisa = np.dstack([monalisa, grad_x, grad_y])

    starry_monalisa = cv2.resize(starry_monalisa, (w, h))

    return torch.from_numpy(monalisa.reshape((h * w, d + 2))), torch.from_numpy(starry_monalisa.reshape((h * w, d)))


def load_image(filename):
    im = cv2.imread(filename)

    h, w, d = im.shape

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=7, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=7, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

    im = np.dstack([im, grad_x, grad_y])

    return torch.from_numpy(im.reshape((h * w, d + 2)))


def train():
    x, y = generate_data()
    model = RGB2GRAY()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    trainer = Trainer(model=model, optimizer=optimizer, loss=loss_fn)
    trainer.train(x, y, 1000)


    exit()

    x, y = load_monalisa()
    model = Styler(width=200)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    trainer = Trainer(model=model, optimizer=optimizer, loss=loss_fn)
    trainer.train(x, y, 10)

    supper = load_image("../../lastsupper.jpg")

    supper_parts = supper.chunk(100)
    predictions = []
    for part in supper_parts:
        predictions.append(model(part))

    predictions = torch.cat(predictions)

    im = predictions.detach().numpy().reshape((960, 1920, 3)).clip(0, 1)

    cv2.imwrite("../../lastsupper2.jpg", (cv2.cvtColor(im, cv2.COLOR_RGB2BGR) * 255).astype(np.uint8))


if __name__ == '__main__':
    train()
