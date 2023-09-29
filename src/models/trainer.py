import os

import numpy as np
import torch

from src.config import device
from src.dataloaders.SateliteDataLoader import PositionalEncode


class Trainer:
    def __init__(self, model, optimizer, loss, train_loader, test_loader, checkpoint=None, validation=None):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.checkpoint = checkpoint
        self.validation = validation

    def train(self, epochs):
        for epoch in range(epochs):
            loss = self.train_one_epoch(epoch)
            print(f"EPOCH {epoch}, train_loss = {loss:.5f}")
            if self.checkpoint and self.checkpoint.should_do_checkpoint(epoch):
                self.checkpoint.checkpoint(epoch, self.model, self.optimizer, loss)
            if self.validation and self.validation.should_do_validation(epoch):
                self.validation.validate(self.model, self.loss, self.test_loader)

    def train_one_epoch(self, epoch):
        running_loss = 0.
        last_loss = 0.
        l = len(self.train_loader)
        for i, data in enumerate(self.train_loader):
            print(f"Batch {i + 1}/{l}")

            # Every data instance is an input + label pair
            inputs, labels = data

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model(inputs).squeeze()

            # Compute the loss and its gradients
            loss = self.loss(outputs, labels)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()

        return running_loss / l


class RenderLossTrainer(Trainer):

    def __init__(self, model, optimizer, loss, train_loader, test_loader, *, height, width, **kwargs):
        super().__init__(model, optimizer, loss, train_loader, test_loader, **kwargs)
        self.height = height
        self.width = width

    def train_one_epoch(self, epoch):
        running_loss = 0.0
        l = len(self.train_loader)
        for i, data in enumerate(self.train_loader):
            timestamp, image = data

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            print(f"Rendering {i + 1}/{l}")
            outputs = self.render(self.model, timestamp)

            print(f"Computing loss {i + 1}/{l}")
            # Compute the loss and its gradients
            loss = self.loss(outputs, image.squeeze().to(device))
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()

            print(f"Done {i + 1}/{l}")

        return running_loss / l

    def render(self, model, when):
        model.eval()
        se = PositionalEncode(5)
        v_step = 2 / self.height
        h_step = 2 / self.width
        query = torch.zeros((self.height * self.width, 30), device=device)
        for i, y in enumerate(np.arange(-1, 1, v_step)):
            for j, x in enumerate(np.arange(-1, 1, h_step)):
                q = se.do_positional_encoding(torch.tensor([x, y, when], device=device))
                query[i * self.width + j, :] = q

        im = model(query).reshape((self.height, self.width, 3))

        return im


class CheckPoint:
    def __init__(self, base_path):
        self.base_path = base_path
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

    def should_do_checkpoint(self, epoch) -> bool:
        return epoch % 10 == 9

    def checkpoint(self, epoch, model, optimizer, loss):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, self.base_path + f"checkpoint_epoch_{epoch}.ckpt")


class Validation:
    def should_do_validation(self, epoch) -> bool:
        return epoch % 10 == 9

    def validate(self, model, loss_fn, data_loader):
        running_loss = 0.0
        for i, data in enumerate(data_loader):
            inputs, labels = data
            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            running_loss += loss.item()

        last_loss = running_loss / len(data_loader)
        print(f"VALIDATION, loss = {last_loss:.5f}")
