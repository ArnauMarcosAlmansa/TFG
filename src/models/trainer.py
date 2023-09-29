import os

import numpy as np
import torch

from src.config import device
from src.dataloaders.SateliteDataLoader import PositionalEncode
from abc import ABC, abstractmethod


class AbstractTrainer(ABC):

    @abstractmethod
    def train(self, epochs):
        pass

    @abstractmethod
    def train_one_epoch(self, epoch):
        pass

    @abstractmethod
    def model(self):
        pass

    @abstractmethod
    def loss(self):
        pass

    @abstractmethod
    def optimizer(self):
        pass

    @abstractmethod
    def name(self):
        pass


class Trainer(AbstractTrainer):
    def __init__(self, model, optimizer, loss, train_loader, name):
        self._model = model
        self._optimizer = optimizer
        self._loss = loss
        self.train_loader = train_loader
        self._name = name

    def train(self, epochs):
        for epoch in range(epochs):
            loss = self.train_one_epoch(epoch)

    def train_one_epoch(self, epoch):
        running_loss = 0.
        last_loss = 0.
        l = len(self.train_loader)
        for i, data in enumerate(self.train_loader):
            print(f"Batch {i + 1}/{l}")

            # Every data instance is an input + label pair
            inputs, labels = data

            # Zero your gradients for every batch!
            self._optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self._model(inputs).squeeze()

            # Compute the loss and its gradients
            loss = self._loss(outputs, labels)
            loss.backward()

            # Adjust learning weights
            self._optimizer.step()

            # Gather data and report
            running_loss += loss.item()

        print(f"EPOCH {epoch}, train_loss = {running_loss / l:.5f}")

        return running_loss / l

    def model(self):
        return self._model

    def loss(self):
        return self._loss

    def optimizer(self):
        return self._optimizer

    def name(self):
        return self._name


class RenderLossTrainer(Trainer):

    def __init__(self, model, optimizer, loss, train_loader, name, *, height, width):
        super().__init__(model, optimizer, loss, train_loader, name)
        self.height = height
        self.width = width

    def train_one_epoch(self, epoch):
        running_loss = 0.0
        l = len(self.train_loader)
        for i, data in enumerate(self.train_loader):
            timestamp, image = data

            # Zero your gradients for every batch!
            self._optimizer.zero_grad()

            print(f"Rendering {i + 1}/{l}")
            outputs = self.render(self._model, timestamp)

            print(f"Computing loss {i + 1}/{l}")
            # Compute the loss and its gradients
            loss = self._loss(outputs, image.squeeze())
            loss.backward()

            # Adjust learning weights
            self._optimizer.step()

            # Gather data and report
            running_loss += loss.item()

            print(f"Done {i + 1}/{l}")

        print(f"EPOCH {epoch}, train_loss = {running_loss / l:.5f}")

        return running_loss / l

    def render(self, model, when):
        model.eval()
        im = torch.zeros((self.height, self.width, 3))
        se = PositionalEncode(10)
        v_step = 2 / self.height
        h_step = 2 / self.width
        for i, y in enumerate(np.arange(-1, 1, v_step)):
            for j, x in enumerate(np.arange(-1, 1, h_step)):
                q = torch.tensor([x, y, when])
                im[i, j, :] = (model(se.do_positional_encoding(q)))

        return im


class TrainerDecorator(AbstractTrainer):

    def __init__(self, trainer: AbstractTrainer):
        self._trainer = trainer

    def train(self, epochs):
        for epoch in range(epochs):
            self.train_one_epoch(epoch)

    def train_one_epoch(self, epoch):
        self._trainer.train_one_epoch(epoch)

    def model(self):
        return self._trainer.model()

    def loss(self):
        return self._trainer.loss()

    def optimizer(self):
        return self._trainer.optimizer()

    def name(self):
        return self._trainer.name()


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


class Checkpoint(TrainerDecorator):
    def __init__(self, trainer: AbstractTrainer, base_path):
        super().__init__(trainer)
        self.base_path = base_path + self.name() + "/"
        self.epoch = 0
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
        else:
            self.load_latest_checkpoint()

    def train(self, epochs):
        epoch = 1
        try:
            for epoch in range(self.epoch, epochs):
                self.train_one_epoch(epoch)
        except KeyboardInterrupt as e:
            self.checkpoint(epoch - 1, self.model(), self.optimizer(), self.loss())

    def train_one_epoch(self, epoch):
        self.epoch = epoch
        self._trainer.train_one_epoch(epoch)
        if self.should_do_checkpoint(epoch):
            self.checkpoint(epoch, self.model(), self.optimizer(), self.loss())

    def should_do_checkpoint(self, epoch) -> bool:
        return epoch % 10 == 9

    def checkpoint(self, epoch, model, optimizer, loss):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, self.base_path + f"checkpoint_epoch_{epoch:010d}.ckpt")

    def load_latest_checkpoint(self):
        checkpoint_filenames = [self.base_path + filename for filename in os.listdir(self.base_path) if filename.endswith(".ckpt")]
        if len(checkpoint_filenames) == 0:
            return

        last_checkpoint = list(sorted(checkpoint_filenames))[-1]

        self.load_checkpoint(last_checkpoint)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model().load_state_dict(checkpoint['model_state_dict'])
        self.optimizer().load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']


class Validation(TrainerDecorator):
    def __init__(self, trainer: AbstractTrainer, data_loader):
        super().__init__(trainer)
        self._data_loader = data_loader

    def train_one_epoch(self, epoch):
        self._trainer.train_one_epoch(epoch)
        if self.should_do_validation(epoch):
            self.do_validation()

    def should_do_validation(self, epoch) -> bool:
        return epoch % 10 == 9

    def do_validation(self):
        self.validate(self.model(), self.loss(), self._data_loader)

    def validate(self, model, loss_fn, data_loader):
        running_loss = 0.0
        for i, data in enumerate(data_loader):
            inputs, labels = data
            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            running_loss += loss.item()

        last_loss = running_loss / len(data_loader)
        print(f"VALIDATION, loss = {last_loss:.5f}")
