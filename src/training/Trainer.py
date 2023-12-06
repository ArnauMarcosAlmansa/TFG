import os

import torch

from src.training.AbstractTrainer import AbstractTrainer, AbstractTrainLoop
from src.training.EpochSummary import EpochSummary
from src.training.decorators.TrainerDecorator import TrainerDecorator


class Trainer(AbstractTrainer):
    def __init__(self, model, optimizer, loss, train_loader, name):
        self._model = model
        self._optimizer = optimizer
        self._loss = loss
        self.train_loader = train_loader
        self._name = name

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


class TrainLoopWithCheckpoints(AbstractTrainLoop, TrainerDecorator):
    def __init__(self, trainer: AbstractTrainer, base_path):
        TrainerDecorator.__init__(self, trainer)
        self.base_path = base_path + self.name() + "/"
        self.epoch = 0
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
        else:
            self.load_latest_checkpoint()

    def train(self, epochs):
        epoch = self.epoch
        try:
            for epoch in range(self.epoch, epochs):
                self.train_one_epoch(epoch)

            self.checkpoint(epoch, self.model(), self.optimizer(), self.loss())
        except KeyboardInterrupt as e:
            self.checkpoint(epoch, self.model(), self.optimizer(), self.loss())

    def train_one_epoch(self, epoch):
        self.epoch = epoch
        summary = self._trainer.train_one_epoch(epoch)
        if self.should_do_checkpoint(epoch):
            self.checkpoint(epoch, self.model(), self.optimizer(), self.loss())

        return summary

    def should_do_checkpoint(self, epoch) -> bool:
        return epoch % 5 == 0

    def checkpoint(self, epoch, model, optimizer, loss):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, self.base_path + f"checkpoint_epoch_{epoch:010d}.ckpt")

    def load_latest_checkpoint(self):
        checkpoint_filenames = [self.base_path + filename for filename in os.listdir(self.base_path) if
                                filename.endswith(".ckpt")]
        if len(checkpoint_filenames) == 0:
            return

        last_checkpoint = list(sorted(checkpoint_filenames))[-1]

        self.load_checkpoint(last_checkpoint)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model().load_state_dict(checkpoint['model_state_dict'])
        self.optimizer().load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
