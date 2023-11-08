import os

import torch

from src.training.AbstractTrainer import AbstractTrainer
from src.training.EpochSummary import EpochSummary
from src.training.decorators.TrainerDecorator import TrainerDecorator


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
        for epoch in range(self.epoch, epochs):
            self.train_one_epoch(epoch)

    def train_one_epoch(self, epoch):
        try:
            self.epoch = epoch
            summary = self._trainer.train_one_epoch(epoch)
            if self.should_do_checkpoint(epoch):
                self.checkpoint(epoch, self.model(), self.optimizer(), self.loss())

            return summary
        except KeyboardInterrupt as e:
            self.checkpoint(epoch - 1, self.model(), self.optimizer(), self.loss())

        return EpochSummary(epoch)

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