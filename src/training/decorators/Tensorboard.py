from torch.utils.tensorboard import SummaryWriter

from src.training.AbstractTrainer import AbstractTrainer
from src.training.EpochSummary import EpochSummary
from src.training.decorators.TrainerDecorator import TrainerDecorator


class Tensorboard(TrainerDecorator):
    def __init__(self, trainer: AbstractTrainer):
        super().__init__(trainer)
        self.summary = SummaryWriter()

    def train_one_epoch(self, epoch):
        summary: EpochSummary = self._trainer.train_one_epoch(epoch)

        if summary.training_loss:
            self.summary.add_scalar("Training/Loss", summary.training_loss, summary.epoch)

        if summary.validation_loss:
            self.summary.add_scalar("Validation/Loss", summary.validation_loss, summary.epoch)

        for name, image in summary.tensorboard_images.items():
            self.summary.add_image(name, image, summary.epoch)
