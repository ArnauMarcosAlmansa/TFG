from src.training.AbstractTrainer import AbstractTrainer


class TrainerDecorator(AbstractTrainer):

    def __init__(self, trainer: AbstractTrainer):
        self._trainer = trainer

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