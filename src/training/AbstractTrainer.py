from abc import ABC, abstractmethod

from src.training.EpochSummary import EpochSummary


class AbstractTrainer(ABC):

    @abstractmethod
    def train(self, epochs):
        pass

    @abstractmethod
    def train_one_epoch(self, epoch) -> EpochSummary:
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