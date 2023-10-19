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