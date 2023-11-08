from src.training.AbstractTrainer import AbstractTrainer
from src.training.decorators.TrainerDecorator import TrainerDecorator


class Validation(TrainerDecorator):
    def __init__(self, trainer: AbstractTrainer, data_loader):
        super().__init__(trainer)
        self._data_loader = data_loader

    def train_one_epoch(self, epoch):
        summary = self._trainer.train_one_epoch(epoch)
        if self.should_do_validation(epoch):
            loss = self.do_validation()
            summary = summary.with_validation_loss(loss)

        return summary

    def should_do_validation(self, epoch) -> bool:
        return epoch % 10 == 9

    def do_validation(self):
        return self.validate(self.model(), self.loss(), self._data_loader)

    def validate(self, model, loss_fn, data_loader):
        running_loss = 0.0
        for i, data in enumerate(data_loader):
            inputs, labels = data
            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            running_loss += loss.item()

        last_loss = running_loss / len(data_loader)
        print(f"VALIDATION, loss = {last_loss:.5f}")

        return last_loss