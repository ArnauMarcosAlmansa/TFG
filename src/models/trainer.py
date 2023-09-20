import os

import torch


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
