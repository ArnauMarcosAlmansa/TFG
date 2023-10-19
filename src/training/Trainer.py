from src.training.AbstractTrainer import AbstractTrainer


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