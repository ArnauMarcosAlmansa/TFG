class EpochSummary:
    def __init__(self, epoch):
        self.epoch = epoch
        self.training_loss = None
        self.validation_loss = None

    def with_training_loss(self, loss):
        self.training_loss = loss
        return self

    def with_validation_loss(self, loss):
        self.validation_loss = loss
        return self
