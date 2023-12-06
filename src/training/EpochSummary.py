class EpochSummary:
    def __init__(self, epoch):
        self.epoch = epoch
        self.training_loss = None
        self.validation_loss = None
        self.validation_ssim = None
        self.tensorboard_images = dict()

    def with_training_loss(self, loss):
        self.training_loss = loss
        return self

    def with_validation_loss(self, loss):
        self.validation_loss = loss
        return self

    def with_validation_ssim(self, loss):
        self.validation_ssim = loss
        return self

    def with_tensorboard_image(self, name, image):
        self.tensorboard_images[name] = image
        return self
