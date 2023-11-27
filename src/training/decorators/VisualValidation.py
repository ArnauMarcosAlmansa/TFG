from src.training.AbstractTrainer import AbstractTrainer
from src.training.decorators.TrainerDecorator import TrainerDecorator


class VisualValidation(TrainerDecorator):
    def __init__(self, trainer: AbstractTrainer, renderer, camera_pose, sun_dir, j):
        super().__init__(trainer)
        self.j = j
        self.sun_dir = sun_dir
        self.renderer = renderer
        self.camera_pose = camera_pose

    def train_one_epoch(self, epoch):
        summary = self._trainer.train_one_epoch(epoch)
        if self.should_do_validation(epoch):
            image = self.do_validation()
            summary = summary.with_tensorboard_image("Validation/Visual", image)

        return summary

    def should_do_validation(self, epoch) -> bool:
        return True

    def do_validation(self):
        self.renderer.camera.pose = self.camera_pose
        return self.renderer.render(self.sun_dir, self.j)[0]

