import random
from statistics import mean

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from ignite.engine import Engine
from ignite.metrics import SSIM

from src.config import device
from src.dataloaders.NerfDataloader import NerfDataset
from src.training.AbstractTrainer import AbstractTrainer
from src.training.decorators.TrainerDecorator import TrainerDecorator


class VisualValidation(TrainerDecorator):
    def __init__(self, trainer: AbstractTrainer, renderer, dataset: NerfDataset):
        super().__init__(trainer)
        self.renderer = renderer
        self.dataset = dataset

    def train_one_epoch(self, epoch):
        summary = self._trainer.train_one_epoch(epoch)
        if self.should_do_validation(epoch):
            ssim = self.do_validation()
            summary = summary.with_validation_ssim(ssim)

        return summary

    def should_do_validation(self, epoch) -> bool:
        return epoch % 10 == 0

    @torch.no_grad()
    def do_validation(self):
        # def eval_step(engine, batch):
        #     return batch
        #
        # default_evaluator = Engine(eval_step)
        # metric = SSIM(data_range=1.0)
        # metric.attach(default_evaluator, 'ssim')

        ssims = []

        showi = random.randint(0, len(self.dataset.images) - 1)

        for i, image in enumerate(self.dataset.images):
            gt = torch.from_numpy(image[0][:, :, :3]).to(device)
            self.renderer.camera.pose = self.dataset.poses[i].to(device)
            im = self.renderer.render()[0]

            # state = default_evaluator.run([[im, image]])
            # ssims.append(state.metrics['ssim'])
            ssims.append((torch.mean(torch.square(gt - im))).item())

            if i == showi:
                ax = plt.axes()
                ax.set_facecolor("black")
                plt.imshow(im.cpu())
                plt.show()
                plt.imshow(image[0][:, :, :3])
                plt.show()

        return mean(ssims)

