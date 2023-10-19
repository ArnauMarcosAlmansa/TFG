from abc import ABC, abstractmethod


class Camera(ABC):
    def __init__(self, w: int, h: int, pose):
        self.w = w
        self.h = h
        self.pose = pose

    @abstractmethod
    def get_rays(self):
        pass
