from abc import ABC, abstractmethod


class Camera(ABC):
    def __init__(self, w: int, h: int, pose):
        self.near = None
        self.far = None
        self.w = w
        self.h = h
        self.pose = pose

    @abstractmethod
    def get_rays(self):
        pass
