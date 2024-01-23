import numpy as np
import pyrender


class Sun:
    def __init__(self, node: pyrender.Node):
        self.node = node
        self.pose = np.eye(4)

    def solar_color(self, when):
        '''Aquí habria que obtener la temperatura CCT y convertirla a RGB según el momento del dia'''
        return np.clip(np.sin(np.ones(3) * when), 0, 1)

    def cycle(self, when):
        return rotx(self.pose, when - np.pi / 2), self.solar_color(when)