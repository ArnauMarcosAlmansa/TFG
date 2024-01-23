import cv2
import numpy as np
import pyrender
import trimesh

from scripts.utils import roty, scale


class CubeMaker:

    def make_cube(self, pose, texture):
        mesh = trimesh.creation.box(transform=pose)
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [0, 1], [1, 0], [1, 1], ]),
            image=texture
        )
        return mesh


    def make_cubes(self):
        poses = [
            roty(scale(np.eye(4), 0.05), np.pi / 2),
            roty(scale(np.eye(4), 0.05), np.pi / 2),
            roty(scale(np.eye(4), 0.05), np.pi / 2),
            roty(scale(np.eye(4), 0.05), np.pi / 2),
            roty(scale(np.eye(4), 0.05), np.pi / 2),
            roty(scale(np.eye(4), 0.05), np.pi / 2),
        ]
        properties = [
            (0.9, 0),
            (0.8, 0.2),
            (0.6, 0.4),
            (0.6, 0.6),
            (0.6, 0.8),
            (0.6, 0.9),
        ]
        textures = [
            cv2.imread("roof-2x2.png", cv2.IMREAD_GRAYSCALE),
            cv2.imread("roof-4x4.png", cv2.IMREAD_GRAYSCALE),
            cv2.imread("roof-8x8.png", cv2.IMREAD_GRAYSCALE),
        ]

        cubes = []
        for pose, (rough, metal), texture in zip(poses, properties, textures * 2):
            cube = pyrender.Mesh.from_trimesh(self.make_cube(pose, texture))
            cube.primitives[0].material.RoughnessFactor = rough
            cube.primitives[0].material.metallicFactor = metal
            cubes.append(cube)

        return cubes

