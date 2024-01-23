import dataclasses
import json
import os
import random

import cv2
import numpy as np
import pyrender
import pickle

from pyrender import RenderFlags

from scripts.CubeMaker import CubeMaker
from scripts.Loader import Loader
from scripts.TerrainMaker import TerrainMaker
from scripts.utils import lookAt, roty, scale


# S2B_MSIL2A_20170709T094029_78_59

@dataclasses.dataclass
class Source:
    dem_file: str
    bands_folder: str
    bands_name: str


@dataclasses.dataclass
class CameraMovementParams:
    height: float
    xy_range: float
    fov: float


@dataclasses.dataclass
class Configuration:
    shadows: bool
    cubes: bool


def ncamera_poses(n, z, xy_range):
    for _ in range(n):
        x = np.random.uniform(-xy_range, xy_range, 1)[0]
        y = np.random.uniform(-xy_range, xy_range, 1)[0]

        camera_pose = np.array([
            [1.0, 0.0, 0.0, x],
            [0.0, 1.0, 0.0, y],
            [0.0, 0.0, 1.0, z],
            [0.0, 0.0, 0.0, 1.0],
        ])

        eye = np.array([camera_pose[0, 3], camera_pose[1, 3], camera_pose[2, 3]])
        center = np.array([0, 0, 0])
        up = np.array([0, 1, 0])

        rotation = np.linalg.inv(lookAt(eye, center, up))

        camera_pose[:3, :3] = rotation[:3, :3]

        yield camera_pose, f"{x:.4f}_{y:.4f}"


def day():
    for t in np.arange(0, np.pi, 0.1):
        yield t, f"{t:.4f}"


def pkl_save(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


class DatasetGenerator:
    @staticmethod
    def folder_name(base, camera_movement: CameraMovementParams, config: Configuration, source: Source):
        shadows = "shadows" if config.shadows else "noshadows"
        cubes = "cubes" if config.cubes else "nocubes"
        return f"{base}/{source.bands_name}/dist_{camera_movement.height}_{shadows}_{cubes}/"

    def __init__(self, folder: str, camera_movement: CameraMovementParams, config: Configuration):
        self.folder = folder
        self.camera_movement = camera_movement
        self.config = config

    def setup_scene(self, scene):
        camera = pyrender.PerspectiveCamera(yfov=self.camera_movement.fov, aspectRatio=1.0)
        camn = scene.add(camera)

        light = pyrender.DirectionalLight(intensity=10)
        if self.config.shadows:
            sunlight = scene.add(light, pose=roty(np.eye(4), np.pi / 4))
        else:
            sunlight = scene.add(light, pose=roty(np.eye(4), 0))

        if self.config.cubes:
            cubes = CubeMaker().make_cubes()
            cube_pose = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ])

            for cube in cubes:
                quarter = 960 // 4
                point = pts[random.randint(quarter, 960 - quarter), random.randint(quarter, 960 - quarter)]
                cube_pose_copy = np.copy(cube_pose)
                cube_pose_copy[:3, 3] = point
                scene.add(cube, pose=cube_pose_copy)

        return camn

    def generate_multispectral_eo_dataset(self, scene, renderer, meshes):

        camn = self.setup_scene(scene)

        modes = ["train", "val", "test"]

        for mode in modes:
            image_index = 1

            jsdoc = dict(frames=[], camera_angle_x=self.camera_movement.fov)

            os.makedirs(f"{self.folder}/{mode}", exist_ok=True)

            for pose, posename in ncamera_poses(20, self.camera_movement.height, self.camera_movement.xy_range):
                scene.set_pose(camn, pose)

                for mesh, bandname in zip(meshes,
                                          ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B11", "B12",
                                           "B8A", ]):
                    node = scene.add(mesh)

                    if self.config.shadows:
                        color, depth = renderer.render(scene, flags=RenderFlags.SHADOWS_ALL | RenderFlags.ALL_SOLID)
                    else:
                        color, depth = renderer.render(scene, flags=RenderFlags.ALL_SOLID)

                    bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(f"{self.folder}/{mode}/{image_index:010d}_{bandname}.png", bgr)

                    scene.remove_node(node)

                node = scene.add(mesh)

                color, depth = renderer.render(scene)
                np.save(f"{self.folder}/{mode}/{image_index:010d}_DEPTH", depth)

                scene.remove_node(node)

                jsdoc['frames'].append({
                    "transform_matrix": [
                        [float(pose[0, 0]), float(pose[0, 1]), float(pose[0, 2]), float(pose[0, 3])],
                        [float(pose[1, 0]), float(pose[1, 1]), float(pose[1, 2]), float(pose[1, 3])],
                        [float(pose[2, 0]), float(pose[2, 1]), float(pose[2, 2]), float(pose[2, 3])],
                        [float(pose[3, 0]), float(pose[3, 1]), float(pose[3, 2]), float(pose[3, 3])],
                    ],
                    "file_path": f"./{mode}/{image_index:010d}"
                })

                image_index += 1

            json.dump(jsdoc, open(f"{self.folder}/transforms_{mode}.json", "wt"), indent=4)


def interact(scene):
    pyrender.Viewer(scene, render_flags={"shadows": True})


sources = [
    Source(
        "/data1tb/BigEarthNet-S2-v1.0/BigEarthNet-S2-v1.0/dem/S2B_MSIL2A_20170709T094029_78_59_dem.tif",
        "/data1tb/BigEarthNet-S2-v1.0/BigEarthNet-S2-v1.0/BigEarthNet-v1.0/",
        "S2B_MSIL2A_20170709T094029_78_59"
    ),
    # Source(
    #     "/data1tb/BigEarthNet-S2-v1.0/BigEarthNet-S2-v1.0/dem/S2A_MSIL2A_20180430T094031_38_57_dem.tif",
    #     "/data1tb/BigEarthNet-S2-v1.0/BigEarthNet-S2-v1.0/BigEarthNet-v1.0/",
    #     "S2A_MSIL2A_20180430T094031_38_57"
    # ),
    # Source(
    #     "/data1tb/BigEarthNet-S2-v1.0/BigEarthNet-S2-v1.0/dem/S2B_MSIL2A_20180525T94031_14_48_dem.tif",
    #     "/data1tb/BigEarthNet-S2-v1.0/BigEarthNet-S2-v1.0/BigEarthNet-v1.0/",
    #     "S2B_MSIL2A_20180525T94031_14_48"
    # ),
    # Source(
    #     "/data1tb/BigEarthNet-S2-v1.0/BigEarthNet-S2-v1.0/dem/S2B_MSIL2A_20180525T94030_48_73_dem.tif",
    #     "/data1tb/BigEarthNet-S2-v1.0/BigEarthNet-S2-v1.0/BigEarthNet-v1.0/",
    #     "S2B_MSIL2A_20180525T94030_48_73"
    # ),
]

movement_params = [
    # CameraMovementParams(5, 5 / 10, np.pi / (3 * 8)),
    CameraMovementParams(10, 10 / 10, np.pi / (4 * 10)),
    CameraMovementParams(20, 20 / 10, np.pi / (4 * 20)),
    CameraMovementParams(40, 40 / 10, np.pi / (4 * 40)),
    CameraMovementParams(80, 80 / 10, np.pi / (4 * 80)),
    CameraMovementParams(160, 160 / 10, np.pi / (4 * 160)),
    CameraMovementParams(320, 320 / 10, np.pi / (4 * 320)),
]

configurations = [
    Configuration(False, False),
    # Configuration(False, True),
    # Configuration(True, False),
    # Configuration(True, True),
]

if __name__ == '__main__':
    for source in sources:
        meshes, pts = TerrainMaker(Loader()).make_terrains(
            source.dem_file,
            source.bands_folder,
            source.bands_name,
        )
        for movement_param in movement_params:
            for configuration in configurations:
                scene = pyrender.Scene(ambient_light=np.array([1.0, 1.0, 1.0]))

                r = pyrender.OffscreenRenderer(800, 800)
                DatasetGenerator(
                    DatasetGenerator.folder_name(
                        "/home/amarcos/workspace/TFG/scripts/generated_parametrized_multispectral_data/",
                        movement_param,
                        configuration,
                        source,
                    ),
                    movement_param,
                    configuration
                ).generate_multispectral_eo_dataset(
                    scene,
                    r,
                    meshes,
                )

    # interact(scene)
    print("END")
