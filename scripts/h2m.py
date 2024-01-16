import json
import os
import random
import time

import cv2
import glm
import numpy as np
import rasterio
import trimesh
import pyrender
from matplotlib import pyplot as plt
import pickle

from pyrender import RenderFlags


class Sun:
    def __init__(self, node: pyrender.Node):
        self.node = node
        self.pose = np.eye(4)

    def solar_color(self, when):
        '''Aquí habria que obtener la temperatura CCT y convertirla a RGB según el momento del dia'''
        return np.clip(np.sin(np.ones(3) * when), 0, 1)

    def cycle(self, when):
        return rotx(self.pose, when - np.pi / 2), self.solar_color(when)


# S2B_MSIL2A_20170709T094029_78_59

def load_dem(filename, upscale=8, blur=17):
    dem = np.squeeze(rasterio.open(filename).read()).astype(np.float32)

    plt.imsave("dem.png", (dem - dem.min()) / (dem.max() - dem.min()), cmap='gray')

    row, col = dem.shape
    mean = 0
    sigma = 0.5
    gauss = np.random.uniform(-sigma, sigma, (row, col))
    gauss = gauss.reshape(row, col)
    dem = dem + gauss
    # dem2 = cv2.GaussianBlur(dem2, (blur, blur), 0)
    # dem = cv2.resize(dem2, (dem.shape[1], dem.shape[0]))
    dem2 = cv2.resize(dem, (dem.shape[1] * upscale, dem.shape[0] * upscale))
    dem2 = cv2.GaussianBlur(dem2, (blur, blur), 0)
    # plt.imsave("dem2.png", (dem2 - dem2.min()) / (dem2.max() - dem2.min()), cmap='gray')
    return dem2


def load_albedo(base_directory, name):
    b = np.squeeze(rasterio.open(base_directory + name + "/" + name + "_B02.tif").read())
    g = np.squeeze(rasterio.open(base_directory + name + "/" + name + "_B03.tif").read())
    r = np.squeeze(rasterio.open(base_directory + name + "/" + name + "_B04.tif").read())

    rgb = np.stack([r, g, b], -1) // 5
    return rgb.astype(np.uint8)


def load_bands(base_directory, name):
    bands = [
        np.squeeze(rasterio.open(base_directory + name + "/" + name + "_B01.tif").read()).astype(np.float32),
        np.squeeze(rasterio.open(base_directory + name + "/" + name + "_B02.tif").read()).astype(np.float32),
        np.squeeze(rasterio.open(base_directory + name + "/" + name + "_B03.tif").read()).astype(np.float32),
        np.squeeze(rasterio.open(base_directory + name + "/" + name + "_B04.tif").read()).astype(np.float32),
        np.squeeze(rasterio.open(base_directory + name + "/" + name + "_B05.tif").read()).astype(np.float32),
        np.squeeze(rasterio.open(base_directory + name + "/" + name + "_B06.tif").read()).astype(np.float32),
        np.squeeze(rasterio.open(base_directory + name + "/" + name + "_B07.tif").read()).astype(np.float32),
        np.squeeze(rasterio.open(base_directory + name + "/" + name + "_B08.tif").read()).astype(np.float32),
        np.squeeze(rasterio.open(base_directory + name + "/" + name + "_B09.tif").read()).astype(np.float32),
        np.squeeze(rasterio.open(base_directory + name + "/" + name + "_B11.tif").read()).astype(np.float32),
        np.squeeze(rasterio.open(base_directory + name + "/" + name + "_B12.tif").read()).astype(np.float32),
        np.squeeze(rasterio.open(base_directory + name + "/" + name + "_B8A.tif").read()).astype(np.float32),
    ]

    max_rgb = max(bands[1].max(), bands[2].max(), bands[3].max())
    bands[1] = (bands[1] / max_rgb * 255).astype(np.uint8)
    bands[2] = (bands[2] / max_rgb * 255).astype(np.uint8)
    bands[3] = (bands[3] / max_rgb * 255).astype(np.uint8)

    for i in [0, 4, 5, 6, 7, 8, 9, 10, 11]:
        bands[i] = (bands[i] / bands[i].max() * 255).astype(np.uint8)

    return bands


def load_image_gray(filename):
    return cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)


def load_image(filename):
    return cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)


def make_points(image, scale=1, z_scale=2):
    points = np.zeros((image.shape[0], image.shape[1], 3))
    w, h = image.shape[1], image.shape[0]
    max_z = image.max()
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            z = image[image.shape[0] - y - 1, x]
            points[y, x] = [(x - w / 2) / w * scale, (y - h / 2) / h * scale, (z / max_z) * z_scale]

    return points


def make_triangles(points):
    uvs = np.zeros((points.shape[0] * points.shape[1] * 3 * 2, 2))
    vertices = np.zeros((points.shape[0] * points.shape[1] * 3 * 2, 3))

    h = points.shape[0]
    w = points.shape[1]

    vertex_index = 0
    for y in range(points.shape[0] - 1):
        for x in range(points.shape[1] - 1):
            vertices[vertex_index] = points[y, x]
            vertices[vertex_index + 1] = points[y, x + 1]
            vertices[vertex_index + 2] = points[y + 1, x]

            uvs[vertex_index] = [x / w, y / h]
            uvs[vertex_index + 2] = [x / w, (y + 1) / h]
            uvs[vertex_index + 1] = [(x + 1) / w, y / h]

            vertex_index += 3

    for y in range(1, points.shape[0]):
        for x in range(1, points.shape[1]):
            vertices[vertex_index] = points[y, x]
            vertices[vertex_index + 1] = points[y, x - 1]
            vertices[vertex_index + 2] = points[y - 1, x]

            uvs[vertex_index] = [x / w, y / h]
            uvs[vertex_index + 2] = [x / w, (y - 1) / h]
            uvs[vertex_index + 1] = [(x - 1) / w, y / h]

            vertex_index += 3

    return vertices, uvs


def make_material(image):
    texture = pyrender.Texture(source=image, source_channels='RGB')
    material = pyrender.MetallicRoughnessMaterial(baseColorTexture=texture, metallicFactor=0)
    return material


def make_primitive(points, uvs, material):
    # return pyrender.Primitive(points)
    return pyrender.Primitive(points, texcoord_0=uvs, material=material)


def make_mesh(primitives):
    return pyrender.Mesh(primitives)


def make_terrain(dem_file, bands_folder, bands_name):
    hm = load_dem(dem_file)
    # hm = np.array([[1, .75], [.75, 0]])
    texture = load_albedo(bands_folder, bands_name)
    pts = make_points(hm)
    vrtxs, uvs = make_triangles(pts)
    mtl = make_material(texture)
    prim = make_primitive(vrtxs, uvs, mtl)
    mesh = make_mesh([prim])

    return mesh


def make_terrains(dem_file, bands_folder, bands_name):
    hm = load_dem(dem_file)
    # hm = np.array([[1, .75], [.75, 0]])
    bands = load_bands(bands_folder, bands_name)
    pts = make_points(hm)
    vrtxs, uvs = make_triangles(pts)

    meshes = []
    for band in bands:
        mtl = make_material(np.stack([band, band, band], -1))
        prim = make_primitive(vrtxs, uvs, mtl)
        mesh = make_mesh([prim])
        meshes.append(mesh)

    return meshes, pts


def make_cube(pose, texture):
    mesh = trimesh.creation.box(transform=pose)
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [0, 1], [1, 0], [1, 1], ]),
        image=texture
    )
    return mesh

def make_cubes():
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
    image = cv2.imread("tejado.jpg", cv2.IMREAD_GRAYSCALE)

    cubes = []
    for pose, (rough, metal) in zip(poses, properties):
        cube = pyrender.Mesh.from_trimesh(make_cube(pose, image))
        cube.primitives[0].material.RoughnessFactor = rough
        cube.primitives[0].material.metallicFactor = metal
        cubes.append(cube)

    return cubes


def lookAt(center, target, up):
    f = (target - center)
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    u = u / np.linalg.norm(u)

    m = np.zeros((4, 4))
    m[0, :-1] = s
    m[1, :-1] = u
    m[2, :-1] = -f
    m[-1, -1] = 1.0

    return m


def scale(pose, factor):
    return pose @ np.array([[factor, 0, 0, 0], [0, factor, 0, 0], [0, 0, factor, 0], [0, 0, 0, factor]])

def rotx(pose, a):
    rot = np.eye(4)
    rot[1, 1] = np.cos(a)
    rot[1, 2] = -np.sin(a)
    rot[2, 1] = np.sin(a)
    rot[2, 2] = np.cos(a)
    return np.dot(rot, pose)


def roty(pose, a):
    rot = np.eye(4)
    rot[0, 0] = np.cos(a)
    rot[0, 2] = np.sin(a)
    rot[2, 0] = -np.sin(a)
    rot[2, 2] = np.cos(a)
    return np.dot(rot, pose)


def rotz(pose, a):
    rot = np.eye(4)
    rot[0, 0] = np.cos(a)
    rot[0, 1] = -np.sin(a)
    rot[1, 0] = np.sin(a)
    rot[1, 1] = np.cos(a)
    return np.dot(rot, pose)


def camera_poses():
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 7.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    for x in np.arange(-1, 1, 300):
        for y in np.arange(-1000 * 2, 1000 * 2, 300):
            current_pose = np.copy(camera_pose)

            current_pose[0, 3] += x
            current_pose[1, 3] += y

            yield current_pose, f"{x:.4f}_{y:.4f}"


def ncamera_poses(n):
    for _ in range(n):
        x = np.random.uniform(-0.5, 0.5, 1)[0]
        y = np.random.uniform(-0.5, 0.5, 1)[0]

        camera_pose = np.array([
            [1.0, 0.0, 0.0, x],
            [0.0, 1.0, 0.0, y],
            [0.0, 0.0, 1.0, 6.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        eye = np.array([camera_pose[0, 3], camera_pose[1, 3], camera_pose[2, 3]])
        center = np.array([0, 0, 0])
        up = np.array([0, 1, 0])

        rotation = np.linalg.inv(lookAt(eye, center, up))

        # camera_pose = rotation * camera_pose
        camera_pose[:3, :3] = rotation[:3, :3]

        yield camera_pose, f"{x:.4f}_{y:.4f}"


def day():
    for t in np.arange(0, np.pi, 0.1):
        yield t, f"{t:.4f}"


def pkl_save(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def generate_eo_dataset(scene, renderer, sun):
    image_index = 1

    for pose, posename in ncamera_poses(20):
        # for (sunpose, ambient), time in [(sun.cycle(time), time) for time in np.arange(0, np.pi, 0.1)]:
        scene.set_pose(camn, pose)
        # scene.set_pose(sun.node, sunpose)
        # scene.ambient_light = ambient

        color, depth = renderer.render(scene)

        band = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(f"/home/amarcos/workspace/TFG/scripts/generated_eo_test_data/{image_index:010d}.png", band)
        # pkl_save({'camera_pose': pose, 'sun_pose': sunpose, 'time': time}, f"/home/amarcos/workspace/TFG/scripts/generated_eo_data/{image_index:010d}.pkl")
        pkl_save({'camera_pose': pose},
                 f"/home/amarcos/workspace/TFG/scripts/generated_eo_test_data/{image_index:010d}.pkl")

        image_index += 1


def generate_multispectral_eo_dataset(scene, renderer, sun, meshes):
    folder = "/home/amarcos/workspace/TFG/scripts/generated_neo_multispectral_cubes_data/"
    modes = ["train", "val", "test"]

    for mode in modes:
        image_index = 1

        jsdoc = dict(frames=[], camera_angle_x=np.pi / 3 / 8)

        os.makedirs(f"{folder}/{mode}", exist_ok=True)

        for pose, posename in ncamera_poses(20):
            # for (sunpose, ambient), time in [(sun.cycle(time), time) for time in np.arange(0, np.pi, 0.1)]:
            scene.set_pose(camn, pose)

            for mesh, bandname in zip(meshes,
                                      ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B11", "B12",
                                       "B8A", ]):
                node = scene.add(mesh)

                color, depth = renderer.render(scene, flags=RenderFlags.SHADOWS_ALL | RenderFlags.ALL_SOLID)

                bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"{folder}/{mode}/{image_index:010d}_{bandname}.png", bgr)

                scene.remove_node(node)

            node = scene.add(mesh)

            color, depth = renderer.render(scene)
            np.save(f"{folder}/{mode}/{image_index:010d}_DEPTH", depth)

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
            # scene.set_pose(sun.node, sunpose)
            # scene.ambient_light = ambient

            image_index += 1

        json.dump(jsdoc, open(f"{folder}/transforms_{mode}.json", "wt"), indent=4)


def generate_depth_eo_dataset(scene, renderer, sun, meshes):
    folder = "/home/amarcos/workspace/TFG/scripts/generated_neo_multispectral_cubes_data/"
    modes = ["train", "val", "test"]

    for mode in modes:
        image_index = 1

        os.makedirs(f"{folder}/{mode}", exist_ok=True)

        jsdoc = json.load(open(f"{folder}/transforms_{mode}.json", "rt"))

        for frame in jsdoc['frames']:
            pose = np.array(frame["transform_matrix"])
            # for (sunpose, ambient), time in [(sun.cycle(time), time) for time in np.arange(0, np.pi, 0.1)]:
            scene.set_pose(camn, pose)

            for mesh, bandname in zip([meshes[0]],
                                      ["DEPTH"]):
                node = scene.add(mesh)

                color, depth = renderer.render(scene)
                np.save(f"{folder}/{mode}/{image_index:010d}_DEPTH", depth)

                scene.remove_node(node)
            # scene.set_pose(sun.node, sunpose)
            # scene.ambient_light = ambient

            image_index += 1

def interact(scene):
    pyrender.Viewer(scene, render_flags={"shadows": True})


if __name__ == '__main__':
    meshes, pts = make_terrains(
        "/data1tb/BigEarthNet-S2-v1.0/BigEarthNet-S2-v1.0/dem/S2B_MSIL2A_20170709T094029_78_59_dem.tif",
        "/data1tb/BigEarthNet-S2-v1.0/BigEarthNet-S2-v1.0/BigEarthNet-v1.0/",
        "S2B_MSIL2A_20170709T094029_78_59"
    )

    light = pyrender.DirectionalLight(intensity=10)
    # cam = pyrender.PerspectiveCamera(1, 0.05, 1000.0)

    scene = pyrender.Scene(ambient_light=np.array([1.0, 1.0, 1.0]))
    # scene = pyrender.Scene()
    scene.add(meshes[1])
    #     scene.add(cam)
    # sunlight = scene.add(light, pose=rotx(np.eye(4), np.pi / 2 - 0.3))
    sunlight = scene.add(light, pose=roty(np.eye(4), np.pi / 4))
    # pyrender.Viewer(scene, render_flags={'shadows': True})
    #
    # exit()

    sun = Sun(sunlight)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3 / 8, aspectRatio=1.0)
    # camera = pyrender.OrthographicCamera(40, 40, zfar=1000)
    s = np.sqrt(2) / 2
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 5.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    eye = np.array([camera_pose[0, 3], camera_pose[1, 3], camera_pose[2, 3]])
    center = np.array([-1, 0, 0])
    up = np.array([0, 1, 0])

    rotation = np.linalg.inv(lookAt(eye, center, up))

    # camera_pose = rotation * camera_pose
    camera_pose[:3, :3] = rotation[:3, :3]
    camn = scene.add(camera, pose=camera_pose)

    cubes = make_cubes()
    cube_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    for cube in cubes:
        point = pts[random.randint(0, 959), random.randint(0, 959)]
        cube_pose_copy = np.copy(cube_pose)
        cube_pose_copy[:3, 3] = point
        scene.add(cube, pose=cube_pose_copy)

    r = pyrender.OffscreenRenderer(800, 800)
    generate_multispectral_eo_dataset(scene, r, sun, meshes)
    # generate_depth_eo_dataset(scene, r, sun, meshes)
    #
    # interact(scene)
    # r = pyrender.OffscreenRenderer(120, 120)
    # generate_eo_dataset(scene, r, sun)

    print("END")
