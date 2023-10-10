import cv2
import numpy as np
import rasterio
import trimesh
import pyrender


# S2B_MSIL2A_20170709T094029_78_59

def load_dem(filename):
    return np.squeeze(rasterio.open(filename).read())


def load_albedo(base_directory, name):
    b = np.squeeze(rasterio.open(base_directory + name + "/" + name + "_B02.tif").read())
    g = np.squeeze(rasterio.open(base_directory + name + "/" + name + "_B03.tif").read())
    r = np.squeeze(rasterio.open(base_directory + name + "/" + name + "_B04.tif").read())

    rgb = np.stack([r, g, b], -1) // 5
    return rgb.astype(np.uint8)


def load_image_gray(filename):
    return cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)


def load_image(filename):
    return cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)


def make_points(image):
    points = np.zeros((image.shape[0], image.shape[1], 3))
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            z = image[image.shape[0] - y - 1, x]
            points[y, x] = [x, y, z / 2]

    return points


def make_triangles(points):
    uvs = np.zeros((points.shape[0] * points.shape[1] * 3 * 2, 2))
    vertices = np.zeros((points.shape[0] * points.shape[1] * 3 * 2, 3))

    inv_x = 1
    inv_y = 1

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
    return pyrender.MetallicRoughnessMaterial(baseColorTexture=texture)


def make_primitive(points, uvs, material):
    # return pyrender.Primitive(points)
    return pyrender.Primitive(points, texcoord_0=uvs, material=material)


def make_mesh(primitives):
    return pyrender.Mesh(primitives)


if __name__ == '__main__':
    hm = load_dem("/home/amarcos/Downloads/BigEarthNet-S2-v1.0/BigEarthNet-S2-v1.0/dem/S2B_MSIL2A_20170709T094029_78_59_dem.tif")
    # hm = np.array([[1, .75], [.75, 0]])
    texture = load_albedo("/home/amarcos/Downloads/BigEarthNet-S2-v1.0/BigEarthNet-S2-v1.0/BigEarthNet-v1.0/", "S2B_MSIL2A_20170709T094029_78_59")
    pts = make_points(hm)
    vrtxs, uvs = make_triangles(pts)
    mtl = make_material(texture)
    prim = make_primitive(vrtxs, uvs, mtl)
    mesh = make_mesh([prim])

    light = pyrender.PointLight(intensity=10.0)

    # fuze_trimesh = trimesh.load('./models/fuze.obj')
    # fuze_mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)

    scene = pyrender.Scene()
    scene.add(mesh)
    # scene.add(fuze_mesh)
    scene.add(light, pose=np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 10],
        [0, 0, 0, 1],
    ]))
    pyrender.Viewer(scene, use_raymond_lighting=True, render_flags={'shadows': False})
