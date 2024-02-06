import numpy as np
import pyrender

from scripts.Loader import Loader


class TerrainMaker:

    def __init__(self, loader: Loader):
        self.loader = loader

    def make_points(self, image, scale=1, z_scale=2):
        points = np.zeros((image.shape[0], image.shape[1], 3))
        w, h = image.shape[1], image.shape[0]
        max_z = image.max()
        min_z = image.min()
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                z = image[image.shape[0] - y - 1, x]
                points[y, x] = [(x - w / 2) / w * scale, (y - h / 2) / h * scale, ((z - min_z) / max_z) * z_scale]

        return points


    def make_triangles(self, points):
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


    def make_material(self, image):
        texture = pyrender.Texture(source=image, source_channels='RGB')
        material = pyrender.MetallicRoughnessMaterial(baseColorTexture=texture, metallicFactor=0)
        return material


    def make_primitive(self, points, uvs, material):
        # return pyrender.Primitive(points)
        return pyrender.Primitive(points, texcoord_0=uvs, material=material)


    def make_mesh(self, primitives):
        return pyrender.Mesh(primitives)


    def make_terrain(self, dem_file, bands_folder, bands_name):
        hm = self.loader.load_dem(dem_file)
        # hm = np.array([[1, .75], [.75, 0]])
        texture = self.loader.load_albedo(bands_folder, bands_name)
        pts = self.make_points(hm)
        vrtxs, uvs = self.make_triangles(pts)
        mtl = self.make_material(texture)
        prim = self.make_primitive(vrtxs, uvs, mtl)
        mesh = self.make_mesh([prim])

        return mesh


    def make_terrains(self, dem_file, bands_folder, bands_name):
        hm = self.loader.load_dem(dem_file)
        # hm = np.array([[1, .75], [.75, 0]])
        bands = self.loader.load_bands(bands_folder, bands_name)
        pts = self.make_points(hm)
        vrtxs, uvs = self.make_triangles(pts)

        meshes = []
        for band in bands:
            mtl = self.make_material(np.stack([band, band, band], -1))
            prim = self.make_primitive(vrtxs, uvs, mtl)
            mesh = self.make_mesh([prim])
            meshes.append(mesh)

        return meshes, pts
