import cv2
import numpy as np
import rasterio
from matplotlib import pyplot as plt


class Loader:
    def load_dem(self, filename):
        dem = np.squeeze(rasterio.open(filename).read()).astype(np.float32)

        plt.imsave("dem.png", (dem - dem.min()) / (dem.max() - dem.min()), cmap='gray')

        h, w = dem.shape
        k = 2
        dem2 = dem
        dem2 = dem2 + (np.random.uniform(size=(h, w)) * k - (k / 2))
        k = k / 4
        dem2 = cv2.resize(dem2, (h * 2, w * 2))
        dem2 = dem2 + (np.random.uniform(size=(h * 2, w * 2)) * k - (k / 2))
        k = k / 4
        dem2 = cv2.resize(dem2, (h * 4, w * 4))
        dem2 = dem2 + (np.random.uniform(size=(h * 4, w * 4)) * k - (k / 2))
        k = k / 4
        dem2 = cv2.resize(dem2, (h * 8, w * 8))
        dem2 = dem2 + (np.random.uniform(size=(h * 8, w * 8)) * k - (k / 2))

        # dem2 = cv2.resize(dem, (h * 32, w * 32))
        # s = 51
        # dem2 = cv2.GaussianBlur(dem2, (s, s), sigmaX=s / 6)

        plt.imsave("dem2.png", (dem2 - dem2.min()) / (dem2.max() - dem2.min()), cmap='gray')
        return dem2

    def load_albedo(self, base_directory, name):
        b = np.squeeze(rasterio.open(base_directory + name + "/" + name + "_B02.tif").read())
        g = np.squeeze(rasterio.open(base_directory + name + "/" + name + "_B03.tif").read())
        r = np.squeeze(rasterio.open(base_directory + name + "/" + name + "_B04.tif").read())

        rgb = np.stack([r, g, b], -1) // 5
        return rgb.astype(np.uint8)

    def load_bands(self, base_directory, name):
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

    def load_image_gray(self, filename):
        return cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)

    def load_image(self, filename):
        return cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
