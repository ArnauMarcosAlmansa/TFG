import cv2
import matplotlib.pyplot as plt

dem = cv2.imread("dem.png")
dem_suave = cv2.imread("dem2.png")

dem_zoom = dem[50:90, 30:70]
dem_suave_zoom = dem_suave[50 * 8:90 * 8, 30 * 8:70 * 8]

dem[49:50, 29:71] = [0, 0, 255]
dem[90:91, 29:71] = [0, 0, 255]
dem[49:91, 29:30] = [0, 0, 255]
dem[49:91, 70:71] = [0, 0, 255]

cv2.imwrite("demfig.png", dem)
cv2.imwrite("demfigzoom.png", dem_zoom)
cv2.imwrite("demfigzoom2.png", dem_suave_zoom)

plt.imshow(dem)
plt.show()

plt.imshow(dem_zoom)
plt.show()

plt.imshow(dem_suave)
plt.show()

plt.imshow(dem_suave_zoom)
plt.show()