import os


import cv2
import matplotlib.pyplot as plt
import numpy as np


def norm(c: np.ndarray) -> np.ndarray:
    return (c - c.min()) / (c.max() - c.min())


alturas_folders = os.listdir("/home/arnau-marcos-almansa/workspace/TFG/scripts/MSNerfCorrected/")
os.chdir("/home/arnau-marcos-almansa/workspace/TFG/scripts/MSNerfCorrected/")

for alturas_folder in sorted(alturas_folders):
    os.chdir(alturas_folder)
    angulos_folders = os.listdir(".")
    for angulo_folder in sorted(angulos_folders):
        os.chdir(angulo_folder)
        B = cv2.imread("2-BP635-27.tiff", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        G = cv2.imread("4-BP525-27.tiff", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        R = cv2.imread("6-BP470-27.tiff", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255

        B = norm(B)
        G = norm(G)
        B = norm(B)

        RGB = np.dstack([R, G, B])

        cv2.imwrite(f"../../{alturas_folder}-{angulo_folder}-RGB.png", (RGB * 255).astype(np.uint8))

        os.chdir("..")
    os.chdir("..")
