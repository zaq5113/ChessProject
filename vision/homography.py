#!/usr/bin/env python3
import cv2
import numpy as np

def compute_homography(pixelPoints, realWorldPoints):
    """
    Calcule la matrice d'homographie H reliant les points en pixels aux coordonnées réelles.
    """
    pixelPoints = np.array(pixelPoints, dtype="float32")
    realWorldPoints = np.array(realWorldPoints, dtype="float32")
    H, status = cv2.findHomography(pixelPoints, realWorldPoints)
    return H

def pixel_to_real(H, point):
    """
    Convertit un point (u, v) en coordonnées réelles grâce à la matrice d'homographie H.
    """
    pt = np.array([[point[0]], [point[1]], [1]], dtype="float32")
    real_pt = H.dot(pt)
    real_pt /= real_pt[2]
    return (real_pt[0][0], real_pt[1][0])
