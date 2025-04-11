#!/usr/bin/env python3
import cv2

def preprocess_image(image, blur_kernel=(5,5)):
    """
    Applique un flou gaussien sur l'image en BGR et retourne l'image floutée.
    Les dimensions restent inchangées.
    """
    image_blurred = cv2.GaussianBlur(image, blur_kernel, 0)
    return image_blurred
