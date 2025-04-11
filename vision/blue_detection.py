#!/usr/bin/env python3
import cv2
import numpy as np

def detect_blue_pieces(image, board_origin, square_size=55):
    """
    Détecte les pions bleus dans l'image originale.
    Si l'image est grande, elle est redimensionnée pour accélérer le traitement.
    Renvoie une liste de tuples ((cx, cy), "") pour chaque pièce détectée et le masque bleu.
    """
    scale_factor = 1.0
    if image.shape[1] > 1000:
        scale_factor = 0.5
        image_small = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
    else:
        image_small = image.copy()
    
    hsv = cv2.cvtColor(image_small, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 100, 50])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    kernel = np.ones((3,3), np.uint8)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pieces_small = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.7:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        pieces_small.append(((cx, cy), ""))
    
    pieces = []
    if scale_factor != 1.0:
        for (cx, cy), _ in pieces_small:
            pieces.append(((int(cx/scale_factor), int(cy/scale_factor)), ""))
        mask_blue = cv2.resize(mask_blue, (image.shape[1], image.shape[0]))
    else:
        pieces = pieces_small
    return pieces, mask_blue
