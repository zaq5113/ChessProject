#!/usr/bin/env python3
import cv2
import numpy as np
import os

#############################
# Fonctions du Pipeline     #
#############################

def preprocess_image(image, blur_kernel=(5,5)):
    """
    Applique un flou gaussien sur l'image en BGR et retourne l'image floutée.
    (Les dimensions restent inchangées.)
    """
    image_blurred = cv2.GaussianBlur(image, blur_kernel, 0)
    return image_blurred

def detect_red_crosses(image):
    """
    À partir de l'image prétraitée, détecte les zones rouges (croix) servant de repère.
    Renvoie une liste de points (en pixels) et le masque binaire associé.
    """
    imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lowerRed1 = np.array([0, 100, 100])
    upperRed1 = np.array([10, 255, 255])
    lowerRed2 = np.array([160, 100, 100])
    upperRed2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(imageHSV, lowerRed1, upperRed1)
    mask2 = cv2.inRange(imageHSV, lowerRed2, upperRed2)
    maskRed = cv2.bitwise_or(mask1, mask2)
    
    contours, _ = cv2.findContours(maskRed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:  # Filtrage du bruit
            continue
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            points.append((cx, cy))
    return points, maskRed

def order_points(points):
    """
    Ordonne 4 points dans l'ordre suivant :
      - Coin supérieur gauche,
      - Coin supérieur droit,
      - Coin inférieur droit,
      - Coin inférieur gauche.
    """
    pts = np.array(points, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered

def compute_homography(pixelPoints, realWorldPoints):
    """
    Calcule la matrice d'homographie H reliant les points pixels aux coordonnées réelles.
    (Ici, pour un plateau de 10x10 cases, de taille totale 550x550 pixels.)
    """
    pixelPoints = np.array(pixelPoints, dtype="float32")
    realWorldPoints = np.array(realWorldPoints, dtype="float32")
    H, status = cv2.findHomography(pixelPoints, realWorldPoints)
    return H

def pixel_to_real(H, point):
    """
    Convertit un point (u,v) de l'image en coordonnées réelles via la matrice H.
    """
    pt = np.array([[point[0]], [point[1]], [1]], dtype="float32")
    real_pt = np.dot(H, pt)
    real_pt /= real_pt[2]
    return (real_pt[0][0], real_pt[1][0])

def extract_square_roi(image, H_inv, square_center, caseSize_pixels):
    """
    Transforme le centre d'une case (en unités réelles) en coordonnées pixels via H_inv,
    puis extrait la ROI de dimension caseSize_pixels x caseSize_pixels.
    """
    pt_real = np.array([[square_center[0]], [square_center[1]], [1]], dtype="float32")
    pt_pixel = np.dot(H_inv, pt_real)
    pt_pixel /= pt_pixel[2]
    cx, cy = int(pt_pixel[0][0]), int(pt_pixel[1][0])
    roi = image[cy - caseSize_pixels//2 : cy + caseSize_pixels//2,
                cx - caseSize_pixels//2 : cx + caseSize_pixels//2]
    return roi

def detect_blue_pieces(image, board_origin, square_size=55):
    """
    Détecte tous les pions bleus dans l'image originale.
    Convertit l'image en HSV et applique un seuillage pour la couleur bleue.
    Renvoie une liste de tuples ((cx,cy), _) pour chaque pièce détectée,
    sans affecter ici directement la case.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 100, 50])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    kernel = np.ones((3,3), np.uint8)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pieces = []
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
        pieces.append(((cx, cy), ""))  # Nous ne nous servirons plus de la case calculée ici
    return pieces, mask_blue

def assign_pieces_to_grid(pieces, board_origin, square_size=55, tolerance=5):
    """
    Pour chaque case du plateau (10x10), calcule le centre attendu 
    à partir de board_origin. Si une pièce bleue détectée a un centre qui
    est à moins de 'tolerance' pixels du centre attendu, on marque cette case.
    
    Retourne un dictionnaire avec pour chaque case (ex. "A1") la valeur :
      - "B" + la case (ex. "BA1") si une pièce a été détectée,
      - "0" sinon.
    """
    grid_results = {}
    # Itération sur les 10 lignes et 10 colonnes.
    for i in range(10):   # lignes : A à J (i = 0 -> A, 9 -> J)
        for j in range(10):  # colonnes : 1 à 10
            square_name = chr(ord('A') + i) + str(j+1)
            # Le centre attendu de la case dans l'espace en pixels
            # board_origin correspond au coin supérieur gauche effectif interne du plateau.
            expected_center = (board_origin[0] + j*square_size + square_size/2,
                               board_origin[1] + i*square_size + square_size/2)
            grid_results[square_name] = "0"  # Par défaut, pas de pièce.
            # Pour chaque pièce détectée
