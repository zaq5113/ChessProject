#!/usr/bin/env python3
import cv2
import numpy as np
import os

#############################
# Fonctions du pipeline     #
#############################

def preprocess_image(image, blur_kernel=(5,5)):
    """
    Applique un flou gaussien sur l'image (en BGR) et retourne l'image floutée.
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
    # Plages HSV pour le rouge
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
        if area < 50:
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
    Calcule la matrice d'homographie H reliant les points pixels aux coordonnées réelles (en cm) du plateau.
    """
    pixelPoints = np.array(pixelPoints, dtype="float32")
    realWorldPoints = np.array(realWorldPoints, dtype="float32")
    H, status = cv2.findHomography(pixelPoints, realWorldPoints)
    return H

def pixel_to_real(H, point):
    """
    Convertit un point (u, v) de l'image en coordonnées réelles via la matrice H.
    """
    pt = np.array([[point[0]], [point[1]], [1]], dtype="float32")
    real_pt = np.dot(H, pt)
    real_pt /= real_pt[2]
    return (real_pt[0][0], real_pt[1][0])

def extract_square_roi(image, H_inv, square_center, caseSize_pixels):
    """
    Convertit le centre d'une case (en unités réelles, ex. cm) en coordonnées pixels via H_inv,
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
    - Convertit l'image en HSV et applique un seuillage pour la couleur bleue.
    - Cherche des contours correspondant à des formes circulaires.
    Pour chaque pion bleu détecté, calcule sa position relative par rapport à board_origin
    et détermine la case correspondante (notation : ligne de A à J et colonne de 1 à 10).
    
    board_origin est le pixel représentant le coin supérieur gauche interne du plateau.
    Retourne une liste de tuples ((cx, cy), square) et le masque bleu.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 100, 50])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Nettoyage du masque (ouvrir puis fermer)
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
        # Calcul de la position relative par rapport à board_origin
        dx = cx - board_origin[0]
        dy = cy - board_origin[1]
        if dx < 0 or dy < 0:
            continue  # en dehors du plateau
        # Calcul de l'index de colonne (0 à 9) et de ligne (0 à 9)
        col_index = int(dx / square_size)
        row_index = int(dy / square_size)
        if col_index < 0 or col_index > 9 or row_index < 0 or row_index > 9:
            continue  # en dehors des bornes du plateau
        # Conversion en notation: les lignes de A à J et les colonnes de 1 à 10.
        square = chr(ord('A') + row_index) + str(col_index + 1)
        pieces.append(((cx, cy), square))
    return pieces, mask_blue

def save_detection_results(results, output_file):
    """
    Sauvegarde les résultats de détection dans un fichier texte.
    Chaque ligne contient la position du pion (centre en pixels) et la case correspondante.
    """
    with open(output_file, 'w') as f:
        for center, square in results:
            f.write(f"Pièce détectée à {center} -> case {square}\n")

#############################
#  Pipeline main            #
#############################

def main():
    # Chemin de l'image source
    imgPath = "/home/zakaria/Workspace/ChessProject/data/damier.png"
    if not os.path.isfile(imgPath):
        print("Erreur : fichier inexistant à", imgPath)
        return
    image_orig = cv2.imread(imgPath)
    if image_orig is None:
        print("Erreur lors du chargement de l'image.")
        return
    print("Dimensions image originale :", image_orig.shape)
    
    # Étape 1 : Prétraitement (floutage)
    image_preproc = preprocess_image(image_orig, blur_kernel=(5,5))
    
    # Étape 2 : Détection des repères (croix rouges) sur l'image prétraitée
    points, maskRed = detect_red_crosses(image_preproc)
    print("Points rouges détectés (non ordonnés) :", points)
    if len(points) >= 4:
        ordered_pts = order_points(points[:4])
        print("Points rouges ordonnés :", ordered_pts)
        for pt in ordered_pts:
            cv2.circle(image_preproc, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
    else:
        print("Moins de 4 points rouges détectés. Homographie impossible.")
        ordered_pts = None
    
    # Affichage des images de repère
    cv2.namedWindow("Image Originale", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Image Prétraitée", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Masque Rouge", cv2.WINDOW_NORMAL)
    cv2.imshow("Image Originale", image_orig)
    cv2.imshow("Image Prétraitée", image_preproc)
    cv2.imshow("Masque Rouge", maskRed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Étape 3 : Calcul de l'homographie si 4 coins ont été détectés
    # Pour un plateau de 10x10 cases, on définit les coins réels avec une taille de plateau de 550 pixels (10 x 55).
    real_world_points = [(0, 0), (550, 0), (550, 550), (0, 550)]
    if ordered_pts is not None:
        H = compute_homography(ordered_pts, real_world_points)
        print("Matrice d'homographie H :", H)
        test_point = (250, 250)  # Exemple
        coord_real = pixel_to_real(H, test_point)
        print("Le point", test_point, "correspond aux coordonnées réelles :", coord_real)
    else:
        print("Homographie non calculable faute de 4 points détectés.")
        H = None
    
    # Nouvelle étape importante :
    # Détermination de board_origin (origine interne du plateau)
    # Board_origin est défini comme le coin supérieur gauche ordonné + (55, 55) pixels.
    if ordered_pts is not None:
        board_origin = (int(ordered_pts[0][0] + 55), int(ordered_pts[0][1] + 55))
        print("Board origin (coin supérieur gauche interne) :", board_origin)
    else:
        board_origin = (0, 0)
        print("Board origin non défini, utilisation de (0,0).")
    
    # Étape 4 : Détection des pions bleus sur l'image originale
    pieces, mask_blue = detect_blue_pieces(image_orig, board_origin, square_size=55)
    print("Pions bleus détectés (centre et case) :")
    for center, square in pieces:
        print(f"Pièce détectée à {center} -> case {square}")
        cv2.circle(image_orig, center, 5, (0, 255, 0), 2)
        cv2.putText(image_orig, square, (center[0]-10, center[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    
    cv2.namedWindow("Pions bleus", cv2.WINDOW_NORMAL)
    cv2.imshow("Pions bleus", image_orig)
    cv2.namedWindow("Masque Bleu", cv2.WINDOW_NORMAL)
    cv2.imshow("Masque Bleu", mask_blue)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Enregistrement des résultats dans un fichier texte
    output_txt = "/home/zakaria/Workspace/ChessProject/data/detection_results.txt"
    save_detection_results(pieces, output_txt)
    print("Résultats de détection sauvegardés dans :", output_txt)
    
    # Optionnel : Extraction d'une ROI pour vérification (exemple pour la case A1)
    # Ici, nous supposons que la case A1 dans le système interne correspond à (0,0) en unités réelles.
    square_center = (0, 0)
    if H is not None:
        H_inv = np.linalg.inv(H)
        roi_square = extract_square_roi(image_orig, H_inv, square_center, 55)
        cv2.imshow("ROI de la case A1", roi_square)
        cv2.waitKey(0)
        piece_found, thresh_roi = detect_chess_piece(roi_square, threshold_value=127, area_min=100, area_max=10000)
        print("Pion détecté dans la case A1 ?", piece_found)
        cv2.imshow("ROI avec détection", roi_square)
        cv2.imshow("Threshold de ROI", thresh_roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Impossible d'extraire la ROI car l'homographie n'a pas pu être calculée.")

if __name__ == '__main__':
    main()
