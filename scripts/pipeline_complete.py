#!/usr/bin/env python3
import cv2
import numpy as np
import os

#######################
#  Fonctions du Pipeline  #
#######################

def preprocess_image(image, blur_kernel=(5,5)):
    """
    Applique un flou gaussien sur l'image (en BGR) et retourne l'image floutée.
    Les dimensions de l'image ne sont pas modifiées.
    """
    image_blurred = cv2.GaussianBlur(image, blur_kernel, 0)
    return image_blurred

def detect_red_crosses(image):
    """
    Utilise l'image prétraitée pour détecter des zones rouges (croix) servant
    de repères pour déterminer les coins de l'échiquier.
    
    Retourne une liste de points (en pixels) et le masque binaire correspondant.
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
    Cet ordre est nécessaire pour le calcul correct de l'homographie.
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
    Calcule la matrice d'homographie H pour transformer des points dans le système pixels
    en coordonnées réelles (exprimées ici en cm) de l'échiquier.
    """
    pixelPoints = np.array(pixelPoints, dtype="float32")
    realWorldPoints = np.array(realWorldPoints, dtype="float32")
    H, status = cv2.findHomography(pixelPoints, realWorldPoints)
    return H

def pixel_to_real(H, point):
    """
    Convertit un point (u, v) dans l'image en coordonnées réelles via la matrice d'homographie H.
    """
    pt = np.array([[point[0]], [point[1]], [1]], dtype="float32")
    real_pt = np.dot(H, pt)
    real_pt /= real_pt[2]
    return (real_pt[0][0], real_pt[1][0])

def extract_square_roi(image, H_inv, square_center, caseSize_pixels):
    """
    Convertit le centre d'une case (exprimé en unités réelles, par exemple cm) en coordonnées pixels,
    puis extrait la région d'intérêt (ROI) de la case de taille caseSize_pixels.
    """
    pt_real = np.array([[square_center[0]], [square_center[1]], [1]], dtype="float32")
    pt_pixel = np.dot(H_inv, pt_real)
    pt_pixel /= pt_pixel[2]
    cx, cy = int(pt_pixel[0][0]), int(pt_pixel[1][0])
    roi = image[cy - caseSize_pixels//2 : cy + caseSize_pixels//2,
                cx - caseSize_pixels//2 : cx + caseSize_pixels//2]
    return roi

def detect_chess_piece(roi, threshold_value=127, area_min=100, area_max=10000):
    """
    Analyse la ROI d'une case pour détecter la présence d'un pion.
    Retourne un booléen indiquant la présence d'un pion et l'image seuillée.
    """
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.GaussianBlur(gray_roi, (5,5), 0)
    _, thresh = cv2.threshold(gray_roi, threshold_value, 255, cv2.THRESH_BINARY_INV)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    piece_detected = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_min or area > area_max:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity > 0.5:
            piece_detected = True
            cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
    return piece_detected, thresh

def detect_blue_pieces(image, board_origin, square_size=55):
    """
    Détecte tous les pions bleus dans l'image originale.
    - Convertit l'image en HSV et applique un seuillage pour la couleur bleue.
    - Cherche des contours de formes circulaires (puisque les pions sont ronds).
    - Pour chaque pion détecté, calcule la case dans laquelle il se trouve en fonction
      de l'origine du plateau (coin supérieur gauche) et de la taille d'une case (square_size en pixels).
    
    On suppose ici que la notation est standard avec le coin supérieur gauche = "A8".
    Ainsi, la colonne (index 0 -> "A", 1 -> "B", etc.) est
    calculée par : int((cx - board_origin_x) / square_size)
    et la rangée par : 8 - int((cy - board_origin_y) / square_size).
    
    Retourne une liste de tuples (center, square) et le masque utilisé.
    """
    # Convertir en HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Définir la plage HSV pour le bleu (ajustez ces valeurs si besoin)
    lower_blue = np.array([100, 100, 50])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Nettoyage du masque
    kernel = np.ones((3,3), np.uint8)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pieces = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:  # Filtrer le bruit
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.7:  # Exiger une forme circulaire
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        # Calcul de la case.
        dx = cx - board_origin[0]
        dy = cy - board_origin[1]
        if dx < 0 or dy < 0:
            continue  # en dehors du plateau
        col_index = int(dx / square_size)
        row_index = int(dy / square_size)
        if col_index < 0 or col_index > 7 or row_index < 0 or row_index > 7:
            continue  # en dehors des 8 colonnes ou 8 rangées
        # Notation standard : colonnes: A-H, rangées: 8 à 1 (du haut vers le bas)
        square = chr(ord('A') + col_index) + str(8 - row_index)
        pieces.append(((cx, cy), square))
    return pieces, mask_blue

#######################
#     Pipeline Main   #
#######################

def main():
    # Chemin de l'image source (originale)
    imgPath = "/home/zakaria/Workspace/ChessProject/data/damier.png"
    
    if not os.path.isfile(imgPath):
        print("Erreur : le fichier n'existe pas à cet emplacement :", imgPath)
        return
    image_orig = cv2.imread(imgPath)
    if image_orig is None:
        print("Erreur lors du chargement de l'image :", imgPath)
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
    
    # Affichage des images (Originale, Prétraitée et Masque Rouge)
    cv2.namedWindow("Image Originale", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Image Prétraitée", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Masque Rouge", cv2.WINDOW_NORMAL)
    cv2.imshow("Image Originale", image_orig)
    cv2.imshow("Image Prétraitée", image_preproc)
    cv2.imshow("Masque Rouge", maskRed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Étape 3 : Calcul de l'homographie (si 4 coins ont été détectés)
    # Ici, on définit les coins réels d'un échiquier standard de 40x40 cm (8 cases x 5 cm).
    real_world_points = [(0, 0), (40, 0), (40, 40), (0, 40)]
    if ordered_pts is not None:
        H = compute_homography(ordered_pts, real_world_points)
        print("Matrice d'homographie H :", H)
        test_point = (250, 250)  # Point de test (à ajuster si besoin)
        coord_real = pixel_to_real(H, test_point)
        print("Le point", test_point, "correspond aux coordonnées réelles :", coord_real)
    else:
        print("Homographie non calculable faute de 4 points détectés.")
        H = None
    
    # Étape 4 : Détection des pions bleus sur l'image originale.
    # On suppose que le coin supérieur gauche du plateau est donné par le premier point
    # détecté (après ordonnancement). Si non disponible, il faut le définir manuellement.
    if ordered_pts is not None:
        board_origin = (int(ordered_pts[0][0]), int(ordered_pts[0][1]))
    else:
        board_origin = (100, 100)  # Valeur d'exemple à adapter
    
    # Ici, chaque case vaut 55 pixels en hauteur et en largeur sur l'image originale.
    pieces, mask_blue = detect_blue_pieces(image_orig, board_origin, square_size=55)
    print("Pions bleus détectés (centre et case) :")
    for center, square in pieces:
        print(f"Pièce à {center} dans la case {square}")
        cv2.circle(image_orig, center, 5, (0, 255, 0), 2)
        cv2.putText(image_orig, square, (center[0]-10, center[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    
    cv2.namedWindow("Pions bleus", cv2.WINDOW_NORMAL)
    cv2.imshow("Pions bleus", image_orig)
    cv2.namedWindow("Masque Bleu", cv2.WINDOW_NORMAL)
    cv2.imshow("Masque Bleu", mask_blue)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Optionnel : Extraction d'une ROI d'une case donnée et détection d'un pion via seuil.
    # Par exemple, pour la case dont le centre est (20,20) en unités réelles.
    square_center = (20, 20)
    caseSize_pixels = 50  # Vous pouvez changer si nécessaire.
    if H is not None:
        H_inv = np.linalg.inv(H)
        roi_square = extract_square_roi(image_orig, H_inv, square_center, caseSize_pixels)
        cv2.imshow("ROI de la case", roi_square)
        cv2.waitKey(0)
        piece_found, thresh_roi = detect_chess_piece(roi_square, threshold_value=127, area_min=100, area_max=10000)
        print("Pion détecté dans la case ?", piece_found)
        cv2.imshow("ROI avec détection", roi_square)
        cv2.imshow("Threshold de ROI", thresh_roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Impossible d'extraire la ROI car l'homographie n'a pas pu être calculée.")

if __name__ == '__main__':
    main()
