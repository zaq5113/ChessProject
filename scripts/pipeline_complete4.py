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
    Calcule la matrice d'homographie H reliant les points pixels aux coordonnées réelles.
    Pour un plateau de 10x10 cases, la taille totale est 550x550 pixels.
    """
    pixelPoints = np.array(pixelPoints, dtype="float32")
    realWorldPoints = np.array(realWorldPoints, dtype="float32")
    H, status = cv2.findHomography(pixelPoints, realWorldPoints)
    return H

def pixel_to_real(H, point):
    """
    Convertit un point (u,v) de l'image en coordonnées réelles via H.
    """
    pt = np.array([[point[0]], [point[1]], [1]], dtype="float32")
    real_pt = np.dot(H, pt)
    real_pt /= real_pt[2]
    return (real_pt[0][0], real_pt[1][0])

def extract_square_roi(image, H_inv, square_center, caseSize_pixels):
    """
    Convertit le centre d'une case (en unités réelles) en coordonnées pixels via H_inv,
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
    Pour accélérer le traitement, si l'image est très grande, on la redimensionne.
    Renvoie une liste de tuples ((cx,cy), _) pour chaque pièce détectée et le masque bleu.
    """
    # Si l'image est grande, on la redimensionne pour accélérer la détection
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
    # Si l'image a été redimensionnée, remettre à l'échelle les coordonnées
    pieces = []
    if scale_factor != 1.0:
        for (cx, cy), _ in pieces_small:
            pieces.append(((int(cx/scale_factor), int(cy/scale_factor)), ""))
        mask_blue = cv2.resize(mask_blue, (image.shape[1], image.shape[0]))
    else:
        pieces = pieces_small
    return pieces, mask_blue

def assign_pieces_to_grid(pieces, board_origin, square_size=55, tolerance=5):
    """
    Pour chaque case d'un plateau 10x10, calcule le centre attendu à partir de board_origin.
    Avec une tolérance de ±tolerance pixels, si une pièce bleue détectée est proche du centre attendu,
    on marque la case par "B" + case, sinon "0".
    Retourne un dictionnaire : { "A1": "BA1" ou "0", ... }.
    """
    grid_results = {}
    # Pour chaque case, le centre attendu est board_origin + (j*square_size + square_size/2, i*square_size + square_size/2)
    for i in range(10):   # lignes A à J (i = 0 -> A, 9 -> J)
        for j in range(10):  # colonnes 1 à 10 (j = 0 -> 1, 9 -> 10)
            square_name = chr(ord('A') + i) + str(j+1)
            expected_center = (board_origin[0] + j*square_size + square_size/2,
                               board_origin[1] + i*square_size + square_size/2)
            grid_results[square_name] = "0"
            for (cx, cy), _ in pieces:
                dx = cx - expected_center[0]
                dy = cy - expected_center[1]
                if np.sqrt(dx*dx + dy*dy) <= tolerance:
                    grid_results[square_name] = "B" + square_name
                    break
    return grid_results

def save_detection_results_grid(grid_results, output_file):
    """
    Enregistre les résultats dans le fichier texte, une ligne par case :
    ex: "A1,BA1" ou "A1,0"
    """
    with open(output_file, 'w') as f:
        for square in sorted(grid_results.keys(), key=lambda s: (s[0], int(s[1:]))):
            f.write(f"{square},{grid_results[square]}\n")

def create_mosaic(images, labels=None, cols=2, gap=5):
    """
    Regroupe une liste d'images en une seule image sous forme de mosaïque.
    images : liste d'images (mêmes dimensions recommandées)
    labels (optionnel) : liste de textes à superposer sur chaque image.
    cols : nombre de colonnes dans la mosaïque.
    gap : espace (en pixels) entre les images.
    Retourne l'image composite.
    """
    if len(images) == 0:
        return None
    
    # Normalisation : redimensionner toutes les images à la même taille
    h, w = images[0].shape[:2]
    resized = [cv2.resize(img, (w, h)) for img in images]
    
    # Appliquer un label (facultatif)
    if labels is not None:
        for i, label in enumerate(labels):
            cv2.putText(resized[i], label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    # Calculer le nombre de rangées
    rows = (len(resized) + cols - 1) // cols
    # Créer la mosaïque
    mosaic = []
    for r in range(rows):
        row_imgs = []
        for c in range(cols):
            idx = r * cols + c
            if idx < len(resized):
                row_imgs.append(resized[idx])
            else:
                row_imgs.append(np.zeros_like(resized[0]))
        row = cv2.hconcat(row_imgs)
        mosaic.append(row)
    mosaic = cv2.vconcat(mosaic)
    # Optionnel : ajouter un gap entre les images (non implémenté ici)
    return mosaic

#############################
#  Pipeline Main            #
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
    
    # Étape 2 : Détection des repères (croix rouges)
    points, maskRed = detect_red_crosses(image_preproc)
    print("Points rouges détectés :", points)
    if len(points) >= 4:
        ordered_pts = order_points(points[:4])
        print("Points rouges ordonnés :", ordered_pts)
        for pt in ordered_pts:
            cv2.circle(image_preproc, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
    else:
        print("Moins de 4 points rouges détectés. Homographie impossible.")
        ordered_pts = None

    # Création d'une mosaïque pour afficher les images de repère
    mosaic1 = create_mosaic([image_orig, image_preproc, cv2.cvtColor(maskRed, cv2.COLOR_GRAY2BGR)],
                              labels=["Originale", "Prétraitée", "Masque Rouge"], cols=3)
    cv2.namedWindow("Mosaïque - Repères", cv2.WINDOW_NORMAL)
    cv2.imshow("Mosaïque - Repères", mosaic1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Étape 3 : Homographie (pour éventuelle extraction de ROI)
    real_world_points = [(0, 0), (550, 0), (550, 550), (0, 550)]
    if ordered_pts is not None:
        H = compute_homography(ordered_pts, real_world_points)
        print("Matrice d'homographie H :", H)
        test_point = (250, 250)
        coord_real = pixel_to_real(H, test_point)
        print("Le point", test_point, "correspond aux coordonnées réelles :", coord_real)
    else:
        print("Homographie non calculable.")
        H = None
    
    # Étape 4 : Détermination de board_origin
    # Ici, board_origin est le coin supérieur gauche interne = ordered_pts[0] + (55,55)
    if ordered_pts is not None:
        board_origin = (int(ordered_pts[0][0] + 55), int(ordered_pts[0][1] + 55))
        print("Board origin :", board_origin)
    else:
        board_origin = (0, 0)
        print("Board origin non défini, utilisation de (0,0)")
    
    # Étape 5 : Détection des pions bleus (optimisée)
    pieces, mask_blue = detect_blue_pieces(image_orig, board_origin, square_size=55)
    print("Pions bleus détectés (centres) :", [p[0] for p in pieces])
    # Tracer les détections sur une copie de l'image originale pour affichage
    image_detection = image_orig.copy()
    for center, _ in pieces:
        cv2.circle(image_detection, center, 5, (0, 255, 0), 2)
    mosaic2 = create_mosaic([image_detection, cv2.cvtColor(mask_blue, cv2.COLOR_GRAY2BGR)],
                              labels=["Détections", "Masque Bleu"], cols=2)
    cv2.namedWindow("Mosaïque - Pions Bleus", cv2.WINDOW_NORMAL)
    cv2.imshow("Mosaïque - Pions Bleus", mosaic2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Étape 6 : Affectation des pions à chaque case avec tolérance ±5 pixels
    tolerance = 5
    grid_results = assign_pieces_to_grid(pieces, board_origin, square_size=55, tolerance=tolerance)
    print("Résultats assignés :", grid_results)
    
    # Sauvegarde des résultats dans un fichier texte
    output_txt = "/home/zakaria/Workspace/ChessProject/data/detection_results.txt"
    save_detection_results_grid(grid_results, output_txt)
    print("Résultats sauvegardés dans :", output_txt)
    
    # Affichage final de la mosaïque complète (regroupant repères et détections)
    mosaic_final = create_mosaic([mosaic1, mosaic2],
                                 labels=["Repères", "Pions Bleus"], cols=1)
    cv2.namedWindow("Mosaïque Finale", cv2.WINDOW_NORMAL)
    cv2.imshow("Mosaïque Finale", mosaic_final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
