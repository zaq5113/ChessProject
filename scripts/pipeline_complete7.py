#!/usr/bin/env python3
import cv2
import numpy as np
import os
import time

#############################
# Fonctions du Pipeline     #
#############################

def preprocess_image(image, blur_kernel=(5,5)):
    """Applique un flou gaussien sur l'image en BGR et retourne l'image floutée."""
    image_blurred = cv2.GaussianBlur(image, blur_kernel, 0)
    return image_blurred

def detect_red_crosses(image):
    """Détecte les zones rouges (repères) dans l'image prétraitée et renvoie les centres."""
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
    """Ordonne 4 points dans l'ordre : supérieur gauche, supérieur droit, inférieur droit, inférieur gauche."""
    pts = np.array(points, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered = np.zeros((4,2), dtype="float32")
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered

def compute_homography(pixelPoints, realWorldPoints):
    """Calcule la matrice d'homographie H reliant pixelPoints aux coordonnées réelles (pour un plateau 550x550)."""
    pixelPoints = np.array(pixelPoints, dtype="float32")
    realWorldPoints = np.array(realWorldPoints, dtype="float32")
    H, _ = cv2.findHomography(pixelPoints, realWorldPoints)
    return H

def pixel_to_real(H, point):
    """Convertit un point (u,v) en coordonnées réelles à l'aide de H."""
    pt = np.array([[point[0]], [point[1]], [1]], dtype="float32")
    real_pt = np.dot(H, pt)
    real_pt /= real_pt[2]
    return (real_pt[0][0], real_pt[1][0])

def detect_blue_pieces(image, board_origin, square_size=55):
    """
    Détecte les pions bleus dans l'image originale.
    Si l'image est très grande, elle est redimensionnée pour accélérer le traitement.
    Les coordonnées sont ramenées à l'échelle d'origine.
    Renvoie une liste de tuples ((cx, cy), "") pour chaque pièce et le masque bleu.
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

def assign_pieces_to_grid(pieces, board_origin, square_size=55, tolerance=30):
    """
    Parcourt chaque case du plateau (10x10) et, pour chaque case,
    compare le centre attendu (board_origin + (col * square_size + square_size/2, row * square_size + square_size/2))
    aux centres détectés dans 'pieces'. Si la distance est inférieure à 'tolerance',
    la case est marquée par son nom avec la mention d'une pièce (ex: "BA1").
    Renvoie un dictionnaire {case: valeur} pour toutes les cases.
    """
    grid_results = {}
    for i in range(10):   # lignes A à J
        for j in range(10):  # colonnes 1 à 10
            square_name = chr(ord('A') + i) + str(j+1)
            expected_center = (board_origin[0] + j*square_size + square_size/2,
                               board_origin[1] + i*square_size + square_size/2)
            # Par défaut, sans pièce
            grid_results[square_name] = None
            for (cx, cy), _ in pieces:
                dx = cx - expected_center[0]
                dy = cy - expected_center[1]
                if np.sqrt(dx*dx + dy*dy) <= tolerance:
                    grid_results[square_name] = square_name
                    break
    return grid_results

def save_detected_cells(grid_results, output_file):
    """
    Enregistre dans le fichier output_file uniquement les cases contenant une pièce.
    Les cases sont séparées par des virgules.
    Par exemple : "E4,F8"
    """
    cells_with_pieces = [square for square, value in grid_results.items() if value is not None]
    with open(output_file, 'w') as f:
        f.write(",".join(cells_with_pieces))

def create_mosaic(images, labels=None, cols=2):
    """
    Regroupe plusieurs images dans une seule fenêtre (mosaïque).
    """
    if not images:
        return None
    h, w = images[0].shape[:2]
    resized = [cv2.resize(img, (w, h)) for img in images]
    if labels is not None:
        for i, label in enumerate(labels):
            cv2.putText(resized[i], label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    rows = (len(resized) + cols - 1) // cols
    mosaic_rows = []
    for r in range(rows):
        row_imgs = []
        for c in range(cols):
            idx = r * cols + c
            if idx < len(resized):
                row_imgs.append(resized[idx])
            else:
                row_imgs.append(np.zeros_like(resized[0]))
        mosaic_rows.append(cv2.hconcat(row_imgs))
    mosaic = cv2.vconcat(mosaic_rows)
    return mosaic

def draw_grid_labels(image, board_origin, square_size=55):
    """
    Dessine les labels de chaque case (A1 à J10) sur l'image, positionnés au centre attendu de la case.
    """
    for i in range(10):
        for j in range(10):
            square_name = chr(ord('A') + i) + str(j+1)
            expected_center = (int(board_origin[0] + j*square_size + square_size/2),
                               int(board_origin[1] + i*square_size + square_size/2))
            cv2.putText(image, square_name, (expected_center[0]-15, expected_center[1]+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    return image

#############################
#  Pipeline Main            #
#############################

def main():
    t0 = time.time()
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

    mosaic1 = create_mosaic([image_orig, image_preproc, cv2.cvtColor(maskRed, cv2.COLOR_GRAY2BGR)],
                              labels=["Originale", "Prétraitée", "Masque Rouge"], cols=3)
    cv2.namedWindow("Mosaïque - Repères", cv2.WINDOW_NORMAL)
    cv2.imshow("Mosaïque - Repères", mosaic1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Étape 3 : Calcul de l'homographie (pour information)
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
    # Nous définissons board_origin comme le coin supérieur gauche interne du plateau
    # = ordered_pts[0] + (55, 55). Ceci correspond au coin de la case A1.
    if ordered_pts is not None:
        board_origin = (int(ordered_pts[0][0] + 55), int(ordered_pts[0][1] + 55))
        print("Board origin :", board_origin)
    else:
        board_origin = (0,0)
        print("Board origin non défini, utilisation de (0,0)")

    # Étape 5 : Détection des pions bleus (optimisée)
    pieces, mask_blue = detect_blue_pieces(image_orig, board_origin, square_size=55)
    print("Pions bleus détectés (centres) :", [p[0] for p in pieces])
    image_detection = image_orig.copy()
    for center, _ in pieces:
        cv2.circle(image_detection, center, 5, (0, 255, 0), 2)
    mosaic2 = create_mosaic([image_detection, cv2.cvtColor(mask_blue, cv2.COLOR_GRAY2BGR)],
                              labels=["Détections", "Masque Bleu"], cols=2)
    cv2.namedWindow("Mosaïque - Pions Bleus", cv2.WINDOW_NORMAL)
    cv2.imshow("Mosaïque - Pions Bleus", mosaic2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Étape 6 : Affectation des pions aux cases (seulement les cases où un pion est proche du centre)
    grid_results = assign_pieces_to_grid(pieces, board_origin, square_size=55, tolerance=30)
    print("Résultats assignés :", grid_results)
    
    # On peut dessiner les labels sur l'image pour vérification
    image_labeled = image_orig.copy()
    image_labeled = draw_grid_labels(image_labeled, board_origin, square_size=55)
    # Marquer les cases contenant un pion
    for square, result in grid_results.items():
        if result is not None:
            row = ord(square[0]) - ord('A')
            col = int(square[1:]) - 1
            expected_center = (int(board_origin[0] + col*55 + 55/2),
                               int(board_origin[1] + row*55 + 55/2))
            cv2.putText(image_labeled, "B", (expected_center[0]-20, expected_center[1]+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.namedWindow("Plateau Noté", cv2.WINDOW_NORMAL)
    cv2.imshow("Plateau Noté", image_labeled)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Étape 7 : Sauvegarde dans un fichier texte uniquement les cases contenant des pions
    output_txt = "/home/zakaria/Workspace/ChessProject/data/detection_results.txt"
    # On ne garde que les cases où grid_results[square] n'est pas None
    cells_with_pieces = [square for square, val in grid_results.items() if val is not None]
    with open(output_txt, 'w') as f:
        f.write(",".join(cells_with_pieces))
    print("Résultats sauvegardés dans :", output_txt)
    
    mosaic_final = create_mosaic([mosaic1, mosaic2],
                                 labels=["Repères", "Pions Bleus"], cols=1)
    cv2.namedWindow("Mosaïque Finale", cv2.WINDOW_NORMAL)
    cv2.imshow("Mosaïque Finale", mosaic_final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    t1 = time.time()
    print("Temps total de traitement : {:.2f} secondes".format(t1 - t0))

if __name__ == '__main__':
    main()
