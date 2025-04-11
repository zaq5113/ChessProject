#!/usr/bin/env python3
import cv2
import os
import time
from vision.preprocessing import preprocess_image
from vision.detection import detect_red_crosses
from vision.ordering import order_points
from vision.homography import compute_homography, pixel_to_real
from vision.blue_detection import detect_blue_pieces
from vision.grid import assign_pieces_to_grid, save_detection_results_grid, draw_grid_labels
from vision.mosaic import create_mosaic

def main():
    t0 = time.time()
    # Chemin vers l'image (adapté à ton projet)
    imgPath = "./data/damier.png"
    if not os.path.isfile(imgPath):
        print("Erreur : fichier inexistant à", imgPath)
        return
    image_orig = cv2.imread(imgPath)
    if image_orig is None:
        print("Erreur lors du chargement de l'image.")
        return
    print("Dimensions image originale :", image_orig.shape)

    # 1. Prétraitement
    image_preproc = preprocess_image(image_orig)

    # 2. Détection des repères rouges
    points, maskRed = detect_red_crosses(image_preproc)
    print("Points rouges détectés :", points)
    if len(points) >= 4:
        ordered_pts = order_points(points[:4])
        print("Points rouges ordonnés :", ordered_pts)
        for pt in ordered_pts:
            cv2.circle(image_preproc, (int(pt[0]), int(pt[1])), 5, (0,255,0), -1)
    else:
        print("Moins de 4 points rouges détectés. Homographie impossible.")
        ordered_pts = None

    # Création d'une mosaïque pour visualiser les étapes
    mosaic1 = create_mosaic([image_orig, image_preproc, cv2.cvtColor(maskRed, cv2.COLOR_GRAY2BGR)],
                              labels=["Originale", "Prétraitée", "Masque Rouge"], cols=3)
    cv2.namedWindow("Mosaïque - Repères", cv2.WINDOW_NORMAL)
    cv2.imshow("Mosaïque - Repères", mosaic1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 3. Calcul de l'homographie
    real_world_points = [(0, 0), (550, 0), (550, 550), (0, 550)]
    if ordered_pts is not None:
        H = compute_homography(ordered_pts, real_world_points)
        print("Matrice d'homographie H :", H)
    else:
        H = None

    # 4. Définition de board_origin (exemple simple)
    if ordered_pts is not None:
        board_origin = (int(ordered_pts[0][0] + 55), int(ordered_pts[0][1] + 55))
    else:
        board_origin = (0, 0)
    print("Board origin :", board_origin)

    # 5. Détection des pions bleus
    pieces, mask_blue = detect_blue_pieces(image_orig, board_origin, square_size=55)
    print("Pions bleus détectés (centres) :", [p[0] for p in pieces])
    image_detection = image_orig.copy()
    for center, _ in pieces:
        cv2.circle(image_detection, center, 5, (0,255,0), 2)
    mosaic2 = create_mosaic([image_detection, cv2.cvtColor(mask_blue, cv2.COLOR_GRAY2BGR)],
                              labels=["Détections", "Masque Bleu"], cols=2)
    cv2.namedWindow("Mosaïque - Pions Bleus", cv2.WINDOW_NORMAL)
    cv2.imshow("Mosaïque - Pions Bleus", mosaic2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 6. Affectation des pions aux cases
    grid_results = assign_pieces_to_grid(pieces, board_origin, square_size=55, tolerance=30)
    print("Résultats assignés :", grid_results)
    
    # Sauvegarde des résultats dans un fichier texte
    output_txt = "./data/detection_results.txt"
    save_detection_results_grid(grid_results, output_txt)
    print("Résultats sauvegardés dans :", output_txt)
    
    # 7. Dessin des labels sur l'image
    image_labeled = image_orig.copy()
    image_labeled = draw_grid_labels(image_labeled, board_origin, square_size=55)
    cv2.namedWindow("Plateau Noté", cv2.WINDOW_NORMAL)
    cv2.imshow("Plateau Noté", image_labeled)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    t1 = time.time()
    print("Temps total de traitement : {:.2f} secondes".format(t1-t0))

if __name__ == '__main__':
    main()
