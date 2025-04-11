#!/usr/bin/env python3
import numpy as np

def assign_pieces_to_grid(pieces, board_origin, square_size=55, tolerance=30):
    """
    Pour chaque case d'un plateau 10x10, compare le centre attendu aux positions détectées.
    Si la distance est inférieure à 'tolerance', on considère la case occupée.
    Renvoie un dictionnaire où la clé est le nom de la case (ex: "A1") et la valeur la marque (ex: "A1" ou None).
    """
    grid_results = {}
    for i in range(10):   # lignes (A à J)
        for j in range(10):  # colonnes (1 à 10)
            square_name = chr(ord('A') + i) + str(j+1)
            expected_center = (board_origin[0] + j*square_size + square_size/2,
                               board_origin[1] + i*square_size + square_size/2)
            grid_results[square_name] = None
            for (cx, cy), _ in pieces:
                dx = cx - expected_center[0]
                dy = cy - expected_center[1]
                if np.sqrt(dx*dx + dy*dy) <= tolerance:
                    grid_results[square_name] = square_name
                    break
    return grid_results

def save_detection_results_grid(grid_results, output_file):
    """
    Enregistre dans un fichier texte les cases contenant une pièce.
    Les cases sont séparées par une virgule.
    """
    cells_with_pieces = [square for square, val in grid_results.items() if val is not None]
    with open(output_file, 'w') as f:
        f.write(",".join(cells_with_pieces))

def draw_grid_labels(image, board_origin, square_size=55):
    """
    Dessine sur l'image le nom de chaque case, en se basant sur board_origin et square_size.
    """
    import cv2
    for i in range(10):
        for j in range(10):
            square_name = chr(ord('A') + i) + str(j+1)
            expected_center = (int(board_origin[0] + j*square_size + square_size/2),
                               int(board_origin[1] + i*square_size + square_size/2))
            cv2.putText(image, square_name, (expected_center[0]-15, expected_center[1]+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    return image
