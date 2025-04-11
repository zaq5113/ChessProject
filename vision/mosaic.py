#!/usr/bin/env python3
import cv2
import numpy as np

def create_mosaic(images, labels=None, cols=2):
    """
    Regroupe une liste d'images dans une mosaïque.
    labels : (optionnel) liste de textes à superposer sur chaque image.
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
