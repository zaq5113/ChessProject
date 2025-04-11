#!/usr/bin/env python3
import cv2
from vision.preprocessing import preprocess_image

def main():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Erreur : impossible de capturer une image.")
        return
    image_preproc = preprocess_image(frame)
    cv2.imshow("Image prétraitée", image_preproc)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
