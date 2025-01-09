import cv2
import numpy as np
from Frame import *

##############################################################################################
#       SETTINGS
##############################################################################################
ZOOM = 0.25
SHOW_BOX = True
SCALE_FACTOR = 1.22
MIN_NEIGHBORS = 8
MINSIZE = (60, 60)
##############################################################################################

# Chemins vers vos fichiers de détection
FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH = "haarcascade_eye.xml"

faceCascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
eyeCascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erreur : impossible d'accéder à la webcam.")
    exit(1)

box = BoundingBox(-1, -1, -1, -1)

print("Appuyez sur 'ESC' pour quitter.")
print("Appuyez sur '1' pour (dés)activer la box de visage.")
print("Appuyez sur '2' pour réduire le zoom.")
print("Appuyez sur '3' pour augmenter le zoom.")

while True:
    ret, img = cap.read()
    if not ret:
        print("Impossible de lire l'image depuis la webcam.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    boxes = faceCascade.detectMultiScale(
        gray,
        scaleFactor=SCALE_FACTOR,
        minNeighbors=MIN_NEIGHBORS,
        minSize=MINSIZE
    )
    boxes = np.array(boxes)

    if boxes.size > 0:
        # On récupère la bounding box la plus grande
        boxLrg = largestBox(boxes)

        if box.dim[0] == -1:
            box = boxLrg
        else:
            box.lerpShape(boxLrg)

        # --- DOLLY ZOOM ---
        frame = Frame(img, box)
        frame.boxIsVisible = SHOW_BOX
        frame.setZoom(ZOOM)
        frame.filter()
        box = frame.box  # mettre à jour la box globale

        # --- DÉTECTION DES YEUX ---
        # On travaille sur l'image "transformée" par le Dolly Zoom (frame.img).
        gray_dz = cv2.cvtColor(frame.img, cv2.COLOR_BGR2GRAY)
        
        # On détecte les yeux dans l'image Dolly Zoomée.
        # Ajustez éventuellement scaleFactor (1.1, 1.2, etc.) et minNeighbors selon la précision désirée.
        eyes = eyeCascade.detectMultiScale(gray_dz, 1.1, 4)

        # Pour chaque œil, tracer un cercle rouge au centre.
        for (ex, ey, ew, eh) in eyes:
            center = (ex + ew // 2, ey + eh // 2)
            # (0, 0, 255) => rouge, -1 => cercle rempli
            cv2.circle(frame.img, center, 5, (0, 0, 255), -1)

        # On affiche le résultat final (Dolly Zoom + cercles rouges sur les yeux).
        frame.show()

    else:
        # AUCUN VISAGE => réinitialise la box et affiche l'image brute
        box = BoundingBox(-1, -1, -1, -1)
        cv2.imshow("Dolly Zoom", img)

    # Gestion des touches
    k = cv2.waitKey(30)
    if k == 27:  # Echap pour quitter
        break
    elif k == 49:  # Touche '1'
        SHOW_BOX = not SHOW_BOX
        print("SHOW_BOX =", SHOW_BOX)
    elif k == 50:  # Touche '2'
        ZOOM = max(ZOOM - 0.05, 0.01)
        print("ZOOM =", ZOOM)
    elif k == 51:  # Touche '3'
        ZOOM = min(ZOOM + 0.05, 0.99)
        print("ZOOM =", ZOOM)

cap.release()
cv2.destroyAllWindows()
