# XtendPLEX_test
Ce projet applique un effet Dolly Zoom en temps réel à partir d'une webcam. Lorsqu'un visage est détecté, un cadre rouge l'entoure et un cercle se dessine sur chaque œil. Sans visage, la vue revient à la taille normale. Les touches 1, 2, 3 permettent de gérer l'affichage du cadre et le facteur de zoom.
# Zoom + Détection de Visage et d'Yeux

Ce projet illustre l'utilisation de la webcam pour appliquer un effet de **Dolly Zoom** lorsqu'un visage est détecté, et affiche un **cercle rouge** sur chaque œil.  

## Contenu

- **Frame.py** : Contient la logique de Dolly Zoom (classe `Frame`) et la gestion de la bounding box.  
- **main.py** : Fichier principal pour exécuter l'application (capture webcam, détection visage, détection yeux).  
- **haarcascade_frontalface_default.xml** : Classifieur HaarCascade pour détecter les visages.  
- **haarcascade_eye.xml** : Classifieur HaarCascade pour détecter les yeux.

## Dépendances

- Python 3.6+
- OpenCV (cv2)
- NumPy

Installez-les avec :

```bash
pip install opencv-python numpy
Dolly-Zoom/
├─ README.md
├─ main.py
├─ Frame.py
├─ haarcascade_frontalface_default.xml
└─ haarcascade_eye.xml

pip install opencv-python numpy
python -m venv venv
python main.py

