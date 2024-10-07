#!/usr/bin/python3

import cv2
import dlib
import numpy as np
from picamera2 import Picamera2

# Chemins vers les fichiers de mod�le
shape_predictor_path = '/home/alexm/shape_predictor_68_face_landmarks.dat'
face_rec_model_path = '/home/alexm/dlib_face_recognition_resnet_model_v1.dat'

# Liste des images et des noms
face_images = {
    "Alex": ["/home/alexm/face_images/face_1.jpg"],
    "Lidia": [
        "/home/alexm/face_images/lidia1.jpeg",
        "/home/alexm/face_images/lidia2.jpeg",
        "/home/alexm/face_images/lidia3.jpeg"
    ]
}

# Chargement des mod�les d'apprentissage
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(shape_predictor_path)
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

# Fonction pour charger et encoder les visages
def load_face_encodings(image_paths):
    encodings = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = detector(rgb_image)
        for face in faces:
            shape = sp(rgb_image, face)
            face_descriptor = face_rec_model.compute_face_descriptor(rgb_image, shape)
            encodings.append(np.array(face_descriptor))
    return encodings

# Charger les encodages pour chaque personne
known_face_encodings = []
known_face_names = []

for name, image_paths in face_images.items():
    encodings = load_face_encodings(image_paths)
    known_face_encodings.extend(encodings)
    known_face_names.extend([name] * len(encodings))

# Initialisation de la cam�ra
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

while True:
    # Capture d'image depuis la cam�ra
    im = picam2.capture_array()
    rgb_image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # D�tection des visages
    faces = detector(rgb_image)

    for face in faces:
        shape = sp(rgb_image, face)
        face_descriptor = face_rec_model.compute_face_descriptor(rgb_image, shape)
        face_encoding = np.array(face_descriptor)

        # Comparaison avec les visages connus
        matches = [np.linalg.norm(face_encoding - known_face) < 0.6 for known_face in known_face_encodings]
        name = "Unknown"
        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]

        # Dessiner le rectangle autour du visage et ajouter le nom
        (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(im, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Afficher l'image avec les d�tections
    cv2.imshow("Camera", im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()