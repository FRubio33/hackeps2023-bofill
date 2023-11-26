import numpy as np
import cv2
import os

import cv2
import os

dataPath = 'C:/Users/fill_/OneDrive/Escritorio/lleidahack/test_segregacion/Data2' #Cambia a la ruta donde hayas almacenado Data
imagePaths = os.listdir(dataPath)
print('imagePaths=',imagePaths)

#face_recognizer = cv2.face.EigenFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Leyendo el modelo
#face_recognizer.read('modeloEigenFace.xml')
#face_recognizer.read('modeloFisherFace.xml')
face_recognizer.read('modeloLBPHFace_test_Felipe_Andrea_Paula.xml')

#cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap = cv2.VideoCapture('C:/Users/fill_/OneDrive/Escritorio/lleidahack/test_segregacion/video_analisis/paula_andrea_felipe2.mp4')

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
# Crear una lista para almacenar las coordenadas de las caras
face_locations = []

for (x,y,w,h) in faces:
    # Aquí va tu código existente para procesar cada rostro

    # Añadir las coordenadas del rostro a la lista
    face_locations.append((x, y, w, h))

# Convertir la lista a un array de numpy para facilitar los cálculos
face_locations = np.array(face_locations)

# Calcular todas las distancias entre pares de rostros
distances = np.sqrt(np.sum((face_locations[:, None] - face_locations)**2, axis=-1))

# Definir la distancia máxima para considerar que dos rostros están en el mismo grupo
max_distance = 100  # Ajusta este valor a lo que necesites

# Crear una matriz que indica qué rostros están en el mismo grupo
same_group = distances < max_distance

# Para cada rostro, encontrar los otros rostros que están en el mismo grupo
for i in range(len(faces)):
    group = np.where(same_group[i])[0]
    if len(group) > 1:
        # Encontrar el cuadro delimitador que contiene a todos los rostros en el grupo
        x_min = np.min(face_locations[group, 0])
        y_min = np.min(face_locations[group, 1])
        x_max = np.max(face_locations[group, 0] + face_locations[group, 2])
        y_max = np.max(face_locations[group, 1] + face_locations[group, 3])

        # Dibujar el cuadro delimitador del grupo
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.putText(frame, 'Grupo {}'.format(i+1), (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
