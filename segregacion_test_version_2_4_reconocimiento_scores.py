import cv2
import os
from collections import deque, Counter

dataPath = 'C:/Users/fill_/OneDrive/Escritorio/lleidahack/test_segregacion/Data2'
imagePaths = os.listdir(dataPath)
print('imagePaths=',imagePaths)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('modeloLBPHFace_test_Felipe_Andrea_Paula.xml')

cap = cv2.VideoCapture('C:/Users/fill_/OneDrive/Escritorio/lleidahack/test_segregacion/video_analisis/paula_andrea_felipe2.mp4')

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

# Creamos un diccionario para almacenar las últimas n predicciones para cada rostro
face_memory = {}

# Creamos un conjunto para almacenar los rostros que ya han sido asignados
assigned_faces = set()

while True:
    ret,frame = cap.read()
    if ret == False: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    faces = faceClassif.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)

        # Si el rostro ya está en la memoria, añadimos la nueva predicción
        # Si no, inicializamos la memoria para ese rostro
        if result[0] in face_memory:
            face_memory[result[0]].append(result[0])
            if len(face_memory[result[0]]) > 100:  # ajusta este valor a tus necesidades
                face_memory[result[0]].popleft()
        else:
            face_memory[result[0]] = deque([result[0]], maxlen=100)  # ajusta este valor a tus necesidades

        # Obtenemos la predicción más común en la memoria para ese rostro
        most_common_prediction = Counter(face_memory[result[0]]).most_common(1)[0][0]

        cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)

        # Si el rostro más común ya ha sido asignado, lo dibujamos con un cuadro verde
        # Si no, lo añadimos al conjunto de rostros asignados
        if most_common_prediction in assigned_faces:
            cv2.putText(frame,'{}'.format(imagePaths[most_common_prediction]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        else:
            assigned_faces.add(most_common_prediction)
            cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)

    cv2.imshow('frame',frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
