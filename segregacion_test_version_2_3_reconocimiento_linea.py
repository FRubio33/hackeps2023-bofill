import cv2
import os
import numpy as np

dataPath = 'C:/Users/fill_/OneDrive/Escritorio/lleidahack/test_segregacion/Data2'
imagePaths = os.listdir(dataPath)
print('imagePaths=',imagePaths)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('modeloLBPHFace_test_Felipe_Andrea_Paula.xml')

cap = cv2.VideoCapture('C:/Users/fill_/OneDrive/Escritorio/lleidahack/test_segregacion/video_analisis/paula_andrea_felipe2.mp4')

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

# Definimos la distancia máxima para dibujar la línea
max_distance = 500  # puedes cambiar este valor a tu gusto

while True:
    ret,frame = cap.read()
    if ret == False: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    faces = faceClassif.detectMultiScale(gray,1.3,5)

    # Creamos una lista para almacenar los puntos centrales de los rostros
    centers = []
    
    for (x,y,w,h) in faces:
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)

        # Calculamos el punto central del rostro y lo añadimos a la lista
        center = (x + w//2, y + h//2)
        centers.append(center)

        cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)

        if result[0] < 90:
            cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
        else:
            cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)

    # Dibujamos una línea entre todos los rostros detectados si la distancia es menor que max_distance
    # y cambiamos el color del cuadro a verde, de lo contrario, el color del cuadro será rojo
    for i in range(len(centers)):
        if i < len(centers) - 1:
            distance = np.linalg.norm(np.array(centers[i]) - np.array(centers[i+1]))
            if distance < max_distance:
                cv2.line(frame, centers[i], centers[i+1], (0,255,0), 2)
                color = (0,255,0)  # verde
            else:
                color = (0,0,255)  # rojo

            # Dibujamos la longitud de la línea en el frame
            midpoint = ((centers[i][0]+centers[i+1][0])//2, (centers[i][1]+centers[i+1][1])//2)
            cv2.putText(frame, f'{distance:.2f}', midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        else:
            color = (0,0,255)  # rojo

        # Dibujamos el cuadro de seguimiento con el color correspondiente
        x, y, w, h = faces[i]
        cv2.rectangle(frame, (x,y),(x+w,y+h), color, 2)

    cv2.imshow('frame',frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
