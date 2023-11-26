import cv2 #libreria de OpenCv
import os #nos da herramientas para crear carpetas, editar nombres de archivos, etc, todo ello interaccionando con el OS
import numpy as np #NumPy es una librería de Python especializada en el cálculo 
                   #numérico y el análisis de datos, especialmente para un gran volumen de datos

dataPath = 'C:/Users/fill_/OneDrive/Escritorio/lleidahack/test_segregacion/Data2'
imagePaths = os.listdir(dataPath)
print('imagePaths=',imagePaths)


face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('modeloLBPHFace_test_Felipe_Andrea_Paula.xml')

cap = cv2.VideoCapture('C:/Users/fill_/OneDrive/Escritorio/lleidahack/test_segregacion/video_analisis/paula_andrea_felipe2.mp4')

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

def check_distance(faces):
    groups = []
    for i in range(len(faces)):
        for j in range(i+1, len(faces)):
            dist = np.sqrt((faces[i][0]-faces[j][0])**2 + (faces[i][1]-faces[j][1])**2)
            if dist < 250:  # DISTANCIA PARA CONSIDERAR GRUPO ***IMPORTANTE***
                merged = False
                for group in groups:
                    if any(np.array_equal(face, group_face) for face in [faces[i], faces[j]] for group_face in group):
                        group.extend([tuple(face) for face in [faces[i], faces[j]] if tuple(face) not in map(tuple, group)])
                        merged = True
                        break
                if not merged:
                    groups.append([faces[i], faces[j]])
    return groups

offset = 20  #offset bordes de las cajas de grupo, para evitar que esten a ras de las cajas de rostro

# inicializar contadores a cero
counters_in_group = {name: 0 for name in imagePaths}
counters_out_group = {name: 0 for name in imagePaths}

while True:
    ret,frame = cap.read()
    if ret == False: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    faces = faceClassif.detectMultiScale(gray,1.3,5)

    groups = check_distance(faces)

    for group in groups:
        x_min = min([face[0] for face in group]) - offset
        y_min = min([face[1] for face in group]) - offset
        x_max = max([face[0]+face[2] for face in group]) + offset
        y_max = max([face[1]+face[3] for face in group]) + offset
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255,0,0), 2)
        cv2.putText(frame,'Grupo_1',(x_min,y_min-10),2,0.8,(255,0,0),1,cv2.LINE_AA)

    for (x,y,w,h) in faces:
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)

        cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)

        color = (0,0,255) #azul
        in_group = False
        for group in groups:
            if (x, y, w, h) in map(tuple, group): 
                color = (0,255,0) #verde hola
                in_group = True
                break

        cv2.rectangle(frame, (x,y),(x+w,y+h),color,2)

        
        if in_group:
            counters_in_group[imagePaths[result[0]]] += 1 #contador_ingroup
        else:
            counters_out_group[imagePaths[result[0]]] += 1 #contador_outgroup

    # nombre_contador_pantalla
    for i, name in enumerate(imagePaths):
        cv2.putText(frame, f'{name}: In group = {counters_in_group[name]}', (10, (i*2+1)*20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(frame, f'Out group = {counters_out_group[name]}', (10, (i*2+2)*20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    cv2.imshow('frame',frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
