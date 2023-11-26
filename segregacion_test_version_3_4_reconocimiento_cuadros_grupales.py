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

def check_distance(faces):
    groups = []
    for i in range(len(faces)):
        for j in range(i+1, len(faces)):
            dist = np.sqrt((faces[i][0]-faces[j][0])**2 + (faces[i][1]-faces[j][1])**2)
            if dist < 300:  # Set your distance threshold here
                merged = False
                for group in groups:
                    if any(np.array_equal(face, group_face) for face in [faces[i], faces[j]] for group_face in group):
                        group.extend([face for face in [faces[i], faces[j]] if face not in group])
                        merged = True
                        break
                if not merged:
                    groups.append([faces[i], faces[j]])
    return groups

offset = 20  # Set your offset here

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

        color = (0,0,255)
        for group in groups:
            if (x, y, w, h) in map(tuple, group):  # Convert lists to tuples before checking
                color = (0,255,0)
                break

        cv2.rectangle(frame, (x,y),(x+w,y+h),color,2)

    cv2.imshow('frame',frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
