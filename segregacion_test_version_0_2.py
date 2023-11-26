import cv2
import os


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

#niños de clase (lista de niños + carpetas para guardar sus fotos)
person_names = ["Juan", "Pepita"]
for name in person_names:
    os.makedirs(name, exist_ok=True)


while True:
    #leer frame
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #cascade de rostro
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    #rectangulo
    for i, (x,y,w,h) in enumerate(faces):
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        cv2.imwrite(f"{person_names[i % len(person_names)]}/{i}.jpg", roi_color)
        cv2.putText(img, person_names[i % len(person_names)], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # imagen por recuadro
    cv2.imshow('img',img)

    #exit
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

cap.release()
cv2.destroyAllWindows()
