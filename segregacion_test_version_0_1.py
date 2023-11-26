import cv2

# Cargamos el clasificador pre-entrenado de OpenCV para la detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Iniciamos la captura de video
cap = cv2.VideoCapture(0)

while True:
    # Leemos cada frame de la cámara
    ret, img = cap.read()

    # Convertimos la imagen a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detectamos los rostros
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Dibujamos un rectángulo alrededor de cada rostro detectado
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    # Mostramos la imagen
    cv2.imshow('img',img)

    # Si se presiona la tecla 'q', salimos del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberamos la cámara y destruimos todas las ventanas
cap.release()
cv2.destroyAllWindows()
