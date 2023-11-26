import cv2
import os
import tensorflow as tf

# Cargamos el modelo pre-entrenado
model = tf.keras.models.load_model('my_model.h5')

# Iniciamos la captura de video
cap = cv2.VideoCapture(0)

# Definimos los nombres de las personas que queremos detectar
person_names = ["Persona1", "Persona2"]

# Creamos las carpetas para guardar las imágenes de los rostros
for name in person_names:
    os.makedirs(name, exist_ok=True)

while True:
    # Leemos cada frame de la cámara
    ret, img = cap.read()

    # Convertimos la imagen a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detectamos los rostros
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Dibujamos un rectángulo alrededor de cada rostro detectado y guardamos la imagen
    for i, (x,y,w,h) in enumerate(faces):
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        cv2.imwrite(f"{person_names[i % len(person_names)]}/{i}.jpg", roi_color)

        # Predecimos la persona en la imagen
        prediction = model.predict(roi_color)
        person_name = person_names[prediction.argmax()]

        # Añadimos el nombre de la persona encima del rectángulo
        cv2.putText(img, person_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Mostramos la imagen
    cv2.imshow('img',img)

    # Si se presiona la tecla 'q', salimos del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberamos la cámara y destruimos todas las ventanas
cap.release()
cv2.destroyAllWindows()
