import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Cargamos el conjunto de datos de rostros (debes proporcionar tus propios datos)
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalizamos los valores de los píxeles a [0, 1]
train_images, test_images = train_images / 255.0, test_images / 255.0

# Definimos el modelo
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

# Compilamos el modelo
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Entrenamos el modelo
model.fit(train_images, train_labels, epochs=10, 
          validation_data=(test_images, test_labels))

# Guardamos el modelo
model.save('my_model.h5')
