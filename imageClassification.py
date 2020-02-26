model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#model defination and layer definition
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])
#Optimizer definiation and matics definition
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#code for training the model
model.fit(train_images, train_labels, epochs=10)
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images = train_images/255.0
test_images = test_images/255.0
model = keras.Sequential([
                       keras.layers.Flatten(input_shape=(28,28)),
                       keras.layers.Dense(units=200,activation='relu'),
                       keras.layers.Dense(units=10,activation='softmax')  
])

model.compile(optimizer="adam",loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)
test_l, test_ac = model.evaluate(test_images,test_labels, verbose=2)
model.summary()


# the below lines of code are used for image classification.
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])



predictions = probability_model.predict(test_images)
