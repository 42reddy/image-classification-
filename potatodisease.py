import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

data_generator = ImageDataGenerator(rescale=1.0/255.0,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

data = data_generator.flow_from_directory('PlantVillage',target_size=(150,150),batch_size=32)
N =2152
images,labels = next(data)

def plotimg(images, labels, n):

    for i in range(n):
        plt.subplot(3,3,i+1)
        plt.imshow(images[i])
    plt.show()

plotimg(images,labels,n=10)

# neural net

model = keras.Sequential([
    keras.layers.Input(shape=(150,150,3)),
    keras.layers.Conv2D(32,(3,3),activation='relu'),
    keras.layers.MaxPool2D((3,3)),
    keras.layers.Conv2D(32,(3,3),activation='relu'),
    keras.layers.MaxPool2D((3,3)),
    keras.layers.GlobalAvgPool2D(),
    keras.layers.Dense(64,activation='relu'),
    keras.layers.Dense(3,activation='softmax')
])

print(model.summary())
# compile
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='accuracy')

images = []
labels = []

for i in range(int(np.ceil(2152/32))):
    batch_images, batch_labels = next(data)
    images.append(batch_images)
    labels.append(batch_labels)

images = np.concatenate(images,axis=0)
labels = np.concatenate(labels,axis=0)
print(labels.shape)
# fit the model

model.fit(images[:1800],labels[:1800],epochs=5,validation_split=0.2)

model.evaluate(images[1801:],labels[1801:])
