import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.layers import Conv2D,Dense,MaxPool2D,Flatten,BatchNormalization,GlobalAvgPool2D
if __name__ == '__main__':

    (x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
    print(x_train.shape)

def display(examples,labels):

    plt.figure(figsize=(10,10))

    for i in range(10):
        a = np.random.randint(0,examples.shape[0]-1)
        img = examples[a]
        label = labels[a]
        plt.subplot(5,5,i+1)
        plt.title(str(label))
        plt.tight_layout()
        plt.imshow(img,cmap='gray')

display(x_train, y_train)

model = keras.Sequential([
    keras.layers.Input(shape=(28,28,1)),

    Conv2D(32,(3,3),activation='relu'),
    MaxPool2D((2,2)),
    BatchNormalization(),

    Conv2D(32,(3,3),activation='relu'),
    MaxPool2D((2,2)),
    BatchNormalization(),

    GlobalAvgPool2D(),
    Dense(64,activation='relu'),
    Dense(10,activation='softmax')
])

#print(model.summary())
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
x_train = np.expand_dims(x_train,axis=-1)
x_test = np.expand_dims(x_test,axis=-1)


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics='accuracy')

model.fit(x_train,y_train,epochs=2,validation_split=0.2)

model.evaluate(x_test,y_test)


df = tf.keras.preprocessing.image_dataset_from_directory("PlantVillage")
print(df.class_names)