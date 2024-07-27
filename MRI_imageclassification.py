import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_generator = ImageDataGenerator(rescale=1.0/255.0,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
data_train = data_generator.flow_from_directory('archive-2/Train',target_size=(250,250),class_mode='categorical',batch_size=10)
N = 4737

data_test = data_generator.flow_from_directory('archive-2/Val',target_size=(250,250),class_mode='categorical',batch_size=10)
N1 = 512

img_train,label_train = next(data_train)
def plotting(img,label,n):

    for i in range(n):
        plt.subplot(3,3,i+1)
        plt.imshow(img[i+1])
    plt.show()

plotting(img_train,label_train,9)

base_model = keras.applications.ResNet50(weights='imagenet', include_top=False)
x  = base_model.output
x = keras.layers.GlobalAvgPool2D()(x)
x = keras.layers.Dense(64,activation='relu')(x)
x= keras.layers.Dense(4,activation='softmax')(x)
model = keras.Model(base_model.input, x)
for layer in base_model.layers:
    layer.trainable = False

model1 = keras.Sequential([
    keras.layers.Input(shape=(250,250,3)),
    keras.layers.Conv2D(filters=32,kernel_size=(3,3), activation='relu'),
    keras.layers.MaxPool2D(pool_size=(3,3)),
    keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu'),
    keras.layers.MaxPool2D(pool_size=(3,3)),
    keras.layers.GlobalAvgPool2D(),
    keras.layers.Dense(64,activation='relu'),
    keras.layers.Dense(4,activation='softmax')
])
model1.compile(optimizer='adam',loss='categorical_crossentropy',metrics='accuracy')
model.summary()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='accuracy')

images = []
labels = []

for i in range(int(np.ceil(N/10))):
    img_train,label_train = next(data_train)
    images.append(img_train)
    labels.append(label_train)

images = np.concatenate(images,axis=0)
labels = np.concatenate(labels,axis=0)
print(labels.shape)
model.fit(images,labels,epochs=1,validation_split=0.2)

images_test = []
labels_test = []

for i in range(int(np.ceil(N1/10))):
    img_test,label_test = next(data_test)
    images_test.append(img_test)
    labels_test.append(label_test)

images_test = np.concatenate(images_test,axis=0)
labels_test = np.concatenate(labels_test,axis=0)
labels_test.shape
model.evaluate(images_test, labels_test)

model1.fit(images,labels,epochs=4,validation_split=0.2)
model1.evaluate(images_test,labels_test)