import numpy as np
import pandas as pd
import keras 
import os
import cv2
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

train_img_path=r"E:\backup\python_image_classification\images\train_set"
test_img_path=r"E:\backup\python_image_classification\images\test_set"

def read_img_train(path):
    label=[]
    data1=[]
    counter=0
    for file in os.listdir(path):
        image_data=cv2.imread(os.path.join(path,file), cv2.IMREAD_GRAYSCALE)
        image_data=cv2.resize(image_data,(150,150))
        if file.startswith("one"):
            label.append(1)
        elif file.startswith("two"):
            label.append(2)
        elif file.startswith("five"):
            label.append(0)
        try:
            data1.append(image_data/255)
        except:
            label=label[:len(label)-1]
        counter+=1
        if counter % 10==0:
            print (counter," image data retreived")
    return label,data1

label_train,data_train=read_img_train(train_img_path)

def read_img_test(test_img_path):
    test_data=[]
    label=[]
    counter=0
    for file in os.listdir(test_img_path):
        image_data=cv2.imread(os.path.join(test_img_path,file), cv2.IMREAD_GRAYSCALE)
        try:
            image_data=cv2.resize(image_data,(150,150))
            test_data.append(image_data/255)
            if file.startswith("one"):
                label.append(1)
            elif file.startswith("two"):
                label.append(2)
            elif file.startswith("five"):
                label.append(0)
        except:
            print ("ek gaya")
        counter+=1
        if counter%4==0:
            print (counter," image data retreived")
    return label,test_data

label_test,data_test=read_img_test(test_img_path)

def createModel():
    model=Sequential()
    model.add(Conv2D(kernel_size=(3,3),filters=32,input_shape=(150,150,1),activation="relu"))
    model.add(Conv2D(kernel_size=(3,3),filters=32,activation="relu",padding="same"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(kernel_size=(3,3),filters=64,activation="relu"))
    model.add(Conv2D(kernel_size=(5,5),filters=64,activation="relu"))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(Conv2D(kernel_size=(2,2),strides=(2,2),filters=32))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(512,activation="relu"))
    model.add(Dense(3,activation="softmax"))
    model.summary()
     
    return model


data_train=np.array(data_train)
print (data_train.shape)
data_train=data_train.reshape((data_train.shape)[0],(data_train.shape)[1],(data_train.shape)[2],1)
#data1=data1/255
label_train=np.array(label_train)
print (data_train.shape)
print (label_train.shape)
label_train

one_hot_labels= to_categorical(label_train[0:])

data_test=np.array(data_test)
print (data_test.shape)
data_test=data_test.reshape((data_test.shape)[0],(data_test.shape)[1],(data_test.shape)[2],1)

one_hot_labels_test= to_categorical(label_test[0:])


datagen = ImageDataGenerator(
        zoom_range=0.2, # randomly zoom into images
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

model1 = createModel()
batch_size = 5
epochs = 100
model1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
 
history = model1.fit_generator(datagen.flow(data_train,one_hot_labels, batch_size=batch_size),steps_per_epoch=int(np.ceil(data_train.shape[0] / float(batch_size))), epochs=epochs, verbose=1,validation_data=(data_test[:-1],one_hot_labels_test))
 

predicted_labels=model1.predict(data_test)
predicted_labels=np.round(predicted_labels,decimals=2)
print(predicted_labels)
model1.evaluate(data_test[0:-1],one_hot_labels_test)
print(model1.metrics_names)

