import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from random import shuffle


datasetDir = "PetImages"
categories = ["Dog", "Cat"]

imgSize = 100;

trainingData = []

def createTrainingData():
    for category in categories:
        path = os.path.join(datasetDir, category) #gets the path to the folder with images
        classNum = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                newArray = cv2.resize(img_array, (imgSize, imgSize))
                trainingData.append([newArray, classNum])
            except Exception as e:
                pass
            

createTrainingData()  


import random

random.shuffle(trainingData)
X = []
y = []
for features, labels in trainingData:
    X.append(features)
    y.append(labels) 
X = np.array(X).reshape(-1, imgSize, imgSize, 1) #the rehsape is pretty much to make is usable for neural network 
X = X/255 #normalising X to a value b/w 0 and 1
import tensorflow
from tensorflow.keras.models import Sequential;
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten;

#Creating the model
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid')) #this could also have been passed into Dense

model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

y= np.array(y)
#fitting the model
model.fit(X, y, batch_size = 32, epochs = 10, validation_split = 0.15)



#testing the model
valToTest = 16

plt.imshow(X[valToTest], cmap = 'gray')
categories[round(model.predict(X[valToTest:valToTest +1])[0][0])]



#testing the model from an outside source(random image I found on google)
random_imgArray = cv2.imread("RandomImage/Cat.jpeg", cv2.IMREAD_GRAYSCALE)
def prepare(imageArray):
    newImgArray = cv2.resize(imageArray, (imgSize, imgSize))
    return (newImgArray.reshape(-1,imgSize, imgSize, 1))/255
toCheck = prepare(random_imgArray)
categories[round(model.predict(toCheck)[0][0])]
