import tensorflow 
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
import cv2
import os
from keras.layers import Activation
from keras.optimizers import SGD

path = "archive/train"
X = []
y = []

def prepareDataset(): #loading the dataset, normalising it, and shaping it for neural network use
    X = []
    y = []
    for i in range(len(os.listdir(path))):
        nameOfImg = os.listdir(path)[i]
        imgArray = cv2.imread(os.path.join(path, nameOfImg), cv2.IMREAD_GRAYSCALE)
        numFingers = numFingers = nameOfImg[len(nameOfImg) - 6:len(nameOfImg)-5]
        X.append(imgArray)
        y.append(numFingers)
    X = np.array(X).reshape(-1, 128, 128, 1)
    y = np.array(y)
    newY = y.astype('float32')
    newX = X.astype('float32')
    X = X/255
    y = to_categorical(newY)
    return X, y
    
X, y = prepareDataset()

def makeModel(): #making the model with all the differnet layers
    model = Sequential()
    model.add(Conv2D(64, (3,3),activation = "relu", kernel_initializer='he_uniform', input_shape = (128, 128, 1)))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(6, activation = "softmax"))
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])
    return model

model = makeModel()
model.fit(X,y, batch_size = 32, epochs = 10, validation_split = 0.15, verbose = 0) #to see progress of model, remove verbose = 0

#This is to test the model against a set of test images

#pathTest = "/Users/soumilrathi/Desktop/Python/MachineLearning/NumberOfFingers/archive/test"
#testX = []
#testXToShow = []
#testY = []
#def prepareTestDataset():
#    for i in range(10):
#        nameOfImg = os.listdir(pathTest)[i]
#        imageArray = cv2.imread(os.path.join(pathTest, nameOfImg), cv2.IMREAD_GRAYSCALE)
#        testXToShow.append(imageArray)
#        numFingers = numFingers = nameOfImg[len(nameOfImg) - 6:len(nameOfImg)-5]
#        testX.append(((imageArray.reshape(-1,128, 128, 1))/255))
#        testY.append(numFingers)


#prepareTestDataset()
#numberToTest = 0 #this number will be the number I test from the test images(can also use for loop to loop thru all the images in test set)
#plt.imshow(testXToShow[numberToTest])
#model.predict_classes(testX[numberToTest])

#This is to remodel a custom image for predicting number of fingers in it. 

def prepareImage(img_array):
    new_array = cv2.resize(img_array, (128, 128))
    return (new_array.reshape(-1,128, 128, 1))/255

imgToTest = cv2.imread("ThreeFingers.png", cv2.IMREAD_GRAYSCALE) #just taking in a random, self-made image
newImgToTest = prepareImage(imgToTest)
plt.figure(0)
plt.imshow(imgToTest, cmap = 'gray')
print(model.predict_classes(newImgToTest)[0])
