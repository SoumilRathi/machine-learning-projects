from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from numpy import where


df = pd.read_csv("Evaluation.csv")
#seperating non-question features
df = df.iloc[:,5:33]


#dropping features from 28 to 2
pca = PCA(n_components = 2)
reduced_train = pca.fit_transform(df)

#For visualisation purposes
#X = []
#y = []
#for i in range(reduced_train.shape[0]):
#    X.append(reduced_train[i][0])
#    y.append(reduced_train[i][1])
#plt.scatter(X, y)

#checking how many clusters we should have
#problems = []
#rangeToPlot = range(1,6)
#for k in rangeToPlot:
#    model = KMeans(n_clusters = k)
#    model.fit(reduced_train)
#    problems.append(model.inertia_)
#plt.plot(rangeToPlot, problems)
#plt.show()


#since the 'elbow' is at 3, we will choose 3 as a natural number of clusters
model = KMeans(n_clusters = 3)
model.fit(reduced_train)
y = model.predict(reduced_train)

#seperating into 3 different arrays for graphing
zero = []
two = []
one = []
for i in range(3):
    toPrint = where(y == i)
    for j in toPrint:
        for k in j:
            if i == 0:
                zero.append(k)
            if i == 1:
                one.append(k)
                
            if i == 2:
                two.append(k)
 
#graphing it               
for value in one[1:500]:
    plt.scatter(reduced_train[value][0], reduced_train[value][1], c = 'blue')
    
for value in two[1:500]:
    plt.scatter(reduced_train[value][0], reduced_train[value][1], c = 'red')
    
for value in zero[1:500]:
    plt.scatter(reduced_train[value][0], reduced_train[value][1], c = 'green')


plt.show()