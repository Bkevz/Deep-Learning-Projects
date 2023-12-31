# -*- coding: utf-8 -*-
"""7_DL_Backpropagation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1IHZaw_JYoJ_r65QSYGdHSNh0w4RAgAG9

##BackPropagation

###Importing Libraries
"""

import numpy as np #array Operations
import pandas as pd #Handling Data
from sklearn.datasets import load_iris #Plant Iris Dataset
from sklearn.model_selection import train_test_split #Splitting Dataset into Train & Test
import matplotlib.pyplot as plt #plotting Graph

"""###Load Dataset"""

dataset = load_iris()
dataset

"""###Segregating Data into X(Imput) & Y(Output)"""

x=dataset.data
y=dataset.target
print(x)
print(y)

"""###Convert categorical variable into dummy/indicator variables"""

y = pd.get_dummies(y).values
y

"""###Split dataset into Train & Test"""

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=20, random_state=4)

"""###Initializing Hyper Parameters"""

learningRate = 0.1
epoch = 5000
N = y_train.size
input_size = 4
hidden_size = 2 
output_size = 3  
results = pd.DataFrame(columns=["MeanSquareError", "accuracy"])

"""###Initializing Weights"""

np.random.seed(10)
firstWeight = np.random.normal(scale=0.5, size=(input_size, hidden_size))   
lastWeight = np.random.normal(scale=0.5, size=(hidden_size , output_size)) 

print(firstWeight)
print(lastWeight)

"""###Activation Function"""

def activationFn(x):
    return 1 / (1 + np.exp(-x))

"""###Mean Squared Error"""

def mse(y_pred, y_true):
    return ((y_pred - y_true)**2).sum() / (2*y_pred.size)

"""###Evaluation Function - Accuracy"""

def accuracy(y_pred, y_true):
    acc = y_pred.argmax(axis=1) == y_true.argmax(axis=1)
    return acc.mean()

"""###Training"""

for i in range(epoch):    
    
    # feedforward propagation on hidden layer
    firstNetInput = np.dot(X_train, firstWeight)
    firstAFnOp = activationFn(firstNetInput)

    # feedforward propagation on output layer
    lastNetInput = np.dot(firstAFnOp, lastWeight)
    lastAFnOp = activationFn(lastNetInput)
    
    # Calculating error
    mseValue = mse(lastAFnOp, y_train)
    acc = accuracy(lastAFnOp, y_train)
    results=results.append({"MeanSquareError":mseValue, "accuracy":acc},ignore_index=True)
    
    # backpropagation
    E1 = lastAFnOp - y_train
    dfirstWeight = E1 * lastAFnOp * (1 - lastAFnOp)

    E2 = np.dot(dfirstWeight, lastWeight.T)
    dlastWeight = E2 * firstAFnOp * (1 - firstAFnOp)

    # weight updates
    lastWeight_update = np.dot(firstAFnOp.T, dfirstWeight) / N
    firstWeight_update = np.dot(X_train.T, dlastWeight) / N

    lastWeight = lastWeight - learningRate * lastWeight_update
    firstWeight = firstWeight - learningRate * firstWeight_update

results.MeanSquareError.plot(title="Mean Squared Error")

results.accuracy.plot(title="Accuracy")

# feedforward
firstNetInput = np.dot(X_test, firstWeight)
firstAFnOp = activationFn(firstNetInput)

lastNetInput = np.dot(firstAFnOp, lastWeight)
lastAFnOp = activationFn(lastNetInput)

acc = accuracy(lastAFnOp, y_test)
print("Accuracy: {}".format(acc))