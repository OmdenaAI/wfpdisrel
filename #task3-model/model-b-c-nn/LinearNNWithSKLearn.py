# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 11:34:07 2020

@author: JOE LIGHTM
"""
#Import all needed libraries for this project
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

#Sample datasets are already imported as presented by Xavier_Torres
df = pd.read_csv('sampleData1.csv', sep=";")

X = df.iloc[:,:5] 
X.head()
X.shape
print(X.shape)
Y = df.iloc[:,5:]
#print(Y)
Y.head()
Y.shape
print(Y.shape)
# Split train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=50, shuffle=True)
sf = StandardScaler()
X_train = sf.fit_transform(X_train)

X_test = sf.transform(X_test)

#Design a linear perceptron. It doesn't use an activation function thats why it's linear
valueTron = Perceptron(random_state = 1, max_iter=500, tol = 0.005)
valueTron.fit(X_train, Y_train)
#predicting with the aid of the perceptron
ypredictTrain = valueTron.predict(X_train)
ypredictTest = valueTron.predict(X_test)

print(f"Accuracy score for train: %2f" %(accuracy_score(Y_train, ypredictTrain)))
print(f"Accuracy score for test: %2f" %(accuracy_score(Y_test, ypredictTest)))

#Evaluating algorithm performance
#MSE Result
print('Mean Squared Error:', mean_squared_error(Y_test, ypredictTest))
#Variance score: 1 is perfect prediction score
print('Test Variance Score: %.2f' % r2_score(Y_test, ypredictTest))

#Run the model against the test data presented through a plot
fig, pX = plt.subplots()

pX.scatter(Y_test, ypredictTest, edgecolors=(0, 0, 0))
pX.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'm--', lw=3)
pX.set_xlabel('Actual Data')
pX.set_ylabel('Predicted Data')
pX.set_title("Verified vs Predicted")
plt.show()