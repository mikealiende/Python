import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model
import pydot
import numpy as np

dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter = ',')
X = dataset[:, 0:8]
y = dataset[:, 8]
print(X)
print(y)

model  = Sequential()
model .add(Dense(12, input_dim = 8, activation = 'relu'))  #Primera capa oculta 12 neuronas
model.add(Dense(8, activation='relu'))  #Segunda capa oculta 8 neuronas
model.add(Dense(1, activation = 'sigmoid')) #Capa de salida

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

model.fit(X,y,epochs=150,batch_size=10)
accuaray = model.evaluate(X,y)







