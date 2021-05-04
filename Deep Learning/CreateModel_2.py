from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import model_from_json
import numpy
import os
import pandas as pd

filename = 'Gain_ULB_P3P4.csv'
data = pd.read_csv(filename,  sep=';', header=0)

valores_pol = {"Pol": {'P45': 1, 'N45': 2,}}
data.replace(valores_pol, inplace = True)

data = data.dropna(how='any')

X = data
y = X.pop('GAIN (dBi)')
X = data.values
y = y.values




# create model
model = Sequential()

#Multicapa
model.add(Dense(200, input_dim=16, kernel_initializer='normal', activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(30, activation='relu'))


model.add(Dense(1, kernel_initializer='normal'))
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')




estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=model, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10)
model.fit(X, y, epochs=5000, batch_size=10)
scores = model.evaluate(X, y, verbose=0)
print(scores)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model")