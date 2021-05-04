from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

from keras.wrappers.scikit_learn import KerasRegressor

filename = 'DatosULB_def.csv'
data = pd.read_csv(filename,  sep=';', header=0)

valores_pol = {"Pol": {'P45': 1, 'N45': 2,}}
data.replace(valores_pol, inplace = True)




#data = data.drop(columns=['TILT_DEV (deg) '])
#data = data.drop(columns=['SQUINT(dg)'])
#data = data.drop(columns=['SQUINT_ABS'])
#data = data.drop(columns=['T_meas(deg) '])
#data = data.drop(columns=['TILT (deg)'])
#data = data.drop(columns=['FIRST_SL(dB)'])

#data = data.drop(columns=['USLS20(dB)'])
#data = data.drop(columns=['XPD_0 (dB)'])
#data = data.drop(columns=['XPD_60(dB)'])

#data = data.drop(columns=['FTB_0 (dB)'])
#data = data.drop(columns=['FTB_30(dB)'])
#data = data.drop(columns=['FTB_CO_0  '])
#data = data.drop(columns=['FTB_CO_30 '])


data = data.dropna(how='any')

X = data
y = X.pop('GAIN_UCIII')
X = data.values
y = y.values

print(y)




# create model
model = Sequential()
model.add(Dense(10, input_dim=16, kernel_initializer='normal', activation='relu'))
model.add(Dense(5, activation='relu'))


model.add(Dense(1, kernel_initializer='normal'))
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X, y, epochs=1000, batch_size=10)

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=model, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10)

scores = model.evaluate(X, y, verbose=0)
print(scores)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model")