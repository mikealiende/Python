from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np
import os
import pandas as pd
from sklearn.metrics import mean_squared_error

filename = 'Test_ULB_def.csv'
data = pd.read_csv(filename,  sep=';', header=0)

valores_pol = {"Pol": {'P45': 1, 'N45': 2,}}
data.replace(valores_pol, inplace = True)
data = data.dropna(how='any')

#data = data.drop(columns=['TILT_DEV (deg) '])
#data = data.drop(columns=['T_meas(deg) '])
#data = data.drop(columns=['TILT (deg)'])
#data = data.drop(columns=['FIRST_SL(dB)'])
#data = data.drop(columns=['SQUINT(dg)'])
#data = data.drop(columns=['SQUINT_ABS'])
#data = data.drop(columns=['USLS20(dB)'])
#data = data.drop(columns=['XPD_0 (dB)'])
#data = data.drop(columns=['XPD_60(dB)'])

#data = data.drop(columns=['FTB_0 (dB)'])
#data = data.drop(columns=['FTB_30(dB)'])
#data = data.drop(columns=['FTB_CO_0  '])
#data = data.drop(columns=['FTB_CO_30 '])
X = data
y = X.pop('GAIN_UCIII ')
X = data.values
y = y.values

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")
print("Modelo cargado")

loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
predicciones = loaded_model.predict(X)
predicciones_nan = predicciones[0:24]
predicciones_nan.tofile("predictions.csv",  sep=";")
print(predicciones)

error = np.sqrt(mean_squared_error(y, predicciones_nan))
print(error)