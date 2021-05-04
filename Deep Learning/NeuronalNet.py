import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold




filename = 'Gain_ULB_P3P4.csv'
data = pd.read_csv(filename,  sep=';', header=0)

test = 'Prueba.csv'
data_test = pd.read_csv(test,  sep=';', header=0)
'''
data = data.drop(columns=['SQUINT_3(dg)'])
data = data.drop(columns=['T_dev(dg)'])
data = data.drop(columns=['T_meas(dg)'])
data = data.drop(columns=['TILT (deg)'])

data_test = data_test.drop(columns=['SQUINT_3(dg)'])
data_test = data_test.drop(columns=['T_dev(dg)'])
data_test = data_test.drop(columns=['T_meas(dg)'])
data_test = data_test.drop(columns=['TILT (deg)'])
'''
valores_pol = {"Pol": {'P45': 1, 'N45': 2,}}
data.replace(valores_pol, inplace = True)
data_test.replace(valores_pol, inplace=True)

data_training = data
tag_training = data_training.pop('GAIN (dBi)')

tag_test = data_test.pop('GAIN (dBi)')


def base_model():
    model = Sequential()
    model.add(Dense(16, input_dim=16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

seed = 7
np.random.seed(seed)
estimator = KerasRegressor(build_fn=base_model, epochs=100, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, data_training, tag_training, cv=kfold)
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))