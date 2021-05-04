import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from keras.models import model_from_json
from keras.utils import plot_model
import pydot
import numpy as np

filename = 'Gain_ULB_P3P4.csv'
data = pd.read_csv(filename,  sep=';', header=0)

test = 'Prueba.csv'
data_test = pd.read_csv(test,  sep=';', header=0)

valores_pol = {"Pol": {'P45': 1, 'N45': 2,}}
data.replace(valores_pol, inplace = True)
data_test.replace(valores_pol, inplace=True)

X = data
y = X.pop('GAIN (dBi)')
tag_test = data_test.pop('GAIN (dBi)')
print(tag_test)
X = data.values
y = y.values
data_test = data_test.values
print(data_test)


'''
data_frame = pd.read_csv("housing.csv", delim_whitespace=True, header=None)
data_set = data_frame.values

X = data_set[:,0:13]
y = data_set[:,13]

'''

'''



def baseline_model():
    model = Sequential()
    model.add(Dense(15,input_dim=15,kernel_initializer='normal' , activation='relu'))
    model.add(Dense(1,kernel_initializer='normal'))

    model.compile(loss='mean_squared_error',optimizer='adam')
    return model

#Evaluamos el modelo


estimator = KerasRegressor(build_fn=baseline_model, epochs = 100, batch_size =5, verbose = 5)
kfold = KFold(n_splits=10)
results = cross_val_score(estimator,X,y,cv=kfold)
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))

'''

def wider_model():
	# create model
	model = Sequential()
	model.add(Dense(20, input_dim=16, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')

	return model
# evaluate model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=wider_model, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10)



results = cross_val_score(pipeline, X, y, cv=kfold)
print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))


model = wider_model()
predicciotions = model.predict(data_test)
print(predicciotions)