import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

filename = 'DatosULB_def.csv'
data = pd.read_csv(filename,  sep=';', header=0)

test = 'Test_ULB_def.csv'
data_test = pd.read_csv(test,  sep=';', header=0)


data = data.dropna(how='any')
data_test = data_test.dropna(how = 'any')

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
tag_training = data_training.pop('GAIN_UCIII')

tag_test = data_test.pop('GAIN_UCIII')

print(tag_test)


model_random_forest = RandomForestRegressor()
model_random_forest.fit(data_training,tag_training)

predicciones_random_forest = model_random_forest.predict(data_test)
print("Gain(random_forest): ",predicciones_random_forest)

error_random_forest = error_gradient_boosting= np.sqrt(mean_squared_error(tag_test, predicciones_random_forest))
print("Error random forest:" ,error_random_forest)

model_gradient_boosting = GradientBoostingRegressor()
model_gradient_boosting.fit(data_training,tag_training)

predicciones_gradient_boosting = model_gradient_boosting.predict(data_test)
print(("Gain(Gradient_Boosting):", predicciones_gradient_boosting))

error_gradient_boosting = np.sqrt(mean_squared_error(tag_test, predicciones_gradient_boosting))
print("Error(Gradient_Boosting):", error_gradient_boosting)

predicciones_gradient_boosting.tofile("mi_fichero.csv",  sep=";")