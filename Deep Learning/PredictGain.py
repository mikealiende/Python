import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import tree
import numpy as np

filename = 'Gain_ULB_P3P4.csv'
data = pd.read_csv(filename,  sep=';', header=0)

data = data.drop(columns=['SQUINT_3(dg)'])
data = data.drop(columns=['T_dev(dg)'])
data = data.drop(columns=['T_meas(dg)'])
data = data.drop(columns=['TILT (deg)'])

'''
data = data.drop(columns=['Pol'])
data = data.drop(columns=['First_SL(dB)'])
data = data.drop(columns=['USLS20(dB)'])
data = data.drop(columns=['XPD_0 (dB)'])
data = data.drop(columns=['XPD_60(dB)'])
'''

#print((data['Pol']).unique())
valores_pol = {"Pol": {'P45': 1, 'N45': 2,}}
data.replace(valores_pol, inplace = True)
def PlotGain_vs_VBW():
    plt.scatter(x=data['GAIN (dBi)'], y=data['VBW (deg) '])
    plt.title('GAIN vs VBW')
    plt.xlabel('GAIN')
    plt.ylabel('VBW')
    plt.show()

def PlotGain_vs_HBW():
    plt.scatter(x=data['GAIN (dBi)'], y=data['HBW (deg) '])
    plt.title('GAIN vs HBW')
    plt.xlabel('GAIN')
    plt.ylabel('HBW')
    plt.show()

def PlotGain_vs_Tilt_mes():
    plt.scatter(x=data['GAIN (dBi)'], y=data['TILT (deg)'])
    plt.title('GAIN vs T_meas')
    plt.xlabel('GAIN')
    plt.ylabel('Tilt')
    plt.show()
def PlotGain_vs_FirstSL():
    plt.scatter(x=data['GAIN (dBi)'], y=data['First_SL(dB)'])
    plt.title('GAIN vs First_SL')
    plt.xlabel('GAIN')
    plt.ylabel('FirstSL')
    plt.show()
def PlotGain_vs_XPD_0():
    plt.scatter(x=data['GAIN (dBi)'], y=data['XPD_0 (dB)'])
    plt.title('GAIN vs XPD_0')
    plt.xlabel('GAIN')
    plt.ylabel('XPD_0')
    plt.show()

'''
PlotGain_vs_VBW()
PlotGain_vs_HBW()
PlotGain_vs_Tilt_mes()
PlotGain_vs_FirstSL()#Duda
PlotGain_vs_XPD_0()
'''
#Separamos entre datos de entrenamiento y de test
data_training = data.sample(frac = 0.8, random_state = 0)
data_test = data.drop(data_training.index)

#Separamos columna que queremos calcuar
tag_training = data_training.pop('GAIN (dBi)')
tag_test = data_test.pop('GAIN (dBi)')







#Utilizamos regresion lineal
model = LinearRegression()
model.fit(data_training,tag_training)  #Entrenamos el modelo con los datos de entramiento

predicciones = model.predict(data_test)  #Calculamos predicciones
print("Gain(Linear): ",predicciones)

error = np.sqrt(mean_squared_error(tag_test,predicciones))  #Comparamos las predicciones con los valores de test
print("Porcentaje de error: ", error*100)


model_tree = DecisionTreeRegressor()
model_tree.fit(data_training,tag_training)

predicciones_tree = model_tree.predict(data_test)
print("Gain(Tree): ", predicciones_tree)

error_tree = np.sqrt(mean_squared_error(tag_test, predicciones_tree))
print("Porcentaje de error(Tree):", error_tree)


model_gradient_boosting = GradientBoostingRegressor()
model_gradient_boosting.fit(data_training,tag_training)

predicciones_gradient_boosting = model_gradient_boosting.predict(data_test)
print(("Gain(Gradient_Boosting):", predicciones_gradient_boosting))

error_gradient_boosting= np.sqrt(mean_squared_error(tag_test, predicciones_gradient_boosting))
print("Error(Gradient_Boosting):", error_gradient_boosting)

model_random_forest = RandomForestRegressor()
model_random_forest.fit(data_training,tag_training)

predicciones_random_forest = model_random_forest.predict(data_test)
print("Gain(random_forest): ",predicciones_random_forest)

error_random_forest = error_gradient_boosting= np.sqrt(mean_squared_error(tag_test, predicciones_random_forest))
print("Error random forest:" ,error_random_forest)