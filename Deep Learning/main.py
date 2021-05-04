import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

def MaratonPrediccion(corredor):
    filename = 'MarathonData.csv'
    data = pd.read_csv(filename, header=0)


    #Al comprobar los datos vemos que hay un error en Wall21, lo corregimos de la siguiente manera
    data['Wall21'] = pd.to_numeric(data['Wall21'], errors='coerce')

    #Borramos cosas que no son importantes
    data = data.drop(columns=['Name'])
    data = data.drop(columns=['id'])
    data = data.drop(columns=['Marathon'])
    data = data.drop(columns=['CATEGORY'])

    #Comprobamos si hay datos nulos
    #print(data.isna().sum())
    data["CrossTraining"]  = data["CrossTraining"].fillna(0)  #Rellenamos con cero los nulos
    data = data.dropna(how='any') #Eliminamos el resto de Nan
    #Como en constraining tenemso strings los vamos a cambiar por valores numéricos
    #print((data['CrossTraining']).unique())
    valores_cross = {"CrossTraining": {'ciclista 1h': 1, 'ciclista 3h': 2, 'ciclista 4h': 3, 'ciclista 5h': 4, 'ciclista 13h': 5}}
    data.replace(valores_cross, inplace = True)


    #print((data['Category']).unique())
    valores_categoria = {"Category":  {'MAM':1, 'M45':2, 'M40':3, 'M50':4, 'M55':5,'WAM':6}}
    data.replace(valores_categoria, inplace=True)
    print(data)

    #Ya tenemos la tabla lista,

    '''Plotemaos tiempo de maraton vs los km que el corredor ha hechoo en las 4 semnas previas
 
    plt.scatter(x = data['km4week'], y=data['MarathonTime'])
    plt.title('km4week Vs Marathon Time')
    plt.xlabel('km4week')
    plt.ylabel('Marathon Time')
    plt.show()

    '''
    '''Comparamos velocidad de los entrenamientp con tiempo de maraton, vemos que no nos da ningun resultado 
        por lo tanto vamos a eliminar los datos menores 1000
    plt.scatter(x = data['sp4week'], y=data['MarathonTime'])
    plt.title('sp4week Vs Marathon Time')
    plt.xlabel('sp4week')
    plt.ylabel('Marathon Time')
    plt.show()
    '''
    data = data.query('sp4week<1000')  #Eliminamos datos desvituados


    '''Empezamos a entrenar el modelo de machine learning
Lo primero que haremos sera dividir entre datos de entrenamineto(80%) o test(20%)'''

    data_training = data.sample(frac = 0.8, random_state = 0)
    data_test = data.drop(data_training.index)

    #Separamos de las tablas la variable que queremos predecir (Tiempo de maraton)

    tag_training = data_training.pop('MarathonTime')
    tag_test = data_test.pop('MarathonTime')

    #Utilizamos regresion lineal
    model = LinearRegression()
    model.fit(data_training,tag_training)  #Entrenamos el modelo con los datos de entramiento

    predicciones = model.predict(corredor)  #Calculamos predicciones
    print("Predicciones: ",predicciones,'\n')

    #error = np.sqrt(mean_squared_error(tag_test,predicciones))  #Comparamos las predicciones con los valores de test
    #print("Porcentaje de error: ", error*100)

nuevo_corredor = pd.DataFrame(np.array([[1,400,20,0,1.4]]),columns=['Category', 'km4week','sp4week', 'CrossTraining','Wall21'])
print(nuevo_corredor, '\n')
MaratonPrediccion(nuevo_corredor)


