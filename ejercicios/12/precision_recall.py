import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np
import sklearn.metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def prepare_data():
    # lee los numeros
    numeros = skdata.load_digits()

    # lee los labels
    target = numeros['target']

    # lee las imagenes
    imagenes = numeros['images']

    # cuenta el numero de imagenes total
    n_imagenes = len(target)

    # para poder correr PCA debemos "aplanar las imagenes"
    data = imagenes.reshape((n_imagenes, -1)) # para volver a tener los datos como imagen basta hacer data.reshape((n_imagenes, 8, 8))

    # Split en train/test
    x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.5)

    # todo lo que es diferente de 1 queda marcado como 0
    y_train[y_train!=1]=0
    y_test[y_test!=1]=0

    # Reescalado de los datos
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return {'x_train':x_train, 'x_test':x_test, 'y_train':y_train, 'y_test':y_test}

def transform_pca(data, numeros=[1], n_componentes=10):
    # numeros utilizados para generar el espacio de PCA
    dd = data['y_train']==data['y_train']
    for n in numeros:
        dd &= data['y_train']==n
        
    cov = np.cov(data['x_train'][dd].T)
    valores, vectores = np.linalg.eig(cov)

    # pueden ser complejos por baja precision numerica, asi que los paso a reales
    valores = np.real(valores)
    vectores = np.real(vectores)

    # reordeno de mayor a menor
    ii = np.argsort(-valores)
    valores = valores[ii]
    vectores = vectores[:,ii]

    # encuentro las imagenes en el espacio de los autovectores
    
    test_trans = data['x_test'] @ vectores
    data['x_test_transform'] = test_trans[:,:n_componentes]
    train_trans = data['x_train'] @ vectores
    data['x_train_transform'] = train_trans[:,:n_componentes]
    
    return data

def precision_recall(data)

datos = prepare_data()
datos_transformados = transform_pca(datos)
