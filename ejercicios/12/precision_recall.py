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

def transform_pca(data, numeros=[1], n_componentes=10, model_name='base'):
    # numeros utilizados para generar el espacio de PCA
    dd = data['y_train']!=data['y_train']
    for n in numeros:
        print(n)
        dd |= data['y_train']==n
        
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
    data['x_test_transform_'+model_name] = test_trans[:,:n_componentes]
    train_trans = data['x_train'] @ vectores
    data['x_train_transform_'+model_name] = train_trans[:,:n_componentes]
    
    return data

def precision_recall(data, model_name='base'):
    linear = LinearDiscriminantAnalysis()
    linear.fit(data['x_train_transform_'+model_name], data['y_train'])
    proba_test  = linear.predict_proba(data['x_test_transform_'+model_name])
    prec, rec, th = sklearn.metrics.precision_recall_curve(data['y_test'], proba_test[:,1], pos_label=1)
    data['precision_'+model_name] = prec[:-1]
    data['recall_'+model_name] = rec[:-1]
    data['threshold_'+model_name] = th.copy()
    data['F1_'+model_name] = 2.0*prec[:-1]*rec[:-1]/(prec[:-1]+rec[:-1] +1E-10)
    #ii = np.isnan(data['F1_'+model_name])
    #data['F1_'+model_name][ii] = 0.0
    return data
                                                                
def plot_precision_recall(data, model_name='base'):
    ii = np.argmax(data['F1_'+model_name])
    plt.plot(data['recall_'+model_name], data['precision_'+model_name], label=model_name)
    plt.scatter(data['recall_'+model_name][ii], data['precision_'+model_name][ii], color='red', s=50.0)

def plot_f1_threshold(data, model_name='base'):
    ii = np.argmax(data['F1_'+model_name])
    plt.plot(data['threshold_'+model_name], data['F1_'+model_name], label=model_name)
    plt.scatter(data['threshold_'+model_name][ii], data['F1_'+model_name][ii], color='red', s=50.0)

    
datos = prepare_data()
numeros = [[1], [0], [1,0]]
modelos = ['Unos', 'Otros', 'Todos']
for n, m in zip(numeros, modelos):
    print(n, m)
    datos = transform_pca(datos, numeros=n, model_name=m)
    datos = precision_recall(datos, model_name=m)
    
plt.figure(figsize=(10,5))
plt.subplot(1,2,1) 
for m in modelos:
    plot_f1_threshold(datos, model_name=m)
plt.xlabel("probabilidad")
plt.ylabel("F1")

plt.subplot(1,2,2)
for m in modelos:
    plot_precision_recall(datos, model_name=m)
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend()
plt.savefig("prec_recall.png")
