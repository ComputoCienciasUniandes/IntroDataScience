import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_vectors(data, row_names, col_names, delta_y=3.5, figname=None):
    # renormaliza los datos
    for i in range(len(col_names)):
        print(i, col_names[i], len(data[i,:]))
        data[i,:] = (data[i,:] - np.mean(data[i,:]))/np.std(data[i,:])
    
    # calcula vectores propios
    cov = np.cov(data)
    valores, vectores = np.linalg.eig(cov)
    orden_valores = np.argsort(valores)[::-1] # indices de los autovalores de mayor a menor
    print(valores[orden_valores])

    new_data = data.T @ vectores # esto hace el cambio de base al nuevo sistema de los autovectores
        
    plt.figure(figsize=(9,9))
    # Grafica los nombres
    for i, name in enumerate(row_names):
        plt.text(new_data[i,orden_valores[0]], new_data[i,orden_valores[1]], name, alpha=0.5, fontsize=12, color='Blue')
    
    # Grafica los loading vectors
    for i, name in enumerate(col_names):
        plt.arrow(0, 0, 3*vectores[i,orden_valores[0]], 3*vectores[i, orden_valores[1]], head_width=0.1)
        plt.text(3*vectores[i,orden_valores[0]], 3*vectores[i, orden_valores[1]], name, fontsize=14, color='Red')
    
    plt.xlabel('Primera Componente Principal')
    plt.ylabel('Segunda Componente Principal')
    plt.xlim(-delta_y,delta_y)
    plt.ylim(-3.5,3.5)
    plt.savefig(figname, box_inches='tight')

    plt.figure(figsize=(6,6))
    plt.plot(np.arange(len(col_names))+1, 100*np.cumsum(valores[orden_valores])/np.sum(valores))
    plt.scatter(np.arange(len(col_names))+1, 100*np.cumsum(valores[orden_valores])/np.sum(valores), s=100)
    plt.xlabel("Numero de autovalores")
    plt.ylabel("Porcentaje de Varianza explicada")
    plt.ylim([0,100])
    plt.xticks(np.arange(len(col_names))+1)
    plt.grid()
    plt.savefig("varianza_"+figname, box_inches='tight')


# primera grafica
indata = pd.read_csv('USArrests.csv')
col_names = ['Murder', 'Assault', 'UrbanPop', 'Rape']
data = np.array(indata[col_names]).T
names = np.array(indata['Unnamed: 0'])
plot_vectors(data, names, col_names, figname='arrestos.png')

#segunda grafica
indata = pd.read_csv('Cars93.csv')
col_names = ['MPG.city', 'MPG.highway', 'EngineSize', 'Horsepower', 
           'RPM', 'Rev.per.mile', 'Fuel.tank.capacity', 'Length', 
           'Width', 'Turn.circle', 'Weight']
data = np.array(indata[col_names]).T
names = np.array(indata['Model'])
plot_vectors(data, names, col_names, delta_y=6.0, figname='cars.png')

