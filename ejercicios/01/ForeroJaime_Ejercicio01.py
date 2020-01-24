import numpy as np
import matplotlib.pyplot as plt

# defino una funcion para calcular un polinomio
def poly(x, beta):
    """
    Entradas:
    x : array de entrada.
    beta: coeficientes del polinomio: beta[i] = beta_i. 

    Salida:
    y : array de salida
    """
    y = np.zeros(len(x))
    for i in range(len(beta)):
        y +=  beta[i] * x**i
    return y

def solve_poly(x, y, m):
    """
    Entradas:
    x : array unidimensional de valores de x.
    y : array unidimensiondal de valores de y.
    m : orden del polinomio que queremos ajustar.

    Salida:
    beta : array con los coeficientes del polinotmi.


    Este codigo resuelve el siguiente sistema de ecuaciones
    S * beta = Y, beta = S^{-1} * Y, 
    donde S es una matriz que tiene como vectotes a columnas de potencias de X.

    """
    # matriz que guarda los vectores columna de potencias de x
    S = np.zeros([len(x), m+1])
    for i in range(m+1):
        S[:,i] = x**i
    
    # Calculo la inversa de S.
    S_inv = np.linalg.inv(S)
    
    # calculo beta
    beta = np.dot(S_inv, y)
    
    return beta



# cargo los datos para manipular
data = np.loadtxt("numeros_20.txt")

# diccionario para guardar todos los coeficientes
betas = {}

# distintos valores de m para probas

m_values = [1,2,3,4]

for m in m_values:
    X = data[: m+1, 0]
    Y = data[: m+1, 1]

    betas[m] = solve_poly(X, Y, m)

# preparo finalmene la grafica pedida

plt.figure()

for i,m in enumerate(m_values):
    # vuelvo a sacar los puntos para graficarlos
    X = data[: m+1, 0]
    Y = data[: m+1, 1]

    # array de valores de x que va del minimo al maximo valor considerados
    # para hacer los ajustes
    x_model = np.linspace(np.min(data[:10,0]), np.max(data[:10,0]), 100)

    # valores correspondientes al modelo de orden m
    y_model = poly(x_model, betas[m])

    plt.subplot(2,2,i+1)
    #grafica del modelo
    plt.plot(x_model, y_model)
    #grafica de los puntos
    plt.scatter(X, Y)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("M={}".format(m))

plt.subplots_adjust(hspace=0.5)
plt.savefig("polinomios.png", bbox_inches='tight')

