import numpy as np
import matplotlib.pyplot as plt


class PolyFit:
    def __init__(self, degree=1):
        self.degree = degree
        self.betas = np.ones(degree+1)

    
    def fit(self, x, y):
        """
        Entradas:
        x : array unidimensional de valores de x.
        y : array unidimensiondal de valores de y.

        Este codigo resuelve el siguiente sistema de ecuaciones
        S * beta = Y, beta = S^{-1} * Y, 
        donde S es una matriz que tiene como vectotes a columnas de potencias de X.        
        """
        m = self.degree
    # matriz que guarda los vectores columna de potencias de x
        S = np.zeros([len(x), m+1])
        for i in range(m+1):
            S[:,i] = x**i
            
    # Calculo la inversa de S.
        S_inv = np.linalg.pinv(S)
    
    # calculo beta
        self.betas = np.dot(S_inv, y)

    def predict(self, x):
        """
        Entradas:
        x : array de entrada.


        Salida:
        y : array de salida
        """
        y = np.zeros(len(x))
        for i in range(len(self.betas)):
            y +=  self.betas[i] * x**i
        return y

    def score(self, x, y):
        """
        Calcula el root mean squared error
        
        Entradas:
        x : array de entrada de valores de x.
        y : array unidimensiondal de valores de y.
        """
        y_predict  = self.predict(x)
        return np.sqrt(np.mean((y_predict - y)**2))


# cargo los datos para manipular
data = np.loadtxt("numeros_20.txt")

# divido en training y test
x_train = data[:10,0]
x_test = data[10:,0]
y_train = data[:10,1]
y_test = data[10:,1]

# distintos valores de m para probar
m_values = [0,1,3,9]

plt.figure()

for i,m in enumerate(m_values):
    modelo = PolyFit(degree=m)
    modelo.fit(x_train, y_train)

    plt.subplot(2,2,i+1)
    #grafica del modelo
    x_model = np.linspace(x_train.min(), x_train.max(), 100)
    plt.plot(x_model, modelo.predict(x_model))
    #grafica de los puntos
    plt.scatter(x_train, y_train)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("M={}".format(m))

plt.subplots_adjust(hspace=0.5)
plt.savefig("polinomios.png", bbox_inches='tight')


#segunda figura con los errores
error_train = []
error_test = []

for m in range(10):
    modelo = PolyFit(degree=m)
    modelo.fit(x_train, y_train)
    error_train.append(modelo.score(x_train, y_train))
    error_test.append(modelo.score(x_test, y_test))

plt.figure()
plt.plot(np.array(range(10)), error_train, label='training')
plt.plot(np.array(range(10)), error_test, label='test')
plt.semilogy()
plt.legend()
plt.xlabel("M")
plt.ylabel("$E_{RMS}$")
plt.savefig("train_test_error.png")
