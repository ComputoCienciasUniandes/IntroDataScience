import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model


def compute_betas(X, Y):
    n_points = len(Y)
    # esta es la clave del bootstrapping: la seleccion de indices de "estudiantes"
    indices = np.random.choice(np.arange(n_points), n_points)
    new_X = X[indices, :]
    new_Y = Y[indices]
    regresion = sklearn.linear_model.LinearRegression()
    regresion.fit(new_X, new_Y)
    return regresion.coef_

# lee los datos
data = np.loadtxt("notas_andes.dat", skiprows=1)
Y = data[:,4]
X = data[:,:4]

# bootstrapping con 10000 iteraciones
n_intentos = 10000
betas = np.ones([n_intentos, 4])
for i in range(n_intentos):
    betas[i,:] = compute_betas(X, Y)


# grafica final
plt.figure()
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.hist(betas[:,i], bins=20, density=True)
    plt.title(r"$\beta_{}= {:.2f}\pm {:.2f}$".format(i+1, np.mean(betas[:,i]), np.std(betas[:,i])))
    plt.xlabel(r"$\beta_{}$".format(i+1))
    plt.xlim([-0.2, 0.4])

plt.subplots_adjust(hspace=0.5)
plt.savefig("bootstrapping.png", bbox_inches='tight')
