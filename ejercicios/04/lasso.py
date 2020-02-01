import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model
import sklearn.preprocessing 
import sklearn.model_selection
import itertools


# datos iniciales
data = pd.read_csv('Cars93.csv')


# columnas de X
columns = ['MPG.city', 'MPG.highway', 'EngineSize', 'Horsepower', 
           'RPM', 'Rev.per.mile', 'Fuel.tank.capacity', 'Length', 
           'Width', 'Turn.circle', 'Weight']

# define X, Y
Y = np.array(data['Price'])
X = np.array(data[columns])

# standarize
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

# split en training y test
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X_scaled, Y, test_size=0.5)

# inicializa clase de regresion lineal
regresion = sklearn.linear_model.LinearRegression()

# para guardar resultados
scores = []
predictores = []
n_predict = []
betas = []

# array de indices que representan las columnas
indices = np.array(np.arange(len(columns)))

# iteracion sobre el numero de indices para utilizas
for n_predictores in range(1,len(indices)+1):
    # todas las combinaciones posibles
    perm = itertools.combinations(indices, n_predictores)
    for p in perm:
        c = list(p) # estos son los indices de las columnas que voy a utilizar

        # Entrena solamen con las columnas que devolvio itertools
        regresion.fit(X_train[:, c], Y_train)
        # Guarda los betas correspondientes
        betas.append(regresion.coef_)
        # Guarda el score correspondiente
        score = regresion.score(X_test[:,c], Y_test)
        scores.append(score)
        # guarda el numero de predictores utilizados
        n_predict.append(n_predictores)
        # guarda los indices de las columas utilizadas
        predictores.append(c)


# Primera grafica
plt.figure()
plt.scatter(n_predict, scores)
plt.xlabel("Numero de predictores")
plt.ylabel("R2")
plt.savefig("nparams.png")

print("")
print("Best model (Regresion Lineal)")
print("")
best = np.argmax(scores) 
predict = predictores[best]
beta = betas[best]
ii = np.argsort(np.abs(beta))
for i in ii:
    print(columns[predict[i]], beta[i])


# numbero de valores de alpha para probar
n_alpha = 100
alpha = np.logspace(-3, 1, n_alpha)
scores = []
betas = []

for i, a in enumerate(alpha):
    lasso = sklearn.linear_model.Lasso(alpha=a)
    lasso.fit(X_train, Y_train)
    scores.append(lasso.score(X_test, Y_test))
    betas.append(lasso.coef_)


plt.figure()
plt.plot(np.log10(alpha), scores)
plt.xlabel("log alpha")
plt.ylabel("R2")
plt.savefig("lasso.png")

print("")
print("Best model (LASSO)")
print("")
best = np.argmax(scores) 
beta = betas[best]
ii = np.argsort(np.abs(beta))
for i in ii:
    if(abs(beta[i])>0):
        print(columns[i], beta[i])
