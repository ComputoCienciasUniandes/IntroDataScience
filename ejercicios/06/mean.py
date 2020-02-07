import numpy as np
import matplotlib.pyplot as plt

def prior(mu):
    """
    Densidad de probabilidad de mu
    """
    p = np.ones(len(mu))/(mu.max()-mu.min())
    return p

def like(x, sigma, mu):
    """
    Likelihod de tener un dato x e incertidumbre sigma
    """
    L = np.ones(len(mu))
    for x_i,sigma_i in zip(x, sigma):
        L *= (1.0/np.sqrt(2.0*np.pi*sigma_i**2))*np.exp(-0.5*(x_i-mu)**2/(sigma_i**2))
    return L

def posterior(mu, x, sigma):
    """
    Posterior calculado con la normalizacion adecuada
    """
    post =  like(x, sigma, mu) * prior(mu)
    evidencia = np.trapz(post, mu)
    return  post/evidencia

def maximo_incertidumbre(x, y):
    deltax = x[1] - x[0]

    # maximo de y
    ii = np.argmax(y)

    # segunda derivada
    d = (y[ii+1] - 2*y[ii] + y[ii-1]) / (deltax**2)

    return x[ii], 1.0/np.sqrt(-d)
    

x = [4.6, 6.0, 2.0, 5.8]
sigma = [2.0, 1.5, 5.0, 1.0]
mu = np.linspace(0.0, 10.0, 1000)

post = posterior(mu, x, sigma)

max, incertidumbre = maximo_incertidumbre(mu, np.log(post))


plt.figure()
plt.plot(mu, post)
plt.title('$\mu$= {:.2f} $\pm$ {:.2f}'.format(max, incertidumbre))
plt.xlabel('$\mu$')
plt.ylabel('prob($\mu$|datos)')
plt.savefig('mean.png')
