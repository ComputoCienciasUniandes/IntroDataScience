import numpy as np
import matplotlib.pyplot as plt

def prior(H):
    """
    Densidad de probabilidad de H 
    (H es la probabilidad de sacar cara).
    """
    p = np.ones(len(H))
    return p

def like(secuencia, H):
    """
    Likelihod de sacar una secuencia de caras y sellos, dado H.
    H es la probabilidad de sacar una cara.
    """
    L = np.ones(len(H))
    n_caras = secuencia.count('c')
    n_sello = secuencia.count('s')
    L *= H**n_caras
    L *= (1-H)**n_sello
    return L

def posterior(H, secuencia):
    """
    Posterior calculado con la normalizacion adecuada
    """
    post =  like(secuencia, H) * prior(H)
    evidencia = np.trapz(post, H)
    return  post/evidencia

def maximo_sigma(x, y):
    deltax = x[1] - x[0]

    # maximo de y
    ii = np.argmax(y)

    # segunda derivada
    d = (y[ii+1] - 2*y[ii] + y[ii-1]) / (deltax**2)

    return x[ii], 1.0/np.sqrt(-d)
    

secuencia = 'scccc'
H = np.linspace(1E-4, 1.0-1.0E-4, 100)
post = posterior(H, secuencia)

max, sigma = maximo_sigma(H, np.log(post))

gauss = (1.0/np.sqrt(2.0*np.pi*sigma**2))*np.exp(-0.5*(H-max)**2/(sigma**2))

plt.figure()
plt.plot(H, post, label='datos={}'.format(secuencia))
plt.plot(H, gauss, ':', label='Aproximacion Gaussiana')
plt.title('H= {:.2f} $\pm$ {:.2f}'.format(max, sigma))
plt.xlabel('H')
plt.ylabel('prob(H|datos)')
plt.legend()
plt.savefig('coins')
