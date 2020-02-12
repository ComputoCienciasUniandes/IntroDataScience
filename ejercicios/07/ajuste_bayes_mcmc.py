import numpy as np
import matplotlib.pyplot as plt

def model(x, betas):
    y = betas[0]
    for i in range(1, len(betas)):
        y += betas[i]*x[i-1]
    return y

def loglike(x_obs, y_obs, sigma_y_obs, betas):
    n_obs = len(y_obs)
    l = 0.0
    for i in range(n_obs):
        l += -0.5*(y_obs[i]-model(x_obs[i,:], betas))**2/sigma_y_obs[i]**2
    return l

def run_mcmc(data_file="notas_andes.dat", n_dim=4, n_iterations=20000, sigma_y=0.1):
    data = np.loadtxt(data_file)
    x_obs = data[:,:n_dim]
    y_obs = data[:, n_dim]
    sigma_y_obs = np.ones(len(y_obs))*sigma_y

    betas = np.zeros([n_iterations, n_dim+1])
    for i in range(1, n_iterations):
        current_betas = betas[i-1,:]
        next_betas = current_betas + np.random.normal(scale=0.01, size=n_dim+1)

        loglike_current = loglike(x_obs, y_obs, sigma_y_obs, current_betas)
        loglike_next = loglike(x_obs, y_obs, sigma_y_obs, next_betas)

        r = np.min([np.exp(loglike_next - loglike_current), 1.0])
        alpha = np.random.random()

        if alpha < r:
            betas[i,:] = next_betas
        else:
            betas[i,:] = current_betas
    betas = betas[n_iterations//2:,:]
    return {'betas':betas, 'x_obs':x_obs, 'y_obs':y_obs}

n_dim = 4
results = run_mcmc()
betas = results['betas']

plt.figure()
for i in range(0,n_dim+1):
    plt.subplot(2,3,i+1)
    plt.hist(betas[:,i],bins=15, density=True)
    plt.title(r"$\beta_{}= {:.2f}\pm {:.2f}$".format(i,np.mean(betas[:,i]), np.std(betas[:,i])))
    plt.xlabel(r"$\beta_{}$".format(i))
plt.subplots_adjust(hspace=0.5)
plt.savefig("ajuste_bayes_mcmc.png",  bbox_inches='tight')    



