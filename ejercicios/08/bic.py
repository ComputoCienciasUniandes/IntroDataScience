import numpy as np
import matplotlib.pyplot as plt

def model_A(x, params):
    y = params[0] + x*params[1] + params[2]*x**2
    return y

def model_B(x, params):
    y = params[0]*(np.exp(-0.5*(x-params[1])**2/params[2]**2))
    return y

def model_C(x, params):
    y = params[0]*(np.exp(-0.5*(x-params[1])**2/params[2]**2))
    y += params[0]*(np.exp(-0.5*(x-params[3])**2/params[4]**2))
    return y

def logprior_model_A(params):
    return 0
    
def logprior_model_B(params):
    if params[2]<0:
        return -np.inf
    else:
        return 0
    
def logprior_model_C(params):
    if params[2]<0 or params[4]<0 or params[1]<0 or params[3]<0:
        return -np.inf
    else:
        return 0
    
def loglike(x_obs, y_obs, sigma_y_obs, model, params):
    l = np.sum(-0.5*(y_obs -model(x_obs, params))**2/sigma_y_obs**2)
    return l
def run_mcmc(x_obs, y_obs, sigma_y_obs, 
             model, logprior, n_params=3, n_iterations=20000, scale=0.1):

    params = np.ones([n_iterations, n_params])
    loglike_values = np.ones(n_iterations)*5.0
    loglike_values[0] = loglike(x_obs, y_obs, sigma_y_obs, model, params[0,:])
    for i in range(1, n_iterations):
        current_params = params[i-1,:]
        next_params = current_params + np.random.normal(scale=scale, size=n_params)
        
        loglike_current = loglike(x_obs, y_obs, sigma_y_obs, model, current_params) + logprior(current_params)
        loglike_next = loglike(x_obs, y_obs, sigma_y_obs, model, next_params) + logprior(next_params)
            
        r = np.min([np.exp(loglike_next - loglike_current), 1.0])
        alpha = np.random.random()

        if alpha < r:
            params[i,:] = next_params
            loglike_values[i] = loglike_next
        else:
            params[i,:] = current_params
            loglike_values[i] = loglike_current
        
    params = params[n_iterations//2:,:]
    loglike_values = loglike_values[n_iterations//2:]
    return {'params':params, 'x_obs':x_obs, 'y_obs':y_obs, 'loglike_values':loglike_values}

def BIC(params_model):
    max_loglike = np.max(params_model['loglike_values'])
    n_dim = np.shape(params_model['params'])[1]
    n_points = len(params_model['y_obs'])
    return 2.0*(-max_loglike + 0.5*n_dim*np.log(n_points))

def plot_model(params_model, model, model_name):
    n_dim = np.shape(params_model['params'])[1]
    n_points = len(params_model['y_obs'])
    
    plt.figure(figsize = (4*(n_dim//2+1),6))
    for i in range(n_dim):
        plt.subplot(2, n_dim//2+1, i+1)
        plt.hist(params_model['params'][:,i], density=True, bins=30)
        plt.title(r"$\beta_{}= {:.2f}\pm {:.2f}$".format(i,np.mean(params_model['params'][:,i]), np.std(params_model['params'][:,i])))
        plt.xlabel(r"$\beta_{}$".format(i))
        
    plt.subplot(2,n_dim//2+1, i+2)
    best = np.mean(params_model['params'], axis=0)
    x = params_model['x_obs']
    x_model = np.linspace(x.min(), x.max(), 100)
    y_model = model(x_model, best)
    plt.plot(x_model, y_model)
    plt.errorbar(x, y, yerr=sigma_y, fmt='o')
    plt.title("BIC={:.2f}".format(BIC(params_model)))
    
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(model_name+".png")
    

# Lee los datos
data = np.loadtxt('data_to_fit.txt')
x = data[:,0]
y = data[:,1]
sigma_y = data[:,2]

# ajusta los modelos
params_model_A = run_mcmc(x, y, sigma_y, model_A, logprior_model_A, scale=0.05, n_iterations=1000000)
params_model_B = run_mcmc(x, y, sigma_y, model_B, logprior_model_B, scale=0.05, n_iterations=1000000)
params_model_C = run_mcmc(x, y, sigma_y, model_C, logprior_model_C, n_params=5, scale=0.05, n_iterations=1000000)

# hace las graficas
plot_model(params_model_A, model_A, 'model_A')
plot_model(params_model_B, model_B, 'model_B')
plot_model(params_model_C, model_C, 'model_C')



