import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def gaussian_matrix(n):
    return np.random.normal(0, 1, (n, n))

def symmetric_gaussian_matrix(n):
    G = gaussian_matrix(n)
    return (G + G.T)/np.sqrt(2*n)

def random_spike(n, k):
    v = np.random.normal(0, 1, n)
    v = v /np.linalg.norm(v)  # normalize to unit vector
    return k * np.outer(v, v)

def spiked_gaussian_matrix(n, k):
    G = symmetric_gaussian_matrix(n)
    S = random_spike(n, k)
    return G + S

def two_correlated_spikes(n, k1, k2, rho):    
    cov = [[1, rho],
           [rho, 1]]

    mean = [0, 0]

    X = np.random.multivariate_normal(mean, cov, n)
    u = X[:, 0]
    v = X[:, 1]
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    S1 = k1 * np.outer(u, u)
    S2 = k2 * np.outer(v, v)
    return S1, S2, u, v

def spiked_gaussian_matrix_with_correlated_spikes(n, k1, k2, rho):
    G1, G2 = symmetric_gaussian_matrix(n), symmetric_gaussian_matrix(n)
    S1, S2, x1, x2 = two_correlated_spikes(n, k1, k2, rho)
    Y1 = G1 + S1
    Y2 = G2 + S2
    return Y1, Y2, x1, x2

def spectral_matrix(Y1, Y2, alpha):
    return alpha * Y1 + np.sqrt(1 - alpha**2) * Y2

def spec(n, k1, k2, rho, alpha):
    Y1, Y2, x1, x2 = spiked_gaussian_matrix_with_correlated_spikes(n, k1, k2, rho)
    return spectral_matrix(Y1, Y2, alpha), x1, x2

def plot_Wigner_Semi_Circle():
    x = np.linspace(-2, 2, 1000)
    y = np.sqrt(4 - x**2) / (2 * np.pi)
    plt.plot(x, y, label='Wigner Semi-Circle', color='black')
    return None

def plot_eigenvalue_distribution(M, N, label = False):
    eigenvalues = np.linalg.eigvals(M)
    label = f'n={N}' if label == 0 else '_nolegend_'
    plt.hist(eigenvalues, bins=100, density=True, alpha=0.5, label=label)
    return None


def plot_Max_Eigenval(lamba = 1, colors = ['red'], i = 0):
    x = BBP(lamba)
    plt.axvline(x,label=fr'Max_Eigen_Val for $\lambda_{i}$ = {x:.2f}', color = colors[i])
    return None

def show():
    plt.xlabel('Eigenvalue')
    plt.ylabel('Density')
    plt.show()
    return None

def BBP(lamba):
        return np.where(lamba >= 1, lamba + 1/lamba, 2)


def max_eigen_val(M):
    eigenvalues = np.linalg.eigvals(M)
    return np.max(eigenvalues)

def first_look_at_BBP_transi(N = 500):
    lambas = np.linspace(0.5, 6, 50)
    x = []
    std = []
    x_axis = np.linspace(0.5, 6, 1000)
    y_BBP = BBP(x_axis)
    plt.plot(x_axis, x_axis, label = 'y = x')
    for lamba in lambas:
        avg = []
        for i in range(50):
            avg.append(max_eigen_val(spiked_gaussian_matrix(N, lamba)))
        x.append(np.mean(avg))
        std.append(np.std(avg))
    plt.errorbar(lambas, x, yerr=std, label = 'Max_Eigen_val', color = 'red', marker='.', capsize=2, zorder = 1)
    plt.plot(x_axis, y_BBP, label = 'Theorical BBP transition', color = 'green', linestyle='--', zorder = 2)
    #plt.scatter(lambas, x, label = 'Max_Eigen_val', color = 'red', marker='.')
    plt.ylabel('Eigenvalue')
    plt.xlabel(f'$\lambda$')
    plt.axvline(1, color = 'black', label = r'BBP transition at $\lambda = 1$')
    plt.title(r'Maximum Eigenvalue as a function of $\lambda$')
    plt.legend()
    plt.grid()

    plt.savefig('../Plots/Spike/BBP_transi_obs.png')
    plt.show()


def overlap(D,x1,x2) :
    eigenvalues, eigenvect = np.linalg.eig(D)

    idx = np.argsort(eigenvalues)[::-1]

    v1 = eigenvect[:, idx[0]] 
    v2 = eigenvect[:, idx[1]]  

    overlap11 = abs(np.dot(v1,x1))
    overlap22 = abs(np.dot(v2,x2)) 
    overlap12 = abs(np.dot(v2,x1)) 
    overlap21 = abs(np.dot(v1,x2)) 
    return overlap11, overlap22, overlap12, overlap21


def overlap_2spike_varyparam(vary_param, vary_values, M, n, k1, k2, rho, alpha, xlabel=None):

    M1, M2, M2bis = [], [], []
    
    for val in vary_values:
        params = {'n': n, 'k1': k1, 'k2': k2, 'rho': rho, 'alpha': alpha}
        params[vary_param] = val
        
        over1, over2, over2bis = [], [], []
        for _ in range(M):
            D, x1, x2 = spec(params['n'], params['k1'], params['k2'], params['rho'], params['alpha'])
            m1, m2, m2bis = overlap(D, x1, x2)
            over1.append(m1)
            over2.append(m2)
            over2bis.append(m2bis)
        
        M1.append([np.mean(over1), np.std(over1)/np.sqrt(M)])
        M2.append([np.mean(over2), np.std(over2)/np.sqrt(M)])
        M2bis.append([np.mean(over2bis), np.std(over2bis)/np.sqrt(M)])
    
    M1, M2, M2bis = np.array(M1), np.array(M2), np.array(M2bis)
    
  
    sns.set_theme(style="whitegrid", context="paper")
    plt.errorbar(vary_values, M1[:,0], yerr=M1[:,1], label="x1.v1", capsize=2, linewidth=1)
    plt.errorbar(vary_values, M2[:,0], yerr=M2[:,1], label="x2.v2", capsize=2, linewidth=1)
    plt.errorbar(vary_values, M2bis[:,0], yerr=M2bis[:,1], label="x2.v1", capsize=2, linewidth=1)
    plt.ylabel("Overlaps", fontsize=16)
    plt.xlabel(xlabel if xlabel else vary_param, fontsize=16)
    plt.grid(alpha=0.5)
    plt.legend(frameon=True, fontsize=16)
    plt.tight_layout()
    plt.tick_params(axis='both', labelsize=14)
    plt.show()
    
    return M1, M2, M2bis