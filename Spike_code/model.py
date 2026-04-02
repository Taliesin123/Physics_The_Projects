import numpy as np
import matplotlib.pyplot as plt

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
    return S1, S2

def spiked_gaussian_matrix_with_correlated_spikes(n, k1, k2, rho):
    G1, G2 = symmetric_gaussian_matrix(n), symmetric_gaussian_matrix(n)
    S1, S2 = two_correlated_spikes(n, k1, k2, rho)
    Y1 = G1 + S1
    Y2 = G2 + S2
    return Y1, Y2

def spectral_matrix(Y1, Y2, alpha):
    return alpha * Y1 + np.sqrt(1 - alpha**2) * Y2

def spec(n, k1, k2, rho, alpha):
    Y1, Y2 = spiked_gaussian_matrix_with_correlated_spikes(n, k1, k2, rho)
    return spectral_matrix(Y1, Y2, alpha)

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


def plot_Max_Eigenval(lamba = 1):
    x = BBP(lamba)
    plt.axvline(x,label=f'Max_Eigen_Val = {x:.2f}', color='red')
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
    lambas = np.linspace(1, 6, 50)
    x = []
    std = []
    x_axis = np.linspace(1, 6, 1000)
    y_BBP = BBP(x_axis)
    plt.plot(x_axis, y_BBP, label = 'Max(2, lambda + 1/lambda)')
    plt.plot(x_axis, x_axis, label = 'y = x')
    for lamba in lambas:
        avg = []
        for i in range(50):
            avg.append(max_eigen_val(spiked_gaussian_matrix(N, lamba)))
        x.append(np.mean(avg))
        std.append(np.std(avg))
    plt.scatter(lambas, x, label = 'Max_Eigen_val', color = 'red', marker='.')
    plt.errorbar(lambas, std, color = 'blue')
    plt.ylabel('Eigenvalue')
    plt.xlabel('Lambda')
    plt.legend()
    plt.axvline(2, color = 'black')
    plt.savefig('../Plots/Spike/BBP_transi_obs.png')
    plt.title('Maximum Eigenvalue as a function of Lambda')
    plt.show()
