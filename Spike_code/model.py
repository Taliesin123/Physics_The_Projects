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

def plotWigner():
    x = np.linspace(-2, 2, 1000)
    y = np.sqrt(4 - x**2) / (2 * np.pi)
    plt.plot(x, y, label='Wigner Semi-Circle', color='black')   
    return None