import os
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool


class WignerMatrix:
    def __init__(self, N):
        self.n = N
        self.matrix = self._build()
        self._eigenvalues: np.ndarray | None = None
        self._eigenvectors: np.ndarray | None = None

    def _build(self):
        G = np.random.normal(0, 1, (self.n, self.n))
        return (G + G.T) / np.sqrt(2 * self.n)

    def compute_eigen(self):
        self._eigenvalues, self._eigenvectors = np.linalg.eigh(self.matrix)

        
    @property
    def eigenvalues(self) -> np.ndarray:
        if self._eigenvalues is None:  # only compute if not already cached -> WE COMPUTE IT ONCE 
            self.compute_eigen()
        assert self._eigenvalues is not None
        return self._eigenvalues
    
    @property
    def eigenvectors(self) -> np.ndarray:
        if self._eigenvectors is None:
            self.compute_eigen()
        assert self._eigenvectors is not None
        return self._eigenvectors
    
    def max_eigenvector(self):
        return self.eigenvectors[:, -1]

    def max_eigenvalue(self):
        return self.eigenvalues[-1]
    
    def theoretical_max_eigenvalue(self):
        return 2
    
    def plot_Wigner_Semi_Circle(self):
        x = np.linspace(-2, 2, 1000)
        y = np.sqrt(4 - x**2) / (2 * np.pi)
        plt.plot(x, y, label='Wigner Semi-Circle', color='black')
        return None
    
    def plot_eigenvalue_distribution(self, label = False):
        label = f'n={self.n}' if label == 0 else '_nolegend_'
        plt.hist(self.eigenvalues, bins=100, density=True, alpha=0.5, label=label)
        return None
    
    def plot_Max_Eigenval(self):
        x = self.max_eigenvalue()
        plt.axvline(float(x),label=f'Max_Eigen_Val = {x:.2f}', color='red')
        return None
    
    def ploplote(self, xlabel: str = 'Eigenvalue', ylabel: str = 'Density', title: str = 'Mettre Titre', savefig: str = 'Mettre Nom', grid: bool = False, Wigner: bool = False, Max_Eigen_val: bool = False):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if grid: plt.grid()
        if Wigner: self.plot_Wigner_Semi_Circle()
        if Max_Eigen_val: self.plot_Max_Eigenval()
        plt.legend()
        plt.title(title)
        os.makedirs('../Plots/Spike/', exist_ok=True)
        plt.savefig('../Plots/Spike/' + savefig + '.png')
        plt.show()
        return None


class SpikedWignerMatrix(WignerMatrix):
    def __init__(self, N, lamba):
        super().__init__(N)     #call the constructor of the parent class
        self.lamba = lamba
        self.base_matrix = self.matrix.copy()  #store the unspiked Wigner matrix
        self.spike = self._build_spike()   #name starts by _ that means it's a private method (only callabe when initializing)
        self.matrix = self._add_spike()     #we overwirte the one of the parent class
        self._eigenvalues = None            #reset cache eigenval to none just in case
        
    def _build_spike(self): 
        v = np.random.normal(0, 1, self.n)
        return v / np.linalg.norm(v)

    def _add_spike(self):
        return self.base_matrix + self.lamba * np.outer(self.spike, self.spike)
    
    def theoretical_max_eigenvalue(self):
        if self.lamba >= 1:
            return self.lamba + 1/self.lamba
        else:
            return 2
        
    def overlap1(self, x_hat):
        return np.abs(np.dot(x_hat, self.spike))
    
    def change_lambda(self, new_lambda):
        self.matrix = self.base_matrix + (new_lambda - self.lamba) * np.outer(self.spike, self.spike)  #rebuild from base Wigner
        self.lamba = new_lambda
        self._eigenvalues = None      # reset cache
        self._eigenvectors = None     # reset cache
        return None
    
class TwoSpikedWignerMatrix(WignerMatrix):
    def __init__(self, N, lamba_1, lamba_2, rho, alpha):
        super().__init__(N)
        self.lamba1 = lamba_1
        self.lamba2 = lamba_2
        self.rho = rho
        self.alpha = alpha
        self.Y1, self.Y2 = None, None
        self.spike1, self.spike2 = self._build_2_spikes(self.rho)
        self.matrix = self._add_two_spikes()
        self._eigenvalues, self._eigenvectors = None, None            #reset cache eigenval to None just in case


    def _build_2_spikes(self, rho):
        cov = [[1, rho],
               [rho, 1]]
        mean = [0, 0]
        X = np.random.multivariate_normal(mean, cov, self.n)
        u = X[:, 0]
        v = X[:, 1]
        u = u / np.linalg.norm(u)
        v = v / np.linalg.norm(v)
        return u, v
    
    def _add_two_spikes(self):
        u, v = self.spike1, self.spike2
        G2 = WignerMatrix(self.n)                # second independent Wigner matrix
        Y1 = self.matrix + self.lamba1 * np.outer(u, u)
        Y2 = G2.matrix + self.lamba2 * np.outer(v, v)
        self.Y1, self.Y2 = Y1, Y2
        return self.alpha * self.Y1 + np.sqrt(1 - self.alpha**2) * self.Y2
    
    def max_eigenvalue(self):       #overwrite to block
        raise NotImplementedError("max_eigenvalue is not defined for two correlated spikes")
    
    def loss(self, x_hat):
        return (self.alpha * np.linalg.norm(self.Y1 - np.outer(x_hat,x_hat))**2 + np.sqrt(1-self.alpha**2) * np.linalg.norm(self.Y2 - np.outer(x_hat,x_hat))**2)/self.n

    def gradient_descent_loss_1(self,x_hat, lr, iterations):
        for i in range(iterations):
            grad = 2 * self.alpha * (np.outer(x_hat,x_hat) - self.Y1) @ x_hat + 2 * np.sqrt(1-self.alpha**2) * (np.outer(x_hat,x_hat) - self.Y2) @ x_hat
            x_hat = x_hat - lr * grad
            x_hat = x_hat / np.linalg.norm(x_hat)
        return x_hat
    
    def overlap1(self, x_hat):
        return np.abs(np.dot(x_hat, self.spike1))
    
    def overlap2(self, x_hat):
        return np.abs(np.dot(x_hat, self.spike2))
    
    def new_alpha(self, alpha):
        self.alpha = alpha
        self.matrix = self.alpha * self.Y1 + np.sqrt(1 - self.alpha**2) * self.Y2
        return None
    
    def change_lambda1(self, new_lambda1):
        self.matrix = self.matrix + self.alpha * ( (new_lambda1 - self.lamba1) * np.outer(self.spike1, self.spike1)) 
        self.lamba1 = new_lambda1
        self._eigenvalues = None      # reset cache
        self._eigenvectors = None     # reset cache
        return None
        
    def change_lambda2(self, new_lambda2):
        self.matrix = self.matrix + np.sqrt(1 - self.alpha**2) * ( (new_lambda2 - self.lamba2) * np.outer(self.spike2, self.spike2)) 
        self.lamba2 = new_lambda2
        self._eigenvalues = None      # reset cache
        self._eigenvectors = None     # reset cache
        return None
    
    def power_iteration(self, x_hat, iterations=50, tol=1e-7):
        for _ in range(iterations):
            x_new = self.matrix @ x_hat   # single mat-vec, O(N²)
            x_new /= np.linalg.norm(x_new)
            if np.linalg.norm(x_new - x_hat) < tol:
                break
            x_hat = x_new
        return x_hat

def phase_diagram_(N, rho, alpha, lambas1, lambas2, N_gradient_descent, threshold, N_average):
    colors = []
    x_vals = []
    y_vals = []
    for lamba1 in lambas1:
        for lamba2 in lambas2:
            m_avg_1 = []
            m_avg_2 = []
            for i in range(N_average):
                M = TwoSpikedWignerMatrix(N, lamba1, lamba2, rho, alpha) 
                x_hat = np.random.normal(0, 1, N)
                x_hat = x_hat / np.linalg.norm(x_hat)
                x_hat = M.power_iteration(x_hat, N_gradient_descent)
                m_avg_1.append(M.overlap1(x_hat))
                m_avg_2.append(M.overlap2(x_hat))
            overlap_1 = np.mean(m_avg_1)
            overlap_2 = np.mean(m_avg_2)
            

            x_vals.append(lamba1)
            y_vals.append(lamba2)

            if overlap_1 > threshold and overlap_2 > threshold:
                colors.append("blue")

            elif overlap_1 < threshold and overlap_2 < threshold:
                colors.append("red")

            elif overlap_1 > threshold:
                colors.append("green")

            else:
                colors.append("pink")
    return x_vals, y_vals, colors

def ploplote_moi_ca(xlabel: str = 'Mettre axe x', ylabel: str = 'Mettre axe y', title: str = 'Mettre Titre', savefig: str = 'Mettre Nom', grid: bool = False, legend: bool = False):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if grid: plt.grid()
        if legend:plt.legend()
        plt.title(title)
        os.makedirs('../Plots/Spike/', exist_ok=True)
        plt.savefig('../Plots/Spike/' + savefig + '.png')
        plt.show()
        return None

def phase_diagram(N, rho, alpha, lambas1, lambas2, N_gradient_descent, threshold, N_average):
    colors = []
    x_vals = []
    y_vals = []
    for lamba1 in lambas1:
        for lamba2 in lambas2:
            votes_1 = 0
            votes_2 = 0
            for i in range(N_average):
                M = TwoSpikedWignerMatrix(N, lamba1, lamba2, rho, alpha)
                x_hat = np.random.normal(0, 1, N)
                x_hat /= np.linalg.norm(x_hat)
                x_hat = M.power_iteration(x_hat, N_gradient_descent)
                if M.overlap1(x_hat) > threshold:
                    votes_1 += 1
                if M.overlap2(x_hat) > threshold:
                    votes_2 += 1

            recovered_1 = votes_1 > N_average / 2
            recovered_2 = votes_2 > N_average / 2

            x_vals.append(lamba1)
            y_vals.append(lamba2)

            if recovered_1 and recovered_2:
                colors.append("blue")
            elif not recovered_1 and not recovered_2:
                colors.append("red")
            elif recovered_1:
                colors.append("green")
            else:
                colors.append("pink")
    return x_vals, y_vals, colors

def _evaluate_point(args):
    N, lamba1, lamba2, rho, alpha, N_gradient_descent, threshold, N_average = args
    votes_1 = 0
    votes_2 = 0
    for i in range(N_average):
        M = TwoSpikedWignerMatrix(N, lamba1, lamba2, rho, alpha)
        x_hat = np.random.normal(0, 1, N)
        x_hat /= np.linalg.norm(x_hat)
        x_hat = M.power_iteration(x_hat, N_gradient_descent)
        if M.overlap1(x_hat) > threshold:
            votes_1 += 1
        if M.overlap2(x_hat) > threshold:
            votes_2 += 1
    recovered_1 = votes_1 > N_average / 2
    recovered_2 = votes_2 > N_average / 2
    if recovered_1 and recovered_2:
        return lamba1, lamba2, "blue"
    elif not recovered_1 and not recovered_2:
        return lamba1, lamba2, "red"
    elif recovered_1:
        return lamba1, lamba2, "green"
    else:
        return lamba1, lamba2, "pink"

def phase_diagram_parallel(N, rho, alpha, lambas1, lambas2, N_gradient_descent, threshold, N_average):
    args = [
        (N, l1, l2, rho, alpha, N_gradient_descent, threshold, N_average)
        for l1 in lambas1
        for l2 in lambas2
    ]
    with Pool() as pool:
        results = pool.map(_evaluate_point, args)

    x_vals = [r[0] for r in results]
    y_vals = [r[1] for r in results]
    colors = [r[2] for r in results]
    return x_vals, y_vals, colors



def phase_diagram_continuous(N, rho, alpha, lambas1, lambas2, N_gradient_descent, N_average):
    """Returns raw overlap grids instead of color strings."""
    n1, n2 = len(lambas1), len(lambas2)
    overlap1_grid = np.zeros((n1, n2))
    overlap2_grid = np.zeros((n1, n2))

    for i, lamba1 in enumerate(lambas1):
        for j, lamba2 in enumerate(lambas2):
            o1_acc, o2_acc = 0.0, 0.0
            for _ in range(N_average):
                M = TwoSpikedWignerMatrix(N, lamba1, lamba2, rho, alpha)
                x_hat = np.random.normal(0, 1, N)
                x_hat /= np.linalg.norm(x_hat)
                x_hat = M.power_iteration(x_hat, N_gradient_descent)
                o1_acc += M.overlap1(x_hat)
                o2_acc += M.overlap2(x_hat)
            overlap1_grid[i, j] = o1_acc / N_average
            overlap2_grid[i, j] = o2_acc / N_average

    return overlap1_grid, overlap2_grid