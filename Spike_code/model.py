import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

class SimpleSpike:
    def __init__(self, N, lam):
        self.N = N
        self.lam = lam
        self.x = np.random.normal(0, 1, N)
        self.spike = self.random_spike(self.x, lam)
        self.Y = self.spike + self.symmetric_gaussian_matrix(N)
        self.x_hat = self.power_iteration(self.Y, N, iterations=50, tol=1e-7)

 
    def symmetric_gaussian_matrix(n):
        G = np.random.normal(0, 1, (n, n))
        return (G + G.T)/np.sqrt(2*n)

    def random_spike(v, k):
        v = v /np.linalg.norm(v)  # normalize to unit vector
        return k * np.outer(v, v)

    def power_iteration(M, N, iterations=50, tol=1e-7):
        x_hat = np.ones(N)
        for _ in range(iterations):
            x_new = M @ x_hat   # single mat-vec, O(N²)
            x_new /= np.linalg.norm(x_new)
            if np.linalg.norm(x_new - x_hat) < tol:
                break
            x_hat = x_new
        return x_hat
    
    def x(self):
        return self.x
    
    def Y(self):
        return self.Y
    
    def x_hat(self):
        return self.xhat



class TwoSpikes:
    def __init__(self, N, lam1, lam2, rho, mu = 1, alpha = np.sqrt(0.5)):
        self.N = N
        self.lam1 = lam1
        self.lam2 = lam2
        self.rho = rho
        self.alpha = alpha
        self.mu = mu
        self.Y1, self.Y2, self.x1, self.x2 = self.two_correlated_spikes()


        self.theta = {
            "1": None,
            "naive": None,
            "A": None,
            "B": None
        }

        self.naive_matrix = self.alpha * self.Y1 + np.sqrt(1 - self.alpha**2) * self.Y2
   
    def get_theta(self, method):
        return self.theta[method]
    
    def get_x1(self):
        return self.x1
    
    def Hess(self, theta):
        H = self.Y2 - np.eye(self.N) + 2 * self.mu * self.fisher_MS(theta, self.lam1)
        return H
    
    def two_correlated_spikes(self):    
        cov = [[1, self.rho],
            [self.rho, 1]]

        mean = [0, 0]

        X = np.random.multivariate_normal(mean, cov, self.N)
        x1 = X[:, 0]
        x2 = X[:, 1]
        x1 = x1 / np.linalg.norm(x1)
        x2 = x2 / np.linalg.norm(x2)
        Y1 = self.lam1 * np.outer(x1, x1) + self.symmetric_gaussian_matrix()
        Y2 = self.lam2 * np.outer(x2, x2) + self.symmetric_gaussian_matrix()
        return Y1, Y2, x1, x2

    def symmetric_gaussian_matrix(self):
        n = self.N
        G = np.random.normal(0, 1, (n, n))
        return (G + G.T)/np.sqrt(2*n)
    
    def power_iteration(self, M, iterations=50, tol=1e-7):
        x_hat = np.ones(self.N)
        for _ in range(iterations):
            x_new = M @ x_hat   # single mat-vec, O(N²)
            x_new /= np.linalg.norm(x_new)
            if np.linalg.norm(x_new - x_hat) < tol:
                break
            x_hat = x_new
        return x_hat
    
    def set_mu(self, mu):
        self.mu = mu
        return self.mu
    
    def compute_method(self, method):
        if "1" in method:
            self.compute_theta1()
        if "naive" in method:
            self.compute_naive()
        if "A" in method:
            self.compute_methodA()
        if "B" in method:
            self.compute_methodB()

    def compute_theta1(self):
        self.theta["1"] = self.power_iteration(self.Y1, iterations=50, tol=1e-7)
        return self.theta["1"]

    def compute_naive(self):
        self.theta["naive"] = self.power_iteration(self.naive_matrix,  iterations=50, tol=1e-7)
        return self.theta["naive"]
    
    def compute_methodA(self):
        self.theta["A"], _ = self.x_fisher(method = "A", solve=True)
    
    def compute_methodB(self):
        _, self.theta["B"]  = self.x_fisher(method = "B", solve=True)
    

    def fisher_MS(self, x, lam):  
        N = self.N
        return ((lam - 1)**2 + 1/N)*np.outer(x, x) + np.eye(N)/N


    def x_fisher(self, method, solve = True):
        mu = self.mu
        N = self.N
        self.compute_theta1()
        F1 = self.fisher_MS(self.theta["1"], self.lam1)
        A = self.Y2 - np.eye(self.N) + 2*mu*F1

        thetaA = np.zeros(N)
        thetaB = np.zeros(N)

        if "A" in method:
            thetaA = self.power_iteration(A)

        if "B" in method:
            b = 2*mu * F1 @ self.theta["1"]
            if solve:
                lam = fsolve(norm_constraint, x0=-4, args=(A, b, self.N))[0]

                thetaB = np.linalg.solve(
                    A - lam*np.eye(self.N),
                    b
                )
                # print(np.linalg.norm(thetaB))
                # print(np.linalg.norm(thetaB)-1)
                # thetaB /= np.linalg.norm(thetaB)
            else: 
                if np.linalg.cond(A) < 1e12 :
                    thetaB = np.linalg.inv(A -np.eye(N)) @ b
                else :
                    print( " det A = 0")
                    print( "cond : " , np.linalg.cond(A) )
  
        return thetaA, thetaB
    
    ## OVERLAP/test methodes
    def overlaps(self, method) :
        overlap1 = abs(np.dot(self.theta[method],self.x1))
        overlap2 = abs(np.dot(self.theta[method],self.x2)) 

        return overlap1, overlap2
    
    def Loss_eval(self, method):
        theta = self.theta[method]
        L = self.alpha*np.linalg.norm(self.Y1 - np.outer(theta, theta), 'fro')**2 + np.sqrt(1-self.alpha**2)*np.linalg.norm(self.Y2 - np.outer(theta, theta), 'fro')**2
        o1, o2 = self.overlaps(method)
        L = o1 + o2
        return -L

## NAIVE 



## Fisher
def norm_constraint(lam, A, b, N):
    theta = np.linalg.solve(
        A - lam*np.eye(N),
        b
    )
    return np.linalg.norm(theta) - 1


#comparaison est theo fisher
def ecart_F_xhat_x1(N, rho, lambda1, lambda2, M=10):
    norm = []
    for _ in range (M):
        S = TwoSpikes(N, lambda1, lambda2, rho)
        x1 = S.get_x1()
        theta1 = S.compute_theta1()

        Fhat = S.fisher_MS(theta1, lambda1)
        F = S.fisher_MS(x1, lambda1)

        norm += [np.linalg.norm(Fhat - F, 'fro')/np.linalg.norm(F, 'fro')]
    norm = np.array(norm)
    return np.mean(norm), np.std(norm)/np.sqrt(M)


def analyse_Hess_Fisher(N, rho, lambda1, lambda2, mu,
                        estim=False, spectre=True):

    S = TwoSpikes(N, lambda1, lambda2, rho, mu)

    if estim:
        theta1 = S.compute_theta1()
    else:
        theta1 = S.get_x1

    H = S.Hess(theta1)
    F1 = S.fisher_MS(theta1, lambda1)

    detF1 = np.linalg.det(F1)
    detH = np.linalg.det(H)

    if spectre:

        eigF1 = np.linalg.eigvals(F1)
        eigH = np.linalg.eigvals(H)

        # # ==========================================
        # # Figure Fisher
        # # ==========================================

        plt.figure(figsize=(6, 5))

        plt.hist(eigF1, bins=100, density=False, alpha=0.5)

        plt.vlines(
            max(eigF1),
            ymin=0,
            ymax=N,
            linestyles='-',
            colors='blue',
            label=r'Top eigenvalue'
        )

        plt.vlines(
            (lambda1 - 1)**2,
            ymin=0,
            ymax=N,
            linestyles='--',
            colors='red',
            label=r'$(1-\lambda_1)^2$'
        )
        plt.ylabel("Count",fontsize=13)
        plt.xlabel("Eigenvalues",fontsize=13)
        plt.grid(alpha=0.6)
        plt.legend(loc='upper center',fontsize=13)

        plt.show()

        # ==========================================
        # Figure Hessian
        # ==========================================

        plt.figure(figsize=(6, 5))

        plt.hist(eigH, bins=100, density=False, alpha=0.5)

        plt.vlines(
            (lambda2 - 1) + 1/lambda2,
            ymin=0,
            ymax=N/100,
            linestyles='solid',
            colors='blue',
            label=r'$\lambda_2 + \frac{1}{\lambda_2} - 1$'
        )

        plt.vlines(
            2 * mu * (1 - lambda1)**2
            + 1/(2 * mu * (1 - lambda1)**2)
            - 1,

            ymin=0,
            ymax=N/100,
            linestyles='--',
            colors='red',
            label=r'$\lambda_{1,eff} + \frac{1}{\lambda_{1,eff}} - 1$'
        )
# 2\mu(1-\lambda_1)^2

        plt.ylabel("Count",fontsize=13)
        plt.xlabel("Eigenvalues",fontsize=13)
        plt.grid(alpha=0.6)
        plt.legend(fontsize=13)

        plt.show()

    return detF1, detH


# MODEL/estimateurs

def run_experiment_2D(vary_param_x, vary_values_x, vary_param_y, vary_values_y, 
                       M = 10, n = 100 , k1 = 2, k2 = 3, rho = 0.6, 
                       alpha = np.sqrt(0.5), mu = 0,
                       method = ["naive"], plot = True, mu_opt = True):
    
    
    Z1 = {m: np.zeros((len(vary_values_y), len(vary_values_x))) for m in method}
    Z2 = {m: np.zeros((len(vary_values_y), len(vary_values_x))) for m in method}

    for i, val_y in enumerate(vary_values_y):
        for j, val_x in enumerate(vary_values_x):
            params = {'n': n, 'k1': k1, 'k2': k2, 'rho': rho, 'alpha': alpha, 'mu' : mu}
            params[vary_param_x] = val_x
            params[vary_param_y] = val_y
            # if mu_opt:
            #     if params['k2'] != 1:
            #         params['mu'] = params['k2']/(2*(1 - params['k1'])**2)
            
            overlaps = {m: {"m1": [], "m2": []} for m in method}

            for _ in range(M):
                S= TwoSpikes(params['n'], params['k1'],  params['k2'], params['rho'])
                if mu_opt:
                    mu_opt, _, _ = cv_mu
                if "naive" in method: 
                    S.compute_naive()
                    m1, m2= S.overlaps("naive")
                    overlaps["naive"]["m1"].append(m1)
                    overlaps["naive"]["m2"].append(m2)
                if "A" in method:
                    S.compute_theta1()
                    S.compute_methodA()
                    m1, m2= S.overlaps("A")
                    overlaps["A"]["m1"].append(m1)
                    overlaps["A"]["m2"].append(m2)
                if "B" in method: 
                    m1, m2 = 0  
                    overlaps["B"]["m1"].append(m1)
                    overlaps["B"]["m2"].append(m2)
                
            for m in method:
                Z1[m][i, j] = np.mean(overlaps[m]["m1"])
                Z2[m][i, j] = np.mean(overlaps[m]["m2"])
            
                
    if plot:
        fig, axes = plt.subplots(2, len(method), figsize=(5 * len(method), 8))

        if len(method) == 1:
            axes = axes.reshape(2, 1)

        titles = ["|theta·x1|", "|theta·x2|"]

        for col, m in enumerate(method):
            for row, Z, title in zip(range(2), [Z1[m], Z2[m]], titles):
                ax = axes[row, col]

                im = ax.imshow(
                    Z,
                    origin='lower',
                    aspect='auto',
                    cmap='viridis',
                    extent=[vary_values_x[0], vary_values_x[-1],
                            vary_values_y[0], vary_values_y[-1]],
                    vmin=0, vmax=1
                )

                fig.colorbar(im, ax=ax)
                ax.set_title(f"{m} - {title}")

        plt.tight_layout()
        plt.show()

            # fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            # axes = axes.flatten()
            # titles = ['max over x1', 'max over x2']
            # Zs = [np.maximum(Z11,Z12), np.maximum(Z22,Z21)]

        # for ax, Z, title in zip(axes, Zs, titles):
        #     im = ax.imshow(Z, origin='lower', aspect='auto', cmap='viridis',
        #                 extent=[vary_values_x[0], vary_values_x[-1], 
        #                         vary_values_y[0], vary_values_y[-1]],vmin=0, vmax=1)
        #     fig.colorbar(im, ax=ax, label='Overlap')
        #     ax.set_xlabel(vary_param_x, fontsize=12)
        #     ax.set_ylabel(vary_param_y, fontsize=12)
        #     ax.set_title(title, fontsize=14)
        # plot f(k1) = sqrt( (k2/k1)^2 / (1 + (k2/k1)^2) )
        #    k1_vals = vary_values_x[vary_values_x > 1]
        #    k2=2
            
        #   f_vals = np.sqrt((k2 / k1_vals)**2 / (1 + (k2 / k1_vals)**2))
        #    ax.plot(k1_vals, f_vals, color='red', linewidth=2, label=r'$f(k_1)$')
        #    ax.legend(fontsize=12)
        # plt.tight_layout()
        # plt.show()
    

    return Z1, Z2


def cv_mu(MU, N, rho, lambda1, lambda2, M= 10,  method = "A", plot = False):
    scores = []
    scores_std = []
    for mu in MU:
        score = []
        for _ in range(M): 
            S = TwoSpikes(N, lambda1, lambda2, rho, mu = mu)

            if method == "A":
                S.compute_theta1()
                S.compute_methodA()
            score += [S.Loss_eval(method)]
        score = np.array(score)
        scores += [np.mean(score)]
        scores_std += [np.std(score)/np.sqrt(M)]
    mu_opt = MU[np.argmin(scores)]

    if plot:
        plt.errorbar(MU, scores, yerr = scores_std, label = "Loss_EWC")
        plt.axvline(mu_opt, linestyle=":", label="mu_opt")

        plt.xlabel("mu")
        plt.ylabel("Loss EWC")
        plt.legend()
        plt.title("Cross-validation on mu")
        plt.show()

    return mu_opt, scores, scores_std








