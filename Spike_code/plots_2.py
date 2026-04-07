import numpy as np
import matplotlib.pyplot as plt
import model_2 as m

N = 500
alphas = np.linspace(0.2, 0.8, 4)  
lambas1 = np.linspace(0.1, 7, 20)
lambas2 = np.linspace(0.1, 7, 20)
rhos = np.linspace(0.1, 0.9, 9)
N_gradient_descent = 1000
threshold = 0.1
N_average = 20

if __name__ == '__main__':
    for rho in rhos:
        for alpha in alphas:
            x_vals, y_vals, colors = m.phase_diagram_parallel(N, rho, alpha, lambas1, lambas2, N_gradient_descent, threshold, N_average)
            plt.scatter(x_vals, y_vals, c=colors, s=60, marker='.')
            m.ploplote_moi_ca(xlabel=r'$\lambda_1$', ylabel=r'$\lambda_2$', title=rf'Phase Diagram for $\rho$ = {rho:.2f} and $\alpha$ = {alpha:.2f}', savefig=f'Phase_Diagram_rho={rho:.2f}_and_alpha={alpha:.2f}', grid=False, legend=False)