import numpy as np
import matplotlib.pyplot as plt
import model as md
import model_2 as m
import importlib
importlib.reload(m)

N = 500
rhos = np.linspace(0, 1, 100)
lamba_1, lamba_2 = 5, 7
y1all, y2all = [], []
y1err, y2err = [], []

for rho in rhos:
    a = []
    b = []
    for i in range(50):
        M = m.TwoSpikedWignerMatrix(N, lamba_1, lamba_2, rho, np.sqrt(0.5))
        x1, x2 = M.double_max_eigenvalue()
        a.append(x1)
        b.append(x2)
    
    y1all.append(np.mean(a))
    y2all.append(np.mean(b))
    y1err.append(np.std(a))
    y2err.append(np.std(b))

plt.errorbar(rhos, y1all, yerr=y1err, label=r'$\lambda_1$', capsize=2)
plt.errorbar(rhos, y2all, yerr=y2err, label=r'$\lambda_2$', capsize=2)
plt.xlabel(r'$\rho$')
plt.xlim(0, 1)
plt.axhline(2, color='red', linestyle='--', label=r'BBP threshold')
plt.ylabel('Top two eigenvalues')
plt.title(rf'Top two eigenvalues of Two Spiked Wigner Matrix with $\alpha$ = {np.sqrt(0.5):.2f}')
plt.legend()
plt.grid()
plt.savefig(f'./Plots/Spike/Two_Spiked_Wigner_Top_Eigenvalues_alpha_{np.sqrt(0.5):.2f}.png', dpi=150)
plt.show()
#M.ploplote(title = rf'Eigenvalue Distribution of Two Spiked Wigner Matrix with $\alpha$ = {np.sqrt(0.5):.2f}', savefig = f'Two_Spiked_Wigner_alpha_{np.sqrt(0.5):.2f}', Wigner=True, grid=True)

