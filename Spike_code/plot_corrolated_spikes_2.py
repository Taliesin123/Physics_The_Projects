import numpy as np
import matplotlib.pyplot as plt
import model as md
import model_2 as m
import importlib
importlib.reload(m)
plt.rcParams.update({'font.size': 17})


N = 500
rho = 0.2
lamba_1, lamba_2 = 5, 7
y1all, y2all = [], []
y1err, y2err = [], []
alphas = np.linspace(0, 1, 100)

for alpha in alphas:
    a = []
    b = []
    for i in range(50):
        M = m.TwoSpikedWignerMatrix(N, lamba_1, lamba_2, rho, alpha)
        x1, x2 = M.double_max_eigenvalue()
        a.append(x1)
        b.append(x2)
    
    y1all.append(np.mean(a))
    y2all.append(np.mean(b))
    y1err.append(np.std(a))
    y2err.append(np.std(b))

plt.errorbar(alphas, y1all, yerr=y1err, label='Top Eigenvalue', capsize=2)
plt.errorbar(alphas, y2all, yerr=y2err, label='2nd Eigenvalue', capsize=2)
plt.xlabel(r'$\alpha$')
plt.xlim(0, 1)
plt.axhline(2, color='red', linestyle='--', label=r'BBP threshold')
plt.ylabel('Top two eigenvalues')
#plt.title(rf'Top two eigenvalues of Two Spiked Wigner Matrix with $\rho$ = 0.2')
plt.legend(loc="upper left",
                framealpha=0.8,
                borderaxespad=0,)
plt.grid()
plt.savefig(f'./Plots/Spike/Two_Spiked_Wigner_Top_Eigenvalues_rho_0.2.png', dpi=150)
plt.show()

