
import timeit
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))     

from utils import *
import numpy as np


nx = 100
ny = 50

sol = Sol((nx,ny))
solveur = Solveur(sol)
densiteCourant = 0.01
sol.placerElectrode(46, 1, densiteCourant)
sol.placerElectrode(54, 1, -densiteCourant)

omega= 0.96
solveur.calculerPotentiel(omega=omega)


nxs = []
durations = []

for i in range(200):
    x = 50 + (i)
    nxs.append(x)
    sol = Sol((x,x))
    solveur = Solveur(sol)
    sol.placerElectrode(x//2-4, 1, densiteCourant)
    sol.placerElectrode(x//2+4, 1, -densiteCourant)
    duration = timeit.timeit(lambda: solveur.calculerPotentiel(omega=omega), number=1)
    print(duration)
    durations.append(duration)



nxs = np.array(nxs)
durations = np.array(durations)*1000






def power_law(x, k, p):
    return k * x**p

# Use robust loss function to ignore outliers
params, _ = curve_fit(
    power_law, nxs, durations,
    p0=[1e-3, 1.0],          # initial guess
    maxfev=10000
)
k, p = params

print(f"Estimated parameters: k = {k:.4e}, p = {p:.2f}")
Nfit = np.linspace(nxs[0], nxs[-1], 200)



fontsize = 20
plt.figure(figsize=(10, 6))
plt.plot(nxs, durations, linewidth=4, color='blue', label='Données')
plt.plot(Nfit, k*Nfit**p, '--', color='black', linewidth=4, 
         label=f'Fit: $O(N^{{{p:.2f}}})$')
plt.xlabel('Nombre de pixels (N x N)', fontsize=fontsize)
plt.ylabel('Temps de calcul (ms)', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.legend(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.grid()
plt.tight_layout()
plt.show()





