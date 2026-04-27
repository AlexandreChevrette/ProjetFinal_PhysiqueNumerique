

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
solveur.calculerPotentiel() # caller une fois pour activer JIT


nx = 100
ny = 50

omegas = np.linspace(0.9, 0.99, 20)
nbIterations = 100
durations = []

def calculerPotentielOmega(solver, sol, omega):
    solver.calculerPotentiel(omega=omega)
    sol.matricePotentiel = np.zeros_like(sol.matricePotentiel)  # Reset du potentiel pour éviter les effets de cache entre les itérations 

for omega in omegas: 
    sol = Sol((nx, ny))
    solveur = Solveur(sol)
    sol.placerElectrode(46, 1, densiteCourant)
    sol.placerElectrode(54, 1, -densiteCourant)
    duration = timeit.timeit(lambda: calculerPotentielOmega(solveur, sol, omega), number=nbIterations)
    durations.append(duration / nbIterations)
    print(f"Omega: {omega:.2f}, Temps de calcul moyen: {duration / nbIterations:.4f} secondes")

durations = np.array(durations) * 1000

fontsize = 20
plt.figure(figsize=(10, 6))
plt.scatter(omegas, durations, linewidth=4, color='blue', label='Données')
plt.xlabel('Paramètre de surrelaxation ω', fontsize=fontsize)
plt.ylabel(r'$T_{moy}$ 100 itérations (ms)', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.title("b)", fontsize=30)
plt.grid()
plt.tight_layout()
plt.show()