
import timeit
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))     

from utils import *


nx = 200
ny = 100

sol = Sol((nx,ny))
solveur = Solveur(sol)
vis = Visualisation(sol)
densiteCourant =1
sol.placerElectrodeMesure(49, 1)
sol.placerElectrodeMesure(51, 1)
solveur.calculerResApparente(densiteCourant)
# vis.afficherResistanceApparente()


fontsize = 20
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(1/(sol.matriceSigma[1:,:]), origin='lower', cmap=cmap)
plt.gca().invert_yaxis()
# plt.tight_layout()
plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)
divider = make_axes_locatable(ax)
cax = divider.append_axes("top", size="2%", pad=0.1)  # 5% thickness, 0.1 pad
cbar = plt.colorbar(im, cax=cax, orientation='horizontal', pad=0.05)
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.set_label("Résistivité (Ω·m)", fontsize=fontsize)
cbar.ax.tick_params(axis='x', labelsize=fontsize)
ax.tick_params(axis='x', labelsize=fontsize)
ax.tick_params(axis='y', labelsize=fontsize)
ax.set_xlabel('Position X (m)', fontsize=fontsize)
ax.set_ylabel('Profondeur (m)', fontsize=fontsize)
plt.title("a)", fontsize=30)
plt.show()


fontsize = 20
linewidth = 4

plt.figure(figsize=(10, 6))
plt.scatter(sol.listeAB2*2, sol.listeResistanceApparente, 
        s=100, color='steelblue', edgecolors='black', linewidth=1)

plt.axhline(y=50, color='red', linestyle='--', linewidth=linewidth)
plt.axhline(y=250, color='red', linestyle='--', linewidth=linewidth)
plt.ylabel(r"Résistivité apparente (Ω·m)", fontsize=fontsize)
plt.xlabel("Distance entre les électrodes (m)", fontsize=fontsize)
plt.tick_params(axis='x', labelsize=fontsize)
plt.tick_params(axis='y', labelsize=fontsize)
plt.grid(True, alpha=0.3)
plt.title("b)", fontsize=30)
plt.tight_layout()
plt.show()
