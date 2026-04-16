
import timeit
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))     

from utils import *


nx = 200
ny = 100
################################################
### mettre précision à 1e-3 pour ces calculs ###
################################################

sol = Sol((nx,ny))
solveur = Solveur(sol)
vis = Visualisation(sol)
densiteCourant = 1
pas = 6
# solveur.calculerPseudoSection(densiteCourant, pas)
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

xi = np.linspace(sol.listeX.min(), sol.listeX.max(), sol.nx)
zi = np.linspace(sol.listeZ.min(), sol.listeZ.max(), sol.nx//2)

XI, ZI = np.meshgrid(xi, zi)

# interpolation
RHOI = griddata((sol.listeX, sol.listeZ), sol.listePseudoSection, (XI, ZI), method='cubic')

fontsize = 20
fig, ax = plt.subplots(figsize=(12, 8))
contourf = ax.contourf(XI, ZI, RHOI, levels=100, cmap=cmap)
plt.gca().invert_yaxis()  # profondeur vers le bas
# points de mesure (optionnel mais pro)
ax.scatter(sol.listeX, sol.listeZ, s=10, color='white', edgecolors='black', label='Points de mesure')

plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)
divider = make_axes_locatable(ax)
cax = divider.append_axes("top", size="2%", pad=0.1)
cbar = plt.colorbar(contourf, cax=cax, orientation='horizontal', pad=0.05)
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.set_label("Résistivité apparente (Ω·m)", fontsize=fontsize)
cbar.ax.tick_params(axis='x', labelsize=fontsize)
ax.tick_params(axis='x', labelsize=fontsize)
ax.tick_params(axis='y', labelsize=fontsize)
ax.set_xlabel("Position X (m)", fontsize=fontsize)
ax.set_ylabel("Profondeur Apparente (m)", fontsize=fontsize)
plt.title("b)", fontsize=30)
plt.tight_layout()
plt.show()
