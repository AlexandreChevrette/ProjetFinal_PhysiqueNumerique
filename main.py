from utils import *
import pickle
  

# Exemple d'utilisation (voir dossier "Creation Figures" pour les scripts de visualisation)

nx = 100
ny = 50

sol = Sol((nx,ny))
solveur = Solveur(sol)
vis = Visualisation(sol)
# vis = Visualisation(sol)
inversionPy = PyGimliInversionSolveur(sol, solveur)
inversion = InversionSolveur(sol)
densiteCourant = 0.01
sol.placerElectrode(46, 1, densiteCourant)
sol.placerElectrode(54, 1, -densiteCourant)

vis.afficherSigma()
solveur.calculerPotentiel()
# vis.afficherCourant()
vis.afficherPotentiel()

sol.placerElectrodeMesure(49, 1)
sol.placerElectrodeMesure(51, 1)
solveur.calculerResApparente(1)
vis.afficherResistanceApparente()

inversionPy.inversionPyGimli(1)
vis.afficherInversion()


