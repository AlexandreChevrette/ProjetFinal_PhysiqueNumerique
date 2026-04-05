from utils import *
  
taille = 100 # 100x100m pas utilisé en ce moment

nx = 100
ny = 50

sol = Sol((taille,taille),(nx,ny))

densiteCourant = 0.01

sol.placerElectrode(46, ny-2, densiteCourant)
sol.placerElectrode(54, ny-2, -densiteCourant)
sol.afficherSigma()
sol.calculerPotentiel()
sol.afficherPotentielImSHOW()

# sol.placerElectrodeMesure(49, 98)
# sol.placerElectrodeMesure(51, 98)
# sol.calculerResApparente(1)
# sol.afficherResistanceApparente()

# pas = 2
# sol.calculerPseudoSection(1, pas)
# sol.afficherPseudoSection()
# sol.inversion(pas)
# sol.afficherInversion()