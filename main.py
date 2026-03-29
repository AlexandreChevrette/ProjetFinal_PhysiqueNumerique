from utils import *

taille = 100 # 100x100m
dim = 100


sol = Sol((taille,taille),(dim,dim))
sol.placerElectrode(46, 98, 2)
sol.placerElectrode(54, 98, -2)
sol.afficherSigma()
sol.calculerPotentiel()
sol.afficherPotentiel()

# sol.placerElectrodeMesure(46, 98)
# sol.placerElectrodeMesure(54, 98)
# sol.calculerResApparente(1)
# sol.afficherResistanceApparente()
