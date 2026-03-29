from utils import *

taille = 100 # 100x100m
dim = 100

sol = Sol((taille,taille),(dim,dim))
# sol.placerElectrode(46, 98, 1)
# sol.placerElectrode(54, 98, -1)
# # sol.afficherSigma()
# sol.calculerPotentiel()
# sol.afficherPotentielImSHOW()

sol.placerElectrodeMesure(46, 98)
sol.placerElectrodeMesure(54, 98)
sol.calculerResApparente(1)
sol.afficherResistanceApparente()
