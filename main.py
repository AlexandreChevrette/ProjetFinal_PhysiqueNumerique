from utils import *

taille = 100 # 100x100m
dim = 100

sol = Sol((taille,taille),(dim,dim))
sol.placerElectrode(46, 98, 1)
sol.placerElectrode(54, 98, -1)
sol.afficherSigma()
sol.calculerPotentiel()
sol.afficherPotentiel()
