from utils import *

taille = 100 # 100x100m
dim = 100
PATH = r"C:\Users\obour\OneDrive - Université Laval\Bureau"


sol = Sol((taille,taille),(dim,dim))
sol.placerElectrode(46, 98, 1)
sol.placerElectrode(54, 98, -1)
sol.afficherSigma()
sol.calculerPotentiel()
sol.afficherPotentielImSHOW()

sol.placerElectrodeMesure(49, 98)
sol.placerElectrodeMesure(51, 98)
sol.calculerResApparente(1)
sol.afficherResistanceApparente()

sol.calculerPseudoSection(1)
sol.afficherPseudoSection()

sol.enregistrerData(PATH, 'test.xlsx')


## optmisations faites:
# à chaque itération, je reprend l'ancien potentiel sans le reset
# la réponse est proche donc il y a moins d'itérations nécessaires pour converger
# ensuite la méthode de calcul utilise deux quadrillés. ça marche bien avec numba jit