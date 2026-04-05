from utils import *

taille = 100 # 100x100m
dim = 100
PATH = r"C:\Users\obour\OneDrive - Université Laval\Bureau"


sol = Sol((taille,taille),(dim,dim))

sol.matriceSigma = np.ones((sol.ny,sol.nx))* 1/5000

yy, xx = np.meshgrid(np.arange(sol.ny), np.arange(sol.nx), indexing='ij')
sol.matriceSigma[(yy-90)**2 + (xx-30)**2 <= 5**2] = 1/1000

sol.placerElectrode(46, 98, 1)
sol.placerElectrode(54, 98, -1)
sol.afficherSigma()
sol.calculerPotentiel()
sol.afficherPotentielImSHOW()


sol.placerElectrodeMesure(48, 98)
sol.placerElectrodeMesure(52, 98)
sol.calculerResApparente(1)
sol.afficherResistanceApparente()

sol.calculerPseudoSection(1)
sol.afficherPseudoSection()

# ... tes réglages avec l'anomalie ...
d_obs = sol.calculerPseudoSection(1) # Voici tes données "terrain"

# --- 2. Préparation de l'inversion (Le point de départ de l'IA) ---
# On "efface" l'anomalie pour voir si l'inversion peut la retrouver
sol.matriceSigma = np.ones((dim, dim)) * (1/5000) 
sol.__genererSigma__() # On recalcule les coefficients pour ce sol homogène

# --- 3. Lancement de l'inversion ---
sol.calculerInversion(d_obs, max_iter=10, lam=100, alpha=0.1)
sol.afficherSigma() # Devrait montrer l'anomalie reconstruite


# sol.enregistrerData(PATH, 'test_3.xlsx')


## optmisations faites:
# à chaque itération, je reprend l'ancien potentiel sans le reset
# la réponse est proche donc il y a moins d'itérations nécessaires pour converger
# ensuite la méthode de calcul utilise deux quadrillés. ça marche bien avec numba jit