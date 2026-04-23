from utils import *

taille = 100 # 100x100m
dim = 100
PATH = r"C:\Users\obour\OneDrive - Université Laval\Bureau"


sol = Sol((taille,taille),(dim,dim))

sol.matriceSigma = np.ones((sol.ny,sol.nx))* 1/250

yy, xx = np.meshgrid(np.arange(sol.ny), np.arange(sol.nx), indexing='ij')
sol.matriceSigma[(yy-90)**2 + (xx-50)**2 <= 5**2] = 1/50

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

sol.enregistrerData()
sol.inversion(lamb=100)
sol.afficherInversion()


# # 1. Générer des données synthétiques (modèle vrai inclus dans initialiserModele)
# d_obs = sol.calculerPseudoSection(courantInjection=1.0)

# # Réinitialiser vers modèle homogène
# sol.matriceSigma = np.ones((sol.ny, sol.nx)) * (1/5000)
# sol.__genererSigma__()

# history = sol.calculerInversion(
#     d_obs    = d_obs,
#     max_iter = 10,
#     eps      = 1e-6,
#     lam      = 0.05     # plus faible qu'avant car la normalisation aide déjà
# )

# # 4. Visualiser la convergence
# import matplotlib.pyplot as plt
# plt.semilogy(history["misfit"], 'o-')
# plt.xlabel("Itération"); plt.ylabel("RMS"); plt.title("Convergence"); plt.show()

# # 5. Visualiser le modèle inversé
# plt.figure()
# plt.imshow(1.0 / sol.matriceSigma, origin='lower', cmap='jet_r')
# plt.colorbar(label="Résistivité (Ω·m)")
# plt.title("Modèle inversé"); plt.show()


## optmisations faites:
# à chaque itération, je reprend l'ancien potentiel sans le reset
# la réponse est proche donc il y a moins d'itérations nécessaires pour converger
# ensuite la méthode de calcul utilise deux quadrillés. ça marche bien avec numba jit