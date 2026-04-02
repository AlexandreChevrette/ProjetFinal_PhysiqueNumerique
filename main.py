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


sol.placerElectrodeMesure(48, 98)
sol.placerElectrodeMesure(52, 98)
sol.calculerResApparente(1)
sol.afficherResistanceApparente()

# sol.Jacobien()

sol.calculerPseudoSection(1)
sol.afficherPseudoSection()


sol.afficherModeleResistivite()
d_obs = sol.simulerDonnees(courantInjection=1.0)
sigma_true = sol.matriceSigma.copy()
sol.matriceSigma = np.ones((sol.ny, sol.nx)) * (1/5000)
sol.matriceSigma[-1, :] = 1e-12
sol.__genererSigma__()
sol.afficherModeleResistivite()
misfits = sol.inversion_maison(
    d_obs=d_obs,
    courantInjection=1.0,
    niter=8,
    lam=5000,
    alpha=0.2
)
sol.afficherModeleResistivite()
sol.afficherMisfit(misfits)

# sol.enregistrerData(PATH, 'test_3.xlsx')


## optmisations faites:
# à chaque itération, je reprend l'ancien potentiel sans le reset
# la réponse est proche donc il y a moins d'itérations nécessaires pour converger
# ensuite la méthode de calcul utilise deux quadrillés. ça marche bien avec numba jit