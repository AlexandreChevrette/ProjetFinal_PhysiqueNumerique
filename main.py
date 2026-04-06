from utils import *
import pickle
  
taille = 100 # 100x100m pas utilisé en ce moment

nx = 200
ny = 100

sol = Sol((taille,taille),(nx,ny))

densiteCourant = 0.01

sol.placerElectrode(46, 1, densiteCourant)
sol.placerElectrode(54, 1, -densiteCourant)
# sol.afficherSigma()
# sol.calculerPotentiel()
# sol.afficherCourant()
# sol.afficherPotentielImSHOW()

# sol.placerElectrodeMesure(49, 98)
# sol.placerElectrodeMesure(51, 98)
# sol.calculerResApparente(1)
# sol.afficherResistanceApparente()




load_data = False
load_data = True
if load_data:
    with open('sol_data.pkl', 'rb') as f:
        data = pickle.load(f)
        sol.listeResistanceApparente = data['listeResistanceApparente']
        sol.listeAB2 = data['listeAB2']
        sol.listePseudoSection = data['listePseudoSection']
        sol.listeX = data['listeX']
        sol.listeZ = data['listeZ']
        sol.inverted_x = data['inverted_x']
        sol.inverted_y = data['inverted_y']
        sol.inverted_res = data['inverted_res']

    sol.afficherSimulationComplete()
else:
    pas = 2
    sol.calculerPseudoSection(1, pas)
    sol.afficherPseudoSection()
    sol.inversion(pas)
    sol.afficherInversion()
    sol.afficherSimulationComplete()

    # Save data
    with open('sol_data.pkl', 'wb') as f:
        pickle.dump({
            'listeResistanceApparente': sol.listeResistanceApparente,
            'listeAB2': sol.listeAB2,
            'listePseudoSection': sol.listePseudoSection,
            'listeX': sol.listeX,
            'listeZ': sol.listeZ,
            'inverted_x': sol.inverted_x,
            'inverted_y': sol.inverted_y,
            'inverted_res': sol.inverted_res
        }, f)