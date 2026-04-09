from utils import *
import pickle
  
taille = 100 # 100x100m pas utilisé en ce moment

nx = 200
ny = 100

sol = Sol((taille,taille),(nx,ny))
solveur = Solveur(sol)
vis = Visualisation(sol)
densiteCourant = 0.01

sol.placerElectrode(46, 1, densiteCourant)
sol.placerElectrode(54, 1, -densiteCourant)

# vis.afficherSigma()
solveur.calculerPotentiel()
vis.afficherCourant()
vis.afficherPotentielImSHOW()

# sol.placerElectrodeMesure(49, 98)
# sol.placerElectrodeMesure(51, 98)
# sol.calculerResApparente(1)
# sol.afficherResistanceApparente()



# load_data = True
# load_data = False
# if load_data:
#     with open('sol_data.pkl', 'rb') as f:
#         data = pickle.load(f)
#         sol.listeResistanceApparente = data['listeResistanceApparente']
#         sol.listeAB2 = data['listeAB2']
#         sol.listePseudoSection = data['listePseudoSection']
#         sol.listeX = data['listeX']
#         sol.listeZ = data['listeZ']
#         sol.inverted_x = data['inverted_x']
#         sol.inverted_y = data['inverted_y']
#         sol.inverted_res = data['inverted_res']
# 
#     sol.afficherSimulationComplete()
# else:
#     pas = 2
#     sol.calculerPseudoSection(1, pas)
#    # # ... tes réglages avec l'anomalie ...
#    # d_obs = sol.listePseudoSection.copy() # Voici tes données "terrain"
#    # sol.afficherPseudoSection()
#    # 
#    # 
# 
#    # # --- 2. Préparation de l'inversion (Le point de départ de l'IA) ---
#    # # On "efface" l'anomalie pour voir si l'inversion peut la retrouver
#    # sol.matriceSigma = np.ones((ny,nx), dtype=np.float64) * (1/2500)
#    # sol.__genererSigma__() # On recalcule les coefficients pour ce sol homogène
# 
#     sol.inversionPyGimli(pas)
#     sol.afficherInversion()
#     sol.afficherSimulationComplete()
# 
#     # Save data
#     with open('sol_data.pkl', 'wb') as f:
#         pickle.dump({
#             'listeResistanceApparente': sol.listeResistanceApparente,
#             'listeAB2': sol.listeAB2,
#             'listePseudoSection': sol.listePseudoSection,
#             'listeX': sol.listeX,
#             'listeZ': sol.listeZ,
#             'inverted_x': sol.inverted_x,
#             'inverted_y': sol.inverted_y,
#             'inverted_res': sol.inverted_res
#         }, f)