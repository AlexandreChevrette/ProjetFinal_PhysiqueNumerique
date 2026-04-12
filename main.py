from utils import *
import pickle
  
taille = 100 # 100x100m pas utilisé en ce moment

nx = 300
ny = 50

sol = Sol((taille,taille),(nx,ny))
solveur = Solveur(sol)
# vis = Visualisation(sol)
inversionPy = PyGimliInversionSolveur(sol, solveur)
inversion = InversionSolveur(sol)
densiteCourant = 0.01
sol.placerElectrode(46, 1, densiteCourant)
sol.placerElectrode(54, 1, -densiteCourant)

vis.afficherSigma()
solveur.calculerPotentiel()
# vis.afficherCourant()
vis.afficherPotentiel()

sol.placerElectrodeMesure(49, 1)
sol.placerElectrodeMesure(51, 1)
# solveur.calculerResApparente(1)
# vis.afficherResistanceApparente()



# load_data = True
# load_data = False
# if load_data:
#     with open('sol_data.pkl', 'rb') as f:
#         data = pickle.load(f)
#         sol.matriceSigma = data['sigma']
#         sol.listeResistanceApparente = data['listeResistanceApparente']
#         sol.listeAB2 = data['listeAB2']
#         sol.listePseudoSection = data['listePseudoSection']
#         sol.listeX = data['listeX']
#         sol.listeZ = data['listeZ']
#         sol.inversionX = data['inversionX']
#         sol.inversionY = data['inversionY']
#         sol.inversionRes = data['inversionRes']

#     vis.afficherSimulationComplete()
# else:
    
#    # with open('sol_data.pkl', 'rb') as f:
#    #     data = pickle.load(f)
#    #     sol.matriceSigma = data['sigma']
#    #     sol.listeResistanceApparente = data['listeResistanceApparente']
#    #     sol.listeAB2 = data['listeAB2']
#    #     sol.listePseudoSection = data['listePseudoSection']
#    #     sol.listeX = data['listeX']
#    #     sol.listeZ = data['listeZ']
#    #     sol.inversionX = data['inversionX']
#    #     sol.inversionY = data['inversionY']
#    #     sol.inversionRes = data['inversionRes']
#     pas = 6
#     solveur.calculerPseudoSection(1, pas)

#     # inversionPy.inversionPyGimli(pas)
#     inversion.calculerInversion(pas)
#     matriceSigmaInversee = inversion.obtenirResultatsInversion()
#     matriceSigmaInversee = matriceSigmaInversee[1:,:]
#     # 5. Visualiser le modèle inversé
#     plt.figure()
#     plt.imshow(1.0 / matriceSigmaInversee, origin='lower', cmap='jet_r')
#     plt.colorbar(label="Résistivité (Ω·m)")
#     plt.title("Modèle inversé"); plt.show()

#     vis.afficherInversion()
#     vis.afficherSimulationComplete()

#     # Save data
#     with open('sol_data.pkl', 'wb') as f:
#         pickle.dump({
#             'sigma': sol.matriceSigma,
#             'listeResistanceApparente': sol.listeResistanceApparente,
#             'listeAB2': sol.listeAB2,
#             'listePseudoSection': sol.listePseudoSection,
#             'listeX': sol.listeX,
#             'listeZ': sol.listeZ,
#             'inversionX': sol.inversionX,
#             'inversionY': sol.inversionY,
#             'inversionRes': sol.inversionRes
#         }, f)