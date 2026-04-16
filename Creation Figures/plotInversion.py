
import pickle
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))     

from utils import *
  

nx = 100
ny = 50

sol = Sol((nx,ny))
solveur = Solveur(sol)
vis = Visualisation(sol)
inversionPy = PyGimliInversionSolveur(sol, solveur)

load_data = False
load_data = True
if load_data:
    with open('sol_data.pkl', 'rb') as f:
        data = pickle.load(f)
        sol.matriceSigma = data['sigma']
        sol.listeResistanceApparente = data['listeResistanceApparente']
        sol.listeAB2 = data['listeAB2']
        sol.listePseudoSection = data['listePseudoSection']
        sol.listeX = data['listeX']
        sol.listeZ = data['listeZ']
        sol.inversionX = data['inversionX']
        sol.inversionY = data['inversionY']
        sol.inversionRes = data['inversionRes']
    vis.afficherSimulationComplete()
else:  

    pas = 2
    solveur.calculerPseudoSection(1, pas)
    inversionPy.inversionPyGimli(pas)
    vis.afficherInversion()
    vis.afficherSimulationComplete()
    # Save data
    with open('sol_data.pkl', 'wb') as f:
        pickle.dump({
            'sigma': sol.matriceSigma,
            'listeResistanceApparente': sol.listeResistanceApparente,
            'listeAB2': sol.listeAB2,
            'listePseudoSection': sol.listePseudoSection,
            'listeX': sol.listeX,
            'listeZ': sol.listeZ,
            'inversionX': sol.inversionX,
            'inversionY': sol.inversionY,
            'inversionRes': sol.inversionRes
        }, f)