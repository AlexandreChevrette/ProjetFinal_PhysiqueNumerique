import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import jit



class Sol:
    def __init__(self, tailles: tuple[float, float], nxy : tuple[int, int]):
        self.tailleX = tailles[0]
        self.tailleY = tailles[1]
        self.nx = nxy[0]
        self.ny = nxy[1]
        self.matriceSigma = np.zeros((self.ny,self.nx))
        self.matricePotentiel = np.zeros((self.ny,self.nx)) 
        self.matriceCourant = np.zeros((self.ny,self.nx)) 
        self.__genererSigma__()
        self.electrodeList = np.array([])
        

    def __genererSigma__(self):
        # voir les sources et changer le setting de resistivité
        self.matriceSigma = np.random.uniform(low=1, high=10, size=(self.ny, self.nx))
        xx, yy = np.meshgrid(np.arange(self.ny), np.arange(self.nx), indexing='ij')

        self.matriceSigma[(xx-90)**2 + (yy-50)**2 <= 5**2] = 500
    
    def __genererCourant__(self):
        for electrode in self.electrodeList:
            self.matriceCourant[electrode.posY:self.ny, electrode.posX] = electrode.courant
        
    def afficherSigma(self):
        plt.figure()
        plt.imshow(self.matriceSigma, origin='lower')
        plt.show()

    def afficherCourant(self):
        plt.figure()
        plt.imshow(self.matriceCourant, origin='lower')
        plt.show()
    
    def afficherPotentiel(self):
        plt.figure()
        plt.contour(self.matricePotentiel, levels=500)
        plt.colorbar()
        plt.show()

    def placerElectrode(self, posX, posY, courant):
        self.electrodeList = np.append(self.electrodeList, Electrode(posX,posY,courant))

    def calculerPotentiel(self):      
        # voir sources

        it=0
        tol = 1e-8
        h = 1
        error = 1
        niter= 1000000
        V = self.matricePotentiel #reference
        self.__genererCourant__()
        I = self.matriceCourant.copy() #copie
        Sigma = self.matriceSigma.copy() #copie
        while error > tol and it < niter:
            V_k = V.copy()

            V[0, :] = 0
            V[-1, :] = V[-2, :]
            V[:, 0] = V[:, 1]
            V[:, -1] = V[:, -2]

            s = Sigma[1:-1, 1:-1]

            sigma_ifhs = (2 * s * Sigma[1:-1, 2:])   / (s + Sigma[1:-1, 2:])
            sigma_ibhs = (2 * s * Sigma[1:-1, :-2])  / (s + Sigma[1:-1, :-2])
            sigma_jfhs = (2 * s * Sigma[2:,   1:-1]) / (s + Sigma[2:,   1:-1])
            sigma_jbhs = (2 * s * Sigma[:-2,  1:-1]) / (s + Sigma[:-2,  1:-1])

            deno = sigma_ifhs + sigma_ibhs + sigma_jfhs + sigma_jbhs

            nom = (I[1:-1, 1:-1] * (h**2)) \
                + sigma_ifhs * V_k[1:-1, 2:]  \
                + sigma_ibhs * V_k[1:-1, :-2] \
                + sigma_jfhs * V_k[2:,   1:-1] \
                + sigma_jbhs * V_k[:-2,  1:-1]

            V[1:-1, 1:-1] = nom / deno

            diff = V - V_k
            error = np.sqrt(np.sum(diff**2))#np.linalg.norm(diff)

            it += 1  
            # print(f"Nombre d'itérations: {it}")
            print(f"Erreur: {error}")

class Electrode:
    def __init__(self, posX, posY, courant):
        self.posX = posX
        self.posY = posY
        self.courant = courant