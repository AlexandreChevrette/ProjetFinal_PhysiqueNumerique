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
        self.__genererSigma__()
        self.electrodeList = np.array([])

    def __genererSigma__(self):
        # voir les sources
        self.matriceSigma = np.random.uniform(low=1, high=10, size=(self.ny, self.nx))
        self.matriceSigma[20: 40,:] = 10
        
    def placerElectrode(self, posX, posY, courant):
        np.append(self.electrodeList, Electrode(posX,posY,courant))
        

class Electrode:
    def __init__(self, posX, posY, courant):
        self.posX = posX
        self.posY = posY
        self.courant = courant