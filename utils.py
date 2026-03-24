import numpy as np
import matplotlib.pyplot as plt
import numba


@numba.njit(fastmath=True, parallel=True)
def __accelerateurCalculPotentiel__(V, V_k, Sigma, I, h):

    ny, nx = V.shape
    h2 = h * h
    error = 0.0

    # valeurs frontières
    for i in numba.prange(nx):
        V[0, i] = 0.0
        V[ny-1, i] = V[ny-2, i]

    for j in numba.prange(ny):
        V[j, 0] = V[j, 1]
        V[j, nx-1] = V[j, nx-2]

    # calculs
    for j in numba.prange(1, ny-1):
        for i in numba.prange(1, nx-1):

            s = Sigma[j, i]

            sip = Sigma[j, i+1]
            sim = Sigma[j, i-1]
            sjp = Sigma[j+1, i]
            sjm = Sigma[j-1, i]

            sigma_ifhs = 2.0 * s * sip / (s + sip)
            sigma_ibhs = 2.0 * s * sim / (s + sim)
            sigma_jfhs = 2.0 * s * sjp / (s + sjp)
            sigma_jbhs = 2.0 * s * sjm / (s + sjm)

            deno = sigma_ifhs + sigma_ibhs + sigma_jfhs + sigma_jbhs

            new_val = (
                I[j, i] * h2
                + sigma_ifhs * V_k[j, i+1]
                + sigma_ibhs * V_k[j, i-1]
                + sigma_jfhs * V_k[j+1, i]
                + sigma_jbhs * V_k[j-1, i]
            ) / deno

            V[j, i] = new_val

            d = new_val - V_k[j, i]
            error += d * d

    return np.sqrt(error)


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
        self.electrodeMesuresList = np.array([])
        self.listeResistanceApparente = np.array([])
        self.listeAB2 = np.array([])
        

    def __genererSigma__(self):
        # voir les sources et changer le setting de resistivité
        # self.matriceSigma = np.random.uniform(low=1, high=10, size=(self.ny, self.nx))
        self.matriceSigma = np.ones((self.ny,self.nx))
        xx, yy = np.meshgrid(np.arange(self.ny), np.arange(self.nx), indexing='ij')

        self.matriceSigma[(xx-90)**2 + (yy-50)**2 <= 5**2] = 1/1000
    
    def __genererCourant__(self):
        for electrode in self.electrodeList:
            self.matriceCourant[electrode.posY, electrode.posX] = electrode.courant
        
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

    def placerElectrodeMesure(self, posX, posY):
        self.electrodeMesuresList = np.append(self.electrodeMesuresList, ElectrodeMesure(posX, posY))

    def enleverElectrodes(self):
        self.electrodeList = np.array([])

    def calculerPotentiel(self):      
        # voir sources

        it=0
        tol = 1e-8

        ### C'est quoi h??
        h = 1


        erreur = 1
        niter= 1000000
        V = self.matricePotentiel #reference
        self.__genererCourant__()
        I = self.matriceCourant.copy() #copie
        Sigma = self.matriceSigma.copy() #copie

        V_k = np.zeros_like(V)
        while erreur > tol and it < niter:

            erreur = __accelerateurCalculPotentiel__(V, V_k, Sigma, I, h)

            V, V_k = V_k, V   # swap instead of copy

            it += 1  
            print(f"Erreur: {erreur}")


    def calculerResApparente(self, courantInjection):
        if (len(self.electrodeMesuresList) != 2):
            print("Seulement deux sondes de mesures doivent être utilisés pour calculer la resistance apparente")
            return
        coord_ab = self.__genererPositionsAB__()
        M = self.electrodeMesuresList[0]
        N = self.electrodeMesuresList[1]

        listeRho = []
        listeAB2 = []
        for (a, b) in coord_ab:
            rho, ab2 = self.__calculerUnAB__(a, b, M, N, courantInjection)
            listeRho.append(rho)
            listeAB2.append(ab2)

        self.listeResistanceApparente = np.array(listeRho)
        self.listeAB2 = np.array(listeAB2)


    def __calculerUnAB__(self, a, b, M, N, courantInjection):
        self.enleverElectrodes()
        self.placerElectrode(a, self.ny-2, courantInjection)
        self.placerElectrode(b, self.ny-2, -courantInjection)
        self.__genererCourant__()
        # Solveur
        self.calculerPotentiel()

        

        # Différence de potentiel
        dV = self.matricePotentiel[M.posY, M.posX] - self.matricePotentiel[N.posY, N.posX]

        # Distance AB
        AB = abs(b - a)
        AB_2 = AB / 2

        # Distance MN
        MN_dist = abs(M.posX - N.posX)
        MN_term = (MN_dist / 2)**2

        # Facteur géométrique
        K = np.pi * ((AB_2**2 - MN_term) / MN_dist)

        # Résistivité apparente
        rho = K * dV / courantInjection

        return rho, AB_2
        

    def __genererPositionsAB__(self):
        listeA = []
        listeB = []
        for i in range(2, self.nx//2, 2):
            A = self.nx//2 - i
            B = self.nx//2 + i
            listeA.append(A)
            listeB.append(B)
        return list(zip(listeA, listeB))
    
    def afficherResistanceApparente(self):
        plt.figure()
        # plt.contourf(xx, yy, V_[0], levels=200)
        plt.plot(self.listeAB2, self.listeResistanceApparente)
        plt.yscale('log')
        plt.ylabel(r"Résistivité apparente [$\Omega$m]")
        plt.xlabel("Demi-distance entre les électrodes [m]")
        plt.show()

    

class Electrode:
    def __init__(self, posX, posY, courant):
        self.posX = posX
        self.posY = posY
        self.courant = courant

class ElectrodeMesure:
    def __init__(self, posX, posY):
        self.posX = posX
        self.posY = posY
        self.potentiel = 0