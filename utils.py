import numpy as np
import matplotlib.pyplot as plt
import numba


@numba.njit(fastmath=True, parallel=True)
def rb_gauss_seidel(
    V,
    sigma_ifhs, sigma_ibhs,
    sigma_jfhs, sigma_jbhs,
    sigma_deno,
    I,
    h
):

    ny, nx = V.shape
    h2 = h * h

    # =========================
    # Boundaries (no prange)
    # =========================
    for i in range(nx):
        V[0, i] = 0.0
        V[ny-1, i] = V[ny-2, i]

    for j in range(ny):
        V[j, 0] = V[j, 1]
        V[j, nx-1] = V[j, nx-2]


    # =========================
    # 🔴 RED PASS
    # =========================
    err_red = 0.0

    for j in numba.prange(1, ny-1):
        tmp = 0.0
        start_i = 1 + (j % 2)

        for i in range(start_i, nx-1, 2):

            v_old = V[j, i]

            new_val = (
                I[j, i] * h2
                + sigma_ifhs[j, i] * V[j, i+1]
                + sigma_ibhs[j, i] * V[j, i-1]
                + sigma_jfhs[j, i] * V[j+1, i]
                + sigma_jbhs[j, i] * V[j-1, i]
            ) / sigma_deno[j, i]

            V[j, i] = new_val

            d = new_val - v_old
            tmp += d * d

        err_red += tmp

    # =========================
    # ⚫ BLACK PASS
    # =========================
    err_black = 0.0

    for j in numba.prange(1, ny-1):
        tmp = 0.0
        start_i = 1 + ((j + 1) % 2)

        for i in range(start_i, nx-1, 2):

            v_old = V[j, i]

            new_val = (
                I[j, i] * h2
                + sigma_ifhs[j, i] * V[j, i+1]
                + sigma_ibhs[j, i] * V[j, i-1]
                + sigma_jfhs[j, i] * V[j+1, i]
                + sigma_jbhs[j, i] * V[j-1, i]
            ) / sigma_deno[j, i]

            V[j, i] = new_val

            d = new_val - v_old
            tmp += d * d

        err_black += tmp

    return np.sqrt(err_red + err_black)


class Sol:
    def __init__(self, tailles: tuple[float, float], nxy : tuple[int, int]):
        self.tailleX = tailles[0]
        self.tailleY = tailles[1]
        self.nx = nxy[0]
        self.ny = nxy[1]
        self.matriceSigma = np.zeros((self.ny,self.nx))
        self.sigma_ifhs = np.zeros((self.ny,self.nx)) 
        self.sigma_ibhs = np.zeros((self.ny,self.nx))
        self.sigma_jfhs = np.zeros((self.ny,self.nx))
        self.sigma_jbhs = np.zeros((self.ny,self.nx))
        self.sigma_deno = np.zeros((self.ny,self.nx))
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
        self.matriceSigma = np.ones((self.ny,self.nx))*1/250

        # xx, yy = np.meshgrid(np.arange(self.ny), np.arange(self.nx), indexing='ij')
        # self.matriceSigma[(xx-90)**2 + (yy-50)**2 <= 5**2] = 1/1000

        self.matriceSigma[:90,:] = 1/50
        self.matriceSigma[-1,:] = 0

        # center
        s = self.matriceSigma[1:-1, 1:-1]

        # neighbors
        sip = self.matriceSigma[1:-1, 2:]   # i+1
        sim = self.matriceSigma[1:-1, :-2]  # i-1
        sjp = self.matriceSigma[2:, 1:-1]   # j+1
        sjm = self.matriceSigma[:-2, 1:-1]  # j-1

        # harmonic means
        self.sigma_ifhs[1:-1, 1:-1] = 2.0 * s * sip / (s + sip)
        self.sigma_ibhs[1:-1, 1:-1] = 2.0 * s * sim / (s + sim)
        self.sigma_jfhs[1:-1, 1:-1] = 2.0 * s * sjp / (s + sjp)
        self.sigma_jbhs[1:-1, 1:-1] = 2.0 * s * sjm / (s + sjm)

        # denominator
        self.sigma_deno = (
            self.sigma_ifhs
            + self.sigma_ibhs
            + self.sigma_jfhs
            + self.sigma_jbhs
        )
    
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
        tol = 1e-2

        ### C'est quoi h??
        h = 1


        erreur = 1
        niter= 1000000
        V = self.matricePotentiel #reference
        self.__genererCourant__()
        I = self.matriceCourant.copy()
        sigma_ifhs = self.sigma_ifhs.copy()
        sigma_ibhs = self.sigma_ibhs.copy()
        sigma_jfhs = self.sigma_jfhs.copy()
        sigma_jbhs = self.sigma_jbhs.copy()
        sigma_deno = self.sigma_deno.copy()

        while erreur > tol and it < niter:

            erreur = rb_gauss_seidel(V, sigma_ifhs,sigma_ibhs,sigma_jfhs,sigma_jbhs,sigma_deno, I, h)

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

        # r1 = abs(M.posX - a)
        # r2 = abs(M.posX - b)
        # r3 = abs(N.posX - a)
        # r4 = abs(N.posX - b)

        # Résistivité apparente
        # rho = 2 * np.pi * dV / courantInjection * ((1/r1-1/r2)-(1/r3-1/r4))
        # rho = 2 * np.pi * dV / courantInjection * r1

        AM = abs(M.posX - a)
        BM = abs(M.posX - b)
        AN = abs(N.posX - a)
        BN = abs(N.posX - b)

        # Résistivité apparente (* à la place de diviser)
        rho = 2 * np.pi * dV / (courantInjection) * ((1/AM-1/AN)-(1/BM-1/BN))


        return rho, AB_2
        

    def __genererPositionsAB__(self):
        listeA = []
        listeB = []
        for i in range(2, self.nx//2-6, 2):
            A = self.nx//2-6 - i
            B = self.nx//2+6 + i
            listeA.append(A)
            listeB.append(B)
        return list(zip(listeA, listeB))
    
    def afficherResistanceApparente(self):
        plt.figure()
        # plt.contourf(xx, yy, V_[0], levels=200)
        plt.plot(self.listeAB2, self.listeResistanceApparente)
        # plt.yscale('log')
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