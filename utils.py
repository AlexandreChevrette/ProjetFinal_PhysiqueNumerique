import numpy as np
import matplotlib.pyplot as plt
import numba
from scipy.interpolate import griddata
import pygimli as pg
from pygimli.physics import ert

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


    err_black = 0.0

    for j in numba.prange(1, ny-1):
        tmp = 0.0
        start_i = 1 + ((j + 1) % 2)

        for i in range(start_i, nx-1, 2):

            v_old = V[j, i]

            new_val = (
                I[j, i] 
                + sigma_ifhs[j, i] * V[j, i+1]
                + sigma_ibhs[j, i] * V[j, i-1]
                + sigma_jfhs[j, i] * V[j+1, i]
                + sigma_jbhs[j, i] * V[j-1, i]
            ) / sigma_deno[j, i]

            V[j, i] = new_val

            d = new_val - v_old
            tmp += d * d

        err_black += tmp

    # =========================
    # Boundaries (no prange)
    # =========================
    for i in range(nx):
        V[0, i] = V[1, i]
        V[ny-1, i] = 0.0

    for j in range(ny):
        V[j, 0] = V[j, 1]
        V[j, nx-1] = V[j, nx-2]


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
        self.listeX = np.array([])
        self.listeZ = np.array([])
        self.listePseudoSection = np.array([])

    def __genererSigma__(self):
        # voir les sources et changer le setting de resistivité
        # self.matriceSigma = np.random.uniform(low=1, high=10, size=(self.ny, self.nx))
        self.matriceSigma = np.ones((self.ny,self.nx))* 1/250
        # self.matriceSigma[:90,:] =1/50
        yy, xx = np.meshgrid(np.arange(self.ny), np.arange(self.nx), indexing='ij')
        self.matriceSigma[(yy-90)**2 + (xx-30)**2 <= 5**2] = 1/1000

       
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

    def afficherPotentielImSHOW(self):
        plt.figure()
        plt.imshow(self.matricePotentiel, origin='lower')
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
            # print(f"Erreur: {erreur}")

    def calculerResApparente(self, courantInjection):
        if (len(self.electrodeMesuresList) != 2):
            print("Seulement deux sondes de mesures doivent être utilisés pour calculer la resistance apparente")
            return
        
        M = self.electrodeMesuresList[0]
        N = self.electrodeMesuresList[1]
        coord_ab = self.__genererPositionsAB__(M, N)
        
        
        listeRho = []
        listeAB2 = []
        for (a, b) in coord_ab:
            rho, ab2 = self.__calculerUnAB__(a, b, M.posX, N.posX, courantInjection)
            listeRho.append(rho)
            listeAB2.append(ab2)

        self.listeResistanceApparente = np.array(listeRho)
        self.listeAB2 = np.array(listeAB2)

    def __calculerUnAB__(self, a, b, M, N, courantInjection):
        self.matriceCourant = np.zeros((self.ny,self.nx))
        # self.matricePotentiel = np.zeros((self.ny,self.nx))
        self.enleverElectrodes()

        self.placerElectrode(a, self.ny-2, courantInjection)
        self.placerElectrode(b, self.ny-2, -courantInjection)
        self.__genererCourant__()
        self.calculerPotentiel()
        
        dV = self.matricePotentiel[98, M] - self.matricePotentiel[98, N]

        AB = abs(b - a)
        AB_2 = AB / 2

        AM = abs(M - a)
        BM = abs(M - b)
        AN = abs(N - a)
        BN = abs(N - b)

        # K = 2 * np.pi/((1/AM-1/AN)-(1/BM-1/BN))


        # En 2d, la formule de la résistivité apparente pour un 
        # montage de Schlumberger est donnée par  selon chatGPT:
        K_2D = np.pi / np.log((AN * BM) / (AM * BN)) 
        rho = K_2D * dV / (courantInjection) 

        return rho, AB_2
        
    def __genererPositionsAB__(self, M, N):
        listeA = []
        listeB = []
        A = M.posX - 1
        B = N.posX + 1
        while (A >= 4 and B < self.nx-4):
            listeA.append(A)
            listeB.append(B)
            A -= 1
            B += 1
        return list(zip(listeA, listeB))
    
    def afficherResistanceApparente(self):
        plt.figure()
        plt.plot(self.listeAB2, self.listeResistanceApparente)
        plt.ylabel(r"Résistivité apparente [$\Omega$m]")
        plt.xlabel("Demi-distance entre les électrodes [m]")
        plt.show()

    def calculerPseudoSection(self, courantInjection):
        coord_abmn = self.__genererPositionsABMN__()
        
        listeRho = []
        listeZ = []
        listeX = []
        for (a, b, M, N) in coord_abmn:
            rho, ab2 = self.__calculerUnAB__(a, b, M, N, courantInjection)
            listeRho.append(rho)
            listeZ.append(ab2)
            listeX.append((M + N) / 2)

        self.listePseudoSection = np.array(listeRho)
        self.listeZ = np.array(listeZ)
        self.listeX = np.array(listeX)

    def __genererPositionsABMN__(self):
        listeA = []
        listeB = []
        listeM = []
        listeN = []
        pas = 6

        A_ini = 2
        M_ini = 4
        N_ini = 6
        B_ini = 8
        for i in range((self.nx-B_ini)//(2*pas)):
            A = A_ini
            M = M_ini + i*pas
            N = N_ini + i*pas
            B = B_ini + 2*i*pas
            while (B < self.nx): 
                listeA.append(A)
                listeB.append(B)
                listeM.append(M)
                listeN.append(N)
                A += pas
                M += pas 
                N += pas
                B += pas
        return list(zip(listeA, listeB, listeM, listeN))
    
    def afficherPseudoSection(self):
        xi = np.linspace(self.listeX.min(), self.listeX.max(), 200)
        zi = np.linspace(self.listeZ.min(), self.listeZ.max(), 200)

        XI, ZI = np.meshgrid(xi, zi)

        # interpolation
        RHOI = griddata((self.listeX, self.listeZ), self.listePseudoSection, (XI, ZI), method='cubic')
        plt.figure()
        contourf = plt.contourf(XI, ZI, RHOI, levels=50)
        plt.gca().invert_yaxis()  # profondeur vers le bas
        # points de mesure (optionnel mais pro)
        plt.scatter(self.listeX, self.listeZ, c='k', s=10)
        
        plt.colorbar(contourf, label=r"Résistivité apparente [$\Omega$m]")
        plt.xlabel("Position X [m]")
        plt.ylabel("Profondeur Apparente (AB/2) [m]")
        plt.title("Pseudo-section de résistivité apparente")
        plt.show()

    def inversion(self):
        coord_abmn = self.__genererPositionsABMN__()


        data = pg.DataContainerERT()

        for i in range(self.nx):
            data.createSensor([i, 0, 0])

        for i, (A, B, M, N) in enumerate(coord_abmn):
            data.createFourPointData(i, A, B, M, N)
            
        data["rhoa"] = self.listePseudoSection

        data.set("k", ert.geometricFactor(data))

        data["err"] = pg.Vector(data.size(), 0.03)

        mgr = ert.ERTManager(data)

        model = mgr.invert(data=data, lam=20, verbose=True) # , mesh=mesh

        model_vals = np.array(model).tolist() 
        parad = mgr.paraDomain
        x = [c.x() for c in parad.cellCenters()]
        y = [c.y() for c in parad.cellCenters()]

        self.inverted_x = x
        self.inverted_y = y
        self.inverted_res = model_vals

    
    
    def afficherInversion(self):
        
        xi = np.linspace(min(self.inverted_x), max(self.inverted_x), 200)
        yi = np.linspace(min(self.inverted_y), max(self.inverted_y), 100)
        xi, yi = np.meshgrid(xi, yi)

        zi = griddata((self.inverted_x, self.inverted_y), self.inverted_res, (xi, yi), method='cubic')

        plt.figure(figsize=(12, 5))
        cntr = plt.contourf(xi, yi, zi, levels=100, cmap="Spectral_r")
        plt.colorbar(cntr, label="Résistivité ($\\Omega m$)")

        plt.xlabel("Distance (m)")
        plt.ylabel("Profondeur (m)")
        plt.title("Coupe de Résistivité Inversée")
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