import numpy as np
import matplotlib.pyplot as plt
import numba
from scipy.interpolate import griddata
import pandas as pd
import os
# import pygimli as pg
# from pygimli.physics import ert

@numba.njit(fastmath=True, parallel=True)
def rb_gauss_seidel(
    V,
    sigma_ifhs, sigma_ibhs,
    sigma_jfhs, sigma_jbhs,
    sigma_deno,
    I,
    h
):
    # formule de poisson généralisée

    ny, nx = V.shape
    # h2 = h * h  chat me dit de ne pas utiliser h^2 lorsqu'on est en A

    

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
                I[j, i]
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
        self.matricePotentiel_adj = np.zeros((self.ny,self.nx)) 
        self.matriceCourant = np.zeros((self.ny,self.nx)) 
        # self.__genererSigma__()
        self.initialiserModele()
        self.electrodeList = np.array([])
        self.electrodeMesuresList = np.array([])
        self.listeResistanceApparente = np.array([])
        self.listeAB2 = np.array([])
        self.listeX = np.array([])
        self.listeZ = np.array([])
        self.listePseudoSection = np.array([])
        self.liste_mat_pot = np.array([])
        self.electrodeList_adj = np.array([])
        self.electrodeMesuresList_adj = np.array([])


    def initialiserModele(self):
        """
        Create initial conductivity model only once.
        """
        self.matriceSigma = np.ones((self.ny, self.nx), dtype=np.float64) * (1/5000)

        yy, xx = np.meshgrid(np.arange(self.ny), np.arange(self.nx), indexing='ij')
        self.matriceSigma[(yy-90)**2 + (xx-30)**2 <= 5**2] = 1/1000

        # bottom boundary
        self.matriceSigma[-1, :] = 1e-12

        self.__genererSigma__()

    def __genererSigma__(self):
        # voir les sources et changer le setting de resistivité
        # self.matriceSigma = np.random.uniform(low=1, high=10, size=(self.ny, self.nx))
        self.matriceSigma = np.ones((self.ny,self.nx))* 1/5000
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
    
    def __genererCourant__(self, adj=False):
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
        # self.electrodeMesuresList_adj = np.append(self.electrodeMesuresList_adj, Electrode(posX,posY,courant))

    def placerElectrodeMesure(self, posX, posY):
        self.electrodeMesuresList = np.append(self.electrodeMesuresList, ElectrodeMesure(posX, posY))
        # self.electrodeList_adj = np.append(self.electrodeList_adj, ElectrodeMesure(posX, posY))


    def enleverElectrodes(self):
        self.electrodeList = np.array([])


    def resoudre_potentiel(self, I, V_init=None, tol=1e-2, niter=100000):
        """
        Solve div(sigma grad V) = -I
        Returns a new potential matrix V.
        """
        if V_init is None:
            V = np.zeros((self.ny, self.nx), dtype=np.float64)
        else:
            V = V_init.copy()

        sigma_ifhs = self.sigma_ifhs
        sigma_ibhs = self.sigma_ibhs
        sigma_jfhs = self.sigma_jfhs
        sigma_jbhs = self.sigma_jbhs
        sigma_deno = self.sigma_deno

        erreur = 1.0
        it = 0
        h = 1.0

        while erreur > tol and it < niter:
            erreur = rb_gauss_seidel(
                V,
                sigma_ifhs, sigma_ibhs,
                sigma_jfhs, sigma_jbhs,
                sigma_deno,
                I,
                h
            )
            it += 1

        return V


    def calculerPotentiel(self):
        self.__genererCourant__()
        self.matricePotentiel = self.resoudre_potentiel(self.matriceCourant)
    

    def creer_source(self, pos_plus, pos_minus, amplitude=1.0):
        """
        Create source matrix I with +amplitude at pos_plus and -amplitude at pos_minus.
        pos_plus and pos_minus are tuples (y, x)
        """
        I = np.zeros((self.ny, self.nx), dtype=np.float64)
        I[pos_plus[0], pos_plus[1]] = amplitude
        I[pos_minus[0], pos_minus[1]] = -amplitude
        return I
    

    def gradient_central(self, V, h=1.0):
        dVdx = np.zeros_like(V)
        dVdy = np.zeros_like(V)

        dVdx[:, 1:-1] = (V[:, 2:] - V[:, :-2]) / (2*h)
        dVdy[1:-1, :] = (V[2:, :] - V[:-2, :]) / (2*h)

        # simple boundary handling
        dVdx[:, 0] = dVdx[:, 1]
        dVdx[:, -1] = dVdx[:, -2]
        dVdy[0, :] = dVdy[1, :]
        dVdy[-1, :] = dVdy[-2, :]

        return dVdx, dVdy


    def Jacobien(self, courantInjection=1.0):
        coord_abmn = self.__genererPositionsABMN__()
        nb_config = len(coord_abmn)
        nb_cells = self.nx*self.ny
        # Jacobien de la position et jacobien de la conductivité concaténé
        # Comment un petit changement de position change le voltage
        # Comment un petit changement de conductivité change le voltage

        # one row per measurement, one column per cell
        J = np.zeros((nb_config, nb_cells), dtype=np.float64)

        for k, (a, b, M, N) in enumerate(coord_abmn):
            # print(f"Jacobian row {k+1}/{nb_config}")

            # -----------------------------
            # 1) Forward solve (A,B)
            # -----------------------------
            I_fwd = self.creer_source(
                pos_plus=(self.ny-2, a),
                pos_minus=(self.ny-2, b),
                amplitude=courantInjection
            )

            V_fwd = self.resoudre_potentiel(I_fwd)

            # -----------------------------
            # 2) Adjoint solve (M,N)
            # -----------------------------
            # Since datum is d = V(M) - V(N),
            # adjoint source is +1 at M and -1 at N
            I_adj = self.creer_source(
                pos_plus=(self.ny-2, M),
                pos_minus=(self.ny-2, N),
                amplitude=1.0
            )

            V_adj = self.resoudre_potentiel(I_adj)

            # -----------------------------
            # 3) Compute gradients
            # -----------------------------
            dVdx_fwd, dVdy_fwd = self.gradient_central(V_fwd)
            dVdx_adj, dVdy_adj = self.gradient_central(V_adj)

            # -----------------------------
            # 4) Sensitivity field
            # -----------------------------
            sens = -(dVdx_fwd * dVdx_adj + dVdy_fwd * dVdy_adj)

            J[k, :] = sens.ravel()
        
        return J
    

    def Jacobien_logsigma(self, courantInjection=1.0):
        """
        Jacobian wrt m = log(sigma)
        """
        J_sigma = self.Jacobien(courantInjection)
        sigma_flat = self.matriceSigma.ravel()
        J_logsigma = J_sigma * sigma_flat[np.newaxis, :]
        print(J_logsigma.shape)
        return J_logsigma
    

    def construire_L(self):
        """
        Build first-order smoothness matrix L such that Lm penalizes differences
        between neighboring cells.
        """
        n = self.ny * self.nx
        rows = []

        def idx(j, i):
            return j * self.nx + i

        # horizontal differences
        for j in range(1, self.ny-1):
            for i in range(1, self.nx-2):
                row = np.zeros(n)
                row[idx(j, i)] = -1.0
                row[idx(j, i+1)] = 1.0
                rows.append(row)

        # vertical differences
        for j in range(1, self.ny-2):
            for i in range(1, self.nx-1):
                row = np.zeros(n)
                row[idx(j, i)] = -1.0
                row[idx(j+1, i)] = 1.0
                rows.append(row)

        print(np.array(rows, dtype=np.float64).shape)

        return np.array(rows, dtype=np.float64)


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

        return self.listeResistanceApparente.copy()


    def __calculerUnAB__(self, a, b, M, N, courantInjection, adj=False):

        self.matriceCourant = np.zeros((self.ny,self.nx))
        # self.matricePotentiel = np.zeros((self.ny,self.nx))
        self.enleverElectrodes()

        self.placerElectrode(a, self.ny-2, courantInjection)
        self.placerElectrode(b, self.ny-2, -courantInjection)
        self.__genererCourant__()
        self.calculerPotentiel()
        
        dV = self.matricePotentiel[self.ny-2, M] - self.matricePotentiel[self.ny-2, N]

        AB = abs(b - a)
        AB_2 = AB / 2

        AM = abs(M - a)
        BM = abs(M - b)
        AN = abs(N - a)
        BN = abs(N - b)

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
        pseudo = self.listePseudoSection.copy()
        self.listeZ = np.array(listeZ)
        self.listeX = np.array(listeX)

        return pseudo
    

    def calculerInversion(self, d_obs):
        
        # Modèle initial
        # m = np.ones((self.ny, self.nx))*np.mean(d_obs)
        self.matricePotentiel = np.ones((self.ny, self.nx))*np.mean(d_obs)

        max_iter = 100000
        # for iter in range(max_iter):
        self.calculerPotentiel()
        data_pred = self.calculerResApparente(1.0)

        print(d_obs.shape)
        print(data_pred.shape)

        # residual = data_pred - d_obs

        




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

    def enregistrerData(self, PATH, name):
        coord_abmn = self.__genererPositionsABMN__()
        # Les coordonnées des électrodes A, B, M et N doivent être en indice
        # pour l'inversion, donc j'ai fait commencer les position à 0, 1, 2, ...
        A = [(a[0]//2)-1 for a in coord_abmn]
        B = [(b[1]//2)-1 for b in coord_abmn]
        M = [(m[2]//2)-1 for m in coord_abmn]
        N = [(n[3]//2)-1 for n in coord_abmn]
        data_sensors = {
            'ax': A, 'ay':np.zeros(len(A)),
            'bx': B, 'by':np.zeros(len(B)),
            'mx': M, 'my':np.zeros(len(M)),
            'nx': N, 'ny':np.zeros(len(N)),
            'rho': self.listePseudoSection,
        }
        
        data_measures = {
            'x': np.arange(2, self.nx, 2), 
            'y': np.zeros(len(np.arange(2, self.nx, 2)))
        }

        df_sensors = pd.DataFrame(data_sensors)
        df_measures = pd.DataFrame(data_measures)

        with pd.ExcelWriter(os.path.join(PATH, name)) as writer:
            df_sensors.to_excel(writer, sheet_name="Sensors", index=False)
            df_measures.to_excel(writer, sheet_name="Measurements", index=False)

        print(f"Enregistrement effectué à la destination {os.path.join(PATH, name)}")


    def inversion(self, PATH):
        df_sensors = pd.read_excel(PATH, sheet_name="Sensors")
        df_measure = pd.read_excel(PATH, sheet_name="Measurements")

        data = pg.DataContainerERT()

        for i, row in df_sensors.iterrows():
            data.createSensor([row['x'], row['y'], row.get('z', 0)])

        for i, row in df_measure.iterrows():
            data.createFourPointData(i, 
                                    int(row['ax']), 
                                    int(row['bx']), 
                                    int(row['mx']), 
                                    int(row['nx']))
            
        data["rhoa"] = df_measure["rho"].values

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
        
        return np.array(x), np.array(y), np.array(model_vals)
    
    
    def afficherInversion(self):
        
        xi = np.linspace(min(self.inverted_x), max(self.inverted_x), 200)
        yi = np.linspace(min(self.inverted_y), max(self.inverted_y), 100)
        xi, yi = np.meshgrid(xi, yi)

        zi = griddata((self.inverted_x, self.inverted_y), self.inverted_res, (xi, yi), method='cubic')

        plt.figure(figsize=(12, 5))
        cntr = plt.contourf(xi, yi, zi, levels=100, cmap="Spectral_r")
        plt.colorbar(cntr, label="Résistivité ($\Omega m$)")

        plt.xlabel("Distance (m)")
        plt.ylabel("Profondeur (m)")
        plt.title("Coupe de Résistivité Inversée")
        plt.show()


    def retrosubstitution(A, b): # Pour résoudre un système Ax=b
        m, n = A.shape
        x = np.empty(m, float)
        for i in range(m-1, -1, -1):
            x[i] = b[i]
            for j in range(i+1, n):
                x[i] -= A[i,j] * x[j]
            x[i] = x[i] / A[i,i]

        return x



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