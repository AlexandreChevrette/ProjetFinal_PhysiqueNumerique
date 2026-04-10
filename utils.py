import numpy as np
import matplotlib.pyplot as plt
import numba
from scipy.interpolate import griddata
import pygimli as pg
from pygimli.physics import ert
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.sparse.linalg import cg

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

    # Conditions frontières
    for i in range(nx):
        V[0, i] = 0.0
        V[ny-1, i] = V[ny-2, i]

    for j in range(ny):
        V[j, 0] = V[j, 1]
        V[j, nx-1] = V[j, nx-2]


    return np.sqrt(err_red + err_black)

cmap = "copper"

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
        self.electrodeList = np.array([])
        self.electrodeMesuresList = np.array([])
        self.listeResistanceApparente = np.array([])
        self.listeAB2 = np.array([])
        self.listeX = np.array([])
        self.listeZ = np.array([])
        self.listePseudoSection = np.array([])
        self.__genererSigma__()
        self.__genererCoefficients__()
        
    def __genererSigma__(self):
        self.matriceSigma = np.ones((self.ny, self.nx), dtype=np.float64) * (1/250)

        yy, xx = np.meshgrid(np.arange(self.ny), np.arange(self.nx), indexing='ij')
        self.matriceSigma[(yy-10)**2 + (xx-(self.nx//2))**2 <= 5**2] = 1/1000

        self.matriceSigma[0, :] = 1e-12

    def __genererCoefficients__(self):
        # On calcule seulement les coefficients pour le solveur
        s = self.matriceSigma[1:-1, 1:-1]
        sip = self.matriceSigma[1:-1, 2:]
        sim = self.matriceSigma[1:-1, :-2]
        sjp = self.matriceSigma[2:, 1:-1]
        sjm = self.matriceSigma[:-2, 1:-1]

        # Moyennes harmoniques (stabilité numérique pour le solveur)
        self.sigma_ifhs[1:-1, 1:-1] = 2.0 * s * sip / (s + sip + 1e-20)
        self.sigma_ibhs[1:-1, 1:-1] = 2.0 * s * sim / (s + sim + 1e-20)
        self.sigma_jfhs[1:-1, 1:-1] = 2.0 * s * sjp / (s + sjp + 1e-20)
        self.sigma_jbhs[1:-1, 1:-1] = 2.0 * s * sjm / (s + sjm + 1e-20)

        self.sigma_deno = (self.sigma_ifhs + self.sigma_ibhs + 
                           self.sigma_jfhs + self.sigma_jbhs)
    
    def __genererCourant__(self):
        for electrode in self.electrodeList:
            self.matriceCourant[electrode.posY, electrode.posX] = electrode.courant
        
    def placerElectrode(self, posX, posY, courant):
        self.electrodeList = np.append(self.electrodeList, Electrode(posX,posY,courant))

    def placerElectrodeMesure(self, posX, posY):
        self.electrodeMesuresList = np.append(self.electrodeMesuresList, ElectrodeMesure(posX, posY))

    def enleverElectrodes(self):
        self.electrodeList = np.array([])

class Solveur:
    def __init__(self, sol : Sol):
        # référence vers les valeurs du sol
        self.sol = sol

    def calculerPotentiel(self, I=None):
        # voir sources

        it=0
        tol = 1e-2
        h = 1
        erreur = 1
        niter= 1000000
        V = self.sol.matricePotentiel #reference
        self.sol.__genererCourant__()
        if I is None:
            I = self.sol.matriceCourant.copy()
        sigma_ifhs = self.sol.sigma_ifhs.copy()
        sigma_ibhs = self.sol.sigma_ibhs.copy()
        sigma_jfhs = self.sol.sigma_jfhs.copy()
        sigma_jbhs = self.sol.sigma_jbhs.copy()
        sigma_deno = self.sol.sigma_deno.copy()

        while erreur > tol and it < niter:

            erreur = rb_gauss_seidel(V, sigma_ifhs,sigma_ibhs,sigma_jfhs,sigma_jbhs,sigma_deno, I, h)

            it += 1  
            # print(f"Erreur: {erreur}")

    def calculerResApparente(self, courantInjection):
        if (len(self.sol.electrodeMesuresList) != 2):
            print("Seulement deux sondes de mesures doivent être utilisés pour calculer la resistance apparente")
            return
        
        M = self.sol.electrodeMesuresList[0]
        N = self.sol.electrodeMesuresList[1]
        coord_ab = self.__genererPositionsAB__(M, N)
        
        
        listeRho = []
        listeAB2 = []
        for (a, b) in coord_ab:
            rho, ab2 = self.__calculerUnAB__(a, b, M.posX, N.posX, courantInjection)
            listeRho.append(rho)
            listeAB2.append(ab2)

        self.sol.listeResistanceApparente = np.array(listeRho)
        self.sol.listeAB2 = np.array(listeAB2)

    def __calculerUnAB__(self, a, b, M, N, courantInjection):
        self.sol.matriceCourant = np.zeros((self.sol.ny,self.sol.nx))
        # self.sol.matricePotentiel = np.zeros((self.sol.ny,self.sol.nx))
        self.sol.enleverElectrodes()

        self.sol.placerElectrode(a, 1, courantInjection)
        self.sol.placerElectrode(b, 1, -courantInjection)
        self.sol.__genererCourant__()
        self.calculerPotentiel()
        
        dV = self.sol.matricePotentiel[1, M] - self.sol.matricePotentiel[1, N]

        AB = abs(b - a)
        AB_2 = AB / 2

        AM = abs(M - a)
        BM = abs(M - b)
        AN = abs(N - a)
        BN = abs(N - b)

        K_2D = np.pi / np.log((AN * BM) / (AM * BN)) 
        rho = K_2D * dV / (courantInjection) 

        return rho, AB_2
        
    def __genererPositionsAB__(self, M, N):
        listeA = []
        listeB = []
        A = M.posX - 1
        B = N.posX + 1
        while (A >= 4 and B < self.sol.nx-4):
            listeA.append(A)
            listeB.append(B)
            A -= 1
            B += 1
        return list(zip(listeA, listeB)) 

    def calculerPseudoSection(self, courantInjection, pas = 1):
        coord_abmn = self.__genererPositionsABMN__(pas)
        
        listeRho = []
        listeZ = []
        listeX = []

        # optimisation : (mettre les electrodes les plus similaires proches pour accelerer les calculs    )

        longueur = len(coord_abmn)
        for i, (a, b, M, N) in enumerate(coord_abmn):
            print(f"Pourcentage de calcul de la pseudo-section: {i/longueur*100:.2f} %", end="\r")
            rho, ab2 = self.__calculerUnAB__(a, b, M, N, courantInjection)
            listeRho.append(rho)
            listeZ.append(ab2)
            listeX.append((M + N) / 2)

        self.sol.listePseudoSection = np.array(listeRho)
        self.sol.listeZ = np.array(listeZ)
        self.sol.listeX = np.array(listeX)

    def __genererPositionsABMN__(self, pas):
        listeA = []
        listeB = []
        listeM = []
        listeN = []
        pas = 6

        A_ini = 2
        M_ini = 4
        N_ini = 6
        B_ini = 8
        for i in range((self.sol.nx-B_ini)//(2*pas)):
            A = A_ini
            M = M_ini + i*pas
            N = N_ini + i*pas
            B = B_ini + 2*i*pas
            while (B < self.sol.nx): 
                listeA.append(A)
                listeB.append(B)
                listeM.append(M)
                listeN.append(N)
                A += pas
                M += pas 
                N += pas
                B += pas
        return list(zip(listeA, listeB, listeM, listeN))

class PyGimliInversionSolveur:
    def __init__(self, sol : Sol, solverDirect : Solveur):
        self.sol = sol
        self.solverDirect = solverDirect

    def inversionPyGimli(self, pas):
        coord_abmn = self.solverDirect.__genererPositionsABMN__(pas)


        data = pg.DataContainerERT()

        for i in range(self.sol.nx):
            data.createSensor([i, 0, 0])

        for i, (A, B, M, N) in enumerate(coord_abmn):
            data.createFourPointData(i, A, B, M, N)
            
        data["rhoa"] = self.sol.listePseudoSection

        data.set("k", ert.geometricFactor(data))

        data["err"] = pg.Vector(data.size(), 0.03)

        mgr = ert.ERTManager(data)

        model = mgr.invert(data=data, lam=20, verbose=True) # , mesh=mesh

        model_vals = np.array(model).tolist() 
        parad = mgr.paraDomain
        x = [c.x() for c in parad.cellCenters()]
        y = [c.y() for c in parad.cellCenters()]

        self.sol.inversionX = x
        self.sol.inversionY = y
        self.sol.inversionRes = model_vals

class InversionSolveur:
    def __init__(self, sol : Sol):
        self.solRef = sol
        self.solSolutionne = Sol((self.solRef.tailleX, self.solRef.tailleY), (self.solRef.nx, self.solRef.ny))
        self.solverDirect = Solveur(self.solSolutionne)

    def obtenirResultatsInversion(self):
        return self.solSolutionne.matriceSigma

    def calculerInversion(self, pas, max_iter=10, lam=1.0, alpha=0.1):
        d_obs = self.solRef.listePseudoSection.copy() # données "terrain"
        self.solSolutionne.matriceSigma = np.ones((self.solSolutionne.ny, self.solSolutionne.nx), dtype=np.float64) * (1/2500)

        m = np.log(self.solSolutionne.matriceSigma.ravel())
        L = self.construire_L()
        LtL = L.T @ L
        
        for i in range(max_iter):
            print(f"--- Itération {i+1} ---")
            
            # Forward & Residual
            self.solverDirect.calculerPseudoSection(courantInjection=1.0, pas=pas)
            d_pred = self.solSolutionne.listePseudoSection.copy()
            residual = d_pred - d_obs
            rms = np.sqrt(np.mean(residual**2))
            
            print(f"RMS Error: {rms:.4f}")
            
            if np.isnan(rms):
                print("Erreur : RMS est NaN. L'inversion a divergé.")
                break

            # Jacobien
            J = self.Jacobien_logsigma(pas, courantInjection=1.0)
            print("apres jacobien log simga")
            # Construction du système
            lhs = J.T @ J + lam * LtL
            rhs = J.T @ residual
            print("apres construction systeme") 
            # Résolution robuste
            try:
                dm, _ = cg(lhs, rhs, maxiter=1000)
                dm = np.clip(dm, -0.5, 0.5)

            except Exception as e:
                print(f"Erreur résolution : {e}")
                break
            print("apres resolution systeme")
            # Mise à jour avec relaxation
            m = m + alpha * dm
            
            # Conversion et mise à jour des paramètres physiques
            new_sigma = np.exp(m).reshape((self.solSolutionne.ny, self.solSolutionne.nx))
            
            # Sécurité supplémentaire : on borne la conductivité
            # (Ex: entre 1e-6 et 1 S/m) pour éviter les valeurs absurdes
            self.solSolutionne.matriceSigma = np.clip(new_sigma, 1e-7, 1.0)
            
            self.solSolutionne.__genererCoefficients__()

    def retrosubstitution(A, b): # Pour résoudre un système Ax=b
        m, n = A.shape
        x = np.empty(m, float)
        for i in range(m-1, -1, -1):
            x[i] = b[i]
            for j in range(i+1, n):
                x[i] -= A[i,j] * x[j]
            x[i] = x[i] / A[i,i]

        return x

    def creer_source(self, pos_plus, pos_minus, amplitude=1.0):
        """
        Create source matrix I with +amplitude at pos_plus and -amplitude at pos_minus.
        pos_plus and pos_minus are tuples (y, x)
        """
        I = np.zeros((self.solSolutionne.ny, self.solSolutionne.nx), dtype=np.float64)
        I[pos_plus[0], pos_plus[1]] = amplitude
        I[pos_minus[0], pos_minus[1]] = -amplitude
        return I
    
    def gradient_central(self, V, h=1.0):

        print("Calcul du gradient central...")
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

    def Jacobien(self, pas, courantInjection=1.0):
        print("Calcul du Jacobien...")
        coord_abmn = self.solverDirect.__genererPositionsABMN__(pas)
        nb_config = len(coord_abmn)
        nb_cells = self.solSolutionne.nx * self.solSolutionne.ny
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
                pos_plus=(0, a),
                pos_minus=(0, b),
                amplitude=courantInjection
            )
            self.solverDirect.calculerPotentiel(I = I_fwd)
            V_fwd = self.solSolutionne.matricePotentiel.copy()

            # -----------------------------
            # 2) Adjoint solve (M,N)
            # -----------------------------
            # Since datum is d = V(M) - V(N),
            # adjoint source is +1 at M and -1 at N
            I_adj = self.creer_source(
                pos_plus=(0, M),
                pos_minus=(0, N),
                amplitude=1.0
            )

            self.solverDirect.calculerPotentiel(I = I_adj)
            V_adj = self.solSolutionne.matricePotentiel.copy()
            

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
    
    def Jacobien_logsigma(self, pas, courantInjection=1.0):
        """
        Jacobian wrt m = log(sigma)
        """
        print("Calcul du Jacobien par rapport à log(sigma)...")
        J_sigma = self.Jacobien(pas, courantInjection)
        sigma_flat = self.solSolutionne.matriceSigma.ravel()
        J_logsigma = J_sigma * sigma_flat[np.newaxis, :]
        # print(J_logsigma.shape)
        return J_logsigma
    
    def construire_L(self):
        """
        Build first-order smoothness matrix L such that Lm penalizes differences
        between neighboring cells.
        """
        print("Construction de la matrice de lissage L...")
        n = self.solSolutionne.ny * self.solSolutionne.nx
        rows = []

        def idx(j, i):
            return j * self.solSolutionne.nx + i

        # horizontal differences
        for j in range(1, self.solSolutionne.ny-1):
            for i in range(1, self.solSolutionne.nx-2):
                row = np.zeros(n)
                row[idx(j, i)] = -1.0
                row[idx(j, i+1)] = 1.0
                rows.append(row)

        # vertical differences
        for j in range(1, self.solSolutionne.ny-2):
            for i in range(1, self.solSolutionne.nx-1):
                row = np.zeros(n)
                row[idx(j, i)] = -1.0
                row[idx(j+1, i)] = 1.0
                rows.append(row)

        # print(np.array(rows, dtype=np.float64).shape)

        return np.array(rows, dtype=np.float64)

class Visualisation:
    def __init__(self, sol : Sol):
         self.sol = sol 
    
    def __afficherImage__(self, matrice, label):
        fontsize = 15
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(matrice, origin='lower', cmap=cmap)
        plt.gca().invert_yaxis()
        # plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="2%", pad=0.1)  # 5% thickness, 0.1 pad
        cbar = plt.colorbar(im, cax=cax, orientation='horizontal', pad = 0.05)
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
        cbar.set_label(label, fontsize=fontsize)
        cbar.ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)
        ax.set_xlabel('Position X (m)', fontsize=fontsize)
        ax.set_ylabel('Profondeur (m)', fontsize=fontsize)
        plt.show()

    def afficherSigma(self):
        self.__afficherImage__(self.sol.matriceSigma, "Conductivité (S/m)")

    def afficherCourant(self):
        self.__afficherImage__(self.sol.matriceCourant, "Courant (A)")

    def afficherPotentiel(self):
        plt.figure()
        plt.contour(self.sol.matricePotentiel, levels=500)
        plt.show()
    
    def afficherPotentielImSHOW(self):
        self.__afficherImage__(self.sol.matricePotentiel, "Potentiel (V)")

    def afficherInversion(self):
        
        xi = np.linspace(min(self.sol.inverted_x), max(self.sol.inverted_x), self.sol.nx)
        yi = np.linspace(min(self.sol.inverted_y), max(self.sol.inverted_y), self.sol.nx//2)
        xi, yi = np.meshgrid(xi, yi)

        zi = griddata((self.sol.inverted_x, self.sol.inverted_y), self.sol.inverted_res, (xi, yi), method='cubic')

        fontsize = 15
        fig, ax = plt.subplots(figsize=(12, 8))
        cntr = ax.contourf(xi, yi, zi, levels=100, cmap=cmap)
        # plt.gca().invert_yaxis()
        plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="2%", pad=0.1)
        cbar = plt.colorbar(cntr, cax=cax, orientation='horizontal', pad=0.05)
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
        cbar.set_label("Résistivité ($\\Omega m$)", fontsize=fontsize)
        cbar.ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)
        ax.set_xlabel("Distance (m)", fontsize=fontsize)
        ax.set_ylabel("Profondeur (m)", fontsize=fontsize)
        plt.show()

    def afficherSimulationComplete(self):
        fontsize = 13
        fig = plt.figure(figsize=(22, 14))
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92, hspace=0.4, wspace=0.35)

        # Sigma takes the full top row
        ax1 = fig.add_subplot(2, 1, 1)
        # Pseudo-section and Inversion share the bottom row
        ax2 = fig.add_subplot(2, 2, 3)
        ax3 = fig.add_subplot(2, 2, 4)

        def add_colorbar(fig, ax, mappable, label):
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("top", size="2%", pad=0.1)
            cbar = fig.colorbar(mappable, cax=cax, orientation='horizontal', pad=0.05)
            cbar.ax.xaxis.set_ticks_position('top')
            cbar.ax.xaxis.set_label_position('top')
            cbar.set_label(label, fontsize=fontsize)
            cbar.ax.tick_params(axis='x', labelsize=fontsize)
            return cbar

        def style_ax(ax, xlabel, ylabel, title):
            ax.tick_params(axis='x', labelsize=fontsize)
            ax.tick_params(axis='y', labelsize=fontsize)
            ax.set_xlabel(xlabel, fontsize=fontsize)
            ax.set_ylabel(ylabel, fontsize=fontsize)
            # Place label in figure coordinates above the axes, after colorbar is appended
            ax.annotate(title, xy=(0.5, 1.2), xycoords='axes fraction',
                        fontsize=fontsize + 1, fontweight='bold', ha='left', va='bottom')

        # --- a) Sigma (full top row) ---
        matriceSigma = self.sol.matriceSigma.copy()
        matriceSigma = matriceSigma[1:,:] 

        im_sigma = ax1.imshow(1/matriceSigma, origin='lower', cmap=cmap)
        ax1.invert_yaxis()
        add_colorbar(fig, ax1, im_sigma, "Résistivité ($\\Omega m$)")
        style_ax(ax1, "Position X (m)", "Profondeur (m)", "a)")

        # --- b) Pseudo-section (bottom-left) ---
        xi2 = np.linspace(self.sol.listeX.min(), self.sol.listeX.max(), self.sol.nx)
        zi2 = np.linspace(self.sol.listeZ.min(), self.sol.listeZ.max(), self.sol.nx // 2)
        XI, ZI = np.meshgrid(xi2, zi2)
        RHOI = griddata((self.sol.listeX, self.sol.listeZ), self.sol.listePseudoSection, (XI, ZI), method='cubic')
        contourf_ps = ax2.contourf(XI, ZI, RHOI, levels=100, cmap=cmap)
        ax2.invert_yaxis()
        ax2.scatter(self.sol.listeX, self.sol.listeZ, s=10, color='white', edgecolors='black', label='Points de mesure')
        add_colorbar(fig, ax2, contourf_ps, "Résistivité apparente ($\\Omega m$)")
        style_ax(ax2, "Position X (m)", "Profondeur Apparente (AB/2) (m)", "b)")

        # --- c) Inversion (bottom-right) ---
        self.sol.inverted_y = np.abs(self.sol.inverted_y)
        xi = np.linspace(min(self.sol.inverted_x), max(self.sol.inverted_x), self.sol.nx)
        yi = np.linspace(min(self.sol.inverted_y), max(self.sol.inverted_y), self.sol.nx // 2)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((self.sol.inverted_x, self.sol.inverted_y), self.sol.inverted_res, (xi, yi), method='cubic')
        cntr = ax3.contourf(xi, yi, zi, levels=100, cmap=cmap)
        ax3.invert_yaxis()
        add_colorbar(fig, ax3, cntr, "Résistivité ($\\Omega m$)")
        style_ax(ax3, "Position X (m)", "Profondeur (m)", "c)")

        plt.show()
    
    def afficherResistanceApparente(self):
        plt.figure()
        plt.plot(self.sol.listeAB2, self.sol.listeResistanceApparente)
        plt.ylabel(r"Résistivité apparente ($\Omega$m)")
        plt.xlabel("Demi-distance entre les électrodes (m)")
        plt.gca().invert_yaxis()
        plt.show()

    def afficherPseudoSection(self):
        xi = np.linspace(self.sol.listeX.min(), self.sol.listeX.max(), self.sol.nx)
        zi = np.linspace(self.sol.listeZ.min(), self.sol.listeZ.max(), self.sol.nx//2)

        XI, ZI = np.meshgrid(xi, zi)

        # interpolation
        RHOI = griddata((self.sol.listeX, self.sol.listeZ), self.sol.listePseudoSection, (XI, ZI), method='cubic')
        
        fontsize = 15
        fig, ax = plt.subplots(figsize=(12, 8))
        contourf = ax.contourf(XI, ZI, RHOI, levels=100, cmap=cmap)
        plt.gca().invert_yaxis()  # profondeur vers le bas
        # points de mesure (optionnel mais pro)
        ax.scatter(self.sol.listeX, self.sol.listeZ, s=10, color='white', edgecolors='black', label='Points de mesure')
        
        plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="2%", pad=0.1)
        cbar = plt.colorbar(contourf, cax=cax, orientation='horizontal', pad=0.05)
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
        cbar.set_label("Résistivité apparente ($\\Omega m$)", fontsize=fontsize)
        cbar.ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)
        ax.set_xlabel("Position X [m]", fontsize=fontsize)
        ax.set_ylabel("Profondeur Apparente (AB/2) [m]", fontsize=fontsize)
        
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


