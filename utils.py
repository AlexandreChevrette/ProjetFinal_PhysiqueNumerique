import numpy as np
import matplotlib.pyplot as plt
import numba
from scipy.interpolate import griddata
import pygimli as pg
from pygimli.physics import ert
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
        self.matriceSigma[(yy-10)**2 + (xx-(self.nx//2))**2 <= 5**2] = 1/1000

       
        self.matriceSigma[0,:] = 0

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
        
    def __afficher__image__(self, matrice, label):
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
        self.__afficher__image__(self.matriceSigma, "Conductivité (S/m)")

    def afficherCourant(self):
        self.__afficher__image__(self.matriceCourant, "Courant (A)")

    def afficherPotentiel(self):
        plt.figure()
        plt.contour(self.matricePotentiel, levels=500)
        plt.show()
    
    def afficherPotentielImSHOW(self):
        self.__afficher__image__(self.matricePotentiel, "Potentiel (V)")

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

        self.placerElectrode(a, 1, courantInjection)
        self.placerElectrode(b, 1, -courantInjection)
        self.__genererCourant__()
        self.calculerPotentiel()
        
        dV = self.matricePotentiel[1, M] - self.matricePotentiel[1, N]

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
        while (A >= 4 and B < self.nx-4):
            listeA.append(A)
            listeB.append(B)
            A -= 1
            B += 1
        return list(zip(listeA, listeB))
    
    def afficherResistanceApparente(self):
        plt.figure()
        plt.plot(self.listeAB2, self.listeResistanceApparente)
        plt.ylabel(r"Résistivité apparente ($\Omega$m)")
        plt.xlabel("Demi-distance entre les électrodes (m)")
        plt.gca().invert_yaxis()
        plt.show()

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

        self.listePseudoSection = np.array(listeRho)
        self.listeZ = np.array(listeZ)
        self.listeX = np.array(listeX)

    def __genererPositionsABMN__(self, pas):
        listeA = []
        listeB = []
        listeM = []
        listeN = []

        A_ini = 2 + self.nx//4 
        M_ini = 4 + self.nx//4
        N_ini = 6 + self.nx//4
        B_ini = 8 + self.nx//4
        for i in range((self.nx//2-B_ini)//(2*pas)):
            A = A_ini
            M = M_ini + i*pas
            N = N_ini + i*pas
            B = B_ini + 2*i*pas
            while (B < self.nx/4*3): 
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
        xi = np.linspace(self.listeX.min(), self.listeX.max(), self.nx)
        zi = np.linspace(self.listeZ.min(), self.listeZ.max(), self.nx//2)

        XI, ZI = np.meshgrid(xi, zi)

        # interpolation
        RHOI = griddata((self.listeX, self.listeZ), self.listePseudoSection, (XI, ZI), method='cubic')
        
        fontsize = 15
        fig, ax = plt.subplots(figsize=(12, 8))
        contourf = ax.contourf(XI, ZI, RHOI, levels=100, cmap=cmap)
        plt.gca().invert_yaxis()  # profondeur vers le bas
        # points de mesure (optionnel mais pro)
        ax.scatter(self.listeX, self.listeZ, s=10, color='white', edgecolors='black', label='Points de mesure')
        
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

    def inversion(self, pas):
        coord_abmn = self.__genererPositionsABMN__(pas)


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
        
        xi = np.linspace(min(self.inverted_x), max(self.inverted_x), self.nx)
        yi = np.linspace(min(self.inverted_y), max(self.inverted_y), self.nx//2)
        xi, yi = np.meshgrid(xi, yi)

        zi = griddata((self.inverted_x, self.inverted_y), self.inverted_res, (xi, yi), method='cubic')

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
        im_sigma = ax1.imshow(1/self.matriceSigma, origin='lower', cmap=cmap)
        ax1.invert_yaxis()
        add_colorbar(fig, ax1, im_sigma, "Résistivité ($\\Omega m$)")
        style_ax(ax1, "Position X (m)", "Profondeur (m)", "a)")

        # --- b) Pseudo-section (bottom-left) ---
        xi2 = np.linspace(self.listeX.min(), self.listeX.max(), self.nx)
        zi2 = np.linspace(self.listeZ.min(), self.listeZ.max(), self.nx // 2)
        XI, ZI = np.meshgrid(xi2, zi2)
        RHOI = griddata((self.listeX, self.listeZ), self.listePseudoSection, (XI, ZI), method='cubic')
        contourf_ps = ax2.contourf(XI, ZI, RHOI, levels=100, cmap=cmap)
        ax2.invert_yaxis()
        ax2.scatter(self.listeX, self.listeZ, s=10, color='white', edgecolors='black', label='Points de mesure')
        add_colorbar(fig, ax2, contourf_ps, "Résistivité apparente ($\\Omega m$)")
        style_ax(ax2, "Position X (m)", "Profondeur Apparente (AB/2) (m)", "b)")

        # --- c) Inversion (bottom-right) ---
        self.inverted_y = np.abs(self.inverted_y)
        xi = np.linspace(min(self.inverted_x), max(self.inverted_x), self.nx)
        yi = np.linspace(min(self.inverted_y), max(self.inverted_y), self.nx // 2)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((self.inverted_x, self.inverted_y), self.inverted_res, (xi, yi), method='cubic')
        cntr = ax3.contourf(xi, yi, zi, levels=100, cmap=cmap)
        ax3.invert_yaxis()
        add_colorbar(fig, ax3, cntr, "Résistivité ($\\Omega m$)")
        style_ax(ax3, "Distance (m)", "Profondeur (m)", "c)")

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