"""
Simulation de tomographie de résistivité électrique (ERT) en 2D.

Ce module implémente un simulateur complet de prospection géophysique par méthode
électrique, incluant:

- La résolution de l'équation de Poisson par la méthode de Gauss-Seidel rouge-noir (Red-Black)
  accélérée avec Numba,
- Le calcul de pseudo-sections de résistivité apparente,
- L'intégration avec PyGIMLi pour l'inversion,
- La visualisation des résultats.

Example:
    Utilisation typique::

        sol = Sol((100, 30))
        solveur = Solveur(sol)
        solveur.calculerPseudoSection(courantInjection=1.0, pas=1)
        visu = Visualisation(sol)
        visu.afficherPseudoSection()
"""

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
    """
    Effectue une itération de la méthode de Gauss-Seidel rouge-noir pour résoudre
    l'équation de Poisson discrétisée en 2D.

    La méthode rouge-noir (Red-Black) met à jour les nœuds en deux passes
    alternées (pairs et impairs) pour permettre la parallélisation avec Numba.
    Les conditions aux limites appliquées sont :

    - Haut (j=0) : Dirichlet (V=0, surface isolante)
    - Bas (j=ny-1) : Neumann (dV/dy=0, flux nul)
    - Gauche / Droite : Neumann (dV/dx=0, flux nul)

    Args:
        V (np.ndarray): Matrice du potentiel électrique, de forme (ny, nx). Modifiée en place.
        sigma_ifhs (np.ndarray): Conductivité harmonique moyenne vers i+1 (demi-pas avant en x).
        sigma_ibhs (np.ndarray): Conductivité harmonique moyenne vers i-1 (demi-pas arrière en x).
        sigma_jfhs (np.ndarray): Conductivité harmonique moyenne vers j+1 (demi-pas avant en y).
        sigma_jbhs (np.ndarray): Conductivité harmonique moyenne vers j-1 (demi-pas arrière en y).
        sigma_deno (np.ndarray): Somme des conductivités harmoniques (dénominateur de la mise à jour).
        I (np.ndarray): Matrice des sources de courant (A/m²).
        h (float): Pas de la grille (assumé 1 pour cette simulation).

    Returns:
        float: Erreur relative maximale entre l'ancienne et la nouvelle valeur de V,
               utilisée comme critère de convergence.
    """

    ny, nx = V.shape
    h2 = h * h 

    err_red = 0.0
    max_rel_err = 0.0

    # --- Passe rouge : nœuds (j+i) pair ---
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
            
            if abs(new_val) > 1e-12:
                rel_err = abs(d) / abs(new_val)
                max_rel_err = max(max_rel_err, rel_err)

        err_red += tmp

    # --- Passe noire : nœuds (j+i) impair ---
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
            
            if abs(new_val) > 1e-12:
                rel_err = abs(d) / abs(new_val)
                max_rel_err = max(max_rel_err, rel_err)

        err_black += tmp

    # --- Conditions aux limites ---
    for i in range(nx):
        V[0, i] = 0.0           # Haut : Dirichlet V=0
        V[ny-1, i] = V[ny-2, i] # Bas  : Neumann

    for j in range(ny):
        V[j, 0] = V[j, 1]       # Gauche : Neumann
        V[j, nx-1] = V[j, nx-2] # Droite : Neumann

    return max_rel_err

class Sol:
    """
    Représente le modèle de sol discrétisé sur une grille 2D.

    Stocke la distribution de conductivité électrique, les matrices de coefficients
    pour le solveur, les potentiels, les sources de courant, ainsi que les
    électrodes et les résultats de mesures ERT.

    Attributes:
        nx (int): Nombre de colonnes de la grille (axe x).
        ny (int): Nombre de lignes de la grille (axe y / profondeur).
        matriceSigma (np.ndarray): Matrice des conductivités électriques (S/m), forme (ny, nx).
        sigma_ifhs (np.ndarray): Conductivités harmoniques moyennes en x+ (demi-pas avant).
        sigma_ibhs (np.ndarray): Conductivités harmoniques moyennes en x- (demi-pas arrière).
        sigma_jfhs (np.ndarray): Conductivités harmoniques moyennes en y+ (demi-pas avant).
        sigma_jbhs (np.ndarray): Conductivités harmoniques moyennes en y- (demi-pas arrière).
        sigma_deno (np.ndarray): Somme des conductivités harmoniques (dénominateur du solveur).
        matricePotentiel (np.ndarray): Matrice du potentiel électrique calculé (V), forme (ny, nx).
        matriceCourant (np.ndarray): Matrice des sources de courant (A), forme (ny, nx).
        electrodeList (np.ndarray): Liste des électrodes d'injection (objets :class:`Electrode`).
        electrodeMesuresList (np.ndarray): Liste des électrodes de mesure (objets :class:`ElectrodeMesure`).
        listeResistanceApparente (np.ndarray): Résistivités apparentes calculées pour un sondage.
        listeAB2 (np.ndarray): Demi-espacements AB/2 correspondants.
        listeX (np.ndarray): Positions X des points de pseudo-section.
        listeZ (np.ndarray): Profondeurs apparentes des points de pseudo-section.
        listePseudoSection (np.ndarray): Résistivités apparentes de la pseudo-section.

    """

    def __init__(self, nxy: tuple[int, int]):
        """
        Initialise la grille du sol avec la taille donnée.

        Génère automatiquement la distribution de conductivité et les
        coefficients du solveur via :meth:`__genererSigma__` et
        :meth:`__genererCoefficients__`.

        Args:
            nxy (tuple[int, int]): Dimensions de la grille sous la forme (nx, ny),
                où nx est le nombre de colonnes et ny le nombre de lignes.
        """
        
        self.nx = nxy[0]
        self.ny = nxy[1]
        self.matriceSigma = np.zeros((self.ny, self.nx))
        self.sigma_ifhs = np.zeros((self.ny, self.nx))
        self.sigma_ibhs = np.zeros((self.ny, self.nx))
        self.sigma_jfhs = np.zeros((self.ny, self.nx))
        self.sigma_jbhs = np.zeros((self.ny, self.nx))
        self.sigma_deno = np.zeros((self.ny, self.nx))
        self.matricePotentiel = np.zeros((self.ny, self.nx))
        self.matriceCourant = np.zeros((self.ny, self.nx))
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
        """
        Génère la distribution de conductivité électrique du sol.

        Le modèle par défaut est un demi-espace homogène de résistivité 250 Ω·m
        (conductivité 1/250 S/m), avec une anomalie circulaire conductrice de
        résistivité 50 Ω·m centrée horizontalement à la profondeur 10, de rayon 5.

        La première rangée (surface libre) est rendue quasi-isolante (1e-12 S/m)
        pour simuler la condition aux limites supérieure.

        Note:
            Pour modifier le modèle de sol, redéfinir cette méthode ou modifier
            directement ``self.matriceSigma`` avant d'appeler :meth:`__genererCoefficients__`.
        """
        self.matriceSigma = np.ones((self.ny, self.nx), dtype=np.float64) * (1 / 250)

        # Anomalie conductrice circulaire (rayon 5, centre en (nx//2, 10))
        yy, xx = np.meshgrid(np.arange(self.ny), np.arange(self.nx), indexing='ij')
        self.matriceSigma[(yy - 10) ** 2 + (xx - (self.nx // 2)) ** 2 <= 5 ** 2] = 1 / 50

        # Surface quasi-isolante (condition aux limites Dirichlet V=0 en haut)
        self.matriceSigma[0, :] = 1e-12

    def __genererCoefficients__(self):
        """
        Calcule les coefficients de conductivité harmonique sur la grille intérieure.

        Utilise des moyennes harmoniques entre cellules voisines pour une meilleure
        stabilité numérique aux interfaces de conductivité contrastée. Les tableaux
        ``sigma_ifhs``, ``sigma_ibhs``, ``sigma_jfhs``, ``sigma_jbhs`` et
        ``sigma_deno`` sont mis à jour en place.

        Note:
            Un terme de régularisation (1e-20) est ajouté au dénominateur pour
            éviter les divisions par zéro.
        """
        s = self.matriceSigma[1:-1, 1:-1]
        sip = self.matriceSigma[1:-1, 2:]
        sim = self.matriceSigma[1:-1, :-2]
        sjp = self.matriceSigma[2:, 1:-1]
        sjm = self.matriceSigma[:-2, 1:-1]

        # Moyennes harmoniques pour la stabilité aux interfaces
        self.sigma_ifhs[1:-1, 1:-1] = 2.0 * s * sip / (s + sip + 1e-20)
        self.sigma_ibhs[1:-1, 1:-1] = 2.0 * s * sim / (s + sim + 1e-20)
        self.sigma_jfhs[1:-1, 1:-1] = 2.0 * s * sjp / (s + sjp + 1e-20)
        self.sigma_jbhs[1:-1, 1:-1] = 2.0 * s * sjm / (s + sjm + 1e-20)

        self.sigma_deno = (self.sigma_ifhs + self.sigma_ibhs +
                           self.sigma_jfhs + self.sigma_jbhs)

    def __genererCourant__(self):
        """
        Remplit la matrice de courant ``matriceCourant`` à partir de ``electrodeList``.

        Place le courant de chaque électrode d'injection à sa position (posY, posX)
        dans la matrice. Doit être appelée après avoir placé les électrodes avec
        :meth:`placerElectrode`.
        """
        for electrode in self.electrodeList:
            self.matriceCourant[electrode.posY, electrode.posX] = electrode.courant

    def placerElectrode(self, posX: int, posY: int, courant: float):
        """
        Ajoute une électrode d'injection à la liste.

        Args:
            posX (int): Position en colonne (axe x) de l'électrode sur la grille.
            posY (int): Position en ligne (axe y/profondeur) de l'électrode.
            courant (float): Intensité du courant injecté (A). Utiliser +I et -I
                pour un dipôle source-puits.
        """
        self.electrodeList = np.append(self.electrodeList, Electrode(posX, posY, courant))

    def placerElectrodeMesure(self, posX: int, posY: int):
        """
        Ajoute une électrode de mesure de potentiel à la liste.

        Args:
            posX (int): Position en colonne (axe x) sur la grille.
            posY (int): Position en ligne (axe y) sur la grille.
        """
        self.electrodeMesuresList = np.append(
            self.electrodeMesuresList, ElectrodeMesure(posX, posY)
        )

    def enleverElectrodes(self):
        """
        Supprime toutes les électrodes d'injection de la liste.

        Doit être appelée avant de repositionner des électrodes pour une
        nouvelle configuration de mesure.
        """
        self.electrodeList = np.array([])

class Solveur:
    """
    Solveur direct pour le problème électrostatique ERT.

    Résout l'équation de Poisson discrétisée sur la grille du sol à l'aide de
    la méthode itérative de Gauss-Seidel rouge-noir (voir :func:`rb_gauss_seidel`).
    Fournit également les outils pour calculer des résistances apparentes 1D
    (sondage de Schlumberger) et des pseudo-sections 2D.

    Attributes:
        sol (Sol): Référence vers l'objet :class:`Sol` associé.

    """

    def __init__(self, sol: Sol):
        """
        Initialise le solveur avec une référence vers le modèle de sol.

        Args:
            sol (Sol): Objet :class:`Sol` contenant la géométrie et les propriétés du sous-sol.
        """
        self.sol = sol

    def calculerPotentiel(self, I: np.ndarray = None):
        """
        Calcule le champ de potentiel électrique par itérations de Gauss-Seidel rouge-noir.

        Itère jusqu'à ce que l'erreur relative maximale soit inférieure à la tolérance
        (1e-3) ou que le nombre maximal d'itérations (1 000 000) soit atteint.
        Le résultat est stocké dans ``sol.matricePotentiel``.

        Args:
            I (np.ndarray, optional): Matrice de sources de courant (ny, nx) à utiliser.
                Si ``None``, utilise ``sol.matriceCourant`` après appel de
                :meth:`Sol.__genererCourant__`.
        """
        it = 0
        tol = 1e-3
        h = 1
        erreur = 1
        niter = 1000000
        V = self.sol.matricePotentiel.copy()
        self.sol.__genererCourant__()
        if I is None:
            I = self.sol.matriceCourant.copy()
        sigma_ifhs = self.sol.sigma_ifhs.copy()
        sigma_ibhs = self.sol.sigma_ibhs.copy()
        sigma_jfhs = self.sol.sigma_jfhs.copy()
        sigma_jbhs = self.sol.sigma_jbhs.copy()
        sigma_deno = self.sol.sigma_deno.copy()

        while erreur > tol and it < niter:
            erreur = rb_gauss_seidel(
                V, sigma_ifhs, sigma_ibhs, sigma_jfhs, sigma_jbhs, sigma_deno, I, h
            )
            it += 1

        self.sol.matricePotentiel = V.copy()

    def calculerResApparente(self, courantInjection: float):
        """
        Calcule la courbe de sondage de résistivité apparente (1D, type Schlumberger).

        Deux électrodes de mesure M et N doivent être définies dans
        ``sol.electrodeMesuresList``. Le programme fait varier l'espacement AB
        en déplaçant symétriquement A et B à l'extérieur de MN.
        Les résultats sont stockés dans ``sol.listeResistanceApparente`` et
        ``sol.listeAB2``.

        Args:
            courantInjection (float): Intensité du courant injecté (A).

        Note:
            Requiert exactement deux électrodes de mesure. Un message d'erreur
            est affiché si ce n'est pas le cas.
        """
        if len(self.sol.electrodeMesuresList) != 2:
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

    def __calculerUnAB__(
        self,
        a: int,
        b: int,
        M: int,
        N: int,
        courantInjection: float
    ):
        """
        Calcule la résistivité apparente pour une configuration ABMN donnée.

        Réinitialise les électrodes, résout le potentiel, puis applique le
        facteur géométrique en 2D (K_2D) pour obtenir la résistivité apparente.

        Args:
            a (int): Position en x de l'électrode A (injection +).
            b (int): Position en x de l'électrode B (injection -).
            M (int): Position en x de l'électrode de mesure M.
            N (int): Position en x de l'électrode de mesure N.
            courantInjection (float): Intensité du courant injecté (A).

        Returns:
            tuple[float, float]: ``(rho, ab2)`` où ``rho`` est la résistivité
            apparente (Ω·m) et ``ab2`` est le demi-espacement AB/2 (m).
        """
        self.sol.matriceCourant = np.zeros((self.sol.ny, self.sol.nx))
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

        # Facteur géométrique 2D pour un dispositif ABMN en surface
        K_2D = np.pi / np.log((AN * BM) / (AM * BN))
        rho = K_2D * dV / courantInjection

        return rho, AB_2

    def __genererPositionsAB__(self, M, N):
        """
        Génère les paires de positions (A, B) pour un sondage de type Schlumberger.

        Part de A=M-1, B=N+1 et écarte progressivement A et B tant qu'ils
        restent dans les limites de la grille (marge de 4 nœuds).

        Args:
            M (ElectrodeMesure): Électrode de mesure M.
            N (ElectrodeMesure): Électrode de mesure N.

        Returns:
            list[tuple[int, int]]: Liste de paires ``(A, B)`` pour chaque
            espacement croissant.
        """
        listeA = []
        listeB = []
        A = M.posX - 1
        B = N.posX + 1
        while A >= 4 and B < self.sol.nx - 4:
            listeA.append(A)
            listeB.append(B)
            A -= 1
            B += 1
        return list(zip(listeA, listeB))

    def calculerPseudoSection(self, courantInjection: float, pas: int = 1):
        """
        Calcule la pseudo-section de résistivité apparente 2D.

        Génère toutes les configurations ABMN avec le pas donné via
        :meth:`__genererPositionsABMN__`, puis calcule la résistivité apparente
        pour chacune. Les résultats sont stockés dans ``sol.listePseudoSection``,
        ``sol.listeZ`` et ``sol.listeX``.

        Args:
            courantInjection (float): Intensité du courant injecté (A).
            pas (int, optional): Pas de déplacement entre configurations successives
                (en nombre de nœuds). Par défaut 1.
        """
        coord_abmn = self.__genererPositionsABMN__(pas)

        listeRho = []
        listeZ = []
        listeX = []

        longueur = len(coord_abmn)
        for i, (a, b, M, N) in enumerate(coord_abmn):
            print(f"Pourcentage de calcul de la pseudo-section: {i / longueur * 100:.2f} %", end="\r")
            rho, ab2 = self.__calculerUnAB__(a, b, M, N, courantInjection)
            listeRho.append(rho)
            listeZ.append(ab2)
            listeX.append((M + N) / 2)

        self.sol.listePseudoSection = np.array(listeRho)
        self.sol.listeZ = np.array(listeZ)
        self.sol.listeX = np.array(listeX)

    def __genererPositionsABMN__(self, pas: int):
        """
        Génère toutes les configurations ABMN pour un profil ERT.

        Utilise un dispositif de type Wenner-Schlumberger en déplaçant
        le quadripôle ABMN le long du profil avec le pas donné.

        Args:
            pas (int): Incrément de déplacement entre configurations (nœuds).

        Returns:
            list[tuple[int, int, int, int]]: Liste de quadruplets ``(A, B, M, N)``
            représentant les positions des électrodes pour chaque mesure.
        """
        listeA = []
        listeB = []
        listeM = []
        listeN = []
        offset = 0

        A_ini = 2 + offset
        M_ini = 4 + offset
        N_ini = 6 + offset
        B_ini = 8 + offset

        for i in range((self.sol.nx - B_ini) // (2 * pas)):
            A = A_ini
            M = M_ini + i * pas
            N = N_ini + i * pas
            B = B_ini + 2 * i * pas
            while B < (self.sol.nx - offset):
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
    """
    Solveur d'inversion ERT utilisant la bibliothèque PyGIMLi.

    Construit un conteneur de données ERT à partir des résultats de la
    pseudo-section simulée, puis lance l'inversion avec le gestionnaire
    :class:`pygimli.physics.ert.ERTManager`.

    Attributes:
        sol (Sol): Référence vers le modèle de sol.
        solverDirect (Solveur): Solveur direct associé pour la génération des
            configurations ABMN.

    Example:
        ::

            pygimli_solveur = PyGimliInversionSolveur(sol, solveur)
            pygimli_solveur.inversionPyGimli(pas=1)
    """

    def __init__(self, sol: Sol, solverDirect: Solveur):
        """
        Initialise le solveur d'inversion PyGIMLi.

        Args:
            sol (Sol): Modèle de sol contenant les données de pseudo-section
                (``listePseudoSection`` doit être remplie au préalable).
            solverDirect (Solveur): Solveur direct pour accéder aux configurations ABMN.
        """
        self.sol = sol
        self.solverDirect = solverDirect

    def inversionPyGimli(self, pas: int):
        """
        Lance l'inversion ERT avec PyGIMLi à partir des données de pseudo-section.

        Construit le :class:`pygimli.DataContainerERT` à partir des configurations
        ABMN et des résistivités apparentes calculées, puis effectue l'inversion
        avec un facteur de régularisation lambda=20. Les résultats (résistivité
        inversée et coordonnées du domaine paramétrique) sont stockés dans
        ``sol.inversionX``, ``sol.inversionY`` et ``sol.inversionRes``.

        Args:
            pas (int): Pas utilisé pour générer les configurations ABMN
                (doit être le même que celui utilisé lors du calcul de la pseudo-section).
        """
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
        model = mgr.invert(data=data, lam=20, verbose=True)

        model_vals = np.array(model).tolist()
        parad = mgr.paraDomain
        x = [c.x() for c in parad.cellCenters()]
        y = [c.y() for c in parad.cellCenters()]

        self.sol.inversionX = x
        self.sol.inversionY = y
        self.sol.inversionRes = model_vals

class InversionSolveur:
    """
    Solveur d'inversion ERT maison. Implémentation non fonctionnelle, trop instable. Ne pas utiliser.
    """

    def __init__(self, sol: Sol):
        """
        Initialise le solveur d'inversion maison.

        Args:
            sol (Sol): Sol de référence contenant les données de pseudo-section observées.
        """
        self.solRef = sol
        self.solSolutionne = Sol(
            (self.solRef.nx, self.solRef.ny),
            # (self.solRef.nx, self.solRef.ny)
        )
        self.solverDirect = Solveur(self.solSolutionne)

    def obtenirResultatsInversion(self) -> np.ndarray:
        """
        Retourne la matrice de conductivité inversée.

        Returns:
            np.ndarray: Matrice ``(ny, nx)`` des conductivités (S/m) après inversion.
        """
        return self.solSolutionne.matriceSigma

    def calculerInversion(
        self,
        pas: int,
        max_iter: int = 10,
        lam: float = 1.0,
        alpha: float = 0.1
    ):
        """
        Lance la boucle d'inversion Gauss-Newton linéarisée.

        À chaque itération :

        1. Calcule la pseudo-section prédite (problème direct).
        2. Calcule le résidu entre données observées et prédites.
        3. Calcule le Jacobien en log-conductivité.
        4. Résout le système linéaire ``(J^T J + lambda * L^T L) dm = J^T r``
           via le gradient conjugué.
        5. Met à jour le vecteur modèle ``m = log(sigma)`` avec relaxation ``alpha``.

        Args:
            pas (int): Pas pour la génération des configurations ABMN.
            max_iter (int, optional): Nombre maximum d'itérations. Par défaut 10.
            lam (float, optional): Facteur de régularisation (poids du lissage). Par défaut 1.0.
            alpha (float, optional): Facteur de relaxation (longueur du pas de mise à jour).
                Par défaut 0.1.
        """
        d_obs = self.solRef.listePseudoSection.copy()
        self.solSolutionne.matriceSigma = np.ones(
            (self.solSolutionne.ny, self.solSolutionne.nx), dtype=np.float64
        ) * (1 / 250)

        m = np.log(self.solSolutionne.matriceSigma.ravel())
        L = self.construire_L()
        LtL = L.T @ L

        for i in range(max_iter):
            print(f"--- Itération {i + 1} ---")

            # Problème direct et résidu
            self.solverDirect.calculerPseudoSection(courantInjection=1.0, pas=pas)
            d_pred = self.solSolutionne.listePseudoSection.copy()
            residual = d_pred - d_obs
            rms = np.sqrt(np.mean(residual ** 2))
            print(f"RMS Error: {rms:.4f}")

            if np.isnan(rms):
                print("Erreur : RMS est NaN. L'inversion a divergé.")
                break

            # Jacobien en log-conductivité
            J = self.Jacobien_logsigma(pas, courantInjection=1.0)

            # Système normal avec régularisation
            lhs = J.T @ J + lam * LtL
            rhs = J.T @ residual

            try:
                dm, _ = cg(lhs, rhs, maxiter=1000)
                dm = np.clip(dm, -0.5, 0.5)
            except Exception as e:
                print(f"Erreur résolution : {e}")
                break

            # Mise à jour avec relaxation
            m = m + alpha * dm

            new_sigma = np.exp(m).reshape((self.solSolutionne.ny, self.solSolutionne.nx))
            self.solSolutionne.matriceSigma = np.clip(new_sigma, 1e-7, 1.0)
            self.solSolutionne.__genererCoefficients__()

    @staticmethod
    def retrosubstitution(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Résout le système triangulaire supérieur Ax = b par rétro-substitution.

        Args:
            A (np.ndarray): Matrice carrée triangulaire supérieure (m, m).
            b (np.ndarray): Vecteur membre droit de longueur m.

        Returns:
            np.ndarray: Vecteur solution x de longueur m.
        """
        m, n = A.shape
        x = np.empty(m, float)
        for i in range(m - 1, -1, -1):
            x[i] = b[i]
            for j in range(i + 1, n):
                x[i] -= A[i, j] * x[j]
            x[i] = x[i] / A[i, i]
        return x

    def creer_source(
        self,
        pos_plus: tuple,
        pos_minus: tuple,
        amplitude: float = 1.0
    ) -> np.ndarray:
        """
        Crée la matrice de sources de courant pour une paire d'électrodes.

        Args:
            pos_plus (tuple[int, int]): Position ``(y, x)`` de l'électrode source (+).
            pos_minus (tuple[int, int]): Position ``(y, x)`` de l'électrode puits (-).
            amplitude (float, optional): Intensité du courant (A). Par défaut 1.0.

        Returns:
            np.ndarray: Matrice de sources ``(ny, nx)`` avec +amplitude en pos_plus
            et -amplitude en pos_minus.
        """
        I = np.zeros((self.solSolutionne.ny, self.solSolutionne.nx), dtype=np.float64)
        I[pos_plus[0], pos_plus[1]] = amplitude
        I[pos_minus[0], pos_minus[1]] = -amplitude
        return I

    def gradient_central(
        self,
        V: np.ndarray,
        h: float = 1.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calcule le gradient du potentiel par différences finies centrées.

        Applique des différences finies centrées à l'intérieur du domaine et
        des différences décentrées (copie du voisin) aux bords.

        Args:
            V (np.ndarray): Matrice du potentiel électrique (ny, nx).
            h (float, optional): Pas de la grille. Par défaut 1.0.

        Returns:
            tuple[np.ndarray, np.ndarray]: ``(dVdx, dVdy)`` — composantes du gradient
            selon x et y, de même forme que V.
        """
        dVdx = np.zeros_like(V)
        dVdy = np.zeros_like(V)

        dVdx[:, 1:-1] = (V[:, 2:] - V[:, :-2]) / (2 * h)
        dVdy[1:-1, :] = (V[2:, :] - V[:-2, :]) / (2 * h)

        dVdx[:, 0] = dVdx[:, 1]
        dVdx[:, -1] = dVdx[:, -2]
        dVdy[0, :] = dVdy[1, :]
        dVdy[-1, :] = dVdy[-2, :]

        return dVdx, dVdy

    def Jacobien(self, pas: int, courantInjection: float = 1.0) -> np.ndarray:
        """
        Calcule la matrice Jacobienne de sensibilité par la méthode adjointe.

        Pour chaque configuration ABMN, effectue deux résolutions du problème direct :
        l'une avec les sources d'injection (A, B) et l'autre avec les sources adjoint
        (M, N). La sensibilité de chaque mesure à chaque cellule est ensuite obtenue
        par le produit scalaire des gradients des deux champs.

        La formule utilisée est :

        .. math::

            J_{k,p} = -\\nabla V_{fwd}^{(k)} \\cdot \\nabla V_{adj}^{(k)}

        évalué sur chaque cellule p de la grille.

        Args:
            pas (int): Pas pour la génération des configurations ABMN.
            courantInjection (float, optional): Intensité du courant pour le problème direct. Par défaut 1.0.

        Returns:
            np.ndarray: Matrice Jacobienne de forme ``(nb_configs, nx*ny)``.
        """
        coord_abmn = self.solverDirect.__genererPositionsABMN__(pas)
        nb_config = len(coord_abmn)
        nb_cells = self.solSolutionne.nx * self.solSolutionne.ny

        J = np.zeros((nb_config, nb_cells), dtype=np.float64)

        for k, (a, b, M, N) in enumerate(coord_abmn):
            # 1) Résolution directe (source en A, puits en B)
            I_fwd = self.creer_source(
                pos_plus=(0, a),
                pos_minus=(0, b),
                amplitude=courantInjection
            )
            self.solverDirect.calculerPotentiel(I=I_fwd)
            V_fwd = self.solSolutionne.matricePotentiel.copy()

            # 2) Résolution adjointe (source en M, puits en N)
            I_adj = self.creer_source(
                pos_plus=(0, M),
                pos_minus=(0, N),
                amplitude=1.0
            )
            self.solverDirect.calculerPotentiel(I=I_adj)
            V_adj = self.solSolutionne.matricePotentiel.copy()

            # 3) Gradients par différences centrées
            dVdx_fwd, dVdy_fwd = self.gradient_central(V_fwd)
            dVdx_adj, dVdy_adj = self.gradient_central(V_adj)

            # 4) Champ de sensibilité (produit scalaire des gradients)
            sens = -(dVdx_fwd * dVdx_adj + dVdy_fwd * dVdy_adj)
            J[k, :] = sens.ravel()

        return J

    def Jacobien_logsigma(self, pas: int, courantInjection: float = 1.0) -> np.ndarray:
        """
        Calcule le Jacobien par rapport au paramètre ``m = log(sigma)``.

        Applique la règle de dérivation en chaîne :

        .. math::

            \\frac{\\partial d}{\\partial m} = \\frac{\\partial d}{\\partial \\sigma} \\cdot \\sigma

        Args:
            pas (int): Pas pour la génération des configurations ABMN.
            courantInjection (float, optional): Intensité du courant. Par défaut 1.0.

        Returns:
            np.ndarray: Jacobien en log-conductivité de forme ``(nb_configs, nx*ny)``.
        """
        J_sigma = self.Jacobien(pas, courantInjection)
        sigma_flat = self.solSolutionne.matriceSigma.ravel()
        J_logsigma = J_sigma * sigma_flat[np.newaxis, :]
        return J_logsigma

    def construire_L(self) -> np.ndarray:
        """
        Construit la matrice de lissage du premier ordre L.

        L encode les différences finies entre cellules voisines (horizontales
        et verticales) sur la grille intérieure (en excluant les bords).
        Le terme de régularisation ``lambda * L^T L`` pénalise les variations
        brusques de conductivité.

        Returns:
            np.ndarray: Matrice L de forme ``(nb_contraintes, nx*ny)``, où
            ``nb_contraintes`` est le nombre de paires de voisins considérées.
        """
        n = self.solSolutionne.ny * self.solSolutionne.nx
        rows = []

        def idx(j, i):
            return j * self.solSolutionne.nx + i

        # Différences horizontales (x)
        for j in range(1, self.solSolutionne.ny - 1):
            for i in range(1, self.solSolutionne.nx - 2):
                row = np.zeros(n)
                row[idx(j, i)] = -1.0
                row[idx(j, i + 1)] = 1.0
                rows.append(row)

        # Différences verticales (y)
        for j in range(1, self.solSolutionne.ny - 2):
            for i in range(1, self.solSolutionne.nx - 1):
                row = np.zeros(n)
                row[idx(j, i)] = -1.0
                row[idx(j + 1, i)] = 1.0
                rows.append(row)

        return np.array(rows, dtype=np.float64)

#: Colormap Matplotlib utilisée pour toutes les visualisations.
cmap = "copper"

class Visualisation:
    """
    Outils de visualisation des résultats du simulateur ERT.

    Fournit des méthodes pour afficher la conductivité du sol, le potentiel
    électrique (avec lignes de courant), la résistance apparente (sondage 1D),
    la pseudo-section 2D, les résultats d'inversion, et une figure synthétique
    complète.

    Attributes:
        sol (Sol): Référence vers le modèle de sol à visualiser.

    """

    def __init__(self, sol: Sol):
        """
        Initialise la visualisation avec le modèle de sol.

        Args:
            sol (Sol): Objet :class:`Sol` contenant les données à afficher.
        """
        self.sol = sol

    def __afficherImage__(self, matrice: np.ndarray, label: str):
        """
        Affiche une matrice 2D sous forme d'image avec une barre de couleur en haut.

        Args:
            matrice (np.ndarray): Données 2D à afficher.
            label (str): Étiquette de la barre de couleur (ex. "Conductivité (S/m)").
        """
        fontsize = 15
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(matrice, origin='lower', cmap=cmap)
        plt.gca().invert_yaxis()
        plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="2%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax, orientation='horizontal', pad=0.05)
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
        """Affiche la distribution de conductivité électrique du sol (S/m)."""
        self.__afficherImage__(self.sol.matriceSigma, "Conductivité (S/m)")

    def afficherCourant(self):
        """Affiche la matrice des sources de courant injectées (A)."""
        self.__afficherImage__(self.sol.matriceCourant, "Courant (A)")

    def afficherPotentiel(self):
        """
        Affiche le champ de potentiel électrique avec les lignes de courant superposées.

        Les lignes de courant sont calculées à partir du gradient du potentiel
        (J = -∇V) et tracées avec ``plt.streamplot``.
        """
        V = self.sol.matricePotentiel.copy()
        dVy, dVx = np.gradient(V)
        Jx = -dVx
        Jy = -dVy

        x, y = np.meshgrid(np.arange(self.sol.nx), np.arange(self.sol.ny))

        fontsize = 25
        fig, ax = plt.subplots(figsize=(10, 8))
        im = plt.imshow(
            V, origin='lower',
            extent=[0, self.sol.nx, 0, self.sol.ny],
            aspect='equal', cmap=cmap
        )
        plt.streamplot(x, y, Jx, Jy, color='white')
        plt.xlim(0, self.sol.nx - 1)
        plt.ylim(0, self.sol.ny - 1)
        plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="2%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax, orientation='horizontal', pad=0.05)
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
        cbar.set_label("Potentiel (V)", fontsize=fontsize)
        cbar.ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)
        ax.invert_yaxis()
        ax.set_xlabel('Position X (m)', fontsize=fontsize)
        ax.set_ylabel('Profondeur (m)', fontsize=fontsize)
        plt.show()

    def afficherPotentielImSHOW(self):
        """Affiche le potentiel électrique sous forme d'image simple (sans lignes de courant)."""
        self.__afficherImage__(self.sol.matricePotentiel, "Potentiel (V)")

    def afficherInversion(self):
        """
        Affiche le résultat de l'inversion interpolé sur une grille régulière.

        Utilise une interpolation cubique (:func:`scipy.interpolate.griddata`)
        pour projeter les résistivités inversées sur une grille régulière avant
        affichage en courbes de niveau.

        Note:
            Nécessite que ``sol.inversionX``, ``sol.inversionY`` et
            ``sol.inversionRes`` soient remplis au préalable.
        """
        xi = np.linspace(min(self.sol.inversionX), max(self.sol.inversionX), self.sol.nx)
        yi = np.linspace(min(self.sol.inversionY), max(self.sol.inversionY), self.sol.nx // 2)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata(
            (self.sol.inversionX, self.sol.inversionY),
            self.sol.inversionRes, (xi, yi), method='cubic'
        )

        fontsize = 15
        fig, ax = plt.subplots(figsize=(12, 8))
        cntr = ax.contourf(xi, yi, zi, levels=100, cmap=cmap)
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
        """
        Affiche une figure synthétique en trois panneaux :

        - **a)** Résistivité vraie du sol (modèle de référence).
        - **b)** Pseudo-section de résistivité apparente avec points de mesure.
        - **c)** Résultats de l'inversion interpolés.

        Nécessite que la pseudo-section et l'inversion aient été calculées au préalable.
        """
        fontsize = 17
        fig = plt.figure(figsize=(22, 14))
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92, hspace=0.4, wspace=0.35)

        ax1 = fig.add_subplot(2, 1, 1)
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
            ax.annotate(title, xy=(0.5, 1.25), xycoords='axes fraction',
                        fontsize=fontsize + 1, fontweight='bold', ha='left', va='bottom')

        # a) Résistivité vraie
        matriceSigma = self.sol.matriceSigma.copy()
        matriceSigma = matriceSigma[1:, :]
        im_sigma = ax1.imshow(1 / matriceSigma, origin='lower', cmap=cmap)
        ax1.invert_yaxis()
        add_colorbar(fig, ax1, im_sigma, "Résistivité (Ω·m)")
        style_ax(ax1, "Position X (m)", "Profondeur (m)", "a)")

        # b) Pseudo-section interpolée
        xi2 = np.linspace(self.sol.listeX.min(), self.sol.listeX.max(), self.sol.nx)
        zi2 = np.linspace(self.sol.listeZ.min(), self.sol.listeZ.max(), self.sol.nx // 2)
        XI, ZI = np.meshgrid(xi2, zi2)
        RHOI = griddata(
            (self.sol.listeX, self.sol.listeZ),
            self.sol.listePseudoSection, (XI, ZI), method='cubic'
        )
        contourf_ps = ax2.contourf(XI, ZI, RHOI, levels=100, cmap=cmap)
        ax2.invert_yaxis()
        ax2.scatter(self.sol.listeX, self.sol.listeZ, s=10,
                    color='white', edgecolors='black', label='Points de mesure')
        add_colorbar(fig, ax2, contourf_ps, "Résistivité apparente (Ω·m)")
        style_ax(ax2, "Position X (m)", "Profondeur Apparente (AB/2) (m)", "b)")

        # c) Résultats d'inversion
        self.sol.inversionY = np.abs(self.sol.inversionY)
        xi = np.linspace(min(self.sol.inversionX), max(self.sol.inversionX), self.sol.nx)
        yi = np.linspace(min(self.sol.inversionY), max(self.sol.inversionY), self.sol.nx // 2)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata(
            (self.sol.inversionX, self.sol.inversionY),
            self.sol.inversionRes, (xi, yi), method='cubic'
        )
        cntr = ax3.contourf(xi, yi, zi, levels=100, cmap=cmap)
        ax3.invert_yaxis()
        add_colorbar(fig, ax3, cntr, "Résistivité (Ω·m)")
        style_ax(ax3, "Position X (m)", "Profondeur (m)", "c)")

        plt.show()

    def afficherResistanceApparente(self):
        """
        Affiche la courbe de sondage de résistivité apparente en fonction de AB/2.

        L'axe Y est inversé pour représenter la profondeur croissante vers le bas,
        convention usuelle en géophysique.

        Note:
            Nécessite que :meth:`Solveur.calculerResApparente` ait été appelée au préalable.
        """
        fontsize = 20

        plt.figure(figsize=(10, 6))
        plt.scatter(
            self.sol.listeAB2, self.sol.listeResistanceApparente,
            s=50, color='steelblue', edgecolors='black', linewidth=1
        )
        plt.ylabel(r"Résistivité apparente ($\Omega$m)", fontsize=fontsize)
        plt.xlabel("Demi-distance entre les électrodes (m)", fontsize=fontsize)
        plt.tick_params(axis='x', labelsize=fontsize)
        plt.tick_params(axis='y', labelsize=fontsize)
        plt.grid(True, alpha=0.3)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    def afficherPseudoSection(self):
        """
        Affiche la pseudo-section de résistivité apparente 2D interpolée.

        Interpole les données sur une grille régulière (méthode cubique) et
        les affiche en courbes de niveau avec les points de mesure superposés.

        Note:
            Nécessite que :meth:`Solveur.calculerPseudoSection` ait été appelée.
        """
        xi = np.linspace(self.sol.listeX.min(), self.sol.listeX.max(), self.sol.nx)
        zi = np.linspace(self.sol.listeZ.min(), self.sol.listeZ.max(), self.sol.nx // 2)
        XI, ZI = np.meshgrid(xi, zi)
        RHOI = griddata(
            (self.sol.listeX, self.sol.listeZ),
            self.sol.listePseudoSection, (XI, ZI), method='cubic'
        )

        fontsize = 15
        fig, ax = plt.subplots(figsize=(12, 8))
        contourf = ax.contourf(XI, ZI, RHOI, levels=100, cmap=cmap)
        plt.gca().invert_yaxis()
        ax.scatter(self.sol.listeX, self.sol.listeZ, s=10,
                   color='white', edgecolors='black', label='Points de mesure')
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
    """
    Représente une électrode d'injection de courant sur la grille.

    Attributes:
        posX (int): Position en colonne (axe x) sur la grille.
        posY (int): Position en ligne (axe y) sur la grille.
        courant (float): Courant injecté (A). Positif pour la source, négatif pour le puits.
    """

    def __init__(self, posX: int, posY: int, courant: float):
        """
        Initialise une électrode d'injection.

        Args:
            posX (int): Position en colonne de l'électrode.
            posY (int): Position en ligne de l'électrode.
            courant (float): Intensité du courant injecté (A).
        """
        self.posX = posX
        self.posY = posY
        self.courant = courant

class ElectrodeMesure:
    """
    Représente une électrode de mesure de potentiel électrique.

    Attributes:
        posX (int): Position en colonne (axe x) sur la grille.
        posY (int): Position en ligne (axe y) sur la grille.
        potentiel (float): Potentiel électrique lu à cette position (V), initialisé à 0.
    """

    def __init__(self, posX: int, posY: int):
        """
        Initialise une électrode de mesure.

        Args:
            posX (int): Position en colonne de l'électrode.
            posY (int): Position en ligne de l'électrode.
        """
        self.posX = posX
        self.posY = posY
        self.potentiel = 0