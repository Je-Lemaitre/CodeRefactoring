from abc import ABC, abstractmethod
import dataclasses
import numpy as np
import scipy.integrate as sci
import scipy.optimize as sco
from softcam.domain.entities.distribution import Distribution, DistributionLinguet

@dataclasses.dataclass
class DistributionUseCases(ABC):
    distribution : Distribution = dataclasses.field(default_factory=lambda:Distribution())
    distribution_init : Distribution = dataclasses.field(default_factory=lambda:Distribution())

    @abstractmethod
    def verification_cinematique(self):
        """ Vérifie si les angles et les positions des points de contact sont admissibles. """

    @abstractmethod
    def verification_mecanique(self):
        """ Vérifie que les pression de Hertz, les produits PV, les efforts et les couples sont admissibles. """

    @abstractmethod
    def resout_equation_mouvement(self, equation_mouvement, conditions_initales):
        """ Implémente la résolution de l'équation du mouvement. """
    
    @abstractmethod
    def equation_mouvement(self, x, x0) :
        """ Implémente la dynamique de l'équation du mouvement. """

    @abstractmethod
    def produit_pv(self):
        """ Calcul le produit Pression X Vitesse de Balayage. """

    @abstractmethod
    def pression_hertz(self):
        """ Calcul les pression de Hertz aux contacts. """
    
    @abstractmethod
    def vitesse_balayage(self):
        """ Calcul la vitesse de balayage entre deux pièces. """

@dataclasses.dataclass
class DistributionLinguetUseCases(DistributionUseCases):
    distribution : DistributionLinguet = dataclasses.field(default_factory=lambda:DistributionLinguet())
    distribution_init : DistributionLinguet = dataclasses.field(default_factory=lambda:DistributionLinguet())
    
    def __post_init__(self):
        self.lbd0, self.beta0 = self.fermeture_geometrique(self.distribution_init)
    
    def verification_cinematique(self):
        return 42

    def verification_mecanique(self):
        return 42

    def resout_equation_mouvement(self, levee_init, vitessemax_reelle, temps):
        beta_init = self.inclinaison_from_levee(levee_init)
        vitesse_angulaire_init = vitessemax_reelle / self.distribution.levier.longueur / np.cos(beta_init - self.distribution.inclinaison_soupape)

        conditions_init = np.array([beta_init[0], vitesse_angulaire_init[0]])

        vec_sol = sci.odeint(self.equation_mouvement, conditions_init, temps, args = (conditions_init,))

        return vec_sol[0], self.levee_from_inclinaison(vec_sol[0])

    #def resout_equation_mouvement(self, beta_init, vitesse_init):  
    #    pass
    
    def equation_mouvement(self, x, x0) :
        masse_mvmt = self.distribution.soupape.masse_soupape + self.distribution.soupape.masse_soupape + self.distribution.ressort.masse/3
        inertie_linguet = self.distribution.levier.inertie
        longueur_linguet = self.distribution.levier.longueur
        raideur = self.distribution.ressort.raideur
        precharge = self.distribution.ressort.precharge
        inclinaison_soupape = self.distribution.inclinaison_soupape

        inclinaison_linguet_init = x0[0]
        vitesse_angluaire_linguet_init = x0[1]

        inclinaison_linguet = x[0]
        vitesse_angulaire_linguet = x[1]
        
        d2levee_membre1 = masse_mvmt *longueur_linguet**2 *np.cos(inclinaison_linguet -inclinaison_soupape)**2
        d2levee_membre2 = masse_mvmt *longueur_linguet**2 *np.sin(inclinaison_linguet -inclinaison_soupape)**2 *np.cos(inclinaison_linguet -inclinaison_soupape)**2 *vitesse_angulaire_linguet**2

        force_ressort_sans_precharge = raideur *longueur_linguet**2 *np.cos(inclinaison_linguet -inclinaison_soupape) *(np.sin(inclinaison_linguet -inclinaison_soupape) - np.sin(inclinaison_linguet_init -inclinaison_soupape))
        force_ressort_precharge =  precharge *longueur_linguet *np.cos(inclinaison_linguet -inclinaison_soupape)

        frottement_poussoir_guide = 0 #Valeur à paramétrer

        return ( d2levee_membre2 - force_ressort_sans_precharge + force_ressort_precharge -          frottement_poussoir_guide) / ( d2levee_membre1 + inertie_linguet)

    def produit_pv(self):
        return 42

    def pression_hertz(self):
        return 42
    
    def vitesse_balayage(self):
        return 42
    
    def inclinaison_from_levee(self, levee):
        longueur = self.distribution.levier.longueur
        beta0 = self.beta0
        alpha = self.distribution.inclinaison_soupape
        return np.arcsin(levee/(longueur + np.sin(beta0 - alpha))) + alpha
    
    def levee_from_inclinaison(self, inclinaison):
        longueur = self.distribution.levier.longueur
        beta = inclinaison
        beta0 = self.beta0
        alpha = self.distribution.inclinaison_soupape
        return longueur*(np.sin(beta - alpha) - np.sin(beta0 - alpha))

    @classmethod
    def fermeture_geometrique(cls, distribution):
        alpha = distribution.inclinaison_soupape
        longueur_linguet = distribution.levier.longueur
        rayon_patinsoupape = distribution.levier.patin_soupape.rayon_courbure
        coords_soupape = distribution.coords_soupape
        coords_levier = distribution.coords_levier
        diff_x, diff_y = coords_soupape - coords_levier

        def equation_numero1(lbd):
            return ((lbd*np.cos(alpha) + diff_x - rayon_patinsoupape*np.sin(alpha))**2 
                + (lbd*np.sin(alpha) + diff_y + rayon_patinsoupape*np.cos(alpha))**2 
                - longueur_linguet)
        
        lbd = sco.root(equation_numero1, 0).x

        beta = np.arctan((
            lbd *np.sin(alpha) + diff_y + rayon_patinsoupape*np.cos(alpha)  
        ) / (
            lbd *np.cos(alpha) + diff_x - rayon_patinsoupape*np.sin(alpha) 
        ))
        
        return lbd, beta
    
    

    
    