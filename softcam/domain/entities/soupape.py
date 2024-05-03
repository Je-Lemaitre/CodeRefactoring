from abc import ABC, abstractmethod
import dataclasses

@dataclasses.dataclass
class Soupape(ABC):
    masse_soupape : float = 43.5e-3
    masse_coupelle : float = 5.5e-3 #Lunules et Grain compris
    module_young : float = 210e9
    coefficient_poisson : float = 0.3

    @abstractmethod   
    def __post_init__(self):
        """Excecuté à l'initialisation de la classe. Permet notamment de vérifier que les paramètres entrés sont acceptables."""

    @abstractmethod
    def validate_masses(self):
        """Valide que les masse sont bien positives."""

    def validate_module_young(self):
        if self.module_young <= 0:
            raise ValueError("Le module de Young doit être positif.")

    def validate_coefficient_poisson(self):
        if not -1 <= self.coefficient_poisson <= 0.5 :
            raise ValueError("Le coefficient de poisson doit être entre -1 et 0.5.")

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
    
    def to_dict(self):
        return dataclasses.asdict(self)

@dataclasses.dataclass
class SoupapeSansPoussoir(Soupape):
    diametre_soupape : float = 6e-3
        
    def __post_init__(self):
        self.validate_masses()
        self.validate_diametre_soupape()
        self.validate_module_young()
        self.validate_coefficient_poisson()
        
    def validate_masses(self):
        if any((self.masse_soupape<0, self.masse_coupelle<0)):
            raise ValueError("Les masses doivent être positives.")
        
    def validate_diametre_soupape(self):
        if self.diametre_soupape <= 0:
            raise ValueError("Le diamètre doit être positif.")

@dataclasses.dataclass
class SoupapeAvecPoussoir(Soupape):
    masse_poussoir : float = 5e-3
    diametre_poussoir : float = 25e-3
    rayon_courbure : float = 35e-3 #Rayon de courbure au niveau du contact (cas du poussoir courbe)
    largeur_courbure: float = 6e-3
    frottement_poussoir_guide : float = 0
        
    def __post_init__(self):
        self.validate_masses()
        self.validate_diametre_poussoir()
        self.validate_largeur_courbure()
        self.validate_module_young()
        self.validate_coefficient_poisson()
        

    def validate_masses(self):
        if any((self.masse_soupape<0, self.masse_coupelle<0, self.masse_poussoir<0)):
            raise ValueError("Les masses doivent être positives.")
        
    def validate_diametre_poussoir(self):
        if self.diametre_poussoir <= 0:
            raise ValueError("Le diamètre doit être positif.")

    def validate_largeur_courbure(self):
        if self.largeur_courbure <= 0:
            raise ValueError("La largeur doit être positive.")
    