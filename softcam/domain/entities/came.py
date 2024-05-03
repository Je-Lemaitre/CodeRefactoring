import dataclasses
import numpy as np
from softcam.unitees import DEGREE_TO_RADIAN

@dataclasses.dataclass
class Came:
    rayon_base : float = 19e-3
    largeur : float = 8e-3
    module_young : float = 210e9
    coefficient_poisson : float = 0.3
    profil : np.ndarray = dataclasses.field(default_factory=lambda: np.array([np.arange(0,360) *DEGREE_TO_RADIAN, 19e-3*np.ones(360)]))
        
    def __post_init__(self):
        self.validate_rayon_base()
        self.validate_largeur()
        self.validate_module_young()
        self.validate_coefficient_poisson()
        self.validate_profil()

    def validate_rayon_base(self):
        if self.rayon_base <= 0:
            raise ValueError("Le rayon de base doit être positif.")
    
    def validate_largeur(self):
        if self.largeur <= 0:
            raise ValueError("La largeur doit être positive.")

    def validate_module_young(self):
        if self.module_young <= 0:
            raise ValueError("Le module de Young doit être positif.")
    
    def validate_coefficient_poisson(self):
        if not -1 <= self.coefficient_poisson <= 0.5 :
            raise ValueError("Le coefficient de poisson doit être entre -1 et 0.5.")
    
    def validate_profil(self):
        if self.profil.shape[0] != 2 :
            raise ValueError("Le profil doit être de dimension (n,2). Un array pour l'angle et un array pour le rayon en coordonnées polaires.")

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
    
    def to_dict(self):
        return dataclasses.asdict(self)
    