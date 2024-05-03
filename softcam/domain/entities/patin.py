import dataclasses
import numpy as np

@dataclasses.dataclass
class Patin:
    rayon_courbure : float = 27e-3 #Rayon de courbure au niveau du contact (cas du poussoir courbe)
    largeur: float = 6e-3
    module_young : float = 210e9
    coefficient_poisson : float = 0.3
        
    def __post_init__(self):
        self.validate_largeur()
        self.validate_module_young()
        self.validate_coefficient_poisson()

    def validate_largeur(self):
        if self.largeur <= 0:
            raise ValueError("La largeur doit être positive.")

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
    