import dataclasses

""" Module créé en prévision de modification futures.
    Non utilisé dans le reste du logiciel."""

@dataclasses.dataclass
class Materiau:
    nom : str = "MonMateriau"
    densite : float = 8e-3
    module_young : int = 210e9
    coefficient_poisson: float = 0.3

    def __post_init__(self):
        self.validate_nom()
        self.validate_densite()
        self.validate_module_young()
        self.validate_coefficient_poisson()

    def validate_nom(self):
        if not isinstance(self.nom, str):
            raise ValueError("Le nom doit être une chaine de caractères.")
    
    def validate_densite(self):
        if self.densite <= 0:
            raise ValueError("La densité doit être positive.")
    
    def validate_module_young(self):
        if self.module_young <= 0:
            raise ValueError("Le module de Young doit être positif.")
    
    def validate_coefficient_poisson(self):
        if not -1 <self.coefficient_poisson < 0.5 :
            raise ValueError("Le coefficient de poisson doit être entre -1 et 0.5.")

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
    
    def to_dict(self):
        return dataclasses.asdict(self)
    