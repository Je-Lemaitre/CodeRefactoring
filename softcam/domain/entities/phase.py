from abc import ABC, abstractmethod
import numpy as np
import dataclasses

"""La réalisation de ce fichier a été interrompue car l'utilité de ces classes n'est pas claire pour le moment. Trop de paramètres se baladent, il faut surement les rassembler."""

@dataclasses.dataclass
class Phase(ABC):
    angle_came : list = dataclasses.field(default_factory=lambda :list)
    acceleration : list = dataclasses.field(default_factory=lambda :list)
    vitesse : list = dataclasses.field(default_factory=lambda :list)
    levee : list = dataclasses.field(default_factory=lambda :list)
    
    def validate_angle_came(self):
        if any([(angle<0 or angle>2*np.pi) for angle in self.angle_came]) :
            raise ValueError("Les angles doivent être compris entre 0 et 2*PI radians.")
    def validate_taille_listes(self):
        if any([len(self.levee)!=len(self.angle_came), len(self.vitesse)!=len(self.angle_came), len(self.acceleration)!=len(self.angle_came)]):
            raise ValueError("Les listes d'accélérations, de vitesses et de levées doivent avoir la même dimension que la liste d'angles.")

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
    
    def to_dict(self):
        return dataclasses.asdict(self)


@dataclasses.dataclass
class PhaseAccelRampeOuverture(Phase):
    levee_fin_accel: float = 5e-5
    vitesse_rampe: float = 5.15e-4

    def __post_init__(self):
        self.validate_angle_came()
        self.validate_taille_listes()
        self.validate_levee_fin_accel()
        self.validate_vitesse_rampe()
    
    def validate_levee_fin_accel(self):
        if self.levee_fin_accel < 0:
            raise ValueError("La levée en fin d'accélération doit être positive.")
        
    def validate_vitesse_rampe(self):
        if self.vitesse_rampe < 0:
            raise ValueError("La vitesse de la rampe doit être positive.")

@dataclasses.dataclass
class PhaseNulleRampeOuverture(Phase):
    hauteur_rampe: float = 3e-4

    def __post_init__(self):
        self.validate_angle_came()
        self.validate_taille_listes()
        self.validate_hauteur_rampe()

    def validate_hauteur_rampe(self):
        if self.hauteur_rampe < 0:
            raise ValueError("La hauteur de rampe doit être positive.")


