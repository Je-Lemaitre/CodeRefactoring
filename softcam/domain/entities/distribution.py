import dataclasses
from abc import ABC, abstractmethod
import numpy as np

from softcam.unitees import DEGREE_TO_RADIAN, TRPMIN_TO_RADPSEC, MEGAPASCAL_TO_PASCAL
from softcam.domain.entities.came import Came
from softcam.domain.entities.levier import Levier
from softcam.domain.entities.ressort import Ressort
from softcam.domain.entities.soupape import Soupape, SoupapeSansPoussoir, SoupapeAvecPoussoir

@dataclasses.dataclass
class Distribution(ABC):
    angle_came : float = 0
    came : Came = dataclasses.field(default_factory= lambda: Came())
    coords_came : np.ndarray = dataclasses.field(default_factory= lambda: np.array([14.2e-3, 31.7e-3]))
    sens_rotation_came : float = 1
    ressort : Ressort = dataclasses.field(default_factory= lambda: Ressort())
    soupape : Soupape = dataclasses.field(default_factory= lambda: SoupapeSansPoussoir())
    coords_soupape : np.ndarray = dataclasses.field(default_factory= lambda: np.array([0, 0]))
    inclinaison_soupape : float = 0

    @abstractmethod
    def __post_init__(self):
        """Fonction appelée à l'initialisation de l'instance. Permet notamment de faire les tests pour vérifier la cohérence des paramètres."""

    def validate_angle_came(self):
        if not (0 <= self.angle_came < 360 *DEGREE_TO_RADIAN) :
            raise ValueError("L'angle de rotation de la came doit être compris entre 0 et 360 degrées.")
    
    @classmethod
    def from_dict(cls, d):
        return cls(**d)
    
    def to_dict(self):
        return dataclasses.asdict(self)

@dataclasses.dataclass
class DistributionLinguet(Distribution):
    levier : Levier = dataclasses.field(default_factory= lambda: Levier())
    coords_levier : np.ndarray = dataclasses.field(default_factory= lambda: np.array([35.6e-3, 5.4e-3]))
    angle_leviercame_init: float = 5.3 *DEGREE_TO_RADIAN
    angles_limites_patinsoupape : tuple = (25,35)
    angles_limites_patincame : tuple = (20,25)

    def __post_init__(self):
        self.validate_angle_came()

    def validate_angle_came(self):
        if not (0 <= self.angle_came < 360 *DEGREE_TO_RADIAN) :
            raise ValueError("L'angle de rotation de la came doit être compris entre 0 et 360 degrées.")
    
    @classmethod
    def from_dict(cls, d):
        return cls(**d)
    
    def to_dict(self):
        thedict = {
            "angle_came" : self.angle_came,
            "came" : self.came,
            "coords_came" : self.coords_came,
            "sens_rotation_came" : self.sens_rotation_came,
            "ressort" : self.ressort,
            "soupape" : self.soupape,
            "coords_soupape" : self.coords_soupape,
            "inclinaison_soupape" : self.inclinaison_soupape,
            "levier" : self.levier,
            "coords_levier" : self.coords_levier,
            "angle_leviercame_init" : self.angle_leviercame_init,
            "angles_limites_patinsoupape" : self.angles_limites_patinsoupape,
            "angles_limites_patincame" : self.angles_limites_patincame
        }
        return thedict

@dataclasses.dataclass
class DistributionDirecte(Distribution):
    offset : float = 0

    def __post_init__(self):
        self.validate_angle_came()

    def validate_angle_came(self):
        if not (0 <= self.angle_came < 360 *DEGREE_TO_RADIAN) :
            raise ValueError("L'angle de rotation de la came doit être compris entre 0 et 360 degrées.")
    
    @classmethod
    def from_dict(cls, d):
        return cls(**d)
    
    def to_dict(self):
        thedict = {
            "angle_came" : self.angle_came,
            "came" : self.came,
            "coords_came" : self.coords_came,
            "sens_rotation_came" : self.sens_rotation_came,
            "ressort" : self.ressort,
            "soupape" : self.soupape,
            "coords_soupape" : self.coords_soupape,
            "inclinaison_soupape" : self.inclinaison_soupape,
            "offset" : self.offset
        }
        return thedict