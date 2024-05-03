import dataclasses
import uuid
from softcam.domain.entities.distribution import Distribution, DistributionLinguet
from softcam.domain.entities.loicame import LoiCame

from softcam.unitees import DEGREE_TO_RADIAN, TRPMIN_TO_RADPSEC, MEGAPASCAL_TO_PASCAL

@dataclasses.dataclass
class Etude():
    id : uuid.UUID = uuid.uuid4()
    nom : str = "Mon Étude"
    pas_angulaire : float = 1 *DEGREE_TO_RADIAN
    regime_affolement : float = 9500 *TRPMIN_TO_RADPSEC
    regime_utilisation : float = 7500 *TRPMIN_TO_RADPSEC
    distribution : list[DistributionLinguet] = dataclasses.field(default_factory= lambda: [DistributionLinguet()])
    loicame : LoiCame = dataclasses.field(default_factory= lambda: LoiCame())

    def __post_init__(self):
        self.validate_pas_angulaire()
        self.validate_regimes()
        self.validate_distribution()

    def validate_pas_angulaire(self):
        if not 0 < self.pas_angulaire < 360 *DEGREE_TO_RADIAN :
            raise ValueError("Le pas angulaire doit être strictement positif et inférieur à 360 degrés.")
        if 360 % ( self.pas_angulaire / DEGREE_TO_RADIAN ) != 0 : # Le signe % permet de calculer le reste de la division Euclidienne.
            raise ValueError("Le pas angulaire doit être multiple de 360 degrées.")

    def validate_regimes(self):
        if self.regime_utilisation > self.regime_affolement :
            raise ValueError("Le régime d'utilisation est supérieur au régime d'affolement.")

    def validate_distribution(self):
        if not (isinstance(self.distribution, list) and isinstance(self.distribution[0], Distribution)):
            raise ValueError("L'objet entré comme distribution doit être une liste composée d'éléments de type Distribution.")
    
    @classmethod
    def from_dict(cls, d):
        return cls(**d)
    
    def to_dict(self):
        thedict = {
            "id" : self.id,
            "nom" : self.nom,
            "pas_angulaire" : self.pas_angulaire,
            "regime_affolement" : self.regime_affolement,
            "regime_utilisation" : self.regime_utilisation,
            "distribution" : self.distribution,
            "loicame" : self.loicame
        }
        return thedict