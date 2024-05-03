import dataclasses
import numpy as np
from softcam.domain.entities.patin import Patin

@dataclasses.dataclass
class Levier:
    masse : float = 70e-3
    inertie : float = 9e-6
    longueur : float = 36.8e-3
    patin_came : Patin = dataclasses.field(default_factory=lambda:Patin())
    patin_soupape : Patin = dataclasses.field(default_factory=lambda:Patin())
        
    def __post_init__(self):
        self.validate_masse()
        self.validate_inertie()
        self.validate_longueur()
        self.validate_patins()

    def validate_masse(self):
        if self.masse < 0:
            raise ValueError("La masse doit être positive.")
    
    def validate_inertie(self):
        if self.inertie < 0:
            raise ValueError("L'inertie doit être positive.")
    
    def validate_longueur(self):
        if self.longueur <= 0:
            raise ValueError("La longueur doit être strictement positive.")
    
    def validate_patins(self):
        if not all((isinstance(self.patin_came, Patin), isinstance(self.patin_soupape, Patin))) :
            raise ValueError("Les patins doivent être de type Patin")

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
    
    def to_dict(self):
        thedict = {
            "masse" : self.masse,
            "inertie" : self.inertie,
            "longueur" : self.longueur,
            "patin_came" : self.patin_came,
            "patin_soupape" : self.patin_soupape
        }
        return thedict
    