import dataclasses

@dataclasses.dataclass
class Ressort:
    masse : float = 27e-3
    raideur : float = 46.5e3
    precharge : float = 240
        
    def __post_init__(self):
        self.validate_masse()
        self.validate_raideur()

    def validate_masse(self):
        if self.masse < 0:
            raise ValueError("Le rayon de base doit être positif.")
    
    def validate_raideur(self):
        if self.raideur < 0:
            raise ValueError("La raideur doit être positive.")

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
    
    def to_dict(self):
        return dataclasses.asdict(self)
    
    def __add__(self, ressort2):
        """ Cette fonction fait la somme de deux ressorts en parallèle."""

        return Ressort(
            masse = self.masse + ressort2.masse ,
            raideur =  self.raideur + ressort2.raideur ,
            precharge = self.precharge + ressort2.precharge
            )
    