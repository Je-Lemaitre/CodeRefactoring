import dataclasses
import numpy as np

@dataclasses.dataclass
class Roller:
    diametre : 8.5e-3
    offset : 1e-3
    precision : 1e-6
    coords_centre : np.ndarray = dataclasses.field(default_factory=lambda :np.array([1,1]))
        
    def __post_init__(self):
        self.validate_diametre()
        self.validate_precision()
        self.validate_coords_centre()
    
    def validate_diametre(self):
        if self.diametre <= 0:
            raise ValueError("Le diametre doit être strictement positif.")
        
    def validate_precision(self):
        if self.precision < 0:
            raise ValueError("La précision doit être positive.")
    
    def validate_coords_centre(self):
        if not isinstance(self.coords_centre, np.ndarray) :
            raise ValueError("Les coordonnées doivent être des arrays.")

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
    
    def to_dict(self):
        return dataclasses.asdict(self)
    