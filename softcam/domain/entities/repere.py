import dataclasses
import numpy as np

@dataclasses.dataclass
class Repere:
    nom : str = "Piece",
    origine_coords: np.ndarray = dataclasses.field(default_factory=lambda :np.array([1,1])),
    x_coords: np.ndarray = dataclasses.field(default_factory=lambda :np.array([1,0])),
    y_coords: np.ndarray = dataclasses.field(default_factory=lambda :np.array([0,1]))
        
    def __post_init__(self):
        self.validate_xy_norme()
        self.validate_xy_orthogonal()
    
    def validate_xy_norme(self):
        if np.linalg.norm(self.x_coords) != 1:
            raise ValueError("Le vecteur X doit être de norme 1")
        if np.linalg.norm(self.y_coords) != 1:
            raise ValueError("Le vecteur Y doit être de norme 1")
        
    def validate_xy_orthogonal(self):
        if np.dot(self.x_coords, self.y_coords) != 0:
            raise ValueError("Les vecteurs X et Y doivent être orthognaux")

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
    
    def to_dict(self):
        return dataclasses.asdict(self)
    