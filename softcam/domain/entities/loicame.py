import dataclasses
from unitees import DEGREE_TO_RADIAN
from domain.services.calculloiscame import AjustementLoisCame, CalculRampeOuverture, CalculRampeFermeture
import numpy as np

@dataclasses.dataclass
class LoiCame:
    duree_rampe_ouverture : float
    duree_ouverture : float
    duree_rampe_fermeture : float
    rampe_ouverture : AjustementLoisCame
    loicame_horsrampes : CalculRampeOuverture
    rampe_fermeture : CalculRampeFermeture
        
    def __post_init__(self):
        self.validate_durees()
    
    def validate_durees(self):
        if not (0<=self.duree_rampe_ouverture + self.duree_ouverture + self.duree_rampe_fermeture <=360):
            raise ValueError("La durée de la loi doit être comprise entre 0 et 360 degrées et la durée des différentes phases doit être inférieure à la durée totale d'ouverture.")
        
    def a(self, t):
        def a_scalar(x):
            if 0<= x < self.duree_rampe_ouverture:
                return self.rampe_ouverture.a(x)
            elif self.duree_rampe_ouverture <= x < self.duree_rampe_ouverture + self.duree_ouverture:
                return self.loicame_horsrampes.a(x - self.duree_rampe_ouverture)
            elif self.duree_rampe_ouverture + self.duree_ouverture <= x <= self.duree_rampe_ouverture + self.duree_ouverture + self.duree_rampe_fermeture:
                return self.rampe_fermeture.a(x - self.duree_rampe_ouverture - self.duree_ouverture)
            else :
                return 0
            
        a_vect = np.vectorize(a_scalar)    
        return a_vect(t)
    
    def v(self, t):
        def v_scalar(x):
            if 0<= x < self.duree_rampe_ouverture:
                return self.rampe_ouverture.v(x)
            elif self.duree_rampe_ouverture <= x < self.duree_rampe_ouverture + self.duree_ouverture:
                return self.loicame_horsrampes.v(x - self.duree_rampe_ouverture)
            elif self.duree_rampe_ouverture + self.duree_ouverture <= x <= self.duree_rampe_ouverture + self.duree_ouverture + self.duree_rampe_fermeture:
                return self.rampe_fermeture.v(x - self.duree_rampe_ouverture - self.duree_ouverture)
            else :
                return 0
            
        v_vect = np.vectorize(v_scalar)    
        return v_vect(t)
    
    def l(self, t):
        def l_scalar(x):
            if 0<= x < self.duree_rampe_ouverture:
                return self.rampe_ouverture.l(x)
            elif self.duree_rampe_ouverture <= x < self.duree_rampe_ouverture + self.duree_ouverture:
                return self.loicame_horsrampes.l(x - self.duree_rampe_ouverture)
            elif self.duree_rampe_ouverture + self.duree_ouverture <= x <= self.duree_rampe_ouverture + self.duree_ouverture + self.duree_rampe_fermeture:
                return self.rampe_fermeture.l(x - self.duree_rampe_ouverture - self.duree_ouverture)
            else :
                return 0
            
        l_vect = np.vectorize(l_scalar)    
        return l_vect(t)
    

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
    
    def to_dict(self):
        return dataclasses.asdict(self)