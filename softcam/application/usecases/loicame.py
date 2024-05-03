import sys
sys.path.append("c:/Users/stagiaire.be/Documents/SOFTCAM_dvpmt/softcam")

from domain.services.calculloiscame import SqueletteLoisCame, AjustementLoisCame, CalculRampeOuverture, CalculRampeFermeture
from domain.entities.loicame import LoiCame
import numpy as np
import matplotlib.pyplot as plt

class CreateLoiCameUseCase():
    def __init__(self, duree_rampe_ouverture, duree_vitesse_constante_rampe_ouverture, levee_fin_rampe_ouverture, vitesse_fin_rampe_ouverture, duree_ouverture, duree_acceleration_ouverture, angle_accelmax_ouverture, accelmax_ouverture, angle_decelmax, decelmax, leveemax, duree_acceleration_fermeture, angle_accelmax_fermeture, accelmax_fermeture, duree_rampe_fermeture, duree_vitesse_constante_rampe_fermeture, levee_debut_rampe_fermeture, vitesse_debut_rampe_fermeture) :
        self.__dt_ro = duree_rampe_ouverture
        self.__dt_vco = duree_vitesse_constante_rampe_ouverture
        self.__lo = levee_fin_rampe_ouverture
        self.__vo = vitesse_fin_rampe_ouverture
        self.__dt_ouverture = duree_ouverture
        self.__dt_accelo = duree_acceleration_ouverture
        self.__t_accelomax = angle_accelmax_ouverture
        self.__accelomax = accelmax_ouverture
        self.__t_decelmax = angle_decelmax
        self.__decelmax = decelmax
        self.__leveemax = leveemax
        self.__t_accelfmax = angle_accelmax_fermeture
        self.__dt_accelf = duree_acceleration_fermeture
        self.__accelfmax = accelmax_fermeture
        self.__dt_rf = duree_rampe_fermeture    
        self.__dt_vcf = duree_vitesse_constante_rampe_fermeture
        self.__lf = levee_debut_rampe_fermeture
        self.__vf = vitesse_debut_rampe_fermeture

        self.__rampe_ouverture = self.create_rampe_ouverture()
        self.__squelette_horsrampes = self.create_squelette_horsrampes()
        self.__lois_ajustees_horsrampes = self.create_ajustement_horsrampes()
        self.__rampe_fermeture = self.create_rampe_fermeture()

    def create_rampe_ouverture(self):
        return CalculRampeOuverture(
            duree_rampe = self.dt_ro,
            duree_vitesse_constante = self.dt_vco,
            levee_rampe = self.lo,
            vitesse_rampe = self.vo
        )
    
    def create_rampe_fermeture(self):
        return CalculRampeFermeture(
            duree_rampe = self.dt_rf,
            duree_vitesse_constante = self.dt_vcf,
            levee_rampe = self.lf,
            vitesse_rampe = self.vf
        )
    
    def create_squelette_horsrampes(self):
        return SqueletteLoisCame(
            t_inputs = np.array([0, self.t_accelomax, self.dt_accelo, self.t_decelmax, self.dt_ouverture-self.dt_accelf, self.t_accelfmax, self.dt_ouverture]),
            a_inputs= np.array([0, self.accelomax, 0, self.decelmax, 0, self.accelfmax,0]),
            j_inputs= np.array([0, 0, np.nan, 0, np.nan, 0, 0], dtype=float),
            v_init= self.vo,
            l_init= self.lo
        )
    
    def create_ajustement_horsrampes(self):
        return AjustementLoisCame(
            squelette = self.squelette_horsrampes,
            position_ajustement = int(len(self.squelette_horsrampes.t_inputs)/2),
            angles_racines = np.array([0, self.dt_accelo, self.dt_ouverture - self.dt_accelf, self.dt_ouverture]),
            angle_levee_max = self.t_decelmax,
            levee_ouverture = self.lo,
            vitesse_ouverture = self.vo,
            levee_max = self.leveemax,
            levee_fermeture = self.lf,
            vitesse_fermeture = self.vf 
        )
    
    def create_loicame(self):
        return LoiCame(
            duree_rampe_ouverture = self.dt_ro,
            duree_ouverture = self.dt_ouverture,
            duree_rampe_fermeture = self.dt_rf,
            rampe_ouverture = self.rampe_ouverture,
            loicame_horsrampes = self.lois_ajustees_horsrampes,
            rampe_fermeture = self.rampe_fermeture
        )

    @property    
    def dt_ro(self):
        return self.__dt_ro
    @property    
    def dt_vco(self):
        return self.__dt_vco
    @property
    def lo(self):
        return self.__lo
    @property
    def vo(self):
        return self.__vo
    @property
    def dt_ouverture(self):
        return self.__dt_ouverture
    @property
    def dt_accelo(self):
        return self.__dt_accelo
    @property
    def t_accelomax(self):
        return self.__t_accelomax
    @property
    def accelomax(self):
        return self.__accelomax
    @property
    def t_decelmax(self):
        return self.__t_decelmax
    @property
    def decelmax(self):
        return self.__decelmax
    @property
    def leveemax(self):
        return self.__leveemax
    @property
    def dt_accelf(self):
        return self.__dt_accelf
    @property
    def t_accelfmax(self):
        return self.__t_accelfmax
    @property
    def accelfmax(self):
        return self.__accelfmax
    @property    
    def dt_rf(self):
        return self.__dt_rf
    @property
    def dt_vcf(self):
        return self.__dt_vcf
    @property
    def lf(self):
        return self.__lf
    @property
    def vf(self):
        return self.__vf
    
    @property
    def rampe_ouverture(self):
        return self.__rampe_ouverture
    @property
    def squelette_horsrampes(self):
        return self.__squelette_horsrampes
    @property
    def lois_ajustees_horsrampes(self):
        return self.__lois_ajustees_horsrampes
    @property
    def rampe_fermeture(self):
        return self.__rampe_fermeture


if __name__=="__main__":

    # Données Lois Comparaison
    t_comp, levee_comp, vitesse_comp, accel_comp = np.loadtxt("data\\exemple_lois_evo2-E_newgeom.txt").T

    clcuc = CreateLoiCameUseCase(
        duree_rampe_ouverture = 36,
        duree_vitesse_constante_rampe_ouverture = 25,
        levee_fin_rampe_ouverture = 270,
        vitesse_fin_rampe_ouverture = 9 ,
        duree_ouverture = 126.5,
        duree_acceleration_ouverture = 16.5,
        angle_accelmax_ouverture = 8,
        accelmax_ouverture = 34.1,
        angle_decelmax = 63.43,
        decelmax = -8.537,
        leveemax = 10900,
        duree_acceleration_fermeture = 16.7,
        angle_accelmax_fermeture = 118,
        accelmax_fermeture = 34.07,
        duree_rampe_fermeture = 36.5,
        duree_vitesse_constante_rampe_fermeture = 25,
        levee_debut_rampe_fermeture = 280,
        vitesse_debut_rampe_fermeture = -9
    )

    loiscame = clcuc.create_loicame()

    t_calcul = np.linspace(0, 200, 10000)
    accel = loiscame.a(t_calcul)
    vitesse = loiscame.v(t_calcul)
    levee = loiscame.l(t_calcul)

    fig_accel = plt.figure(1)
    plt.plot(t_comp, accel_comp, "-.")
    plt.plot(t_calcul, accel)

    fig_vitesse = plt.figure(2)
    plt.plot(t_comp, vitesse_comp, "-.")
    plt.plot(t_calcul, vitesse)

    fig_levee = plt.figure(3)
    plt.plot(t_comp, levee_comp, "-.")
    plt.plot(t_calcul, levee/1000)

    plt.show()

