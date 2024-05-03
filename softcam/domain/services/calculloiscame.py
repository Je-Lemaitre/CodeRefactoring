import os
from abc import ABC, abstractmethod
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import scipy.integrate as sci
import scipy.optimize as sco
import time
import pytest

class SqueletteLoisCame():
    """Calcul les Lois de Came suivant la méthode choisie
    """
    def __init__(self,t_inputs, a_inputs, j_inputs = None, v_init=0 , l_init=0):
        if len(t_inputs)<7 :
            raise ValueError("Afin que le calcul du profil de came se passe correctement, il est fortement conseillé que le nombre de points soit supérieur ou égal à 7. 9 points sont conseillés.")
        self.__t_inputs = t_inputs
        self.__a_inputs = a_inputs
        if j_inputs is None :
            self.__j_inputs = np.array([np.nan for _ in range(len(t_inputs))], dtype=float)
        else :
            self.__j_inputs = j_inputs     
        self.__t_knots, self.__a_knots, self.__j_knots, self.__kflags, self.__a_coeffs = self.schumaker_quad_spline(self.t_inputs, self.a_inputs, self.j_inputs)
        self.__v_knots, self.__l_knots = self.v_l_knots(v_init, l_init)
    
    def a(self, t):
        """Calcule l'accélération pour un ensemble de points donnés.

        Args:
            t (numpy.ndarray): Angles de came pour lesquels les accélérations sont calculées.

        Returns:
            numpy.ndarray: Accélération pour les angles renseignés.
        """
        def a_scalar(x) :
            i = np.searchsorted(self.t_knots, x, side="right") - 1
            if (0<=i) and (i<len(self.t_knots) - 1) :
                dt = x - self.t_knots[i]

                accel = self.a_coeffs[i,2]*dt**2 + self.a_coeffs[i,1]*dt + self.a_coeffs[i,0]
                return accel
            
            elif (i == len(self.t_knots) - 1) and (x==self.t_knots[-1]):
                i += -1
                dt = x - self.t_knots[i]
                
                accel = self.a_coeffs[i,2]*dt**2 + self.a_coeffs[i,1]*dt + self.a_coeffs[i,0]
                return accel
            
            else :
                raise ValueError(f"L'angle de {x}° n'appartient pas à l'intervalle de définition de la fonction.")
            
        a_vect = np.vectorize(a_scalar)
        return a_vect(t)
    
    def v(self, t):
        """Calcule la vitesse pour un ensemble de points donnés.

        Args:
            t (numpy.ndarray): Angles de came pour lesquels les vitesses sont calculées.

        Returns:
            numpy.ndarray: Vitesses pour les angles renseignés.
        """
        def v_scalar(x) :
            i = np.searchsorted(self.t_knots, x, side="right") - 1
            if (0<=i) and (i<len(self.t_knots) - 1) :
                dt = x - self.t_knots[i]
                
                vitesse = self.a_coeffs[i,2]/3*dt**3 + self.a_coeffs[i,1]/2*dt**2 + self.a_coeffs[i,0]*dt + self.v_knots[i]
                return vitesse
            
            elif (i == len(self.t_knots) - 1) and (x==self.t_knots[-1]):
                i += -1
                dt = x - self.t_knots[i]
                
                vitesse = self.a_coeffs[i,2]/3*dt**3 + self.a_coeffs[i,1]/2*dt**2 + self.a_coeffs[i,0]*dt + self.v_knots[i]
                return vitesse
            
            else :
                raise ValueError(f"L'angle de {x}° n'appartient pas à l'intervalle de définition de la fonction.")

        v_vect = np.vectorize(v_scalar)
        return v_vect(t)

    def l(self, t):
        """Calcule la levée pour un ensemble de points donnés.

        Args:
            t (numpy.ndarray): Angles de came pour lesquels les levées sont calculées.

        Returns:
            numpy.ndarray: Levées pour les angles renseignés.
        """
        def l_scalar(x) :
            i = np.searchsorted(self.t_knots, x, side="right") - 1
            if (0<=i) and (i<len(self.t_knots) - 1) :
                dt = x - self.t_knots[i]
                
                levee = self.a_coeffs[i,2]/3/4*dt**4 + self.a_coeffs[i,1]/2/3*dt**3 + self.a_coeffs[i,0]/2*dt**2 + self.v_knots[i]*dt + self.l_knots[i]
                return levee
            
            elif (i == len(self.t_knots) - 1) and (x==self.t_knots[-1]):
                i += -1
                dt = x - self.t_knots[i]
                
                levee = self.a_coeffs[i,2]/3/4*dt**4 + self.a_coeffs[i,1]/2/3*dt**3 + self.a_coeffs[i,0]/2*dt**2 + self.v_knots[i]*dt + self.l_knots[i]
                return levee

            else :
                raise ValueError(f"L'angle de {x}° n'appartient pas à l'intervalle de définition de la fonction.")

        l_vect = np.vectorize(l_scalar)
        return l_vect(t)
    
    def add_knot(self,pos,t,a,j=np.nan):
        """Permet d'ajouter un point (angle, acceleration, jerk) aux points initiaux.
        Relance automatiquement les calculs des coefficients de la loi d'accélération.

        Args:
            pos (int): Position à laquelle insérer le nouveau point 
            t (float): Angle du nouveau point.
            a (float): Accélération au nouveau point.
            j (float, optional): Jerk au nouveau point. Defaults to np.nan.

        Returns:
            None
        """
        self.t_inputs = np.insert(self.t_inputs, pos, t)
        self.a_inputs = np.insert(self.a_inputs, pos, a)
        self.j_inputs = np.insert(self.j_inputs, pos, j)
        self.__t_knots, self.__a_knots, self.__j_knots, self.__kflags, self.__a_coeffs = self.schumaker_quad_spline(self.t_inputs, self.a_inputs, self.j_inputs)
        self.__v_knots, self.__l_knots = self.v_l_knots(self.v_knots[0], self.l_knots[0])
        pass

    def remove_knot(self,pos):
        """Permet de supprimer un point parmi les entrées.
        Relance automatiquement les calculs des coefficients de la loi d'accélération.

        Args:
            pos (int): Position à laquelle insérer le nouveau point 
        
        Returns:
            None
        """
        self.t_inputs = np.delete(self.t_inputs, pos)
        self.a_inputs = np.delete(self.a_inputs, pos)
        self.j_inputs = np.delete(self.j_inputs, pos)
        self.__t_knots, self.__a_knots, self.__j_knots, self.__kflags, self.__a_coeffs = self.schumaker_quad_spline(self.t_inputs, self.a_inputs, self.j_inputs)
        self.__v_knots, self.__l_knots = self.v_l_knots(self.v_knots[0], self.l_knots[0])
        pass
    
    def modify_knot(self, pos, t, a, j = np.nan):
        """Permet de modifier un point (angle, acceleration, jerk) parmi les noeuds déjà existants.
        Relance automatiquement les calculs des coefficients de la loi d'accélération.

        Args:
            pos (int): Position du point à modifier. 
            t (float): Angle des nouveau points. Defaults to np.nan.
            a (float): Accélération des nouveau points. Defaults to np.nan.
            j (float): Jerk des nouveau points. Defaults to np.nan.

        Returns:
            None
        """
        self.t_inputs[pos] = t 
        self.a_inputs[pos] = a
        self.j_inputs[pos] = j
        self.__t_knots, self.__a_knots, self.__j_knots, self.__kflags, self.__a_coeffs = self.schumaker_quad_spline(self.t_inputs, self.a_inputs, self.j_inputs)
        self.__v_knots, self.__l_knots = self.v_l_knots(self.v_knots[0], self.l_knots[0])
        pass

    def v_l_knots(self, v_init = 0, l_init = 0):
        """Calcule les vitesses et levées pour les points à interpoler.

        Args:
            v_init (int, optional): Vitesse au premier noeud. Defaults to 0.
            l_init (int, optional): Levée au premier noeud. Defaults to 0.

        Returns:
            numpy.ndarray : vitesses aux points à interpoler.
            numpy.ndarray : levees aux points à interpoler.
        """
        dt_knots = self.t_knots[1:] - self.t_knots[:-1]
        v_knots = [v_init]
        l_knots = [l_init]
        for i in range(len(self.t_knots)-1):
            v_next = self.a_coeffs[i,2]/3*dt_knots[i]**3 + self.a_coeffs[i,1]/2*dt_knots[i]**2 + self.a_coeffs[i,0]*dt_knots[i] + v_knots[-1]
            l_next = self.a_coeffs[i,2]/3/4*dt_knots[i]**4 + self.a_coeffs[i,1]/2/3*dt_knots[i]**3 + self.a_coeffs[i,0]/2*dt_knots[i]**2 + v_knots[-1]*dt_knots[i] + l_knots[-1]
            v_knots.append(v_next)
            l_knots.append(l_next)
        return np.array(v_knots), np.array(l_knots)
    
    #Fonction à implémenter basé sur l'algorithme de Schumaker
    def schumaker_quad_spline(self, x, y, dy_inputs=None):
        """Fonction d'interpolation quadratique préservant la forme de la solution.
        
        Args :
            x (numpy.ndarray): Abscisses des points de contrôle initiaux. Dans notre cas l'angle de la came.
            y (numpy.ndarray):  Ordonnées des points de contrôle initiaux. Dans notre cas l'accélération.
            dy (numpy.ndarray), Optional : Dérivées aux points de contrôle. Dans notre cas le Jerk. Si la dérivée en 1 point n'est pas renseigné alors elles sont estimées. 
        
        Returns :
            (int) : Nombre total de points de contrôle finaux
            (numpy.ndarray) : Abscisses des points de contrôle finaux.
            (numpy.ndarray) : Ordonnées des points de contrôle finaux.
            (numpy.ndarray) : Coefficients des splines pour chaque intervalle compris entre 2 points de contôle.

        """
        ##1-Preprocessing
        weights, slopes = self.preprocess_schumaker(x ,y)

        ##2-Slope Calculations
        dy = self.compute_dy(weights, slopes, dy_inputs)
        
        ##3-Compute Knots and Coefficients
        x_knots = []
        y_knots = []
        dy_knots = []
        kflags = []
        coeffs = []
        for i in range(len(x)-1):
            x_new, y_new, dy_new, xflags, coeffs_new = self.compute_coeffs(x, y, dy, slopes, i)
            x_knots.extend(x_new)
            y_knots.extend(y_new)
            dy_knots.extend(dy_new)
            kflags.extend(xflags)
            coeffs.extend(coeffs_new)
        x_knots.append(x[-1])
        y_knots.append(y[-1])
        dy_knots.append(dy[-1])
        return np.array(x_knots), np.array(y_knots), np.array(dy_knots), np.array(kflags), np.array(coeffs)

    def preprocess_schumaker(self,x ,y):
        """Réalise le prétraitement des données entrées en calculant des poids et les pentes entre noeuds.

        Args:
            x (_type_): _description_
            y (_type_): _description_

        Returns:
            _type_: _description_
        """
        weights = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)
        slopes = (y[1:] - y[:-1])/(x[1:] - x[:-1])
        # Une étape supplémentaire est présente dans l'algorithme initial mais je ne l'ai pas compris. 
        return np.array(weights), np.array(slopes)

    def compute_dy(self, weights, slopes, dy = None) :
        """Calcule les dérivées aux différents points grâce à une différence finies avec poids, voir documentation pour plus de détails.
        
        Args :
            weights (numpy.ndarray): Poids calculés pour chaque point intérieur.
            slopes (numpy.ndarray): Pentes calculées pour chaque point intérieur.

        Returns :
            dy (numpy.ndarray): Dérivées calculer pour tous les points (intérieurs et extrêmes). Dans notre ça correspond au Jerk.
        """
        if dy is None :
            dy = np.array([np.nan for _ in range(len(weights)+1)], dtype=float)
        ind_to_compute = np.where(np.isnan(dy[1:-1]))
        dy_computed = (weights[:-1]*slopes[:-1] + weights[1:]*slopes[1:])/(weights[:-1] + weights[1:])
        dy[1:-1][ind_to_compute] = dy_computed[ind_to_compute]
        if dy[0] == np.nan : 
            dy[0] = 0.5*(3*slopes[0] - dy[1])
        if dy[-1] == np.nan : 
            dy[-1] = 0.5*(3*slopes[-1] - dy[-2])
        return dy

    def compute_coeffs(self, x, y, dy, slopes, i) :
        """Ajoute des noeuds si cela est nécessaire et Calcule les coefficients associés aux splines quadratiques de chaque noeuds sur chaque intervalle."""
        x_new = []
        y_new = []
        dy_new = []
        xflags = []
        coeffs = []

        if dy[i] + dy[i+1] == 2*slopes[i]:
            x_new.append(x[i])
            y_new.append(y[i])
            dy_new.append(dy[i])
            xflags.append(0)
            c0 = y[i]
            c1 = dy[i]
            c2 = 0.5*(dy[i+1] - dy[i])/(x[i+1] - x[i])
            coeffs.append([c0, c1, c2])
        else :
            a = dy[i] - slopes[i]
            b = dy[i+1] - slopes[i]
            if a*b >= 0 :
                epsilon = (x[i+1] + x[i])/2
            else :
                if abs(a) > abs(b) :
                    epsilon = x[i+1] + a*(x[i+1] - x[i])/(dy[i+1] - dy[i])
                else :
                    epsilon = x[i] + b*(x[i+1] - x[i])/(dy[i+1] - dy[i])
            slope_epsilon = (2*slopes[i] - dy[i+1]) + (dy[i+1] - dy[i])*(epsilon - x[i])/(x[i+1] - x[i])
            eta = (slope_epsilon - dy[i])/(epsilon - x[i])

            # Point initial
            x_new.append(x[i])
            y_new.append(y[i])
            dy_new.append(dy[i])
            xflags.append(0)
            c0 = y[i]
            c1 = dy[i]
            c2 = eta/2
            coeffs.append([c0, c1, c2])

            # Point intermédiaire
            x_new.append(epsilon)
            y_new.append(y[i] + dy[i]*(epsilon - x[i]) + eta*(epsilon - x[i])**2/2)
            dy_new.append(slope_epsilon)
            xflags.append(1)
            c0 = y[i] + dy[i]*(epsilon - x[i]) + eta*(epsilon - x[i])**2/2 
            c1 = slope_epsilon
            c2 = (dy[i+1] - slope_epsilon)/(x[i+1] - epsilon)/2
            coeffs.append([c0, c1, c2])
            
        return x_new, y_new, dy_new, xflags, coeffs

    @property
    def t_inputs(self):
        return self.__t_inputs
    @property
    def a_inputs(self):
        return self.__a_inputs
    @property
    def j_inputs(self):
        return self.__j_inputs
    
    @property
    def t_knots(self):
        return self.__t_knots
    @property
    def l_knots(self):
        return self.__l_knots
    @property
    def v_knots(self):
        return self.__v_knots
    @property
    def a_knots(self):
        return self.__a_knots
    @property
    def j_knots(self):
        return self.__j_knots
    @property
    def kflags(self):
        return self.__kflags
    @property 
    def a_coeffs(self):
        return self.__a_coeffs
    
    @t_inputs.setter
    def t_inputs(self, new_t_inputs):
        self.__t_inputs = new_t_inputs
    @a_inputs.setter
    def a_inputs(self, new_a_inputs):
        self.__a_inputs = new_a_inputs
    @j_inputs.setter
    def j_inputs(self, new_j_inputs):
        self.__j_inputs = new_j_inputs

class AjustementLoisCame():
    def __init__(self, squelette, position_ajustement, angles_racines, angle_levee_max, levee_ouverture, vitesse_ouverture, levee_max, levee_fermeture, vitesse_fermeture):
        self.__squelette = squelette
        self.__pts_adjust = position_ajustement
        self.__t_roots = angles_racines
        self.__lo = levee_ouverture
        self.__vo = vitesse_ouverture
        self.__tmax = angle_levee_max
        self.__lmax = levee_max
        self.__lf = levee_fermeture
        self.__vf = vitesse_fermeture

        self.__int_l1, self.__int_l2, self.__int_l3, self.__int_l4, self.__int_v1, self.__int_v2, self.__int_v3, self.__int_v4, self.__result1, self.__result2, self.__result3, self.__result4 = self.compute_matrix_elements()
        
        self.validate_solvable()

        self.__sc1, self.__sc2, self.__sc3 = self.compute_scale_factors()
    
    def validate_solvable(self):
        validation_criteria = -self.vo*self.int_l4 - self.int_v1*(self.t_roots[0]*self.vo - self.lo + self.lmax)
        if validation_criteria == 0 :
            raise ValueError("Le profil d'accélération spécifié n'est pas soluble. Modifier les points de contrôle et réessayer.")
        
    def a(self,t):
        def piecewise_function(x):
            if self.t_roots[0] <= x <self.t_roots[1] :
                return self.sc1*self.squelette.a(x)
            elif self.t_roots[1] <= x <self.t_roots[2] :
                return self.sc2*self.squelette.a(x)
            elif self.t_roots[2] <= x <=self.t_roots[3] :
                return self.sc3*self.squelette.a(x)
            else :
                raise ValueError("L'angle de came renseigné n'est pas dans l'intervalle de définition de la fonction.")
        
        vectorized_function = np.vectorize(piecewise_function)

        return vectorized_function(t)

    def v(self, t):
        def piecewise_function(x):
            if x == 0 :
                return self.vo
            elif self.t_roots[0] < x <=self.t_roots[1] :
                return self.vo + self.sc1*(self.squelette.v(x) - self.vo)
            elif self.t_roots[1] <= x <self.t_roots[2] :
                return self.vo + self.sc1*(self.squelette.v(self.t_roots[1]) - self.vo) + self.sc2*(self.squelette.v(x) - self.squelette.v(self.t_roots[1]))
            elif self.t_roots[2] <= x <=self.t_roots[3] :
                return self.vo + self.sc1*(self.squelette.v(self.t_roots[1]) - self.vo) + self.sc2*(self.squelette.v(self.t_roots[2]) - self.squelette.v(self.t_roots[1])) + self.sc3*(self.squelette.v(x) - self.squelette.v(self.t_roots[2]))
            else :
                raise ValueError("L'angle de came renseigné n'est pas dans l'intervalle de définition de la fonction.")
        
        vectorized_function = np.vectorize(piecewise_function)

        return vectorized_function(t)
        
    def l(self, t):
        def piecewise_function(x):
            if x == 0 :
                return self.lo
            elif self.t_roots[0] < x <=self.t_roots[1] :
                return self.lo + self.sc1*(self.squelette.l(x) - self.lo)
            elif self.t_roots[1] <= x <self.t_roots[2] :
                return self.lo + self.sc1*(self.squelette.l(self.t_roots[1]) - self.lo) + self.sc2*(self.squelette.l(x) - self.squelette.l(self.t_roots[1]))
            elif self.t_roots[2] <= x <=self.t_roots[3] :
                return self.lo + self.sc1*(self.squelette.l(self.t_roots[1]) - self.lo) + self.sc2*(self.squelette.l(self.t_roots[2]) - self.squelette.l(self.t_roots[1])) + self.sc3*(self.squelette.l(x) - self.squelette.l(self.t_roots[2]))
            else :
                raise ValueError("L'angle de came renseigné n'est pas dans l'intervalle de définition de la fonction.")
        
        vectorized_function = np.vectorize(piecewise_function)

        return vectorized_function(t)

    def compute_scale_factors(self):
        a_adjust = self.adjust_accel()
        self.squelette.modify_knot(self.pts_adjust, self.squelette.t_inputs[self.pts_adjust], a_adjust, self.squelette.j_inputs[self.pts_adjust])
        self.__int_l1, self.__int_l2, self.__int_l3, self.__int_l4, self.__int_v1, self.__int_v2, self.__int_v3, self.__int_v4, self.__result1, self.__result2, self.__result3, self.__result4 = self.compute_matrix_elements()

        matrix_pb_global = self.matrix_pb()
        matrix_pb_reduced = matrix_pb_global[:3,:3]
        results_reduced = matrix_pb_global[:3,-1]
        sc1, sc2, sc3 = npl.inv(matrix_pb_reduced)@results_reduced

        return sc1, sc2, sc3

    def adjust_accel(self):
        solution = sco.root_scalar(self.det_matrix_pb, method="secant", x0=self.squelette.a_inputs[self.pts_adjust])
        return solution.root

    def det_matrix_pb(self, a_adjust):
        a_adjust = np.atleast_1d(a_adjust)[0]
        self.squelette.modify_knot(self.pts_adjust, self.squelette.t_inputs[self.pts_adjust], a_adjust, self.squelette.j_inputs[self.pts_adjust])
        
        self.__int_l1, self.__int_l2, self.__int_l3, self.__int_l4, self.__int_v1, self.__int_v2, self.__int_v3, self.__int_v4, self.__result1, self.__result2, self.__result3, self.__result4 = self.compute_matrix_elements()

        return npl.det(self.matrix_pb())

    def matrix_pb(self):
        return np.array([
            [self.int_l1, self.int_l2, self.int_l3, self.result1],
            [self.int_v1, self.int_v2, self.int_v3, self.result2],
            [self.int_l1, self.int_l4, 0, self.result3],
            [self.int_v1, self.int_v4, 0, self.result4]
        ])

    def compute_matrix_elements(self):
        """Calcule tous les éléments de la matrice représentant le problème.

        Returns:
            tuple(float): Élément de la Matrice
        """
        int_l1, err_l1 = sci.quad(self.squelette.v, self.t_roots[0], self.t_roots[1], points = self.squelette.t_knots)
        int_l2, err_l2 = sci.quad(self.squelette.v, self.t_roots[1], self.t_roots[2], points = self.squelette.t_knots)
        int_l3, err_l3 = sci.quad(self.squelette.v, self.t_roots[2], self.t_roots[3], points = self.squelette.t_knots)
        int_l4, err_l4 = sci.quad(self.squelette.v, self.t_roots[1], self.tmax, points = self.squelette.t_knots)

        int_v1, err_v1 = sci.quad(self.squelette.a, self.t_roots[0], self.t_roots[1], points = self.squelette.t_knots)
        int_v2, err_v2 = sci.quad(self.squelette.a, self.t_roots[1], self.t_roots[2], points = self.squelette.t_knots)
        int_v3, err_v3 = sci.quad(self.squelette.a, self.t_roots[2], self.t_roots[3], points = self.squelette.t_knots)
        int_v4, err_v4 = sci.quad(self.squelette.a, self.t_roots[1], self.tmax, points = self.squelette.t_knots)

        result1 = self.lf - self.lo
        result2 = self.vf - self.vo
        result3 = self.lmax - self.lo
        result4 = -self.vo

        return int_l1 , int_l2, int_l3, int_l4, int_v1, int_v2, int_v3, int_v4, result1, result2, result3, result4

    @property
    def squelette(self):
        return self.__squelette
    @property
    def pts_adjust(self):
        return self.__pts_adjust
    @property
    def t_roots(self):
        return self.__t_roots
    @property
    def tmax(self):
        return self.__tmax
    @property
    def lo(self):
        return self.__lo
    @property
    def lf(self):
        return self.__lf
    @property
    def lmax(self):
        return self.__lmax
    @property
    def vo(self):
        return self.__vo
    @property
    def vf(self):
        return self.__vf
    @property
    def int_l1(self):
        return self.__int_l1 
    @property
    def int_l2(self):
        return self.__int_l2
    @property
    def int_l3(self):
        return self.__int_l3
    @property
    def int_l4(self):
        return self.__int_l4
    @property
    def int_v1(self):
        return self.__int_v1
    @property
    def int_v2(self):
        return self.__int_v2
    @property
    def int_v3(self):
        return self.__int_v3
    @property
    def int_v4(self):
        return self.__int_v4
    @property
    def result1(self):
        return self.__result1
    @property
    def result2(self):
        return self.__result2
    @property
    def result3(self):
        return self.__result3
    @property
    def result4(self):
        return self.__result4
    @property
    def sc1(self):
        return self.__sc1
    @property
    def sc2(self):
        return self.__sc2
    @property
    def sc3(self):
        return self.__sc3
    
    @squelette.setter
    def squelette(self, new_squelette):
        self.__squelette == new_squelette

class CalculRampe(ABC) :
    """Une classe qui implémente le calcul des rampes d'ouverture et de fermeture. Cette classe est abstraite.

    Args:
        ABC (Object): Classe Abstraite
    
    Methods:
    """
    def __init__(self, duree_rampe, duree_vitesse_constante, levee_rampe, vitesse_rampe):
        """Innitialisation de la classe et construction des attributs.

        Args:
            duree_rampe (float): Durée angulaire de la phase de rampe.
            duree_vitesse_constante (float): Durée angulaire de la sous-phase à vitesse constante dans la phase de rampe.
            levee_rampe (float): Levée caractéristique de la rampe, i.e. levée en fin de rampe pour la rampe d'ouverture et levée en début de rampe pour la rampe de fermeture.
            vitesse_rampe (float): Vitesse caractéristique de la rampe, i.e vitesse du palier à vitesse constante.
        """
        self.__dtr = duree_rampe
        self.__dtvc = duree_vitesse_constante
        self.__dta = duree_rampe - duree_vitesse_constante
        self.__lr = levee_rampe
        self.__vr = vitesse_rampe
    
    def compute_coeffs_accel(self):
        """Calcul les coefficients du polynôme de degré 7 qui modélise la levée pendant les rampes d'ouverture et de fermeture. Cela revient résoudre un système d'équations linéraires d'ordre 4.

        Raises:
            ValueError: Si la matrice du système d'équation n'est pas inversible, il n'est pas possible de trouver des solutions. 

        Returns:
            numpy.ndarray: Coefficients du polynôme de degré 7 modélisant la levée.
        """
        if npl.det(self.matrix_pb()) < 1e-5 :
            raise ValueError("La matrice du problème est difficilement inversible. Il se peut que les solutions du problème n'existent pas ou soient imprécises")
        return npl.inv(self.matrix_pb())@self.bcs_pb()
    
    @property
    def dtr(self):
        return self.__dtr
    @property
    def dtvc(self):
        return self.__dtvc
    @property
    def dta(self):
        return self.__dta
    @property
    def lr(self):
        return self.__lr
    @property
    def vr(self):
        return self.__vr

    @abstractmethod
    def a(self, t):
        pass
    @abstractmethod
    def v(self, t):
        pass
    @abstractmethod
    def l(self, t):
        pass
    @abstractmethod
    def compute_levee_accel(self):
        pass
    @abstractmethod
    def matrix_pb(self):
        pass
    @abstractmethod
    def bcs_pb(self):
        pass
    
    @property
    @abstractmethod
    def la(self):
        pass
    @property
    @abstractmethod
    def a7(self):
        pass
    @property
    @abstractmethod
    def a6(self):
        pass
    @property
    @abstractmethod
    def a5(self):
        pass        
    @property
    @abstractmethod
    def a4(self):
        pass

class CalculRampeOuverture(CalculRampe):
    """Une classe qui implémente le calcul spécifiquement pour le calcul de la rampe d'ouverture. Cette classe est une implémentation de la classe abstraite CalculRampe définie précédemment.
    """
    def __init__(self, duree_rampe, duree_vitesse_constante, levee_rampe, vitesse_rampe):
        super().__init__(duree_rampe, duree_vitesse_constante, levee_rampe, vitesse_rampe)
        
        self.__la = self.compute_levee_accel()
        self.__a7, self.__a6, self.__a5, self.__a4 = self.compute_coeffs_accel()
    
    def a(self, t):
        """Calcul l'accélération spécifiquement pour la phase de rampe d'ouverture.

        Args:
            t (numpy.ndarray): Angles pour lesquels sont calculées les accélérations
        """
        def a_scalar(x) :
            if 0 <= x < self.dtr - self.dtvc :
                return 42*self.a7*x**5 + 30*self.a6*x**4 + 20*self.a5*x**3 + 12*self.a4*x**2
            elif self.dtr - self.dtvc <= x < self.dtr :
                return 0
            else :
                raise ValueError("L'angle renseigné n'appartient pas à l'ensemble de définition de la fonction.")
        a_vect = np.vectorize(a_scalar)
        return a_vect(t)
            
    def v(self, t):
        def v_scalar(x) :
            if 0 <= x < self.dtr - self.dtvc :
                return 7*self.a7*x**6 + 6*self.a6*x**5 + 5*self.a5*x**4 + 4*self.a4*x**3
            elif self.dtr - self.dtvc <= x < self.dtr :
                return self.vr
            else :
                raise ValueError("L'angle renseigné n'appartient pas à l'ensemble de définition de la fonction.")
        v_vect = np.vectorize(v_scalar)
        return v_vect(t)
        
    def l(self, t):
        def l_scalar(x) :
            if 0 <= x < self.dtr - self.dtvc :
                return self.a7*x**7 + self.a6*x**6 + self.a5*x**5 + self.a4*x**4
            elif self.dtr - self.dtvc <= x < self.dtr :
                return self.lr + self.vr*(x - self.dtr)
            else :
                raise ValueError("L'angle renseigné n'appartient pas à l'ensemble de définition de la fonction.")
        l_vect = np.vectorize(l_scalar)
        return l_vect(t)
    
    def compute_levee_accel(self):
        levee_accel = self.lr - self.dtvc*self.vr
        if levee_accel < 0 :
            raise ValueError("La levée renseigné pour la rampe conduit à une levée négative. Afin de contrer ce problème vous pouvez au choix : \n \t Augmenter la levée de rampe \n \t Diminuer la vitesse de rampe \n \t Diminuer la durée de la phase à vitesse constante.")
        return levee_accel
    
    def matrix_pb(self):
        return np.array([
            [self.dta**3, self.dta**2, self.dta, 1],
            [7*self.dta**3, 6*self.dta**2, 5*self.dta, 4],
            [42*self.dta**3, 30*self.dta**2, 20*self.dta, 12],
            [210*self.dta**3, 120*self.dta**2, 60*self.dta, 24]
        ])
    def bcs_pb(self):
        return np.array([
            self.la/self.dta**4, 
            self.vr/self.dta**3,
            0,
            0
        ])
    
    @property
    def la(self):
        return self.__la
    
    @property
    def a7(self):
        return self.__a7
    @property
    def a6(self):
        return self.__a6
    @property
    def a5(self):
        return self.__a5        
    @property
    def a4(self):
        return self.__a4
    
class CalculRampeFermeture(CalculRampe):
    def __init__(self, duree_rampe, duree_vitesse_constante, levee_rampe, vitesse_rampe):
        super().__init__(duree_rampe, duree_vitesse_constante, levee_rampe, vitesse_rampe)
        
        self.__la = self.compute_levee_accel()
        self.__a7, self.__a6, self.__a5, self.__a4 = self.compute_coeffs_accel()
    
    def a(self, t):
        def a_scalar(x) :
            if 0 <= x < self.dtvc :
                return 0
            elif self.dtvc <= x <= self.dtr :
                return 42*self.a7*(x - self.dtr)**5 + 30*self.a6*(x - self.dtr)**4 + 20*self.a5*(x - self.dtr)**3 + 12*self.a4*(x - self.dtr)**2
            else :
                raise ValueError("L'angle renseigné n'appartient pas à l'ensemble de définition de la fonction.")
        a_vect = np.vectorize(a_scalar)
        return a_vect(t)
    
    def v(self, t):
        def v_scalar(x) :
            if 0 <= x < self.dtvc :
                return self.vr
            elif self.dtvc <= x <= self.dtr :
                return 7*self.a7*(x - self.dtr)**6 + 6*self.a6*(x - self.dtr)**5 + 5*self.a5*(x - self.dtr)**4 + 4*self.a4*(x - self.dtr)**3
            else :
                raise ValueError("L'angle renseigné n'appartient pas à l'ensemble de définition de la fonction.")
        v_vect = np.vectorize(v_scalar)
        return v_vect(t)
    
    def l(self, t):
        def l_scalar(x) :
            if 0 <= x < self.dtvc :
                return self.lr + self.vr*x
            elif self.dtvc <= x <= self.dtr :
                return self.a7*(x - self.dtr)**7 + self.a6*(x - self.dtr)**6 + self.a5*(x - self.dtr)**5 + self.a4*(x - self.dtr)**4
            else :
                raise ValueError("L'angle renseigné n'appartient pas à l'ensemble de définition de la fonction.")
        l_vect = np.vectorize(l_scalar)
        return l_vect(t)
    
    def compute_levee_accel(self):
        levee_accel = self.lr + self.dtvc*self.vr
        if levee_accel < 0 :
            raise ValueError("La levée renseigné pour la rampe conduit à une levée négative. Afin de contrer ce problème vous pouvez au choix : \n \t Augmenter la levée de rampe \n \t Diminuer la vitesse de rampe \n \t Diminuer la durée de la phase à vitesse constante.")
        return levee_accel
    
    def matrix_pb(self):
        return np.array([
            [-self.dta**3, self.dta**2, -self.dta, 1],
            [-7*self.dta**3, 6*self.dta**2, -5*self.dta, 4],
            [-42*self.dta**3, 30*self.dta**2, -20*self.dta, 12],
            [-210*self.dta**3, 120*self.dta**2, -60*self.dta, 24]
        ])
    def bcs_pb(self):
        return np.array([
            self.la/self.dta**4, 
            -self.vr/self.dta**3,
            0,
            0
        ])
    
    @property
    def la(self):
        return self.__la
    
    @property
    def a7(self):
        return self.__a7
    @property
    def a6(self):
        return self.__a6
    @property
    def a5(self):
        return self.__a5        
    @property
    def a4(self):
        return self.__a4
    

if __name__=="__main__":
    
    # Données Lois Comparaison
    t_comp, levee_comp, vitesse_comp, accel_comp = np.loadtxt("data\\exemple_lois_1.txt").T

    # Points Contrôle Lois
    # x = np.array([0, 8, 16.5, 23.1, 63.43, 106.52, 109.8, 118, 126.5])
    # y = np.array([0, 34.1, 0, -4.76, -8.537, -3.533, 0, 34.07, 0])
    # dy = np.array([0,0, np.nan, np.nan, 0, np.nan, np.nan, 0,0],dtype=float)
    x = np.array([0, 8, 16.5, 63.43, 109.8, 118, 126.5])
    y = np.array([0, 34.1, 0, -8.537, 0, 34.07, 0])
    dy = np.array([0,0, np.nan, 0, np.nan, 0,0],dtype=float)
    t_interp = np.linspace(0,126.5,1000)

    ## Test Classe
    clc = SqueletteLoisCame(x,y,dy)
    accel_squelette = clc.a(t_interp)
    knots_in_squelette = np.array([clc.t_knots[np.where(clc.kflags==0)], clc.a_knots[np.where(clc.kflags==0)]]).T
    knots_add_squelette = np.array([clc.t_knots[np.where(clc.kflags==1)], clc.a_knots[np.where(clc.kflags==1)]]).T
    vitesse_squelette = clc.v(t_interp)
    levee_squelette = clc.l(t_interp)

    alc = AjustementLoisCame(
        squelette=clc, 
        position_ajustement=3,
        angles_racines=np.array([0,16.5,109.8,126.5]),
        angle_levee_max=63.25,
        levee_ouverture=0,
        vitesse_ouverture=0,
        levee_max=10900, #En micromètre. À modifier.
        levee_fermeture=0,
        vitesse_fermeture=0
        ) 
    accel_adjust = alc.a(t_interp)
    vitesse_adjust = alc.v(t_interp)
    levee_adjust = alc.l(t_interp)

    fig = plt.figure()
    plt.plot(t_comp, accel_comp,"-.")
    plt.plot(t_interp, accel_squelette)
    plt.plot(t_interp, accel_adjust)
    plt.show()

    fig = plt.figure()
    plt.plot(t_comp, vitesse_comp,"-.")
    plt.plot(t_interp, vitesse_squelette)
    plt.plot(t_interp, vitesse_adjust)
    plt.show()

    fig = plt.figure()
    plt.plot(t_comp, levee_comp,"-.")
    plt.plot(t_interp, levee_squelette/1000)
    plt.plot(t_interp, levee_adjust/1000)
    plt.show()

    print(alc.matrix_pb())
    print(npl.det(alc.matrix_pb()))
    print(alc.sc1, alc.sc2, alc.sc3)