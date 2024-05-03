import numpy as np
import scipy.optimize as sco

def lissage(x_array, y1, y2, dy1, dy2, method = "poly3"):
    if method == "poly3":
        return lissage_polynomial(x_array, y1, y2, dy1, dy2) 
    elif method == "bezier":
        return lissage_bezier()
    pass

def lissage_bezier(x1, x2, y1, y2, dy1, dy2):
    
    #Point d'intersection des tangentes aux deux points
    cx = (y2 - y1 + dy1 * x1 - dy2 * x2) / (dy1 - dy2)
    cy = dy1 * (cx - x1) + y1

    #Définition des points de contrôle de la courbe de Bézier
    P0 = np.array([x1,y1])
    P1 = np.array([cx,cy])
    P2 = np.array([x2,y2])

    def courbe_bezier(absc_curv):
        return (1 - absc_curv)**2*P0 + 2*(1-absc_curv)*absc_curv*P1 + absc_curv**2*P2
    
    return lambda x : fct_curv_to_cart(courbe_bezier, x)

def lissage_polynomial(x_array, y1, y2, dy1, dy2):
    """
    xn : coordonnée 1 du point n
    yn : coordonnée 2 du point n
    dyn : dérivée dy/dx au point n
    """ 
    x1 = x_array[0]
    x2 = x_array[-1]
    ## On cherche maintenant à déterminer les 4 coefficients du polynôme de degré 3. Ce qui revient à resoudre un système linéaire de 4 équations à 4 inconnues.

    coefficients = resout_systeme_ordre4(x1, x2, y1, y2, dy1, dy2)

    x_matrice = np.array([
        x_array**3,
        x_array**2,
        x_array,
        np.ones(len(x_array))
    ])

    new_y_array = coefficients@x_matrice

    return new_y_array.flatten()


def resout_systeme_ordre4 (x1, x2, y1, y2, dy1, dy2) :
    matrice_systeme = np.array([
        [x1**3, x1**2, x1, 1],
        [x2**3, x2**2, x2, 1],
        [3*x1**2, 2*x1, 1, 0],
        [3*x2**2, 2*x2, 1, 0]
    ])
    second_membre_systeme = np.transpose(np.array([y1, y2, dy1, dy2]))
    coefficients = np.linalg.inv(matrice_systeme)@second_membre_systeme

    return np.transpose(coefficients)

def fct_curv_to_cart(f, x_cart) :
    def diff(absc_curv):
        return abs(f(absc_curv)[0] - x_cart)
        
    absc_curv = sco.minimize(diff, 0.5)

    return f(absc_curv.x)


def transformation_affine_bezier(x0, x1, x2, y0, y1, y2):
    #Sûrement inutile finalement
    matrice_systeme = np.array([[x0, y0, 1],
                                [x1, y1, 1],
                                [x2, y2, 1]])
    
    x_trans = np.array([-1, 0, 1])
    y_trans = np.array([0, 1, 0])
    coeffs_x = np.linalg.inv(matrice_systeme)@x_trans
    coeffs_y = np.linalg.inv(matrice_systeme)@y_trans

    matrice_trans = np.array([coeffs_x[:2],
                              coeffs_y[:2]])
    
    vecteur_trans = np.array([coeffs_x[2],
                              coeffs_y[2]])

    return matrice_trans, vecteur_trans