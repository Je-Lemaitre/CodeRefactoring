from PySide6 import QtCore, QtWidgets, QtGui

import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import Cursor


class LoiAccelWidget(QWidget):
    def __init__(self, t_inputs, a_inputs, angle_display, accel_display):
        super().__init__()
        self.t_inputs = t_inputs
        self.a_inputs = a_inputs
        self.t_display = angle_display
        self.a_display = accel_display

        # Création du graphique matplotlib
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel('Angle de rotation (°)')
        self.ax.set_ylabel('Accélération (mu/°)')

        # Génération des données initiales
        self.angle_data = np.linspace(0, 200, 100)
        self.acceleration_data = np.random.rand(100) * 10  # Exemple de données aléatoires
        self.points, = self.ax.plot(self.angle_data, self.acceleration_data, 'ro', picker=5)

        # Création du curseur pour afficher les coordonnées
        self.cursor = Cursor(self.ax, useblit=True, color='red', linewidth=1)

        # Connexion des événements
        self.canvas.mpl_connect('pick_event', self.on_pick)

        # Mise en place de la mise à l'échelle pour s'adapter à la résolution de l'écran
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def on_pick(self, event):
        # Méthode appelée lorsque l'utilisateur clique sur un point
        if isinstance(event.artist, type(self.points)):
            point_index = event.ind[0]
            x_data = self.angle_data[point_index]
            y_data = self.acceleration_data[point_index]
            print(f"Point sélectionné : Angle={x_data}, Accélération={y_data}")





if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    