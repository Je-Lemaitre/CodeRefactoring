# This Python file uses the following encoding: utf-8
import sys
sys.path.append("c:/Users/stagiaire.be/Documents/SOFTCAM_dvpmt/softcam")

from PySide6.QtWidgets import QApplication, QMainWindow

from infrastructure.adapters.viewmodel.graph_widgets import LoiAccelWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Application de bureau avec graphique interactif")
        self.setGeometry(100, 100, 800, 600)

        # Ajout du widget du graphique interactif
        self.graph_widget = LoiAccelWidget()
        self.setCentralWidget(self.graph_widget)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
