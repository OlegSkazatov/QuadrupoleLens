from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.uic import loadUi
import sys
import numpy as np
from plot import TrajectoryAnimation


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("ui/mainWindow.ui", self)

        # Connect button
        self.runButton.clicked.connect(self.launch_simulation)
        self.plotting = TrajectoryAnimation()

        # Set default values
        self._set_defaults()

    def _set_defaults(self):
        """Initialize default values for all inputs"""
        # Electron parameters
        self.energySpinBox.setValue(1.0)  # MeV
        self.posXSpinBox.setValue(-60.0)  # mm
        self.posYSpinBox.setValue(0.0)
        self.posZSpinBox.setValue(0.0)
        self.dirXSpinBox.setValue(1.0)
        self.dirYSpinBox.setValue(0.0)
        self.dirZSpinBox.setValue(0.0)

        # Lens parameters
        self.gradientSpinBox.setValue(1.0)  # T/m
        self.radiusSpinBox.setValue(50.0)  # mm
        self.lengthSpinBox.setValue(100.0)  # mm

    def launch_simulation(self):
        """Collect inputs and start simulation"""
        try:
            # 1. Get and validate energy (MeV → J)
            energy_MeV = self.energySpinBox.value()
            if energy_MeV <= 0:
                raise ValueError("Energy must be positive")

            # 2. Get and normalize direction vector
            direction = np.array([
                self.dirXSpinBox.value(),
                self.dirYSpinBox.value(),
                self.dirZSpinBox.value()
            ])
            norm = np.linalg.norm(direction)
            if norm == 0:
                raise ValueError("Direction vector cannot be zero")
            direction /= norm  # Normalize

            # 3. Get initial position (mm → m)
            positions = np.array([
                self.posXSpinBox.value() * 1e-3,
                self.posYSpinBox.value() * 1e-3,
                self.posZSpinBox.value() * 1e-3
            ])

            # 4. Get lens parameters (mm → m)
            lens_params = {
                'gradient': self.gradientSpinBox.value(),  # T/m
                'radius': self.radiusSpinBox.value() * 1e-3,  # m
                'length': self.lengthSpinBox.value() * 1e-3  # m
            }

            # 5. Launch simulation
            self.plotting.run_simulation(energy_MeV, direction, positions, lens_params)

        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Input Error", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())