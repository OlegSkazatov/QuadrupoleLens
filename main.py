# main.py
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.uic import loadUi
import sys
import numpy as np
from plot import TrajectoryAnimation


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("ui/mainWindow.ui", self)

        # Инициализация элементов управления
        self.runButton.clicked.connect(self.launch_simulation)
        self.plotting = TrajectoryAnimation()
        self._set_defaults()

    def _set_defaults(self):
        """Установка значений по умолчанию"""
        # Параметры электрона
        self.energySpinBox.setValue(1.0)  # Энергия (МэВ)
        self.posXSpinBox.setValue(-60.0)  # Начальная позиция X (мм)
        self.posYSpinBox.setValue(0.0)
        self.posZSpinBox.setValue(0.0)
        self.dirXSpinBox.setValue(1.0)  # Направляющий вектор
        self.dirYSpinBox.setValue(0.0)
        self.dirZSpinBox.setValue(0.0)

        # Параметры линзы
        self.gradientSpinBox.setValue(1.0)  # Градиент (Т/м)
        self.radiusSpinBox.setValue(50.0)  # Радиус (мм)
        self.lengthSpinBox.setValue(100.0)  # Длина (мм)

    def launch_simulation(self):
        """Запуск расчета траектории"""
        try:
            # Сбор входных параметров
            energy = self.energySpinBox.value()
            direction = self._get_normalized_direction()
            position = self._get_position()
            lens = self._get_lens_params()

            # Запуск симуляции
            self.plotting.run_simulation(energy, direction, position, lens)

        except Exception as e:
            self._show_error(str(e))

    def _get_normalized_direction(self):
        """Нормализация вектора направления"""
        vec = np.array([self.dirXSpinBox.value(),
                        self.dirYSpinBox.value(),
                        self.dirZSpinBox.value()])
        norm = np.linalg.norm(vec)
        if norm == 0:
            raise ValueError("Нулевой вектор направления")
        return vec / norm

    def _get_position(self):
        """Преобразование позиции в метры"""
        return np.array([self.posXSpinBox.value() * 1e-3,
                         self.posYSpinBox.value() * 1e-3,
                         self.posZSpinBox.value() * 1e-3])

    def _get_lens_params(self):
        """Параметры квадрупольной линзы"""
        return {
            'gradient': self.gradientSpinBox.value(),
            'radius': self.radiusSpinBox.value() * 1e-3,
            'length': self.lengthSpinBox.value() * 1e-3
        }

    def _show_error(self, message):
        """Обработка исключений"""
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.warning(self, "Error", message)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())