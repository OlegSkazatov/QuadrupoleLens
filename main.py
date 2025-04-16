# main.py
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.uic import loadUi
import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation

from beam import Beam
from magnets import QuadrupoleLens, FieldCalculator
from plot import SingleElectron, ElectronBeam


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi(os.path.sep.join(["ui", "mainWindow.ui"]), self)

        # Инициализация элементов управления
        self.runButton.clicked.connect(self.launch_simulation)
        self.beamRunButton.clicked.connect(self.launch_beam_simulation)
        self.plotting = None
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
            self.plotting = SingleElectron()

            # Загружаем магнитную систему в калькулятор поля
            self._load_magnetic_system()
            # Запуск симуляции
            self.plotting.run_simulation(energy, direction, position)

        except Exception as e:
            pass
            # self._show_error(str(e))

    def launch_beam_simulation(self):
        try:
            # Получение параметров из UI
            beam_params = {
                'energy': self.beamEnergySpinBox.value(),
                'current': self.beamCurrentSpinBox.value(),
                'position': self._get_beam_position(),
                'direction': self._get_beam_direction(),
                'ellipse': {
                    'a': self.beamAxisASpinBox.value() * 1e-3,
                    'b': self.beamAxisBSpinBox.value() * 1e-3,
                    'theta': self.beamThetaSpinBox.value()
                },
                'density_profile': self.densityProfileCombo.currentText(),
                'energy_spread': self.energySpreadSpinBox.value() / 100
            }
            # Создание пучка
            beam = Beam(beam_params)
            # Запуск симуляции
            self.plotting = ElectronBeam()
            self._load_magnetic_system()
            self.plotting.run_simulation(
                beam,
                num_samples=500
            )

        except Exception as e:
            pass
            # self._show_error(str(e))

    def _load_magnetic_system(self):
        """Загрузка всех элементов магнитной системы"""
        lenses = []
        # Различные конфигурации для тестов

        # # Одна линза
        # main_lens_params = self._get_lens_params()
        # main_lens = QuadrupoleLens(main_lens_params['gradient'], main_lens_params['radius'],
        #                            main_lens_params['length'])
        # lenses = [main_lens]

        # Тест 1: Две линзы вдоль оси X
        # lens1 = QuadrupoleLens(gradient=6.0, radius=0.05, length=0.1, position=(-0.3, 0, 0))
        # lens2 = QuadrupoleLens(gradient=6.0, radius=0.03, length=0.2, position=(0.2, 0, 0))
        # lenses = [lens1, lens2]

        # Тест 2: Чередуем фокус/дефокус
        lens1 = QuadrupoleLens(gradient=6.0, radius=0.2, length=0.1, position=(0, 0, 0),
                               rotation=Rotation.identity())  # Стандартная ориентация
        lens2 = QuadrupoleLens(gradient=6.0, radius=0.2, length=0.1, position=(0.15, 0, 0),
                               rotation=Rotation.from_euler('x', 90, degrees=True))  # Поворот на 90°
        lenses = [lens1, lens2]
        if self.plotting is not None:
            self.plotting.field_calculator = FieldCalculator(lenses)

    def _get_beam_position(self):
        """Координаты центра пучка"""
        return np.array([self.beamPosXSpinBox.value() * 1e-3,
                         self.beamPosYSpinBox.value() * 1e-3,
                         self.beamPosZSpinBox.value() * 1e-3])

    def _get_beam_direction(self):
        """Направление пучка"""
        vec = np.array([self.beamDirXSpinBox.value(),
                        self.beamDirYSpinBox.value(),
                        self.beamDirZSpinBox.value()])
        norm = np.linalg.norm(vec)
        if norm == 0:
            raise ValueError("Нулевой вектор направления")
        return vec / norm

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



def except_hook(cls, exception, traceback):  # Чтобы видеть где косяк
    sys.__excepthook__(cls, exception, traceback)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.excepthook = except_hook
    sys.exit(app.exec_())
