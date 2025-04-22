# main.py
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.uic import loadUi
import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation

from Editor.editor import ElementEditorWindow
from beam import Beam
from magnets import QuadrupoleLens, FieldCalculator, Dipole
from plot import SingleElectron, ElectronBeam


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi(os.path.sep.join(["ui", "mainWindow.ui"]), self)

        # Инициализация элементов управления
        self.runButton.clicked.connect(self.launch_simulation)
        self.beamRunButton.clicked.connect(self.launch_beam_simulation)
        self.resetButton.clicked.connect(self._set_defaults)
        self.openEditorButton.clicked.connect(self.open_editor)
        self.beamOpenEditorButton.clicked.connect(self.open_editor)
        self.saveButton.clicked.connect(self.save_configuration)
        self.beamSaveButton.clicked.connect(self.save_configuration)
        self.loadButton.clicked.connect(self.load_configuration)
        self.beamLoadButton.clicked.connect(self.load_configuration)
        self.plotting = None
        self.editor_window = None
        self.lenses = []
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
        # self.gradientSpinBox.setValue(1.0)  # Градиент (Т/м)
        # self.radiusSpinBox.setValue(50.0)  # Радиус (мм)
        # self.lengthSpinBox.setValue(100.0)  # Длина (мм)

    def launch_simulation(self):
        """Запуск расчета траектории"""
        try:
            # Сбор входных параметров
            energy = self.energySpinBox.value()
            direction = self._get_normalized_direction()
            position = self._get_position()
            self.plotting = SingleElectron()

            # Загружаем магнитную систему в калькулятор поля
            self.plotting.field_calculator = FieldCalculator(self.lenses.copy())
            # Запуск симуляции
            self.plotting.run_simulation(energy, direction, position)

        except Exception as e:
            self._show_error(str(e))

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
                'energy_spread': self.energySpreadSpinBox.value() / 100,
                'show_3d': self.show_3dCheckbox.isChecked(),
                'show_cross_section': self.showCrossSectionCheckBox.isChecked()
            }
            # Создание пучка
            beam = Beam(beam_params)
            # Запуск симуляции
            self.plotting = ElectronBeam()
            self.plotting.field_calculator = FieldCalculator(self.lenses.copy())
            self.plotting.run_simulation(
                beam,
                num_samples=500
            )

        except Exception as e:
            self._show_error(str(e))

    def load_magnetic_system(self, json_config):
        """Загрузка всех элементов магнитной системы"""
        elements = []

        for elem in json_config["elements"]:
            # Общие параметры для всех элементов
            position = (elem["x"], elem["y"], 0)  # z = 0 по умолчанию
            rotation = Rotation.from_euler('z', -elem["rotation"], degrees=True)
            params = elem["parameters"]

            if elem["type"] == "Quadrupole":
                elements.append(QuadrupoleLens(
                    gradient=params["gradient"],
                    radius=params["radius"],
                    length=params["length"],
                    position=position,
                    rotation=rotation
                ))

            elif elem["type"] == "Dipole":
                elements.append(Dipole(
                    field=params["field"],
                    width=params["width"],  # X-размер
                    length=params["length"],  # Y-размер
                    height=params["height"],  # Z-размер
                    position=position,
                    rotation=rotation
                ))

        self.lenses = elements.copy()

    def save_configuration(self):
        if self.editor_window is None:
            self.open_editor()
        self.editor_window.save_config()

    def load_configuration(self):
        if self.editor_window is None:
            self.open_editor()
        self.editor_window.load_config()
        self.editor_window.set_configuration()

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

    def _show_error(self, message):
        """Обработка исключений"""
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.warning(self, "Error", message)

    def open_editor(self):
        if self.editor_window is not None: return
        self.editor_window = ElementEditorWindow(self)
        self.editor_window.show()




# def except_hook(cls, exception, traceback):  # Чтобы видеть где косяк
#     sys.__excepthook__(cls, exception, traceback)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    # sys.excepthook = except_hook
    sys.exit(app.exec_())
