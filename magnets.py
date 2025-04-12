# magnets.py
import numpy as np
from scipy.spatial.transform import Rotation
from typing import List


class QuadrupoleLens:
    def __init__(self, gradient, radius, length,
                 position=(0, 0, 0), rotation=None):
        self.gradient = gradient  # T/m
        self.radius = radius  # m
        self.length = length  # m
        self.position = np.array(position, dtype=float)

        # Ориентация (Rotation object из scipy)
        self.rotation = rotation or Rotation.identity()

        # Матрица поворота 3x3
        self.rot_matrix = self.rotation.as_matrix()
        # Обратная матрица для преобразования координат
        self.inv_rot_matrix = self.rot_matrix.T

    def magnetic_field(self, x, y, z):
        """Поле в точке (x,y,z) в СК линзы (микротесла)"""
        # Преобразование в локальные координаты
        local_pos = self.inv_rot_matrix @ (np.array([x, y, z]) - self.position)
        x_loc, y_loc, z_loc = local_pos

        if abs(x_loc) > self.length / 2 or y_loc ** 2 + z_loc ** 2 > self.radius ** 2:
            return np.zeros(3)

        # Поле в локальных координатах (микротесла)
        B_local = np.array([0, -self.gradient * 1e6 * z_loc, -self.gradient * 1e6 * y_loc])

        # Поворот поля в глобальную систему координат
        return self.rot_matrix @ B_local

    def render_cylinder(self, ax3d, num_points=30):
        """Отрисовка цилиндра в 3D"""
        # Генерация точек в локальной СК
        theta = np.linspace(0, 2 * np.pi, num_points)
        z = np.linspace(-self.length / 2, self.length / 2, 2)
        theta_grid, z_grid = np.meshgrid(theta, z)

        x_loc = z_grid
        y_loc = self.radius * np.cos(theta_grid)
        z_loc = self.radius * np.sin(theta_grid)

        # Преобразование в глобальные координаты
        points = np.stack([x_loc, y_loc, z_loc], axis=-1)
        points_global = (self.rot_matrix @ points[..., None]).squeeze() + self.position

        # Отрисовка
        ax3d.plot_surface(points_global[..., 0],
                        points_global[..., 1],
                        points_global[..., 2],
                        alpha=0.2, color='red')

    def render_2d(self, ax):
        """Отрисовка прямоугольника в 2D"""
        pass

    def get_bounding_box(self):
        """Возвращает минимальные и максимальные координаты линзы в глобальной системе"""
        # Вершины цилиндра в локальной системе (X вдоль главной оси)
        x = np.array([-self.length / 2, self.length / 2])
        y = np.array([-self.radius, self.radius])
        z = np.array([-self.radius, self.radius])

        # Создаем сетку точек
        points = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)

        # Преобразуем в глобальные координаты
        global_points = (self.rot_matrix @ points.T).T + self.position

        # Находим границы
        return {
            'x_min': np.min(global_points[:, 0]),
            'x_max': np.max(global_points[:, 0]),
            'y_min': np.min(global_points[:, 1]),
            'y_max': np.max(global_points[:, 1]),
            'z_min': np.min(global_points[:, 2]),
            'z_max': np.max(global_points[:, 2])
        }


class FieldCalculator:
    def __init__(self, lenses: List[QuadrupoleLens]):
        self.lenses = lenses

    def total_field(self, x, y, z):
        """Суммарное поле всех линз в точке (x,y,z)"""
        B_total = np.zeros(3)
        for lens in self.lenses:
            B_total += lens.magnetic_field(x, y, z)
        return B_total

    def get_system_bounds(self):
        """Возвращает общие границы всей системы линз"""
        bounds = {
            'x_min': np.inf, 'x_max': -np.inf,
            'y_min': np.inf, 'y_max': -np.inf,
            'z_min': np.inf, 'z_max': -np.inf
        }

        for lens in self.lenses:
            bb = lens.get_bounding_box()
            bounds['x_min'] = min(bounds['x_min'], bb['x_min'])
            bounds['x_max'] = max(bounds['x_max'], bb['x_max'])
            bounds['y_min'] = min(bounds['y_min'], bb['y_min'])
            bounds['y_max'] = max(bounds['y_max'], bb['y_max'])
            bounds['z_min'] = min(bounds['z_min'], bb['z_min'])
            bounds['z_max'] = max(bounds['z_max'], bb['z_max'])

        return bounds