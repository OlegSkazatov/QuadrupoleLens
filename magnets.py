# magnets.py
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import art3d
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from typing import List


class QuadrupoleLens:
    def __init__(self, gradient, radius, length,
                 position=(0, 0, 0), rotation=None, file=''):
        self.interpolator = None
        self.real_field = None
        self.gradient = gradient  # T/m
        self.radius = radius  # m
        self.length = length  # m
        self.position = np.array(position, dtype=float)
        self.file = file

        # Ориентация (Rotation object из scipy)
        self.rotation = rotation or Rotation.identity()

        # Матрица поворота 3x3
        self.rot_matrix = self.rotation.as_matrix()
        # Обратная матрица для преобразования координат
        self.inv_rot_matrix = self.rot_matrix.T

    def import_field(self, f):
        data = []
        with open(f, "r") as file:
            lines = file.read().split("\n")[1:]  # Пропускаем заголовок
            for line in lines:
                if not line.strip(): continue
                parts = line.replace(",", ".").split("\t")
                row = list(map(float, parts))
                data.append(row)
        data = np.array(data)
        self.real_field = data

    def magnetic_field(self, x, y, z):
        """Поле в точке (x,y,z) в СК линзы (микротесла)"""
        # Преобразование в локальные координаты
        local_pos = self.inv_rot_matrix @ (np.array([x, y, z]) - self.position)
        x_loc, y_loc, z_loc = local_pos

        if self.real_field is not None:
            if self.interpolator is None:
                x_grid, y_grid, z_grid, B_values = self.real_field
                self.interpolator = RegularGridInterpolator(
                    (x_grid, y_grid, z_grid),
                    B_values,
                    method='linear',
                    bounds_error=False,
                    fill_value=0.0
                )
            return self.interpolator((x_loc, y_loc))

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

    def render_xy(self, ax):
        self._render_projection(ax, plane='xy')

    def render_xz(self, ax):
        self._render_projection(ax, plane='xz')

    def render_yz(self, ax):
        self._render_projection(ax, plane='yz')

    def _render_projection(self, ax, plane):
        """Универсальный метод проекций для квадруполя"""
        # Генерация точек на цилиндре
        theta = np.linspace(0, 2 * np.pi, 50)
        z = np.linspace(-self.length / 2, self.length / 2, 2)
        theta_grid, z_grid = np.meshgrid(theta, z)

        # Локальные координаты цилиндра
        x_loc = z_grid
        y_loc = self.radius * np.cos(theta_grid)
        z_loc = self.radius * np.sin(theta_grid)

        # Преобразование в глобальные координаты
        points = np.stack([x_loc, y_loc, z_loc], axis=-1)
        points_global = (self.rot_matrix @ points[..., None]).squeeze() + self.position

        # Выбор осей для проекции
        if plane == 'xy':
            proj = points_global[..., [0, 1]]
        elif plane == 'xz':
            proj = points_global[..., [0, 2]]
        elif plane == 'yz':
            proj = points_global[..., [1, 2]]

        # Построение выпуклой оболочки
        hull = ConvexHull(proj.reshape(-1, 2))
        poly = plt.Polygon(
            proj.reshape(-1, 2)[hull.vertices],
            closed=True,
            fill=True,
            color='red',
            alpha=0.3
        )
        ax.add_patch(poly)

        # Для 3D-осей
        if hasattr(ax, 'get_zlim'):
            z_coord = 0 if plane == 'yz' else self.position[2]
            art3d.pathpatch_2d_to_3d(poly, z=z_coord)

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


class Dipole:
    def __init__(self, field, width, length, height,
                 position=(0, 0, 0), rotation=None, file=''):
        self.interpolator = None
        self.real_field = None
        self.field_value = field  # T (константное поле)
        self.width = width  # m (ширина по X)
        self.length = length  # m (длина по Y)
        self.height = height  # m (высота по Z)
        self.position = np.array(position, dtype=float)
        self.file = file

        # Ориентация (Rotation object из scipy)
        self.rotation = rotation or Rotation.identity()
        self.rot_matrix = self.rotation.as_matrix()
        self.inv_rot_matrix = self.rot_matrix.T

        if self.file:
            self.import_field(self.file)

    def import_field(self, f):
        data = []
        with open(f, "r") as file:
            lines = file.read().split("\n")[1:]  # Пропускаем заголовок
            for line in lines:
                if not line.strip(): continue
                parts = line.replace(",", ".").split("\t")
                row = list(map(float, parts))
                data.append(row)
        data = np.array(data)
        self.real_field = data

    def magnetic_field(self, x, y, z):
        """Поле в точке (x,y,z) в СК диполя (микротесла)"""
        # Преобразование в локальные координаты
        local_pos = self.inv_rot_matrix @ (np.array([x, y, z]) - self.position)
        x_loc, y_loc, z_loc = local_pos
        if self.real_field is not None:
            if self.interpolator is None:
                x_grid = np.unique(self.real_field[:, 0])
                y_grid = np.unique(self.real_field[:, 1])
                x_size = len(x_grid)
                y_size = len(y_grid)

                # Сортируем данные сначала по X, затем по Y
                sorted_indices = np.lexsort((self.real_field[:, 1], self.real_field[:, 0]))
                sorted_data = self.real_field[sorted_indices]

                # Проверяем согласованность данных
                if len(sorted_data) != x_size * y_size:
                    raise ValueError("Данные не образуют регулярную сетку")

                B_values = sorted_data[:, 2].reshape(x_size, y_size)

                self.interpolator = RegularGridInterpolator(
                    (x_grid, y_grid),
                    B_values,
                    method='linear',
                    bounds_error=False,
                    fill_value=0.0
                )
            return np.array([0, 0, self.interpolator((x_loc, y_loc)) * 1e6])

        # Проверка нахождения внутри объёма диполя
        in_x = abs(x_loc) <= self.width / 2  # X-границы
        in_y = abs(y_loc) <= self.length / 2  # Y-границы (длина)
        in_z = abs(z_loc) <= self.height / 2  # Z-границы (высота)

        if not (in_x and in_y and in_z):
            return np.zeros(3)

        # Поле в локальной системе (направлено по локальной Z)
        B_local = np.array([0, 0, self.field_value * 1e6])  # конвертация Т → мкТ

        # Поворот поля в глобальную систему
        return self.rot_matrix @ B_local

    def get_bounding_box(self):
        """Возвращает границы диполя в глобальной системе"""
        # Вершины параллелепипеда в локальной СК
        x = np.array([-self.width / 2, self.width / 2])  # X
        y = np.array([-self.length / 2, self.length / 2])  # Y (длина)
        z = np.array([-self.height / 2, self.height / 2])  # Z (высота)

        points = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)

        # Преобразование в глобальные координаты
        global_points = (self.rot_matrix @ points.T).T + self.position

        return {
            'x_min': np.min(global_points[:, 0]),
            'x_max': np.max(global_points[:, 0]),
            'y_min': np.min(global_points[:, 1]),
            'y_max': np.max(global_points[:, 1]),
            'z_min': np.min(global_points[:, 2]),
            'z_max': np.max(global_points[:, 2])
        }

    def render_cylinder(self, ax3d, num_points=30):
        """Отрисовка параллелепипеда в 3D"""
        # Локальные координаты углов параллелепипеда
        corners = np.array([
            [-self.width / 2, -self.length / 2, -self.height / 2],
            [self.width / 2, -self.length / 2, -self.height / 2],
            [self.width / 2, self.length / 2, -self.height / 2],
            [-self.width / 2, self.length / 2, -self.height / 2],
            [-self.width / 2, -self.length / 2, self.height / 2],
            [self.width / 2, -self.length / 2, self.height / 2],
            [self.width / 2, self.length / 2, self.height / 2],
            [-self.width / 2, self.length / 2, self.height / 2],
        ])

        # Преобразование в глобальные координаты
        global_corners = (self.rot_matrix @ corners.T).T + self.position

        # Список граней (индексы углов)
        faces = [
            [0, 1, 2, 3],  # нижняя грань
            [4, 5, 6, 7],  # верхняя грань
            [0, 1, 5, 4],  # передняя грань
            [2, 3, 7, 6],  # задняя грань
            [0, 3, 7, 4],  # левая грань
            [1, 2, 6, 5],  # правая грань
        ]

        # Отрисовка всех граней
        for face in faces:
            verts = global_corners[face]
            ax3d.add_collection3d(
                art3d.Poly3DCollection([verts], alpha=0.3, color='blue')
            )

        # Настройка осей для лучшего отображения
        ax3d.autoscale_view()

    def render_xy(self, ax):
        """Проекция на плоскость XY"""
        self._render_projection(ax, plane='xy')

    def render_xz(self, ax):
        """Проекция на плоскость XZ"""
        self._render_projection(ax, plane='xz')

    def render_yz(self, ax):
        """Проекция на плоскость YZ"""
        self._render_projection(ax, plane='yz')

    def _render_projection(self, ax, plane):
        """Общий метод для проекций"""
        # Получаем локальные координаты углов
        corners = np.array([
            [-self.width / 2, -self.length / 2, -self.height / 2],
            [self.width / 2, -self.length / 2, -self.height / 2],
            [self.width / 2, self.length / 2, -self.height / 2],
            [-self.width / 2, self.length / 2, -self.height / 2],
            [-self.width / 2, -self.length / 2, self.height / 2],
            [self.width / 2, -self.length / 2, self.height / 2],
            [self.width / 2, self.length / 2, self.height / 2],
            [-self.width / 2, self.length / 2, self.height / 2],
        ])

        # Преобразуем в глобальные координаты
        global_corners = (self.rot_matrix @ corners.T).T + self.position

        # Выбираем оси для проекции
        if plane == 'xy':
            proj = global_corners[:, [0, 1]]
        elif plane == 'xz':
            proj = global_corners[:, [0, 2]]
        elif plane == 'yz':
            proj = global_corners[:, [1, 2]]

        # Рисуем выпуклую оболочку
        hull = ConvexHull(proj)
        poly = plt.Polygon(
            proj[hull.vertices],
            closed=True,
            fill=True,
            color='blue',
            alpha=0.4
        )
        ax.add_patch(poly)

        # Для 3D-осей нужно преобразовать патч
        if hasattr(ax, 'get_zlim'):
            art3d.pathpatch_2d_to_3d(poly, z=0 if plane == 'yz' else self.position[2])

class FieldCalculator:
    def __init__(self, lenses):
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