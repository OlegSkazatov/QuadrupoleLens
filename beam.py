# beam.py

import numpy as np


def calculate_rotation_matrix(direction):
    """Создаёт матрицу поворота из направления пучка"""
    main_axis = direction / np.linalg.norm(direction)
    temp_vector = np.array([0, 0, 1]) if not np.allclose(main_axis, [0, 0, 1]) else np.array([1, 0, 0])
    second_axis = np.cross(main_axis, temp_vector)
    second_axis /= np.linalg.norm(second_axis)
    third_axis = np.cross(main_axis, second_axis)
    # Столбцы матрицы: second_axis (X_локальная), third_axis (Y_локальная), main_axis (Z_локальная)
    return np.column_stack([second_axis, third_axis, main_axis])


class Beam:
    def __init__(self, parameters):
        self.energy = parameters['energy']  # Средняя энергия [МэВ]
        self.current = parameters['current']  # Ток пучка [мА]
        self.position = np.array(parameters['position'])  # Центр пучка [м]
        self.direction = parameters['direction']  # Направление пучка (нормированный вектор)

        # Параметры эллипса
        beam_ellipse = parameters['ellipse']
        self.a = beam_ellipse['a']  # Большая полуось [м]
        self.b = beam_ellipse['b']  # Малая полуось [м]
        self.theta = np.radians(beam_ellipse['theta'])  # Угол поворота [рад]

        # Распределения
        self.density_profile = parameters['density_profile']  # 'constant' или 'gaussian'
        self.energy_spread = parameters['energy_spread']  # Относительный разброс (0..1)
        self.parameters = parameters

    def generate_particles(self, num_samples):
        """Генерация репрезентативных частиц"""

        # 1. Генерация координат в системе пучка
        if self.density_profile == 'Gaussian':
            print('bebebe')
            # Генерация по Гауссу с обрезкой на 3 сигма
            u = np.random.normal(0, 1, num_samples)
            v = np.random.normal(0, 1, num_samples)
            r = np.sqrt(u ** 2 + v ** 2)
            mask = r <= 3.0
            u = u[mask]
            v = v[mask]
            phi = np.arctan2(v, u)
            r = np.sqrt(u ** 2 + v ** 2) / 3.0  # Нормализация до [0,1]
        else:  # constant
            phi = np.random.uniform(0, 2 * np.pi, num_samples)
            r = np.sqrt(np.random.uniform(0, 1, num_samples))

        # 2. Преобразование в эллиптические координаты
        x_beam = self.a * r * np.cos(phi)
        y_beam = self.b * r * np.sin(phi)

        # 3. Поворот в плоскости пучка (вокруг локальной Z)
        rotation = np.array([
            [np.cos(self.theta), -np.sin(self.theta), 0],
            [np.sin(self.theta), np.cos(self.theta), 0],
            [0, 0, 1]
        ])
        rotated_coords = rotation @ np.vstack([x_beam, y_beam, np.zeros_like(x_beam)])

        # 4. Преобразование в глобальную систему
        rotation_matrix = calculate_rotation_matrix(self.direction)
        global_coords = (rotation_matrix @ rotated_coords).T + self.position


        # 5. Расчёт весов частиц
        if self.density_profile == 'gaussian':
            weights = np.exp(-0.5 * (r ** 2))  # Вес убывает от центра
        else:
            weights = np.ones_like(r)

        # 6. Применение энергетического разброса
        energies = self.energy * (1 + np.random.normal(
            scale=self.energy_spread,
            size=global_coords.shape[0]
        ))
        return {
            'positions': global_coords,
            'energies': energies,
            'weights': weights / np.sum(weights)  # Нормализация
        }

    def get_bounding_box(self):
        """Вычисляет границы пучка в глобальной системе координат"""
        # Крайние точки эллипса в локальной системе
        points = np.array([
            [self.a, 0, 0],
            [-self.a, 0, 0],
            [0, self.b, 0],
            [0, -self.b, 0],
            [self.a * np.cos(self.theta), self.a * np.sin(self.theta), 0],
            [-self.a * np.cos(self.theta), -self.a * np.sin(self.theta), 0]
        ])

        # Преобразование в глобальную систему
        global_points = (calculate_rotation_matrix(self.direction) @ points.T).T + self.position


        return {
            'x_min': np.min(global_points[:, 0]),
            'x_max': np.max(global_points[:, 0]),
            'y_min': np.min(global_points[:, 1]),
            'y_max': np.max(global_points[:, 1]),
            'z_min': np.min(global_points[:, 2]),
            'z_max': np.max(global_points[:, 2])
        }