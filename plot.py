# plot.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Физические константы
e_phys = 1.602176634e-19  # Заряд электрона (Кл)
m_e_phys = 9.1093837015e-31  # Масса электрона (кг)
c_phys = 2.99792458e8  # Скорость света (м/с)


class TrajectoryAnimation:
    def __init__(self):
        self.ani = None

    def run_simulation(self, energy_MeV, direction, position, lens):
        """Основной метод расчета траектории"""
        # Релятивистские параметры
        gamma = self._calculate_gamma(energy_MeV)
        beta = np.sqrt(1 - 1 / gamma ** 2)

        # Начальные условия
        v0 = beta * c_phys * direction
        r0 = position

        # Магнитное поле линзы
        def field_func(x, y, z):
            return self._quadrupole_field(x, y, z, lens)

        # Численное интегрирование
        trajectory = self._calculate_trajectory(r0, v0, gamma, field_func)

        # Визуализация результатов
        self._create_plots(trajectory, lens)

    def _calculate_gamma(self, energy_MeV):
        """Расчет релятивистского фактора"""
        return 1 + (energy_MeV * 1e6 * e_phys) / (m_e_phys * c_phys ** 2)

    def _quadrupole_field(self, x, y, z, lens):
        """Магнитное поле квадруполя"""
        k = lens['gradient'] * 1e-3 / 5.6856293  # Пересчет в безразмерные единицы
        R = lens['radius']
        L = lens['length']

        if abs(x) > L / 2 or y ** 2 + z ** 2 > R ** 2:
            return np.zeros(3)

        return np.array([0, -k * z, -k * y])  # Фокусировка в Y-направлении

    def _calculate_trajectory(self, r0, v0, gamma, field_func):
        """Интегрирование уравнений движения"""
        # Параметры симуляции
        dt = 1e-13  # Шаг времени (с)
        steps = 50000

        # Инициализация массивов
        pos = np.zeros((steps, 3))
        vel = np.zeros((steps, 3))
        pos[0] = r0
        vel[0] = v0

        # Основной цикл
        for i in range(1, steps):
            B = field_func(*pos[i - 1])
            F = -e_phys * np.cross(vel[i - 1], B)

            # Релятивистское обновление импульса
            p = gamma * m_e_phys * vel[i - 1]  # Текущий импульс
            F = -e_phys * np.cross(vel[i - 1], B)  # Сила Лоренца
            dp = F * dt  # Изменение импульса
            p_new = p + dp

            # Пересчет скорости из нового импульса
            gamma_new = np.sqrt(1 + (np.linalg.norm(p_new) / (m_e_phys * c_phys) ** 2))
            vel[i] = p_new / (gamma_new * m_e_phys)

            # Обновление позиции
            pos[i] = pos[i - 1] + vel[i] * dt

        return pos[:i]  # Обрезаем массив до реального числа шагов

    def _create_plots(self, trajectory, lens):
        """Создание графиков и анимации"""
        fig = plt.figure(figsize=(15, 8))
        ax3d = fig.add_subplot(121, projection='3d')
        ax_yx = fig.add_subplot(222)
        ax_zx = fig.add_subplot(224)

        # Визуализация линзы
        self._draw_lens(ax3d, lens)

        # Настройка анимации
        line3d, = ax3d.plot([], [], [], 'b-')
        line_yx, = ax_yx.plot([], [], 'b-')
        line_zx, = ax_zx.plot([], [], 'b-')

        def update(frame):
            # Рассчитываем текущий индекс данных
            skip_frames = 100
            idx = (frame + 1) * skip_frames

            # Обрезаем индекс до размера массива
            if idx >= len(trajectory):
                idx = len(trajectory) - 1

            # Обновляем 3D траекторию
            line3d.set_data(trajectory[:idx, 0], trajectory[:idx, 1])
            line3d.set_3d_properties(trajectory[:idx, 2])

            # Обновляем 2D проекции
            line_yx.set_data(trajectory[:idx, 0], trajectory[:idx, 1])
            line_zx.set_data(trajectory[:idx, 0], trajectory[:idx, 2])

            return line3d, line_yx, line_zx

        self.ani = FuncAnimation(fig, update, frames=range(len(trajectory) // 100),
                                 interval=20, blit=True)
        plt.show()

    def _draw_lens(self, ax, lens):
        """Отрисовка 3D модели линзы"""
        R = lens['radius']
        L = lens['length']

        # Генерация цилиндрических координат
        theta = np.linspace(0, 2 * np.pi, 30)
        z = np.linspace(-L / 2, L / 2, 10)
        theta_grid, z_grid = np.meshgrid(theta, z)

        # Преобразование в декартовы координаты
        x_cyl = z_grid
        y_cyl = R * np.cos(theta_grid)
        z_cyl = R * np.sin(theta_grid)

        # Отрисовка поверхности
        ax.plot_surface(x_cyl, y_cyl, z_cyl,
                        alpha=0.2,
                        color='red',
                        edgecolor='none')

        # Добавление меток
        ax.set_xlabel('Ось X (м)', labelpad=12)
        ax.set_ylabel('Ось Y (м)', labelpad=12)
        ax.set_zlabel('Ось Z (м)', labelpad=12)