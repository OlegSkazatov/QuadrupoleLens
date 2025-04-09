# plot.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from magnets import FieldCalculator

c = 2.99792e8

T0 = 1e-14 # Единица времени
L0 = c * T0 # Единица длины



class TrajectoryAnimation:
    def __init__(self):
        self.ani = None
        self.field_calculator = None

    def run_simulation(self, energy_MeV, direction, position, lenses):
        """Основной метод расчета траектории"""

        self.field_calculator = FieldCalculator(lenses)

        # Релятивистские параметры
        gamma = self._calculate_gamma(energy_MeV)
        beta = np.sqrt(1 - 1 / gamma ** 2)

        # Начальные условия
        v0 = beta * direction
        r0 = position / L0

        # Магнитное поле
        def field_func(x, y, z):
            return self.field_calculator.total_field(x, y, z)

        bounds = self.field_calculator.get_system_bounds()

        # Расчёт максимального линейного размера системы
        system_size = max(
            bounds['x_max'] - bounds['x_min'],
            bounds['y_max'] - bounds['y_min'],
            bounds['z_max'] - bounds['z_min']
        )

        # Учитываем начальную позицию электрона
        electron_start_size = max(
            abs(r0[0] * L0 - bounds['x_min']),
            abs(r0[0] * L0 - bounds['x_max']),
            abs(r0[1] * L0 - bounds['y_min']),
            abs(r0[1] * L0 - bounds['y_max']),
            abs(r0[2] * L0 - bounds['z_min']),
            abs(r0[2] * L0 - bounds['z_max'])
        )

        total_path = system_size + electron_start_size

        # Расчёт числа шагов
        steps = int(total_path / L0)

        # Ограничиваем минимальное и максимальное число шагов
        steps = max(min(steps, 10 ** 6), 1000)

        # Численное интегрирование
        trajectory = self._calculate_trajectory(r0, v0, gamma, field_func, steps)

        # Визуализация результатов
        self._create_plots(trajectory)

    def _calculate_gamma(self, energy_MeV):
        """Расчет релятивистского фактора"""
        return 1 + 1.95429*energy_MeV

    def _quadrupole_field(self, x, y, z, lens):
        """Магнитное поле квадруполя"""
        k = lens['gradient'] * L0 * 1e6 # мктеслы / L0
        R = lens['radius'] / L0
        L = lens['length'] / L0

        if abs(x) > L / 2 or y ** 2 + z ** 2 > R ** 2:
            return np.zeros(3)

        return np.array([0, -k * z, -k * y])  # Фокусировка в Y-направлении

    def _calculate_trajectory(self, r0, v0, gamma, field_func, steps):
        """Интегрирование уравнений движения"""
        # Параметры симуляции

        # Инициализация массивов
        pos = np.zeros((steps, 3))
        vel = np.zeros((steps, 3))
        pos[0] = r0
        vel[0] = v0

        # Основной цикл
        for i in range(1, steps):
            B = field_func(*pos[i - 1] * L0)
            F = -np.cross(vel[i - 1], B) # [кг * м/с^2 * 1000000/ec]

            # Релятивистское обновление импульса
            p = gamma * vel[i - 1]  # Текущий импульс [кг * м/c * 1/(me * c)]
            dp = F * 1.758821e-9  # Домножаем на e/me * 10^6, т.к. B в микротеслах
            p_new = p + dp

            # Пересчет скорости из нового импульса
            gamma_new = np.sqrt(1 + (np.linalg.norm(p_new)) ** 2)
            beta_new = np.sqrt(1 - 1 / gamma_new ** 2)
            vel[i] = p_new / np.linalg.norm(p_new) * beta_new

            # Обновление позиции
            pos[i] = pos[i - 1] + vel[i]

        pos = pos * L0
        return pos[:i]  # Обрезаем массив до реального числа шагов

    def _create_plots(self, trajectory):
        """Создание графиков и анимации"""
        fig = plt.figure(figsize=(15, 8))
        ax3d = fig.add_subplot(121, projection='3d')
        ax_yx = fig.add_subplot(222)
        ax_zx = fig.add_subplot(224)
        ax_yx.grid(True)
        ax_zx.grid(True)

        # Визуализация линзы
        if self.field_calculator is not None:
            for lens in self.field_calculator.lenses:
                lens.render_cylinder(ax3d) # Пока только на 3д
                lens.render_2d(ax_yx)
                lens.render_2d(ax_zx)

        # Настройка анимации
        line3d, = ax3d.plot([], [], [], 'b-')
        line_yx, = ax_yx.plot([], [], 'b-')
        line_zx, = ax_zx.plot([], [], 'b-')

        # Лимиты осей
        x1, x2 = trajectory[0][0], trajectory[-1][0]
        r_max = max(list(map(lambda x: x.radius, self.field_calculator.lenses)))
        if x2 > x1:
            x1 = x1 - 0.05*(x2 - x1)
            x2 = x2 + 0.05*(x2 - x1)
        else:
            x1 = x1 + 0.05 * (x2 - x1)
            x2 = x2 - 0.05 * (x2 - x1)
        ax_yx.set_xlim(x1, x2)
        ax_yx.set_ylim(-r_max, r_max)
        ax_zx.set_xlim(x1, x2)
        ax_zx.set_ylim(-r_max, r_max)

        def update(frame):
            # Рассчитываем текущий индекс данных
            skip_frames = 500
            idx = (frame + 1) * skip_frames

            # Обрезаем индекс до размера массиваx
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

    def _draw_lens(self, ax3d, ax_yx, ax_zx, lens):
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
        ax3d.plot_surface(x_cyl, y_cyl, z_cyl,
                        alpha=0.2,
                        color='red',
                        edgecolor='none')

        # Отрисовка плоскости линзы в 2D
        for ax in [ax_yx, ax_zx]:
            ax.axvspan(-L / 2, L / 2, color='red', alpha=0.1)
            ax.grid(True)

        # Добавление меток
        ax3d.set_xlabel('Ось X (м)', labelpad=12)
        ax3d.set_ylabel('Ось Y (м)', labelpad=12)
        ax3d.set_zlabel('Ось Z (м)', labelpad=12)