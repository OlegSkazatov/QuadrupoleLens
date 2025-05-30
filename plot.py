# plot.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import QProgressDialog
from mpl_toolkits.mplot3d import Axes3D
from numpy.ma.core import log10
from scipy.spatial import ConvexHull
from matplotlib.patches import Ellipse
import mpl_toolkits.mplot3d.art3d as art3d

from magnets import FieldCalculator, QuadrupoleLens

c = 2.99792e8

T0 = 1e-12  # Единица времени
L0 = c * T0  # Единица длины


def calculate_trajectory(r0, v0, gamma, field_func, steps):
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
        F = -np.cross(vel[i - 1], B)  # [кг * м/с^2 * 1000000/ec]

        # Релятивистское обновление импульса
        p = gamma * vel[i - 1]  # Текущий импульс [кг * м/c * 1/(me * c)]
        dp = F * 1.758821 * 10 ** (log10(T0)-6+11)  # Домножаем на e/me * 10^6, т.к. B в микротеслах
        p_new = p + dp

        # Пересчет скорости из нового импульса
        gamma_new = np.sqrt(1 + (np.linalg.norm(p_new)) ** 2)
        beta_new = np.sqrt(1 - 1 / gamma_new ** 2)
        vel[i] = p_new / np.linalg.norm(p_new) * beta_new

        # Обновление позиции
        pos[i] = pos[i - 1] + vel[i]

    pos = pos * L0
    return pos[:i]  # Обрезаем массив до реального числа шагов


def calculate_gamma(energy_MeV):
    """Расчет релятивистского фактора"""
    return 1 + 1.95429 * energy_MeV


class BeamWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object, object)
    error = pyqtSignal(str)

    def __init__(self, plotter, beam, num_samples=1000):
        super().__init__()
        self.plotter = plotter
        self.beam = beam
        self.num_samples = num_samples

    def run(self):
        particles = self.beam.generate_particles(self.num_samples)

        # Получаем границы всей системы
        lens_bounds = self.plotter.field_calculator.get_system_bounds()
        # beam_bounds = self.beam.get_bounding_box()
        #
        # # Объединяем границы
        # system_bounds = {
        #     'x_min': min(lens_bounds['x_min'], beam_bounds['x_min']),
        #     'x_max': max(lens_bounds['x_max'], beam_bounds['x_max']),
        #     'y_min': min(lens_bounds['y_min'], beam_bounds['y_min']),
        #     'y_max': max(lens_bounds['y_max'], beam_bounds['y_max']),
        #     'z_min': min(lens_bounds['z_min'], beam_bounds['z_min']),
        #     'z_max': max(lens_bounds['z_max'], beam_bounds['z_max'])
        # }
        #
        # # Расчёт максимального линейного размера
        # system_size = max(
        #     system_bounds['x_max'] - system_bounds['x_min'],
        #     system_bounds['y_max'] - system_bounds['y_min'],
        #     system_bounds['z_max'] - system_bounds['z_min']
        # )
        system_size = max(
                lens_bounds['x_max'] - lens_bounds['x_min'],
                lens_bounds['y_max'] - lens_bounds['y_min'],
                lens_bounds['z_max'] - lens_bounds['z_min']
            )

        # Вычисление базового числа шагов
        steps = int(system_size / L0)
        steps = max(min(steps, 10 ** 6), 1000)

        # Корректировка шагов для каждой частицы
        trajectories = []
        for i in range(self.num_samples):
            if self.isInterruptionRequested():
                break
            # Расчёт индивидуального steps
            particle_pos = particles['positions'][i]
            min_dist = min(
                np.linalg.norm(particle_pos - lens.position)
                for lens in self.plotter.field_calculator.lenses
            ) if self.plotter.field_calculator.lenses else system_size

            adaptive_steps = int(min_dist / L0 * 1000)
            final_steps = min(adaptive_steps, steps)
            # Расчёт траектории
            gamma = calculate_gamma(particles['energies'][i])
            beta = np.sqrt(1 - 1 / gamma ** 2)
            v0 = beta * self.beam.direction
            positions = particles['positions'] / L0
            trajectory = calculate_trajectory(
                positions[i],
                v0,
                gamma,
                lambda x, y, z: self.plotter.field_calculator.total_field(x, y, z),
                steps=final_steps
            )
            trajectories.append(trajectory)
            self.progress.emit(int((i + 1) / self.num_samples * 100))

        self.finished.emit(trajectories, particles['weights'])


class ElectronBeam:
    def __init__(self):
        self.field_calculator = None
        self.show_3d = False
        self.show_cross_section = False
        self.animation_fps = 1  # Частота кадров
        self.cross_section_step = 50  # Шаг выборки точек

    def run_simulation(self, beam, num_samples=1000):
        """Основной метод для симуляции пучка"""
        self.show_3d = beam.parameters['show_3d']
        self.show_cross_section = beam.parameters['show_cross_section']
        try:
            # Расчет траекторий для всех частиц
            self._calculate_beam_trajectories(beam, num_samples=num_samples)

        except Exception as e:
            print(f"Ошибка при расчете пучка: {str(e)}")
            raise

    def _calculate_beam_trajectories(self, beam, num_samples=1000):
        """Расчёт траекторий для пучка с адаптивным steps"""
        # Создаем прогресс-диалог
        self.progress_dialog = QProgressDialog(minimum=0, maximum=100,
            labelText="Calculating beam trajectories..."
        )
        self.progress_dialog.setWindowTitle("Progress")
        self.progress_dialog.setWindowModality(Qt.WindowModal)

        # Создаем и настраиваем worker
        self.beam_worker = BeamWorker(self, beam, num_samples=num_samples)
        # Подключаем сигналы
        self.beam_worker.progress.connect(self.progress_dialog.setValue)
        self.beam_worker.finished.connect(self.on_beam_simulation_finished)
        # self.beam_worker.error.connect(self._show_error)
        self.progress_dialog.canceled.connect(self.beam_worker.requestInterruption)

        # Запускаем
        self.beam_worker.start()
        self.progress_dialog.show()

    def on_beam_simulation_finished(self, trajectories, weights):
        self.progress_dialog.close()
        self._create_beam_plots(trajectories, weights)

    def _create_beam_plots(self, trajectories, weights):
        """Визуализация для пучка"""
        # Создаем основное окно с 2D проекциями
        fig_2d = plt.figure(figsize=(15, 8))

        # Настройка сетки графиков
        gs = fig_2d.add_gridspec(2, 2)
        ax_xy = fig_2d.add_subplot(gs[:, 0])  # Большой график XY
        ax_xz = fig_2d.add_subplot(gs[0, 1])  # Верхний правый XZ
        ax_yz = fig_2d.add_subplot(gs[1, 1])  # Нижний правый YZ

        # Отрисовка проекций пучка
        self._plot_2d_projections(ax_xy, ax_xz, ax_yz, trajectories)

        # Отрисовка магнитных элементов
        if self.field_calculator is not None:
            for lens in self.field_calculator.lenses:
                lens.render_xy(ax_xy)
                lens.render_xz(ax_xz)
                lens.render_yz(ax_yz)

        # Настройка подписей осей
        ax_xy.set_xlabel('X (м)')
        ax_xy.set_ylabel('Y (м)')
        ax_xz.set_xlabel('X (м)')
        ax_xz.set_ylabel('Z (м)')
        ax_yz.set_xlabel('Y (м)')
        ax_yz.set_ylabel('Z (м)')
        ax_xy.grid(True)
        ax_xz.grid(True)
        ax_yz.grid(True)

        # Отрисовка 3D графика при необходимости
        if self.show_3d:
            self._create_3d_plot(trajectories)
        if self.show_cross_section:
            self._create_cross_section_animation(trajectories)

        plt.tight_layout()
        plt.show()

    def _create_cross_section_animation(self, trajectories):
        """Создание анимации поперечного сечения"""
        # Подготовка данных
        all_points = np.concatenate(trajectories)
        num_frames = len(all_points) // self.cross_section_step
        # Создание фигуры
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.set_title('Поперечное сечение пучка (95% доверительный интервал)')
        ax.set_xlabel('X (м)')
        ax.set_ylabel('Y (м)')
        ax.grid(True)
        # Инициализация элементов анимации
        scatter = ax.scatter([], [], s=1, alpha=0.3)
        ellipse = ax.add_patch(Ellipse((0, 0), 0, 0, angle=0, color='r', fill=False, lw=2))
        time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)

        # Расчет границ для стабильной анимации
        x = all_points[:, 0]
        y = all_points[:, 1]
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        def animate(frame):
            # Выборка данных для текущего кадра
            idx = frame * self.cross_section_step
            points = all_points[:idx]

            # Обновление рассеяния
            scatter.set_offsets(points[:, [0, 1]])

            # Расчет эллипса
            if len(points) > 10:
                cov = np.cov(points[:, [0, 1]].T)
                lambda_, v = np.linalg.eig(cov)
                angle = np.degrees(np.arctan2(*v[:, 0][::-1]))

                # Доверительный интервал 95% (хи-квадрат с 2 степенями свободы)
                scale = np.sqrt(5.991)
                width, height = 2 * scale * np.sqrt(lambda_)

                # Обновление эллипса
                ellipse.set_center(np.mean(points[:, [0, 1]], axis=0))
                ellipse.set_width(width)
                ellipse.set_height(height)
                ellipse.set_angle(angle)

            # Обновление времени
            time_text.set_text(f'Шаг: {idx}/{len(all_points)}')
            return scatter, ellipse, time_text

        # Создание анимации
        ani = FuncAnimation(
            fig, animate,
            frames=num_frames,
            interval=1000 // self.animation_fps,
            blit=True
        )

        plt.show()

    def _plot_2d_projections(self, ax_xy, ax_xz, ax_yz, trajectories):
        """Отрисовка 2D проекций траекторий"""
        for traj in trajectories:
            # Проекция XY
            ax_xy.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.1, linewidth=0.5)

            # Проекция XZ
            ax_xz.plot(traj[:, 0], traj[:, 2], 'b-', alpha=0.1, linewidth=0.5)

            # Проекция YZ
            ax_yz.plot(traj[:, 1], traj[:, 2], 'b-', alpha=0.1, linewidth=0.5)

    def _create_3d_plot(self, trajectories):
        """Создание отдельного 3D окна"""
        fig_3d = plt.figure(figsize=(10, 8))
        ax_3d = fig_3d.add_subplot(111, projection='3d')

        # Отрисовка оболочки пучка
        all_points = np.concatenate(trajectories)
        hull = ConvexHull(all_points)
        ax_3d.plot_trisurf(
            all_points[:, 0],
            all_points[:, 1],
            all_points[:, 2],
            triangles=hull.simplices,
            alpha=0.5,
            color='blue'
        )

        # Отрисовка магнитной системы
        if self.field_calculator is not None:
            for lens in self.field_calculator.lenses:
                lens.render_cylinder(ax_3d)

        # Настройка осей
        ax_3d.set_xlabel('X (м)')
        ax_3d.set_ylabel('Y (м)')
        ax_3d.set_zlabel('Z (м)')
        plt.title('3D Visualization')



class SingleElectron:
    def __init__(self):
        self.ani = None
        self.field_calculator = None

    def run_simulation(self, energy_MeV, direction, position):
        """Основной метод расчета траектории"""
        if self.field_calculator is None:
            self.field_calculator = FieldCalculator([])

        # Релятивистские параметры
        gamma = calculate_gamma(energy_MeV)
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

        # Расчёт числа шагов, исходя из размеров системы
        steps = int(total_path / L0)

        # Ограничиваем минимальное и максимальное число шагов
        steps = max(min(steps, 10 ** 6), 1000)

        # Численное интегрирование
        trajectory = calculate_trajectory(r0, v0, gamma, field_func, steps)
        # Визуализация результатов
        self._create_plots(trajectory)

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
                lens.render_cylinder(ax3d)
                lens.render_xy(ax_yx)
                lens.render_xz(ax_zx)

        # Настройка анимации
        line3d, = ax3d.plot([], [], [], 'b-')
        line_yx, = ax_yx.plot([], [], 'b-')
        line_zx, = ax_zx.plot([], [], 'b-')

        # Лимиты осей
        x1, x2 = trajectory[0][0], trajectory[-1][0]

        # r_max = max(list(map(lambda x: x.radius, self.field_calculator.lenses)))

        if x2 > x1:
            x1 = x1 - 0.05 * (x2 - x1)
            x2 = x2 + 0.05 * (x2 - x1)
        else:
            x1 = x1 + 0.05 * (x2 - x1)
            x2 = x2 - 0.05 * (x2 - x1)
        ax_yx.set_xlim(x1, x2)
        ax3d.set_xlabel('X (м)', labelpad=12)
        ax3d.set_ylabel('Y (м)', labelpad=12)
        ax3d.set_zlabel('Z (м)', labelpad=12)
        ax_yx.set_xlabel('X (м)', labelpad=12)
        ax_yx.set_ylabel('Y (м)', labelpad=12)
        ax_zx.set_xlabel('X (м)', labelpad=12)
        ax_zx.set_ylabel('Z (м)', labelpad=12)
        # ax_yx.set_ylim(-r_max, r_max)
        # ax_zx.set_xlim(x1, x2)
        # ax_zx.set_ylim(-r_max, r_max)
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
