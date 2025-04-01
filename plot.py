import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams
from axis import DynamicXYGrid

# rcParams['toolbar'] = 'None'          # Hide toolbar
rcParams['animation.embed_limit'] = 10  # Lower FPS during window drags

NO_Z_AXIS = False

# ====== PHYSICAL CONSTANTS (SI) ======
e_phys = 1.602176634e-19  # Elementary charge (C)
m_e_phys = 9.1093837015e-31  # Electron rest mass (kg)
c_phys = 2.99792458e8  # Speed of light (m/s)

# ====== SCALING FACTORS ======
L0 = 1e-3  # Length scale: 1 mm
T0 = L0 / c_phys  # Time scale
B0 = m_e_phys * c_phys / (e_phys * L0)  # Magnetic field scale (~5.7 T)

# ====== RESCALED CONSTANTS ======
e = 1.0  # Charge (units of e)
m_e = 1.0  # Mass (units of m_e)
c = 1.0  # Speed of light (units of c)

class TrajectoryAnimation:
    def __init__(self):
        self.ani = None

    def run_simulation(self, energy_MeV, direction, positions, lens_params):
        """Run simulation with given parameters"""
        # Convert energy to velocity (relativistic)
        gamma = 1 + (energy_MeV * 1e6 * e_phys) / (m_e_phys * c_phys ** 2)
        beta = np.sqrt(1 - 1 / gamma ** 2)  # v/c

        # Calculate initial velocity vector
        v0_phys = beta * c_phys * direction
        r0_phys = positions

        # 3. Set lens parameters
        global R_lens, L_lens, k_phys
        k_phys = lens_params['gradient']
        R_lens = lens_params['radius']
        L_lens = lens_params['length']

        # Rescale lens parameters
        R = R_lens / L0
        L = L_lens / L0
        k = k_phys * L0 / B0  # Rescaled gradient

        def magnetic_field(x, y, z):
            """Quadrupole field (X-axis aligned): B = (0, k*y, k*z)."""
            x_abs = np.abs(x)
            r = np.sqrt(y ** 2 + z ** 2)
            if x_abs <= L / 2 and r <= R:
                if np.allclose([y, z], [0.0, 0.0], atol=1e-15):
                    Bx, By, Bz = 0.0, 0.0, 0.0  # Enforce B=0 on-axis
                else:
                    Bx = 0.0
                    By = k * z
                    Bz = k * y
            else:
                Bx, By, Bz = 0.0, 0.0, 0.0
            return np.array([Bx, By, Bz])

        # Convert to rescaled units
        v0 = v0_phys / c_phys
        r0 = r0_phys / L0

        # ====== SIMULATION PARAMETERS ======
        dt_phys = 1e-13  # Time step (s)
        dt = dt_phys / T0  # Rescaled time step
        steps = 50000  # Number of steps
        skip_frames = 100  # Render every N steps

        # Arrays to store trajectory
        positions = np.zeros((steps, 3))
        velocities = np.zeros((steps, 3))

        # Initialize
        positions[0] = r0
        velocities[0] = v0

        # ====== MAIN LOOP ======
        for i in range(1, steps):
            r = positions[i - 1]
            v = velocities[i - 1]

            B = magnetic_field(*r)
            gamma = 1 / np.sqrt(1 - np.linalg.norm(v) ** 2)
            p = gamma * m_e * v
            F = -e * np.cross(v, B)
            dp = F * dt
            p_new = p + dp
            gamma_new = np.sqrt(1 + np.linalg.norm(p_new) ** 2)  # Correct, but ensure p_new is updated properly
            v_new = p_new / (gamma_new * m_e)  # m_e = 1 in rescaled units
            r_new = r + v_new * dt

            positions[i] = r_new
            velocities[i] = v_new

        # ====== CONVERT TO PHYSICAL UNITS ======
        positions_phys = positions * L0  # Convert to meters

        # ====== VISUALIZATION SETUP ======
        fig = plt.figure(figsize=(15, 8))
        gs = fig.add_gridspec(2, 2)

        # 3D Plot
        ax3d = fig.add_subplot(gs[:, 0], projection='3d')

        # 2D Projection Plots
        ax_yx = fig.add_subplot(gs[0, 1])
        ax_zx = fig.add_subplot(gs[1, 1])

        # Lens visualization (3D)
        theta = np.linspace(0, 2 * np.pi, 30)
        x = np.linspace(-L_lens / 2, L_lens / 2, 10)
        theta_grid, x_grid = np.meshgrid(theta, x)
        y_cyl = R_lens * np.cos(theta_grid)
        z_cyl = R_lens * np.sin(theta_grid)
        ax3d.plot_surface(x_grid, y_cyl, z_cyl, alpha=0.2, color='red')

        # Lens visualization (2D projections)
        for ax in [ax_yx, ax_zx]:
            ax.axvspan(-L_lens / 2, L_lens / 2, color='red', alpha=0.1)
            ax.grid(True)

        # Plot limits
        x_min = min(positions_phys[:, 0]) - 0.1
        x_max = max(positions_phys[:, 0]) + 0.1
        ax_yx.set_xlim(x_min, x_max)
        ax_zx.set_xlim(x_min, x_max)

        # Initialize plot elements
        line3d, = ax3d.plot([], [], [], 'b-', lw=1)
        line_yx, = ax_yx.plot([], [], 'b-', lw=1)
        line_zx, = ax_zx.plot([], [], 'b-', lw=1)

        # Labels
        ax3d.set_xlabel('X (m)')
        ax3d.set_ylabel('Y (m)')
        ax3d.set_zlabel('Z (m)')
        ax_yx.set_ylabel('Y (m)')
        ax_zx.set_ylabel('Z (m)')
        ax_zx.set_xlabel('X (m)')
        ax_yx.set_title('Y-X Projection (Focusing Plane)')
        ax_zx.set_title('Z-X Projection (Defocusing Plane)')

        # Animation function
        def update(frame):
            idx = (frame + 1) * skip_frames
            if idx >= steps:
                idx = steps - 1

            # Update 3D plot
            line3d.set_data(positions_phys[:idx, 0], positions_phys[:idx, 1])
            line3d.set_3d_properties(positions_phys[:idx, 2])

            # Update 2D projections
            line_yx.set_data(positions_phys[:idx, 0], positions_phys[:idx, 1])
            line_zx.set_data(positions_phys[:idx, 0], positions_phys[:idx, 2])

            return line3d, line_yx, line_zx

        self.ani = FuncAnimation(fig, update, frames=range(steps // skip_frames),
                                 interval=20, blit=True)

        plt.tight_layout()
        plt.show()
