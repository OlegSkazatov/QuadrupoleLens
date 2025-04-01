import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ====== PHYSICAL CONSTANTS ======
e_phys = 1.602176634e-19  # C
m_e_phys = 9.1093837015e-31  # kg
c_phys = 2.99792458e8  # m/s

# ====== SIMULATION PARAMETERS ======
energy_MeV = 2.0
initial_position = np.array([-0.1, 1e-3, 1e-3])  # [-100mm, 1mm, 1mm]
initial_direction = np.array([1, 0, 0])  # Mostly x-axis
lens_params = {
    'gradient': 0.1,  # T/m
    'radius': 50e-3,  # m
    'length': 300e-3  # m
}


# ====== NUMERICAL SOLUTION ======
def numerical_trajectory():
    # 1. Convert energy to velocity
    gamma = 1 + (energy_MeV * 1e6 * e_phys) / (m_e_phys * c_phys ** 2)
    beta = np.sqrt(1 - 1 / gamma ** 2)
    v_mag = beta * c_phys
    v0_phys = v_mag * initial_direction / np.linalg.norm(initial_direction)

    # 2. Set lens parameters
    k_phys = lens_params['gradient']
    R_lens = lens_params['radius']
    L_lens = lens_params['length']

    # 3. Magnetic field function (hard-edge)
    def magnetic_field(x, y, z):
        if abs(x) <= L_lens / 2 and (y ** 2 + z ** 2) <= R_lens ** 2:
            return np.array([0, k_phys * y, k_phys * z])  # T
        return np.array([0, 0, 0])

    # 4. Simulation parameters
    dt = 1e-12  # Time step (s)
    steps = 100000
    positions = np.zeros((steps, 3))
    velocities = np.zeros((steps, 3))

    # Initial conditions
    positions[0] = initial_position
    velocities[0] = v0_phys

    # 5. Main loop
    for i in range(1, steps):
        r = positions[i - 1]
        v = velocities[i - 1]

        B = magnetic_field(*r)
        F = -e_phys * np.cross(v, B)  # Lorentz force (N)

        # Relativistic momentum update
        p = gamma * m_e_phys * v
        dp = F * dt
        p_new = p + dp

        # Normalize velocity to maintain |v| = v_mag
        v_new = p_new / (gamma * m_e_phys)
        v_new = v_mag * v_new / np.linalg.norm(v_new)

        positions[i] = r + v_new * dt
        velocities[i] = v_new

        # Stop if electron exits lens
        if r[0] > L_lens / 2 and i > 100:
            positions = positions[:i + 1]
            break

    return positions


# ====== ANALYTICAL SOLUTION ======
def analytical_trajectory():
    gamma = 1 + (energy_MeV * 1e6 * e_phys) / (m_e_phys * c_phys ** 2)
    beta = np.sqrt(1 - 1 / gamma ** 2)
    v_x = beta * c_phys

    k = lens_params['gradient']
    K = (e_phys * k) / (gamma * m_e_phys * v_x ** 2)  # Quadrupole strength

    x = np.linspace(initial_position[0], initial_position[0] + lens_params['length'], 500)

    # Focusing plane (Y)
    y0 = initial_position[1]
    y_prime0 = initial_direction[1] / initial_direction[0] * v_x
    y = y0 * np.cos(np.sqrt(K) * x) + (y_prime0 / np.sqrt(K)) * np.sin(np.sqrt(K) * x)

    # Defocusing plane (Z)
    z0 = initial_position[2]
    z_prime0 = initial_direction[2] / initial_direction[0] * v_x
    z = z0 * np.cosh(np.sqrt(K) * x) + (z_prime0 / np.sqrt(K)) * np.sinh(np.sqrt(K) * x)

    return np.column_stack([x, y, z])


# ====== VISUALIZATION ======
def plot_comparison():
    num_pos = numerical_trajectory()
    ana_pos = analytical_trajectory()

    fig = plt.figure(figsize=(15, 6))

    # 3D Plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(num_pos[:, 0], num_pos[:, 1], num_pos[:, 2], 'b-', label='Numerical')
    ax1.plot(ana_pos[:, 0], ana_pos[:, 1], ana_pos[:, 2], 'r--', label='Analytical')
    ax1.set_xlabel('X (m)');
    ax1.set_ylabel('Y (m)');
    ax1.set_zlabel('Z (m)')
    ax1.legend()

    # Y/X Comparison (Focusing Plane)
    ax2 = fig.add_subplot(222)
    ax2.plot(num_pos[:, 0], num_pos[:, 1], 'b-', label='Numerical')
    ax2.plot(ana_pos[:, 0], ana_pos[:, 1], 'r--', label='Analytical')
    ax2.set_title('Y-X Plane (Focusing)')
    ax2.set_xlabel('X (m)');
    ax2.set_ylabel('Y (m)')

    # Z/X Comparison (Defocusing Plane)
    ax3 = fig.add_subplot(224)
    ax3.plot(num_pos[:, 0], num_pos[:, 2], 'b-', label='Numerical')
    ax3.plot(ana_pos[:, 0], ana_pos[:, 2], 'r--', label='Analytical')
    ax3.set_title('Z-X Plane (Defocusing)')
    ax3.set_xlabel('X (m)');
    ax3.set_ylabel('Z (m)')

    plt.tight_layout()
    plt.show()


plot_comparison()