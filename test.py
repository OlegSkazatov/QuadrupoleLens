import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

# ====== QUADRUPOLE LENS PARAMETERS ======
R_lens = 0.2  # Lens radius (5 mm)
L_lens = 0.1  # Lens length (10 mm)
k_phys = 10.0  # Gradient (T/m)

# Rescale lens parameters
R = R_lens / L0
L = L_lens / L0
k = k_phys / B0  # Rescaled gradient


def magnetic_field(x, y, z):
    """Quadrupole field localized to the lens volume."""
    r = np.sqrt(x ** 2 + y ** 2)
    z_abs = np.abs(z)
    if r <= R and z_abs <= L / 2:
        Bx = k * y
        By = k * x
        Bz = 0.0
    else:
        Bx, By, Bz = 0.0, 0.0, 0.0
    return np.array([Bx, By, Bz])


# ====== INITIAL CONDITIONS ======
v0_phys = np.array([0.0, 0.0, 0.99 * c_phys])  # Initial velocity (m/s)
r0_phys = np.array([1e-4, 1e-4, -20e-3])  # Start 20 mm upstream (z = -20 mm)

# Convert to rescaled units
v0 = v0_phys / c_phys
r0 = r0_phys / L0

# ====== SIMULATION PARAMETERS ======
dt_phys = 1e-13  # Time step (s)
dt = dt_phys / T0  # Rescaled time step
steps = 100000  # Number of steps

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

    B = magnetic_field(r[0], r[1], r[2])
    gamma = 1 / np.sqrt(1 - np.linalg.norm(v) ** 2)
    p = gamma * m_e * v
    F = e * np.cross(v, B)
    dp = F * dt
    p_new = p + dp
    gamma_new = np.sqrt(1 + np.linalg.norm(p_new) ** 2)
    v_new = p_new / (gamma_new * m_e)
    r_new = r + v_new * dt

    positions[i] = r_new
    velocities[i] = v_new

# ====== CONVERT BACK TO PHYSICAL UNITS ======
positions_phys = positions * L0  # Convert to meters

# ====== PLOT TRAJECTORY AND LENS ======
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot trajectory
ax.plot(positions_phys[:, 0], positions_phys[:, 1], positions_phys[:, 2],
        linewidth=1, label='Electron trajectory')

# ====== FIXED CYLINDER PLOTTING ======
# Generate cylinder surface points
theta = np.linspace(0, 2 * np.pi, 50)
z_cyl = np.linspace(-L_lens / 2, L_lens / 2, 10)
theta_grid, z_grid = np.meshgrid(theta, z_cyl)
x_cyl = R_lens * np.cos(theta_grid)
y_cyl = R_lens * np.sin(theta_grid)

# Plot cylinder (lens volume)
ax.plot_surface(x_cyl, y_cyl, z_grid, alpha=0.3, color='red', label='Quadrupole lens')

# Add lens caps (disks at z = ±L/2)
theta = np.linspace(0, 2 * np.pi, 100)
r_disk = np.linspace(0, R_lens, 10)
theta_grid, r_grid = np.meshgrid(theta, r_disk)
x_disk = r_grid * np.cos(theta_grid)
y_disk = r_grid * np.sin(theta_grid)

# Top cap (z = L/2)
ax.plot_surface(x_disk, y_disk, np.full_like(x_disk, L_lens / 2), alpha=0.2, color='red')

# Bottom cap (z = -L/2)
ax.plot_surface(x_disk, y_disk, np.full_like(x_disk, -L_lens / 2), alpha=0.2, color='red')

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Electron Trajectory in Localized Quadrupole Field')
ax.legend()
plt.show()