import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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
R_lens = 5e-3  # Lens radius (5 mm)
L_lens = 10e-3  # Lens length (10 mm)
k_phys = 10.0  # Gradient (T/m)

# Rescale lens parameters
R = R_lens / L0
L = L_lens / L0
k = k_phys / B0  # Rescaled gradient


def magnetic_field(x, y, z):
    """Quadrupole field (aligned along X-axis): B = (0, k*y, k*z)."""
    x_abs = np.abs(x)
    r = np.sqrt(y ** 2 + z ** 2)
    if x_abs <= L / 2 and r <= R:
        # Explicitly enforce B=0 if y=z=0 to avoid numerical noise
        if np.allclose([y, z], [0.0, 0.0], atol=1e-15):
            Bx, By, Bz = 0.0, 0.0, 0.0
        else:
            Bx = 0.0
            By = k * y
            Bz = k * z
    else:
        Bx, By, Bz = 0.0, 0.0, 0.0
    return np.array([Bx, By, Bz])


# ====== INITIAL CONDITIONS ======
v0_phys = np.array([0.99 * c_phys, 0.0, 0.0])  # Purely along X-axis
r0_phys = np.array([-20e-3, 0.0, 0.0])  # Strictly on-axis (y=z=0)

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

# ====== PLOT SETUP ======
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot lens (cylinder along X-axis)
theta = np.linspace(0, 2 * np.pi, 30)
x = np.linspace(-L_lens / 2, L_lens / 2, 10)
theta_grid, x_grid = np.meshgrid(theta, x)
y_cyl = R_lens * np.cos(theta_grid)
z_cyl = R_lens * np.sin(theta_grid)
ax.plot_surface(x_grid, y_cyl, z_cyl, alpha=0.2, color='red')

# Plot trajectory (should be straight along X)
ax.plot(positions_phys[:, 0], positions_phys[:, 1], positions_phys[:, 2], 'b-', lw=1)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Electron Trajectory (On-Axis, No Deflection Expected)')
plt.show()