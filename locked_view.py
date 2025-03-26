import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams
rcParams['toolbar'] = 'None'          # Hide toolbar
rcParams['animation.embed_limit'] = 10  # Lower FPS during window drags
NO_Z_AXIS = True

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
    """Quadrupole field (X-axis aligned): B = (0, k*y, k*z)."""
    x_abs = np.abs(x)
    r = np.sqrt(y ** 2 + z ** 2)
    if x_abs <= L / 2 and r <= R:
        if np.allclose([y, z], [0.0, 0.0], atol=1e-15):
            Bx, By, Bz = 0.0, 0.0, 0.0  # Enforce B=0 on-axis
        else:
            Bx = 0.0
            By = k * y
            Bz = k * z
    else:
        Bx, By, Bz = 0.0, 0.0, 0.0
    return np.array([Bx, By, Bz])


# ====== INITIAL CONDITIONS ======
v0_phys = np.array([0.99 * c_phys, 0.0, 0.0])  # Along X-axis
r0_phys = np.array([-20e-3, 0.0, 0.0])  # Slightly off-axis

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

# ====== CONVERT TO PHYSICAL UNITS ======
positions_phys = positions * L0  # Convert to meters

# ====== VISUALIZATION SETUP ======
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Enable mouse wheel zoom by modifying the axes' event handling
def on_scroll(event):
    if event.inaxes == ax:
        scale = 1.1 if event.button == 'up' else 1/1.1
        ax.set_xlim3d([x/scale for x in ax.get_xlim3d()])
        ax.set_ylim3d([y/scale for y in ax.get_ylim3d()])
        ax.set_zlim3d([z/scale for z in ax.get_zlim3d()])
        fig.canvas.draw_idle()

fig.canvas.mpl_connect('scroll_event', on_scroll)

# Plot lens (cylinder along X-axis)
theta = np.linspace(0, 2 * np.pi, 30)
x = np.linspace(-L_lens / 2, L_lens / 2, 10)
theta_grid, x_grid = np.meshgrid(theta, x)
y_cyl = R_lens * np.cos(theta_grid)
z_cyl = R_lens * np.sin(theta_grid)
ax.plot_surface(x_grid, y_cyl, z_cyl, alpha=0.2, color='red')

# Plot trajectory
line, = ax.plot([], [], [], 'b-', lw=1)

# ====== VIEW CONTROL ======
# 1. Set strict top-down view
ax.view_init(elev=90, azim=-90)  # Perfect XY plane view

if NO_Z_AXIS:
    # 2. Completely hide Z-axis grid
    ax.zaxis._axinfo["grid"].update({"visible": False})  # Hide grid lines
    ax.set_zticks([])  # Remove tick marks
    ax.zaxis.line.set_color((1,1,1,0))  # Fully transparent Z-axis line

    # Hide secondary grid lines too
    ax.xaxis.pane.set_edgecolor('w')  # White edges (invisible)
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')


    # 3. Make XY plane more visible
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # Hide YZ Plane
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # Hide XZ Plane
    ax.zaxis.set_pane_color((0.9, 0.9, 0.9, 1.0))  # Make XY Plane visible
    ax.xaxis._axinfo["grid"].update({"linewidth": 0.5})
    ax.yaxis._axinfo["grid"].update({"linewidth": 0.5})

    # 4. Set orthographic projection
    ax.set_proj_type('ortho')

# Animation (same as before)
def update(frame):
    idx = (frame + 1) * skip_frames
    if idx >= steps:
        idx = steps - 1
    line.set_data(positions_phys[:idx, 0], positions_phys[:idx, 1])
    line.set_3d_properties(positions_phys[:idx, 2])
    return line,

ani = FuncAnimation(fig, update, frames=range(steps//skip_frames),
                    interval=20, blit=False)

plt.tight_layout()
plt.show()