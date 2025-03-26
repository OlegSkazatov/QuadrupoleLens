import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import proj3d, art3d


class DynamicXYGrid:
    def __init__(self, ax, fig):
        self.ax = ax
        self.fig = fig
        self.grid_lines = []
        self._setup_transform_aware_grid()

    def _setup_transform_aware_grid(self):
        """Create grid using proper coordinate transforms"""
        # Clear existing grid
        for artist in self.grid_lines:
            artist.remove()
        self.grid_lines = []

        # Get current axis limits
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        # Create grid lines in data coordinates
        for y in np.linspace(ylim[0], ylim[1], 10):
            line = art3d.Line3D([xlim[0], xlim[1]], [y, y], [0, 0],
                                color='green', lw=0.5, alpha=1.0)
            self.ax.add_line(line)
            self.grid_lines.append(line)

        for x in np.linspace(xlim[0], xlim[1], 10):
            line = art3d.Line3D([x, x], [ylim[0], ylim[1]], [0, 0],
                                color='green', lw=0.5, alpha=1.0)
            self.ax.add_line(line)
            self.grid_lines.append(line)

        # Connect transform updates
        self.fig.canvas.mpl_connect('draw_event', self._update_grid_position)

    def _update_grid_position(self, event):
        """Adjust grid depth based on current transform"""
        try:
            # Get current transform matrix
            M = self.ax.M
            if M is None:
                return

            # Find where z=0 projects in screen coordinates
            x0, y0, z0 = proj3d.proj_transform(0, 0, 0, M)
            x1, y1, z1 = proj3d.proj_transform(0, 0, 1e-6, M)  # Tiny offset

            # Calculate required depth adjustment
            depth_adjustment = 0.01 * (z1 - z0)  # Small fraction of z-range

            # Apply to all grid lines
            for line in self.grid_lines:
                xdata, ydata, zdata = line.get_data_3d()
                line.set_3d_properties(np.array(zdata) + depth_adjustment)

            self.fig.canvas.draw_idle()
        except Exception as e:
            print(f"Transform update skipped: {str(e)}")