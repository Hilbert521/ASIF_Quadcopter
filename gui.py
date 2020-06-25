import numpy as np
import utils
import matplotlib

matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D
import matplotlib.animation as animation

'''
Originally from https://github.com/abhijitmajumdar/Quadcopter_simulator
'''
plt.rcParams["animation.convert_path"] = "C:/Program Files/ImageMagick-7.0.10-Q16-HDRI/magick.exe"


class GUI():
    # 'quad_list' is a dictionary of format: quad_list = {'quad_1_name':{'position':quad_1_position,'orientation':quad_1_orientation,'arm_span':quad_1_arm_span}, ...}
    def __init__(self, quads, quad, display_obstacles=False, plot_sim_trail=False, plot_quad_trail=False, save=False):
        self.quads = quads
        self.quad = quad
        self.display_obstacles = display_obstacles
        self.plot_sim_trail = plot_sim_trail
        self.plot_quad_trail = plot_quad_trail
        self.fig = plt.figure()
        self.ax = Axes3D.Axes3D(self.fig)
        self.ax.set_xlim3d([0.0, 4.0])
        self.ax.set_xlabel('X')
        self.ax.set_ylim3d([0.0, 4.0])
        self.ax.set_ylabel('Y')
        self.ax.set_zlim3d([0, 5.0])
        self.ax.set_zlabel('Z')
        self.ax.set_title('Quadcopter Simulation')
        # Store points for trails, if enabled
        self.x, self.y, self.z = [], [], []
        # Create definitions for actual trail, and simulated trails
        self.trail, = self.ax.plot([], [], [], '--', c='blue', markersize=2)
        self.sim_trail, = self.ax.plot([], [], [], '.', c='red', markersize=2)
        if self.display_obstacles:
            xs, ys, zs = np.indices((4, 1, 5))
            voxel = (xs < 4) & (ys < 0.5) & (zs < 5)
            self.ax.voxels(voxel, facecolors='red', edgecolor='k')

        for key in self.quads:
            self.quads[key]['l1'], = self.ax.plot([], [], [], color='blue', linewidth=2)
            self.quads[key]['l2'], = self.ax.plot([], [], [], color='red', linewidth=2)
            self.quads[key]['hub'], = self.ax.plot([], [], [], marker='o', color='green', markersize=6)

        self.ax.view_init(elev=40, azim=40)
        self.ani = animation.FuncAnimation(self.fig, self.update, frames=400, init_func=None,
                                           interval=10)
        if save:
            # Interval : Amount of time, in ms between generated frames
            # Frames: Number of frames to produce
            # FPS: 1/(interval*0.001) -> multiple by number of seconds needed for number of frames
            writer = animation.ImageMagickFileWriter(fps=1 / (10 * 0.001))
            self.ani.save('test.gif', writer=writer)
        else:
            plt.show()

    def update(self, i):
        for key in self.quads:
            state = self.quad.get_state(key)
            self.quads[key]['position'] = state[0:3]
            self.quads[key]['orientation'] = state[6:9]
            R = utils.rotation_matrix(self.quads[key]['orientation'])
            L = self.quads[key]['L']
            points = R @ np.array([[-L, 0, 0], [L, 0, 0], [0, -L, 0], [0, L, 0], [0, 0, 0], [0, 0, 0]]).T
            points[0, :] += self.quads[key]['position'][0]
            points[1, :] += self.quads[key]['position'][1]
            points[2, :] += self.quads[key]['position'][2]
            self.quads[key]['l1'].set_data_3d(points[0, 0:2], points[1, 0:2], points[2, 0:2])
            self.quads[key]['l2'].set_data_3d(points[0, 2:4], points[1, 2:4], points[2, 2:4])
            self.quads[key]['hub'].set_data_3d(points[0, 5], points[1, 5], points[2, 5])

            if self.plot_quad_trail:
                quad_x, quad_y, quad_z = self.quads[key]['position']
                self.x.append(quad_x)
                self.y.append(quad_y)
                self.z.append(quad_z)
                self.trail.set_data_3d(self.x, self.y, self.z)

            if self.plot_sim_trail:
                sim = np.array(self.quad.last_sim_state)
                sim_x = sim[:, 0]
                sim_y = sim[:, 1]
                sim_z = sim[:, 2]
                self.sim_trail.set_data_3d(sim_x, sim_y, sim_z)