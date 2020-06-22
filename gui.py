import numpy as np
import utils
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D
import matplotlib.animation as animation

'''
Originally from https://github.com/abhijitmajumdar/Quadcopter_simulator
'''
class GUI():
    # 'quad_list' is a dictionary of format: quad_list = {'quad_1_name':{'position':quad_1_position,'orientation':quad_1_orientation,'arm_span':quad_1_arm_span}, ...}
    def __init__(self, quads, quad, display_obstacles=False, plot_sim_trail=False, plot_quad_trail=False):
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
        self.x_sim, self.y_sim, self.z_sim = [], [], []
        # Create definitions for actual trail, and simulated trails
        self.trail, = self.ax.plot([], [], [], '.', c='blue', markevery=2, markersize=2)
        self.sim_trail, = self.ax.plot([], [], [], '-', c='red', markersize=2)
        self.artists = [self.trail, self.sim_trail]
        self.ani = animation.FuncAnimation(self.fig, self.update, init_func=self.init_plot,
                                           interval=0.01, blit=True)

    def init_plot(self):
        self.ax.view_init(elev=40, azim=40)

        # Initialize obstacles
        if self.display_obstacles:
            xs, ys, zs = np.indices((4, 1, 5))
            voxel = (xs < 4) & (ys < 0.5) & (zs < 5)
            self.ax.voxels(voxel, facecolors='red', edgecolor='k')

        for key in self.quads:
            self.quads[key]['l1'], = self.ax.plot([], [], [], color='blue', linewidth=2)
            self.quads[key]['l2'], = self.ax.plot([], [], [], color='red', linewidth=2)
            self.quads[key]['hub'], = self.ax.plot([], [], [], marker='o', color='green', markersize=6)
            self.artists.append(self.quads[key]['l1'])
            self.artists.append(self.quads[key]['l2'])
            self.artists.append(self.quads[key]['hub'])

        for a in self.artists:
            yield a

    def update(self, i):
        for key in self.quads:
            self.quads[key]['position'] = self.quad.get_position(key)
            self.quads[key]['orientation'] = self.quad.get_orientation(key)
            R = utils.rotation_matrix(self.quads[key]['orientation'])
            L = self.quads[key]['L']
            points = R @ np.array([[-L, 0, 0], [L, 0, 0], [0, -L, 0], [0, L, 0], [0, 0, 0], [0, 0, 0]]).T
            points[0, :] += self.quads[key]['position'][0]
            points[1, :] += self.quads[key]['position'][1]
            points[2, :] += self.quads[key]['position'][2]
            self.quads[key]['l1'].set_data(points[0, 0:2], points[1, 0:2])
            self.quads[key]['l1'].set_3d_properties(points[2, 0:2])
            self.quads[key]['l2'].set_data(points[0, 2:4], points[1, 2:4])
            self.quads[key]['l2'].set_3d_properties(points[2, 2:4])
            self.quads[key]['hub'].set_data(points[0, 5], points[1, 5])
            self.quads[key]['hub'].set_3d_properties(points[2, 5])

            if self.plot_quad_trail:
                self.x.append(self.quad.get_position('q1')[0])
                self.y.append(self.quad.get_position('q1')[1])
                self.z.append(self.quad.get_position('q1')[2])
                self.trail.set_data(self.x, self.y)
                self.trail.set_3d_properties(self.z)

            if self.plot_sim_trail:
                state_sim = self.quad.last_sim_state[0:3]
                self.x_sim.append(state_sim[0])
                self.y_sim.append(state_sim[1])
                self.z_sim.append(state_sim[2])
                self.sim_trail.set_data(self.x_sim, self.y_sim)
                self.sim_trail.set_3d_properties(self.z_sim)

        for a in self.artists:
            yield a
