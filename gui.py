import numpy as np
import utils
import os
import matplotlib
matplotlib.use('TKAgg')

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import profilehooks

'''
Originally from https://github.com/abhijitmajumdar/Quadcopter_simulator
'''
plt.rcParams["animation.convert_path"] = "C:/Program Files/ImageMagick-7.0.10-Q16-HDRI/magick.exe"

class GUI:
    def __init__(self, quads, queue):
        self.quads = quads
        self.queue = queue
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1, projection='3d')
        self.fig.tight_layout(rect=[0, 0, 1, 0.97])
        #self.ax_graph = self.fig.add_subplot(2,1,2)
        self.ax.set_xlim3d([0.0, 4.0])
        self.ax.set_xlabel('X')
        self.ax.set_ylim3d([0.0, 4.0])
        self.ax.set_ylabel('Y')
        self.ax.set_zlim3d([0, 4.0])
        self.ax.set_zlabel('Z')
        self.t = 1
        self.ax.set_title('Quadcopter Simulation')
        # Store points for trails, if enabled
        self.x, self.y, self.z = [], [], []
        # Create definitions for actual trail, and simulated trails
        self.trail, = self.ax.plot([], [], [], '--', c='blue', markersize=2)
        self.sim_trail, = self.ax.plot([], [], [], '.', c='red', markersize=2, markevery=4)

        self.artists = []
        for quad in self.quads:
            l1, = self.ax.plot([], [], [], color='blue', linewidth=2)
            l2, = self.ax.plot([], [], [], color='red', linewidth=2)
            hub, = self.ax.plot([], [], [], marker='o', color='green', markersize=6)
            self.artists.append(l1)
            self.artists.append(l2)
            self.artists.append(hub)
        self.artists = tuple(self.artists)
        #self.ax.view_init(elev=30, azim=10)
        self.ax.view_init(elev=0, azim=0)
        self.ani = animation.FuncAnimation(self.fig, self.update, init_func=None,
                                           interval=1, blit=True)
        save=False
        if save:
            # Interval : Amount of time, in ms between generated frames
            # Frames: Number of frames to produce
            # FPS: 1/(interval*0.001) -> multiple by number of seconds needed for number of frames
            writer = animation.ImageMagickFileWriter(fps=1 / (5 * 0.001))
            self.ani.save('test.gif', writer=writer)
        else:
            plt.show()
            os._exit(0)

    #@profilehooks.profile(immediate=True)
    def update(self, i):
        for num, quad in enumerate(self.quads):
            pos = quad.get_position()
            attitude = quad.get_attitude()
            R = utils.rotation_matrix(attitude)
            L = 0.2
            points = R @ np.array([[-L, 0, 0], [L, 0, 0], [0, -L, 0], [0, L, 0], [0, 0, 0], [0, 0, 0]]).T
            points[0, :] += pos[0]
            points[1, :] += pos[1]
            points[2, :] += pos[2]
            self.artists[num*3].set_data_3d(points[0, 0:2], points[1, 0:2], points[2, 0:2])
            self.artists[1+num*3].set_data_3d(points[0, 2:4], points[1, 2:4], points[2, 2:4])
            self.artists[2+num*3].set_data_3d(points[0, 5], points[1, 5], points[2, 5])
        return self.artists
