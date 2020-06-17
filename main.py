import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import quadcopter, gui, controller

'''
Simulation Parameters
'''
TIME_SCALING = 1.0  # Any positive number(Smaller is faster). 1.0->Real Time, 0.0->Run as fast as possible
QUAD_DYNAMICS_UPDATE = 0.002  # seconds
CONTROLLER_DYNAMICS_UPDATE = 0.005  # seconds
dt = 0.01  # seconds
t = 0  # current time (s)

'''
Quadcopter Parameters
'''
# Set goals to go to
GOALS = [(2, 2, 2), (2, 3, 2), (2, 3.5, 2), (1, 1.2, 0)]
YAWS = [0, 3.14, -1.54, 1.54]
# Define the quadcopters
QUADCOPTER = {
    'q1': {'position': [1, 0, 4], 'orientation': [0, 0, 0], 'L': 0.3, 'r': 0.1, 'prop_size': [10, 4.5], 'weight': 1.2}}
# Controller parameters
CONTROLLER_PARAMETERS = {'Motor_limits': [4000, 9000],
                         'Tilt_limits': [-10, 10],
                         'Yaw_Control_Limits': [-900, 900],
                         'Z_XY_offset': 500,
                         'Linear_PID': {'P': [300, 300, 7000], 'I': [0.04, 0.04, 4.5], 'D': [450, 450, 5000]},
                         'Linear_To_Angular_Scaler': [1, 1, 0],
                         'Yaw_Rate_Scaler': 0.18,
                         'Angular_PID': {'P': [22000, 22000, 1500], 'I': [0, 0, 1.2], 'D': [12000, 12000, 0]},
                         }

# Make objects for quadcopter, gui and controller
quad = quadcopter.Quadcopter(QUADCOPTER)
ctrl = controller.Controller_PID_Point2Point(quad.get_state, quad.get_time, quad.set_motor_speeds,
                                             params=CONTROLLER_PARAMETERS, quad_identifier='q1')
# Start the threads
quad.start_thread(dt=QUAD_DYNAMICS_UPDATE, time_scaling=TIME_SCALING)
ctrl.start_thread(update_rate=CONTROLLER_DYNAMICS_UPDATE, time_scaling=TIME_SCALING)

'''
Plot Setup
'''
fig, axs = plt.subplots(ncols=4, nrows=2)

# Make top row into one axes
gs = axs[0, 0].get_gridspec()
for ax in axs[0, 0:2]:
    ax.remove()

ax_quadcopter = fig.add_subplot(gs[0, 0:2], projection='3d')


# Creat Obstacle visuals
xs, ys, zs = np.indices((4, 1, 5))
voxel = (xs < 4) & (ys < 0.5) & (zs < 5)
ax_quadcopter.voxels(voxel, facecolors='red', edgecolor='k')

ui = gui.GUI(quads=QUADCOPTER, ax=ax_quadcopter)
quad_1, quad_2, quad_3 = ui.init_plot()
ax_quadcopter.set_xlim3d([0, 4.0])
ax_quadcopter.set_xlabel('X')
ax_quadcopter.set_ylim3d([0.0, 4.0])
ax_quadcopter.set_ylabel('Y')
ax_quadcopter.set_zlim3d([0, 6.0])
ax_quadcopter.set_zlabel('Z')
ax_quadcopter.view_init(elev=20, azim=10)


# Safe Set: Not near wall
# Definition: If the distance between quadcopter center is greater than radius

'''
Functions
'''

# Takes in x0, final time, spits out final state

def is_safe(x, y):
    return np.sqrt((x - 1) ** 2 + (y - 1) ** 2) > 0.3


def barrier_condition(y, y_dot, T_m1, T_m2, T_m3, T_m4):
    h = abs((y - 1)) - 0.3
    dh_dy = np.sign(y)
    dy_dt = y_dot
    BC = dh_dy * dy_dt + 80 * h ** 7
    return BC, h


def animate(i):
    global quad_1
    global quad_2
    global quad_3
    global t

    for goal, y in zip(GOALS, YAWS):
        ctrl.update_target(goal)
        ctrl.update_yaw_target(y)
        ui.quads['q1']['position'] = quad.get_position('q1')
        ui.quads['q1']['orientation'] = quad.get_orientation('q1')
        quad_1, quad_2, quad_3 = ui.update()
    t += dt
    return quad_1, quad_2, quad_3


ani = animation.FuncAnimation(fig, animate, interval=QUAD_DYNAMICS_UPDATE, blit=True)
plt.show()
