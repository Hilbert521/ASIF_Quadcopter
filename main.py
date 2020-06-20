import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import time
import controller
import gui
import quadcopter

'''
Simulation Parameters
'''
TIME_SCALING = 1.0  # Any positive number(Smaller is faster). 1.0->Real Time, 0.0->Run as fast as possible
QUAD_DYNAMICS_UPDATE = 0.005  # seconds
CONTROLLER_DYNAMICS_UPDATE = 0.002  # seconds
dt = 0.01  # seconds

'''
Quadcopter Parameters
'''
# Set goals to go to
GOALS = [(2, 2, 2)]
YAWS = [0]
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
quad.set_controller(ctrl)
end_state = quad.simulate_dynamics(1)
x, y, z = end_state[-1][0:3]
t = time.time()
i = 0
for _ in range(100):
    quad.simulate_dynamics(1)
print((time.time() - t)/100)

# Start the threads
quad.start_thread(dt=QUAD_DYNAMICS_UPDATE, time_scaling=TIME_SCALING)
ctrl.start_thread(update_rate=CONTROLLER_DYNAMICS_UPDATE, time_scaling=TIME_SCALING)

'''
Plot Setup
'''
fig = plt.figure()
# Make top row into one axes

ax_quadcopter = fig.add_subplot(projection='3d')

# Creat Obstacle visuals
xs, ys, zs = np.indices((4, 1, 5))
voxel = (xs < 4) & (ys < 0.5) & (zs < 5)
ax_quadcopter.voxels(voxel, facecolors='red', edgecolor='k')

ui = gui.GUI(quads=QUADCOPTER, ax=ax_quadcopter)
ax_quadcopter.view_init(elev=20, azim=10)
point_start, = ax_quadcopter.plot([1], [0], [4], 'go')
point_end, = ax_quadcopter.plot([x], [y], [z], 'ro')

'''
Functions
'''

def animate(i):
    for goal, y in zip(GOALS, YAWS):
        ctrl.update_target(goal)
        ctrl.update_yaw_target(y)
        ui.quads['q1']['position'] = quad.get_position('q1')
        ui.quads['q1']['orientation'] = quad.get_orientation('q1')
    quad_1, quad_2, quad_3 = ui.update()
    return quad_1, quad_2, quad_3, point_start, point_end


ani = animation.FuncAnimation(fig, animate, interval=0.01, blit=True)
plt.show()
