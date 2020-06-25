import controller
import gui
import quadcopter

'''
Simulation Parameters
'''
TIME_SCALING = 1.0  # Any positive number (1.0=Real Time, 0.0=Run as fast as possible)
QUAD_DYNAMICS_UPDATE = 0.005  # Update rate for quad dynamics, in seconds
CONTROLLER_DYNAMICS_UPDATE = 0.002  # Update rate for controller dynamics, in seconds
ANIMATION_UPDATE = 0.01  # Update rate for animation, in seconds
time_horizon = 0.5  # Time horizon (seconds) over which the simulator simulates dynamics every update
plot_quad_trail = True  # If true, the simulator will plot the actual trajectory of the quadcopter in blue
plot_sim_trail = True  # If true, the simulator will plot the predicted trajectory some time ahead in red
display_obstacles = False  # If true, the simulator will display obstacles
save = False  # If true, saves animation as an mp4, else it is displayed

'''
Quadcopter Parameters
'''
# Set target sequences of waypoints and yaws to go to
WAYPOINTS = [(2, 2, 4)]
YAWS = [0]

# Define quadcopter properties
QUADCOPTER = {
    'q1': {'position': [3, 3, 1],
           'orientation': [0, 0, 0],
           'L': 0.3,
           'r': 0.1,
           'prop_size': [10, 4.5],
           'weight': 1.2}}

# PID Controller parameters
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
quad = quadcopter.Quadcopter(QUADCOPTER, time_horizon=time_horizon)
ctrl = controller.Controller_PID_Point2Point(quad.get_state, quad.get_time, quad.set_motor_speeds,
                                             params=CONTROLLER_PARAMETERS, quad_identifier='q1')
quad.set_controller(ctrl)

# Start the threads
quad.start_thread(dt=QUAD_DYNAMICS_UPDATE, time_scaling=TIME_SCALING)
ctrl.start_thread(update_rate=CONTROLLER_DYNAMICS_UPDATE, time_scaling=TIME_SCALING)
for goal, y in zip(WAYPOINTS, YAWS):
    ctrl.update_target(goal)
    ctrl.update_yaw_target(y)
ui = gui.GUI(QUADCOPTER, quad, display_obstacles, plot_sim_trail, plot_quad_trail, save)
