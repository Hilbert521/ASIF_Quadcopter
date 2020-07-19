import controller
import gui
import quadcopter

'''
Simulation Parameters
'''
TIME_SCALING = 1.0  # Any positive number (1.0=Real Time, 0.0=Run as fast as possible)
QUAD_DYNAMICS_UPDATE = 0.005  # Update rate for quad dynamics, in seconds
time_horizon = 1  # Time horizon (seconds) over which the simulator simulates dynamics every update
plot_quad_trail = True  # If true, the simulator will plot the actual trajectory of the quadcopter in blue
plot_sim_trail = False  # If true, the simulator will plot the predicted trajectory some time ahead in red
display_obstacles = False  # If true, the simulator will display obstacles
save = False  # If true, saves animation as an mp4, else it is displayed

'''
Quadcopter Parameters
'''
# Set target sequences of waypoints and yaws to go to
WAYPOINTS = [(0, 0, 4)]
YAWS = [0]

# Define quadcopter properties
QUADCOPTER = {
    'q1': {'position': [2, 2, 2],
           'orientation': [3, 2, 0],
           'L': 0.3,
           'r': 0.1,
           'prop_size': [10, 4.5],
           'weight': 1.2}}

# Make objects for quadcopter, gui and controller
quad = quadcopter.Quadcopter(QUADCOPTER, time_horizon=time_horizon)

# Start the threads
quad.start_thread(dt=QUAD_DYNAMICS_UPDATE, time_scaling=TIME_SCALING)
ui = gui.GUI(QUADCOPTER, quad, display_obstacles, plot_sim_trail, plot_quad_trail, save)
