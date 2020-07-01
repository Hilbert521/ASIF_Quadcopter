import numpy as np
import scipy.integrate
import datetime
import time
import threading
import utils

'''
Originally from https://github.com/abhijitmajumdar/Quadcopter_simulator
'''


class Propeller:
    def __init__(self, prop_dia, prop_pitch, thrust_unit='N'):
        self.dia = prop_dia
        self.pitch = prop_pitch
        self.thrust_unit = thrust_unit
        self.speed = 0  # RPM
        self.thrust = 0

    def set_speed(self, speed):
        self.speed = speed
        # From http://www.electricrcaircraftguy.com/2013/09/propeller-static-dynamic-thrust-equation.html
        self.thrust = utils.thrust(self.speed, self.pitch, self.dia)
        if self.thrust_unit == 'Kg':
            self.thrust *= 0.101972


class Quadcopter:
    # State space representation: [x y z x_dot y_dot z_dot theta phi gamma theta_dot phi_dot gamma_dot]
    # From Quadcopter Dynamics, Simulation, and Control by Andrew Gibiansky
    def __init__(self, quads, gravity=9.81, b=0.0245, time_horizon=1):
        self.quads = quads
        self.g = gravity
        self.b = b
        self.thread_object = None
        self.time_horizon = time_horizon
        self.ode = scipy.integrate.ode(self.state_dot).set_integrator('dopri5', atol=1e-3, rtol=1e-6)
        self.controller = None
        self.time = datetime.datetime.now()
        for key in self.quads:
            self.quads[key]['state'] = np.zeros(12)
            self.quads[key]['state'][0:3] = self.quads[key]['position']
            self.quads[key]['state'][6:9] = self.quads[key]['orientation']
            self.quads[key]['m1'] = Propeller(self.quads[key]['prop_size'][0], self.quads[key]['prop_size'][1])
            self.quads[key]['m2'] = Propeller(self.quads[key]['prop_size'][0], self.quads[key]['prop_size'][1])
            self.quads[key]['m3'] = Propeller(self.quads[key]['prop_size'][0], self.quads[key]['prop_size'][1])
            self.quads[key]['m4'] = Propeller(self.quads[key]['prop_size'][0], self.quads[key]['prop_size'][1])

            # From Quadrotor Dynamics and Control by Randal Beard
            ixx = ((2 * self.quads[key]['weight'] * self.quads[key]['r'] ** 2) / 5) + (
                    2 * self.quads[key]['weight'] * self.quads[key]['L'] ** 2)
            iyy = ixx
            izz = ((2 * self.quads[key]['weight'] * self.quads[key]['r'] ** 2) / 5) + (
                    4 * self.quads[key]['weight'] * self.quads[key]['L'] ** 2)
            self.quads[key]['I'] = np.array([[ixx, 0, 0], [0, iyy, 0], [0, 0, izz]])
            self.quads[key]['invI'] = np.linalg.inv(self.quads[key]['I'])
        self.last_sim_state = np.array([self.quads['q1']['state']])
        self.run = True

    def set_controller(self, controller):
        self.controller = controller

    def state_dot(self, t, state, key, sim=False):
        """Calculates d/dt of the state
        :param t: Current time during integration
        :param state: Current state of the quadcopter
        :param key: String key for accessing the quadcopter information dictionary
        :param sim: Where the function is being to used to simulate over a time horizon (True) or to simulate in real
        time (False)
        :returns d/dt of the state
        """

        # Define motor thrusts, based on whether function is used to simulate future or present trajectory
        if sim:
            control_input = self.control(state)
            m1, m2, m3, m4 = utils.thrust(control_input, self.quads[key]['prop_size'][1],
                                          self.quads[key]['prop_size'][0])
        else:
            m1 = self.quads[key]['m1'].thrust
            m2 = self.quads[key]['m2'].thrust
            m3 = self.quads[key]['m3'].thrust
            m4 = self.quads[key]['m4'].thrust

        state_dot = np.zeros(12)

        # The velocities(t+1 x_dots equal the t x_dots)
        state_dot[0:3] = state[3:6]

        # The acceleration (x_dotdot)
        x_dotdot = np.array([0, 0, -self.quads[key]['weight'] * self.g]) + (utils.rotation_matrix(state[6:9])
                                                                            @ np.array([0, 0, (m1 + m2 + m3 + m4)])
                                                                            / self.quads[key]['weight'])
        state_dot[3:6] = x_dotdot

        # The angular rates(t+1 theta_dots equal the t theta_dots)
        state_dot[6:9] = state[9:12]

        # The angular accelerations (omega_dot)
        omega = state[9:12]
        tau = np.array([self.quads[key]['L'] * (m1 - m3),
                        self.quads[key]['L'] * (m2 - m4),
                        self.b * (m1 - m2 + m3 - m4)])
        omega_dot = self.quads[key]['invI'] @ (tau - np.cross(omega, (self.quads[key]['I'] @ omega)))
        state_dot[9:12] = omega_dot
        return state_dot

    def update(self, dt):
        self.simulate_dynamics(self.time_horizon)
        for key in self.quads:
            self.ode.set_initial_value(self.quads[key]['state'], 0).set_f_params(str(key), False, )
            self.quads[key]['state'] = self.ode.integrate(self.ode.t + dt)
            self.quads[key]['state'][6:9] = utils.wrap_angle(self.quads[key]['state'][6:9])

    def control(self, state):
        """Return control inputs for quadcopter, given its current state"""
        return self.controller.update(state, sim=True)

    def simulate_dynamics(self, total_time):
        """Simulate quadcopter dynamics over finite time horizon and return final states"""
        states = []
        self.controller.reset_sim()
        for key in self.quads:
            sol = scipy.integrate.solve_ivp(self.state_dot, [0, total_time], self.quads[key]['state'],
                                            args=(str(key), True), t_eval=[total_time], method='RK45', atol=1e-3,
                                            rtol=1e-6)
            final_state = sol.y[:, -1]
            final_state[6:9] = utils.wrap_angle(final_state[6:9])
            states.append(final_state)
        self.last_sim_state = np.vstack((self.last_sim_state, states[0]))
        return states

    def set_motor_speeds(self, quad_name, speeds):
        self.quads[quad_name]['m1'].set_speed(speeds[0])
        self.quads[quad_name]['m2'].set_speed(speeds[1])
        self.quads[quad_name]['m3'].set_speed(speeds[2])
        self.quads[quad_name]['m4'].set_speed(speeds[3])

    def get_position(self, quad_name):
        return self.quads[quad_name]['state'][0:3]

    def get_linear_rate(self, quad_name):
        return self.quads[quad_name]['state'][3:6]

    def get_orientation(self, quad_name):
        return self.quads[quad_name]['state'][6:9]

    def get_angular_rate(self, quad_name):
        return self.quads[quad_name]['state'][9:12]

    def get_state(self, quad_name):
        return self.quads[quad_name]['state']

    def set_position(self, quad_name, position):
        self.quads[quad_name]['state'][0:3] = position

    def set_orientation(self, quad_name, orientation):
        self.quads[quad_name]['state'][6:9] = orientation

    def get_time(self):
        return self.time

    def thread_run(self, dt, time_scaling):
        rate = time_scaling * dt
        last_update = self.time
        while self.run:
            time.sleep(0)
            self.time = datetime.datetime.now()
            if (self.time - last_update).total_seconds() > rate:
                self.update(dt)
                last_update = self.time

    def start_thread(self, dt=0.01, time_scaling=1):
        self.thread_object = threading.Thread(target=self.thread_run, args=(dt, time_scaling))
        self.thread_object.start()

    def stop_thread(self):
        self.run = False
