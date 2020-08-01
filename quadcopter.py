import numpy as np
import scipy.integrate
import datetime
import time
import threading
import controller
import utils
import profilehooks

'''
Originally from https://github.com/abhijitmajumdar/Quadcopter_simulator
'''

np.set_printoptions(linewidth=np.inf)

class Quadcopter:
    def __init__(self, quads, gravity=9.81, b=0.0245, time_horizon=1, plot_simulation_trail=False, simulate=False):
        self.quads = quads
        self.g = gravity
        self.b = b
        self.thread_object = None
        self.time_horizon = time_horizon
        self.ode = scipy.integrate.ode(self.state_dot).set_integrator('dopri5', atol=1e-3, rtol=1e-6)
        self.time = datetime.datetime.now()
        self.u1_prev = 1
        self.plot_sim_trail = plot_simulation_trail
        self.simulate = simulate
        for key in self.quads:
            self.quads[key]['state'] = np.zeros(14) if self.quads[key]['method'] == "Feedback_Linearization" else np.zeros(12)
            self.quads[key]['state'][0:3] = self.quads[key]['position']
            self.quads[key]['state'][3:6] = self.quads[key]['orientation']
            self.quads[key]['state'][9] = 1 if self.quads[key]['method'] == "Feedback_Linearization" else 0

            # From Quadrotor Dynamics and Control by Randal Beard
            ixx = ((2 * self.quads[key]['weight'] * self.quads[key]['r'] ** 2) / 5) + (
                    2 * self.quads[key]['weight'] * self.quads[key]['L'] ** 2)
            iyy = ixx
            izz = ((2 * self.quads[key]['weight'] * self.quads[key]['r'] ** 2) / 5) + (
                    4 * self.quads[key]['weight'] * self.quads[key]['L'] ** 2)
            self.quads[key]['I'] = np.array([[ixx, 0, 0], [0, iyy, 0], [0, 0, izz]])
            self.quads[key]['invI'] = np.linalg.inv(self.quads[key]['I'])
        #self.last_sim_state = np.array(self.x_bar_to_z(self.quads['q1']['state'], 'q1'))
        self.last_sim_state = [self.quads['q1']['state']]
        self.k_d = 0
        self.run = True

    def state_dot(self, t, state, key, method="Zero_Dynamic_Stabilization"):
        """
        Calculates d/dt of the state
        From Francesco Sabatino's thesis: Quadrotor control: modeling, nonlinear control design, and simulation

        :param t: Current time during integration
        :param state: Current state of the quadcopter
        :param key: String key for accessing the quadcopter information dictionary
        :param sim: Whether the function is being to used to simulate over a time horizon (True) or to simulate in real
        time (False)
        :param method:
            Feedback_Linearization:
            (Quadrotor control: modeling, nonlinear control design, and simulation by Francesco Sabatino)
            [x, y, z, psi, theta, phi, x_dot, y_dot, z_dot, zeta, xi, p, q, r]

            Zero_Dynamic_Stabilization:
            (Quadrotor control: modeling, nonlinear control design, and simulation by Francesco Sabatino)
            [x, y, z, psi, theta, phi, x_dot, y_dot, z_dot, 0, 0, p, q, r]

            Regular_Dynamics:
            (Quadcopter Dynamics, Simulation, and Control by Andrew Gibiansky)
            [x y z x_dot y_dot z_dot theta phi gamma theta_dot phi_dot gamma_dot]
        :return: d/dt of state
        """

        if method not in ["Feedback_Linearization", "Zero_Dynamic_Stabilization", "Regular_Dynamics"]:
            raise SyntaxError("parameter 'method' must be one of "
                              + "\"Feedback_Linearization\", "
                              + "\"Zero_Dynamic_Stabilization\", or"
                                "\"Regular_Dynamics\"")

        m = self.quads[key]['weight']
        Ix, Iy, Iz = self.quads[key]['I'].diagonal()

        if method.lower() == "Feedback_Linearization".lower():
            if np.array(state).shape[0] != 14:
                raise ValueError("State should be 14-dimensional!")
            x, y, z, psi, theta, phi, x_dot, y_dot, z_dot, zeta, xi, p, q, r = state

            cs, ct, cp = np.cos(state[3:6])
            ss, st, sp = np.sin(state[3:6])
            ts, tt, tp = np.tan(state[3:6])

            state_dot = np.zeros(14)

            u1, u2, u3, u4 = self.control(t, state, key, "Feedback_Linearization")

            # Linear velocities
            state_dot[0:3] = state[6:9]

            # Angular velocities
            state_dot[3] = q * sp / ct + r * cp / ct
            state_dot[4] = q * cp - r * sp
            state_dot[5] = p + q * sp * tt + r * cp * tt

            # Linear Accelerations
            state_dot[6] = - (sp * ss + cp * cs * st) * zeta / m
            state_dot[7] = - (cs * sp - cp * ss * st) * zeta / m
            state_dot[8] = - (cp * ct) * zeta / m

            # zeta_dot/xi_dot
            state_dot[9] = xi
            state_dot[10] = 0 + u1

            # Angular Accelerations
            state_dot[11] = q * r * (Iy - Iz) / Ix + u2 / Ix
            state_dot[12] = p * r * (Iz - Ix) / Iy + u3 / Iy
            state_dot[13] = 0 + u4 / Iz

            return state_dot

        elif method.lower() == "Zero_Dynamic_Stabilization".lower():
            if np.array(state).shape[0] != 12:
                raise ValueError("State should be 12-dimensional!")

            a = 1

            x_desireds = np.array([1 * np.cos(a * t) + 2,
                                   -1 * a * np.sin(a * t),
                                   -1 * a ** 2 * np.cos(a * t),
                                   1 * a ** 3 * np.sin(a * t),
                                   1 * a ** 4 * np.cos(a * t)])

            y_desireds = np.array([1 * np.sin(a * t) + 2,
                                   1 * a * np.cos(a * t),
                                   -1 * a ** 2 * np.sin(a * t),
                                   -1 * a ** 3 * np.cos(a * t),
                                   1 * a ** 4 * np.sin(a * t)])

            z_desireds = np.array([2, 0, 0])

            psi_desireds = np.array([0, 0, 0])

            u1, u2, u3, u4 = self.control(t, state, key, self.quads[key]['method'],
                                          x_desireds=x_desireds, y_desireds=y_desireds,
                                          z_desireds=z_desireds, psi_desireds=psi_desireds)

            self.u1_prev = u1

            cs, ct, cp = np.cos(state[3:6])
            ss, st, sp = np.sin(state[3:6])
            ts, tt, tp = np.tan(state[3:6])

            p, q, r = state[9:12]

            state_dot = np.zeros(12)

            # Linear velocities
            state_dot[0:3] = state[6:9]

            # Angular velocities
            state_dot[3] = q * sp / ct + r * cp / ct
            state_dot[4] = q * cp - r * sp
            state_dot[5] = p + q * sp * tt + r * cp * tt

            # Linear Accelerations
            state_dot[6] = - u1 * (sp * ss + cp * cs * st) / m
            state_dot[7] = - u1 * (cs * sp - cp * ss * st) / m
            state_dot[8] = self.g - u1 * cp * ct / m

            # Angular Accelerations
            state_dot[9] = q * r * (Iy - Iz) / Ix + u2 / Ix
            state_dot[10] = p * r * (Iz - Ix) / Iy + u3 / Iy
            state_dot[11] = p * q * (Ix - Iy) / Iz + u4 / Iz
            return state_dot

        elif method.lower() == "Regular_Dynamics".lower():
            if np.array(state).shape[0] != 12:
                raise ValueError("State should be 12-dimensional!")

            # Define motor thrusts, based on whether function is used to simulate future or present trajectory
            control_input = self.control(state)
            m1, m2, m3, m4 = utils.thrust(control_input, self.quads[key]['prop_size'][1],
                                          self.quads[key]['prop_size'][0])

            state_dot = np.zeros(12)

            # The velocities(t+1 x_dots equal the t x_dots)
            state_dot[0:3] = state[3:6]

            # The acceleration (x_dotdot)
            x_dotdot = np.array([0, 0, -self.quads[key]['weight'] * self.g]) + \
                       (utils.rotation_matrix(state[6:9]) @ np.array([0, 0, (m1 + m2 + m3 + m4)]) - self.k_d * state[
                                                                                                               3:6]) \
                       / self.quads[key]['weight']
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

    def control(self, t, state, key, method="Zero_Dynamic_Stabilization", **kwargs):
        """
        Returns control input for the quadcopter, given the parameters

        :param t: The current time, t (s)
        :param state: A 12- or 14-D state, depending on the method
        :param key: String key for accessing the quadcopter information dictionary
        :param method:
            Feedback_Linearization:
            (Quadrotor control: modeling, nonlinear control design, and simulation by Francesco Sabatino)
            [x, y, z, psi, theta, phi, x_dot, y_dot, z_dot, zeta, xi, p, q, r]

            Zero_Dynamic_Stabilization:
            (Quadrotor control: modeling, nonlinear control design, and simulation by Francesco Sabatino)
            [x, y, z, psi, theta, phi, x_dot, y_dot, z_dot, 0, 0, p, q, r]

            PID:
            (Quadcopter Dynamics, Simulation, and Control by Andrew Gibiansky)
            [x y z x_dot y_dot z_dot theta phi gamma theta_dot phi_dot gamma_dot]
        :keyword x_desireds (Required for Zero_Dynamic_Stabilization)
            A 5 x 1 array-like object containing the desired values for x, x', x'', x^(3), x^(4)
        :keyword y_desireds (Required for Zero_Dynamic_Stabilization)
            A 5 x 1 array-like object containing the desired values for y, y', y'', y^(3), y^(4)
        :keyword z_desireds (Required for Zero_Dynamic_Stabilization)
            A 3 x 1 array-like object containing the desired values for z, z', z''
        :keyword psi_desireds (Required for Zero_Dynamic_Stabilization)
            A 3 x 1 array-like object containing the desired values for psi, psi', psi''
        :return:
        """

        if method not in ["Feedback_Linearization", "Zero_Dynamic_Stabilization", "PID"]:
            raise SyntaxError("parameter 'method' must be one of "
                              + "\"Feedback_Linearization\", "
                              + "\"Zero_Dynamic_Stabilization\", or"
                                "\"Regular_Dynamics\"")

        m = self.quads[key]['weight']
        Ix, Iy, Iz = self.quads[key]['I'].diagonal()

        if method.lower() == "Feedback_Linearization".lower():
            if np.array(state).shape[0] != 14:
                raise ValueError("State should be 14-dimensional!")

            z = self.x_bar_to_z(state, key)
            u_bar = controller.Feedback_Linearization(0, self.v(z, t), state, m, Ix, Iy, Iz)
            return u_bar

        elif method.lower() == "Zero_Dynamic_Stabilization".lower():
            if np.array(state).shape[0] != 12:
                raise ValueError("State should be 12-dimensional!")
            required_kwargs = ["x_desireds", "y_desireds", "z_desireds", "psi_desireds"]
            if len(kwargs) != 4:
                raise SyntaxError(
                    "Zero_Dynamic Stabilization requires 4 kwargs: x_desireds, y_desireds, z_desireds, psi_desireds")
            if any([arg not in required_kwargs for arg in required_kwargs]):
                raise SyntaxError(
                    "Zero_Dynamic Stabilization requires 4 kwargs: x_desireds, y_desireds, z_desireds, psi_desireds")

            x_desireds = kwargs.get("x_desireds")
            assert np.array(x_desireds).shape[0] == 5, "x_desireds must be 5-dimensional"

            y_desireds = kwargs.get("y_desireds")
            assert np.array(y_desireds).shape[0] == 5, "y_desireds must be 5-dimensional"

            z_desireds = kwargs.get("z_desireds")
            assert np.array(z_desireds).shape[0] == 3, "z_desireds must be 3-dimensional"

            psi_desireds = kwargs.get("psi_desireds")
            assert np.array(z_desireds).shape[0] == 3, "psi_desireds must be 3-dimensional"

            u = self.u1_prev
            k_11 = 4
            k_12 = 4
            x_d, x_dot_d, x_dot_dot_d, x_3_d, x_4_d = x_desireds
            y_d, y_dot_d, y_dot_dot_d, y_3_d, y_4_d = y_desireds
            z_d, z_dot_d, z_dot_dot_d = z_desireds
            psi_d, psi_dot_d, psi_dot_dot_d = psi_desireds

            x, y, z, psi, theta, phi, x_dot, y_dot, z_dot, p, q, r = state

            # Outer Layer 3
            theta_d = -m / u * (x_dot_dot_d + k_11 * (x_dot_d - x_dot) + k_12 * (x_d - x))
            phi_d = -m / u * (y_dot_dot_d + k_11 * (y_dot_d - y_dot) + k_12 * (y_d - y))

            theta_dot_d = -m / u * (x_3_d + k_11 * (x_dot_dot_d + u / m * theta) + k_12 * (x_dot_d - x_dot))
            phi_dot_d = -m / u * (y_3_d + k_11 * (y_dot_dot_d + u / m * phi) + k_12 * (y_dot_d - y_dot))

            theta_dot_dot_d = -m / u * (x_4_d + k_11 * (x_3_d + u / m * q) + k_12 * (x_dot_dot_d + u / m * theta))
            phi_dot_dot_d = -m / u * (y_4_d + k_11 * (y_3_d + u / m * p) + k_12 * (y_dot_dot_d + u / m * phi))

            # Inner Layer 2
            K_v = 4*np.eye(4)

            K_p = 4*np.eye(4)

            x_vec_q = np.array([z, phi, theta, psi])
            x_vec_q_d = np.array([z_d, phi_d, theta_d, psi_d])

            x_vec_dot_q = np.array([z_dot, p, q, r])
            x_vec_dot_q_d = np.array([z_dot_d, phi_dot_d, theta_dot_d, psi_dot_d])

            x_vec_dot_dot_q_d = np.array([z_dot_dot_d, phi_dot_dot_d, theta_dot_dot_d, psi_dot_dot_d])

            v = x_vec_dot_dot_q_d - K_v @ (x_vec_dot_q - x_vec_dot_q_d) - K_p @ (x_vec_q - x_vec_q_d)

            # FBL Layer 1
            u_vec = controller.Feedback_Linearization_Zero_Dynamics(v, state, m, Ix, Iy, Iz)
            if any([component > 1e7 for component in u_vec]):
                raise ValueError("Control input is blowing up", u_vec > 1e7)
            return u_vec

        elif method.lower() == "Regular_Dynamics".lower():
            if np.array(state).shape[0] != 12:
                raise ValueError("State should be 12-dimensional!")
            u = [0, 0, 0, 0]
            return u

    def z_dot_bar(self, t, z_state):
        v = self.v(z_state)
        z_dot = np.zeros(14)
        z_dot[0:3] = z_state[1:4]
        z_dot[4:7] = z_state[5:8]
        z_dot[8:11] = z_state[9:12]
        z_dot[12] = z_state[13]
        z_dot[3] += v[0]
        z_dot[7] += v[1]
        z_dot[11] += v[2]
        return z_dot

    def decomposition_function(self, z, v, w):
        """
        :param z: 14-dimensional vector representing state
        :param v: 4-dimensional vector representing feedback-linearized inputs
        :param w: 3-dimensional vector representing disturbances
        :return:
        """
        d = np.zeros(14)
        d[0] = z[1]
        d[1] = z[2] + w[0]
        d[2] = z[3]
        d[3] = v[0]
        d[4] = z[5]
        d[5] = z[6] + w[1]
        d[6] = z[7]
        d[7] = v[1]
        d[8] = z[9]
        d[9] = z[10] + w[2]
        d[10] = z[11]
        d[11] = v[2]
        d[12] = z[13]
        d[13] = v[3]
        return d

    def simulate_embedding(self, t, dt, z_initial, v_signal, w_lower, w_upper):
        """

        :param t: time to run simulation for
        :param dt: time step
        :param z_initial: 14-dimensional feedback-linearized initial state
        :param v_signal: collection of 4d vectors with control input for each time step
        :param w_lower: lower bound of noise
        :param w_upper: upper bound of noise
        :return:
        """
        z_emb_1 = np.copy(z_initial)
        z_emb_2 = np.copy(z_initial)
        for v in v_signal:
            z_emb_1 += dt * self.decomposition_function(z_emb_1, v, w_lower)
            z_emb_2 += dt * self.decomposition_function(z_emb_2, v, w_upper)

        return np.array([z_emb_1, z_emb_2])

    def update(self, dt):
        if self.plot_sim_trail or self.simulate:
            self.simulate_dynamics(self.time_horizon)
        for key in self.quads:
            self.ode.set_initial_value(self.quads[key]['state'], self.ode.t)
            self.ode.set_f_params(str(key), self.quads[key]['method'])
            self.quads[key]['state'] = self.ode.integrate(self.ode.t + dt)
            self.quads[key]['state'][3:6] = utils.wrap_angle(self.quads[key]['state'][3:6])

    def v(self, z, t=0):
        ref = np.array([2 * np.cos(0.1 * t) + 2, 2 * np.sin(0.1 * t) + 2, 2])
        return self.v_to_ref(z, ref)

    def v_to_ref(self, z, ref):
        x_ref, y_ref, z_ref = ref
        v_vec = np.array(
            [(-24 * (z[0] - x_ref) - 50 * z[1] - 35 * z[2] - 10 * z[3]),
             (-24 * (z[4] - y_ref) - 50 * z[5] - 35 * z[6] - 10 * z[7]),
             (-24 * (z[8] - z_ref) - 50 * z[9] - 35 * z[10] - 10 * z[11]),
             0])
        return v_vec

    def x_bar_to_z(self, x_state, key):
        x, y, z, psi, theta, phi, x_dot, y_dot, z_dot, zeta, xi, p, q, r = x_state
        m = self.quads[key]['weight']

        cs, ct, cp = np.cos(x_state[3:6])
        ss, st, sp = np.sin(x_state[3:6])
        ts, tt, tp = np.tan(x_state[3:6])

        z_vec = np.zeros(14)
        z_vec[0] = x
        z_vec[1] = x_dot
        z_vec[2] = -zeta * (sp * ss + st * cp * cs) / m
        z_vec[3] = -xi * (sp * ss + st * cp * cs) / m - zeta * (q * cp - r * sp) * cp * cs * ct / m \
                   - zeta * (sp * cs - ss * st * cp) * (q * sp / ct + r * cp / ct) / m - zeta * (
                           -sp * st * cs + ss * cp) * (p + q * sp * tt + r * cp * tt) / m
        z_vec[4] = y
        z_vec[5] = y_dot
        z_vec[6] = -zeta * (sp * cs - ss * st * cp) / m
        z_vec[7] = -xi * (sp * cs - ss * st * cp) / m + zeta * (q * cp - r * sp) * ss * cp * ct / m \
                   - zeta * (-sp * ss - st * cp * cs) * (q * sp / ct + r * cp / ct) / m - zeta * (
                           sp * ss * st + cp * cs) * (p + q * sp * tt + r * cp * tt) / m
        z_vec[8] = z
        z_vec[9] = z_dot
        z_vec[10] = -zeta * cp * ct / m
        z_vec[11] = -xi * cp * ct / m + zeta * (q * cp - r * sp) * st * cp / m + zeta * (
                p + q * sp * tt + r * cp * tt) * sp * ct / m
        z_vec[12] = psi
        z_vec[13] = q * sp / ct + r * cp / ct

        return z_vec

    def simulate_dynamics(self, total_time):
        """Simulate quadcopter dynamics over finite time horizon and return final states"""
        for key in self.quads:
            '''z_init = self.x_bar_to_z(self.quads[key]['state'], key)
            sol = scipy.integrate.solve_ivp(self.z_dot_bar, [0, total_time], z_init,
                                            t_eval=[total_time], method='RK45', atol=1e-3,
                                            rtol=1e-6)'''
            sol = scipy.integrate.solve_ivp(self.state_dot, [self.ode.t, self.ode.t+total_time], self.quads[key]['state'],
                                            args=(key, self.quads[key]['method'],),
                                            t_eval=[self.ode.t+total_time], method='RK45', atol=1e-3,
                                            rtol=1e-6)
            final_state = sol.y[:, -1]
            final_state[3:6] = utils.wrap_angle(final_state[3:6])
            self.last_sim_state.append(final_state)

    def get_bounds(self, t, dt):
        a = 3
        return self.simulate_embedding(t, dt, self.last_sim_state[-1],
                                       np.repeat([self.v(self.last_sim_state[-1])], (t / dt), axis=0),
                                       a * np.array([-.1, -.1, -.1]), a * np.array([.1, .1, .1]))

    def get_state(self, quad_name):
        return self.quads[quad_name]['state']

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
