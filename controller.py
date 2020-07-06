import numpy as np
import math
import time
import threading
import utils

'''
Originally from https://github.com/abhijitmajumdar/Quadcopter_simulator
'''

def Feedback_Linearization(t, v, state, m, u1_prev, zeta, xi, Ix, Iy, Iz):
    '''Credit to https://www.kth.se/polopoly_fs/1.588039.1550155544!/Thesis%20KTH%20-%20Francesco%20Sabatino.pdf'''
    p, q, r = state[9:12]

    ct, cp, cs = np.cos(state[3:6])
    st, sp, ss = np.sin(state[3:6])
    tt, tp, ts = np.tan(state[3:6])

    # d_i,j = L_gj * L_f^(r_i-1) (h_i(x))
    delta = np.array([[-(sp * ss + st * cp * cs)/m, 
                       zeta * (sp * st * ss - ss * sp) / (Ix * m),
                       -(zeta * cp * ct) / (Iy * m),
                       0],
                      [(-sp * cs + ss * st * cp) / m,
                       -zeta * (sp * ss * st + cp * cs) / (Ix * m),
                       zeta * ss * ct / (Iy * m),
                       0],
                      [-ss * st / m,
                       0, 
                       zeta * (sp * ss + st * cp * ss) / (Iy * m),
                       zeta * (-sp * st * cs + ss * sp) / (Iz * m)],
                      [0, 
                       (-4*Ix*p*r*cp*ct - 6*Ix*q*r*sp*st*cp + 6*Ix*r**2*sp**2*st - 3*Ix*r**2*st - 2*Iy*p*q*sp*ct - 2*Iy*p*r*cp*ct - 6*Iy*q**2*sp**2*st + 3*Iy*q**2*st - 12*Iy*q*r*sp*st*cp + 6*Iy*r**2*sp**2*st - 3*Iy*r**2*st + 4*Iz*p*r*cp*ct + 6*Iz*q*r*sp*st*cp - 6*Iz*r**2*sp**2*st + 3*Iz*r**2*st)/(Ix*Iy*ct**2),
                       (-6*Ix**2*p*r*sp*cp*tt - Ix*Iy*p**2*sp - 12*Ix*Iy*p*q*sp**2*tt + 6*Ix*Iy*p*q*tt - 12*Ix*Iy*p*r*sp*cp*tt + 18*Ix*Iy*q**2*sp**3 - 24*Ix*Iy*q**2*sp**3/ct**2 - 12*Ix*Iy*q**2*sp + 18*Ix*Iy*q**2*sp/ct**2 - 36*Ix*Iy*q*r*cp**3 + 48*Ix*Iy*q*r*cp**3/ct**2 + 28*Ix*Iy*q*r*cp - 36*Ix*Iy*q*r*cp/ct**2 - 18*Ix*Iy*r**2*sp**3 + 24*Ix*Iy*r**2*sp**3/ct**2 + 13*Ix*Iy*r**2*sp - 18*Ix*Iy*r**2*sp/ct**2 + 6*Ix*Iz*p*r*sp*cp*tt + Ix*Iz*r**2*sp + 2*Iy**2*q*r*cp - Iy**2*r**2*sp - 2*Iy*Iz*q*r*cp + 2*Iy*Iz*r**2*sp - Iz**2*r**2*sp)/(Ix*Iy**2*ct),
                       (-2*Ix**2*p**2*cp - 6*Ix**2*p*q*sp*cp*tt + 12*Ix**2*p*r*sp**2*tt - 6*Ix**2*p*r*tt - Ix*Iy*p**2*cp - 12*Ix*Iy*p*q*sp*cp*tt + 12*Ix*Iy*p*r*sp**2*tt - 6*Ix*Iy*p*r*tt - 18*Ix*Iy*q**2*cp**3 + 24*Ix*Iy*q**2*cp**3/ct**2 + 14*Ix*Iy*q**2*cp - 18*Ix*Iy*q**2*cp/ct**2 - 36*Ix*Iy*q*r*sp**3 + 48*Ix*Iy*q*r*sp**3/ct**2 + 26*Ix*Iy*q*r*sp - 36*Ix*Iy*q*r*sp/ct**2 + 18*Ix*Iy*r**2*cp**3 - 24*Ix*Iy*r**2*cp**3/ct**2 - 12*Ix*Iy*r**2*cp + 18*Ix*Iy*r**2*cp/ct**2 + 2*Ix*Iz*p**2*cp + 6*Ix*Iz*p*q*sp*cp*tt - 12*Ix*Iz*p*r*sp**2*tt + 6*Ix*Iz*p*r*tt + 2*Ix*Iz*q*r*sp + Iy**2*q**2*cp - 2*Iy**2*q*r*sp - Iy*Iz*q**2*cp + 4*Iy*Iz*q*r*sp - 2*Iz**2*q*r*sp)/(Ix*Iy*Iz*ct)]])
    b = np.array([(Ix*Iy*(-xi*(-p*sp*st*cs + p*ss*cp + q*cs*ct) + (p + q*sp*tt + r*cp*tt)*(p*zeta*sp*ss + p*zeta*st*cp*cs + xi*sp*st*cs - xi*ss*cp))*ct**2 + Ix*Iy*(-(q*sp + r*cp)*(p*zeta*sp*ss*st + p*zeta*cp*cs - q*zeta*ss*ct + xi*sp*cs - xi*ss*st*cp)*ct + (q*cp - r*sp)*(p*zeta*sp*ct**3 - q*zeta*st**3 + q*zeta*st - xi*cp*ct**3)*cs) + Ix*p*r*zeta*(Ix - Iz)*cs*ct**3 + Iy*q*r*zeta*(Iy - Iz)*(sp*st*cs - ss*cp)*ct**2)/(Ix*Iy*m*ct**2),
                  (Ix*Iy*(xi*(-p*sp*ss*st - p*cp*cs + q*ss*ct) - (p + q*sp*tt + r*cp*tt)*(-p*zeta*sp*cs + p*zeta*ss*st*cp + xi*sp*ss*st + xi*cp*cs))*ct**2 + Ix*Iy*((q*sp + r*cp)*(-p*zeta*sp*st*cs + p*zeta*ss*cp + q*zeta*cs*ct + xi*sp*ss + xi*st*cp*cs)*ct + (q*cp - r*sp)*(-p*zeta*sp*ct**3 + q*zeta*st**3 - q*zeta*st + xi*cp*ct**3)*ss) - Ix*p*r*zeta*(Ix - Iz)*ss*ct**3 - Iy*q*r*zeta*(Iy - Iz)*(sp*ss*st + cp*cs)*ct**2)/(Ix*Iy*m*ct**2),
                  (-Ix*p*r*zeta*sp*ss - Ix*p*r*zeta*st*cp*cs - Iy*p*q*zeta*sp*st*cs + Iy*p*q*zeta*ss*cp - Iy*p*r*zeta*sp*ss - Iy*p*r*zeta*st*cp*cs + Iy*q**2*zeta*cs*ct + 2*Iy*q*xi*sp*ss + 2*Iy*q*xi*st*cp*cs + Iy*r**2*zeta*cs*ct - 2*Iy*r*xi*sp*st*cs + 2*Iy*r*xi*ss*cp + Iz*p*r*zeta*sp*ss + Iz*p*r*zeta*st*cp*cs)/(Iy*m),
                  (q*cp - r*sp)*((q*cp - r*sp)*((q*cp - r*sp)*(6*q*sp*st**3/ct**4 + 5*q*sp*st/ct**2 + 6*r*st**3*cp/ct**4 + 5*r*st*cp/ct**2) + 2*(q*(tt**2 + 1)*sp + r*(tt**2 + 1)*cp)*(q*st*cp/ct**2 - r*sp*st/ct**2) + (q*cp/ct - r*sp/ct)*(q*(2*tt**2 + 2)*sp*tt + r*(2*tt**2 + 2)*cp*tt) + (p + q*sp*tt + r*cp*tt)*(2*q*st**2*cp/ct**3 + q*cp/ct - 2*r*sp*st**2/ct**3 - r*sp/ct) + 2*p*r*(-Ix + Iz)*sp*st**2/(Iy*ct**3) + p*r*(-Ix + Iz)*sp/(Iy*ct)) + (q*(tt**2 + 1)*sp + r*(tt**2 + 1)*cp)*((-q*sp - r*cp)*(q*sp*st/ct**2 + r*st*cp/ct**2) + (q*cp - r*sp)*(q*st*cp/ct**2 - r*sp*st/ct**2) + (-q*sp/ct - r*cp/ct)*(p + q*sp*tt + r*cp*tt) + (q*cp/ct - r*sp/ct)*(q*cp*tt - r*sp*tt) + p*r*(-Ix + Iz)*cp/(Iy*ct)) + (p + q*sp*tt + r*cp*tt)*((-q*sp - r*cp)*(2*q*sp*st**2/ct**3 + q*sp/ct + 2*r*st**2*cp/ct**3 + r*cp/ct) + (q*cp - r*sp)*(2*q*st**2*cp/ct**3 + q*cp/ct - 2*r*sp*st**2/ct**3 - r*sp/ct) + (q*(tt**2 + 1)*sp + r*(tt**2 + 1)*cp)*(-q*sp/ct - r*cp/ct) + (q*(tt**2 + 1)*cp - r*(tt**2 + 1)*sp)*(q*cp/ct - r*sp/ct) + (q*cp*tt - r*sp*tt)*(q*st*cp/ct**2 - r*sp*st/ct**2) + (-q*sp*st/ct**2 - r*st*cp/ct**2)*(p + q*sp*tt + r*cp*tt) + p*r*(-Ix + Iz)*st*cp/(Iy*ct**2)) + p*r*(-Ix + Iz)*(2*(q*cp - r*sp)*sp*st**2/ct**3 + (q*cp - r*sp)*sp/ct + (q*(tt**2 + 1)*sp + r*(tt**2 + 1)*cp)*cp/ct + (q*cp/ct - r*sp/ct)*(tt**2 + 1)*sp + (q*st*cp/ct**2 - r*sp*st/ct**2)*sp*tt + (p + q*sp*tt + r*cp*tt)*st*cp/ct**2 + (2*q*sp*st**2/ct**3 + q*sp/ct + 2*r*st**2*cp/ct**3 + r*cp/ct)*cp)/Iy + q*r*(Iy - Iz)*(q*st*cp/ct**2 - r*sp*st/ct**2 + r*(-Ix + Iz)*sp*st/(Iy*ct**2))/Ix) + (p + q*sp*tt + r*cp*tt)*((-q*sp - r*cp)*((q*cp - r*sp)*(2*q*sp*st**2/ct**3 + q*sp/ct + 2*r*st**2*cp/ct**3 + r*cp/ct) + (q*(tt**2 + 1)*sp + r*(tt**2 + 1)*cp)*(q*cp/ct - r*sp/ct) + (q*st*cp/ct**2 - r*sp*st/ct**2)*(p + q*sp*tt + r*cp*tt) + p*r*(-Ix + Iz)*sp*st/(Iy*ct**2)) + (q*cp - r*sp)*((-q*sp - r*cp)*(2*q*sp*st**2/ct**3 + q*sp/ct + 2*r*st**2*cp/ct**3 + r*cp/ct) + (q*cp - r*sp)*(2*q*st**2*cp/ct**3 + q*cp/ct - 2*r*sp*st**2/ct**3 - r*sp/ct) + (q*(tt**2 + 1)*sp + r*(tt**2 + 1)*cp)*(-q*sp/ct - r*cp/ct) + (q*(tt**2 + 1)*cp - r*(tt**2 + 1)*sp)*(q*cp/ct - r*sp/ct) + (q*cp*tt - r*sp*tt)*(q*st*cp/ct**2 - r*sp*st/ct**2) + (-q*sp*st/ct**2 - r*st*cp/ct**2)*(p + q*sp*tt + r*cp*tt) + p*r*(-Ix + Iz)*st*cp/(Iy*ct**2)) + (q*cp*tt - r*sp*tt)*((-q*sp - r*cp)*(q*sp*st/ct**2 + r*st*cp/ct**2) + (q*cp - r*sp)*(q*st*cp/ct**2 - r*sp*st/ct**2) + (-q*sp/ct - r*cp/ct)*(p + q*sp*tt + r*cp*tt) + (q*cp/ct - r*sp/ct)*(q*cp*tt - r*sp*tt) + p*r*(-Ix + Iz)*cp/(Iy*ct)) + (p + q*sp*tt + r*cp*tt)*(2*(-q*sp - r*cp)*(q*st*cp/ct**2 - r*sp*st/ct**2) + (-q*cp + r*sp)*(q*sp*st/ct**2 + r*st*cp/ct**2) + (q*cp - r*sp)*(-q*sp*st/ct**2 - r*st*cp/ct**2) + 2*(-q*sp/ct - r*cp/ct)*(q*cp*tt - r*sp*tt) + (-q*sp*tt - r*cp*tt)*(q*cp/ct - r*sp/ct) + (-q*cp/ct + r*sp/ct)*(p + q*sp*tt + r*cp*tt) - p*r*(-Ix + Iz)*sp/(Iy*ct)) + p*r*(-Ix + Iz)*((-q*sp - r*cp)*sp*st/ct**2 + (q*cp - r*sp)*st*cp/ct**2 + (-q*sp/ct - r*cp/ct)*sp*tt + (q*cp/ct - r*sp/ct)*cp*tt + (q*cp*tt - r*sp*tt)*cp/ct - (q*sp*st/ct**2 + r*st*cp/ct**2)*sp + (q*st*cp/ct**2 - r*sp*st/ct**2)*cp - (p + q*sp*tt + r*cp*tt)*sp/ct)/Iy + q*r*(Iy - Iz)*(-q*sp/ct - r*cp/ct + r*(-Ix + Iz)*cp/(Iy*ct))/Ix) + p*r*(-Ix + Iz)*((q*cp - r*sp)*((q*cp - r*sp)*(2*sp*st**2/ct**3 + sp/ct) + (q*(tt**2 + 1)*sp + r*(tt**2 + 1)*cp)*cp/ct + (q*cp/ct - r*sp/ct)*(tt**2 + 1)*sp + (q*st*cp/ct**2 - r*sp*st/ct**2)*sp*tt + (p + q*sp*tt + r*cp*tt)*st*cp/ct**2 + (2*q*sp*st**2/ct**3 + q*sp/ct + 2*r*st**2*cp/ct**3 + r*cp/ct)*cp) + (p + q*sp*tt + r*cp*tt)*((-q*sp - r*cp)*sp*st/ct**2 + (q*cp - r*sp)*st*cp/ct**2 + (-q*sp/ct - r*cp/ct)*sp*tt + (q*cp/ct - r*sp/ct)*cp*tt + (q*cp*tt - r*sp*tt)*cp/ct - (q*sp*st/ct**2 + r*st*cp/ct**2)*sp + (q*st*cp/ct**2 - r*sp*st/ct**2)*cp - (p + q*sp*tt + r*cp*tt)*sp/ct) + ((q*cp - r*sp)*(2*q*sp*st**2/ct**3 + q*sp/ct + 2*r*st**2*cp/ct**3 + r*cp/ct) + (q*(tt**2 + 1)*sp + r*(tt**2 + 1)*cp)*(q*cp/ct - r*sp/ct) + (q*st*cp/ct**2 - r*sp*st/ct**2)*(p + q*sp*tt + r*cp*tt) + p*r*(-Ix + Iz)*sp*st/(Iy*ct**2))*cp + ((-q*sp - r*cp)*(q*sp*st/ct**2 + r*st*cp/ct**2) + (q*cp - r*sp)*(q*st*cp/ct**2 - r*sp*st/ct**2) + (-q*sp/ct - r*cp/ct)*(p + q*sp*tt + r*cp*tt) + (q*cp/ct - r*sp/ct)*(q*cp*tt - r*sp*tt) + p*r*(-Ix + Iz)*cp/(Iy*ct))*sp*tt + p*r*(-Ix + Iz)*(2*sp*st*cp/ct**2 + 2*sp*cp*tt/ct)/Iy + q*r*(Iy - Iz)*cp/(Ix*ct) + r*(Iy - Iz)*(q*cp/ct - r*sp/ct + r*(-Ix + Iz)*sp/(Iy*ct))/Ix)/Iy + q*r*(Iy - Iz)*((-q*sp - r*cp)*(q*sp*st/ct**2 + r*st*cp/ct**2) + (q*cp - r*sp)*(q*st*cp/ct**2 - r*sp*st/ct**2) + (q*cp - r*sp)*(q*st*cp/ct**2 - r*sp*st/ct**2 + r*(-Ix + Iz)*sp*st/(Iy*ct**2)) + (-q*sp/ct - r*cp/ct)*(p + q*sp*tt + r*cp*tt) + (q*cp/ct - r*sp/ct)*(q*cp*tt - r*sp*tt) + (p + q*sp*tt + r*cp*tt)*(-q*sp/ct - r*cp/ct + r*(-Ix + Iz)*cp/(Iy*ct)) + 2*p*r*(-Ix + Iz)*cp/(Iy*ct) + r*(-Ix + Iz)*((q*cp - r*sp)*sp*st/ct**2 + (q*cp/ct - r*sp/ct)*sp*tt + (q*sp*st/ct**2 + r*st*cp/ct**2)*cp + (p + q*sp*tt + r*cp*tt)*cp/ct)/Iy)/Ix]).T
    # u = alpha + beta * v

    try:
        d_inv = -np.linalg.inv(delta)
        alpha = -d_inv @ b
        beta = -d_inv
        u_bar = alpha + beta @ v

        dt = 0.005
        u1_dot_dot = xi + u_bar[0] * dt
        u1_dot = zeta + u_bar[0] * dt
        u_bar[0] = u1_prev + u_bar[0] * dt
        return u_bar, [u_bar[0], u1_dot, u1_dot_dot]
    except Exception:
        print("!")
        return [1,0,1,0], [0,0,0]

class Controller_PID_Point2Point:
    def __init__(self, get_state, get_time, actuate_motors, params, quad_identifier):
        self.quad_identifier = quad_identifier
        self.actuate_motors = actuate_motors
        self.get_state = get_state
        self.get_time = get_time
        self.MOTOR_LIMITS = params['Motor_limits']
        self.TILT_LIMITS = [(params['Tilt_limits'][0] / 180.0) * np.pi, (params['Tilt_limits'][1] / 180.0) * np.pi]
        self.YAW_CONTROL_LIMITS = params['Yaw_Control_Limits']
        self.Z_LIMITS = [self.MOTOR_LIMITS[0] + params['Z_XY_offset'], self.MOTOR_LIMITS[1] - params['Z_XY_offset']]
        self.LINEAR_P, self.LINEAR_I, self.LINEAR_D = params['Linear_PID'].values()
        self.ANGULAR_P, self.ANGULAR_I, self.ANGULAR_D = params['Angular_PID'].values()
        self.LINEAR_TO_ANGULAR_SCALER = params['Linear_To_Angular_Scaler']
        self.YAW_RATE_SCALER = params['Yaw_Rate_Scaler']
        self.xi_term = 0
        self.yi_term = 0
        self.zi_term = 0
        self.thetai_term = 0
        self.phii_term = 0
        self.gammai_term = 0
        self.xi_term_sim = 0
        self.yi_term_sim = 0
        self.zi_term_sim = 0
        self.thetai_term_sim = 0
        self.phii_term_sim = 0
        self.gammai_term_sim = 0
        self.thread_object = None
        self.target = [0, 0, 0]
        self.yaw_target = 0.0
        self.run = True

    def update(self, state=None, sim=False):
        if sim:
            xi, yi, zi = self.xi_term_sim, self.yi_term_sim, self.zi_term_sim
            thetai, phii, gammai = self.thetai_term_sim, self.phii_term_sim, self.gammai_term_sim
            x, y, z, x_dot, y_dot, z_dot, theta, phi, gamma, theta_dot, phi_dot, gamma_dot = state
        else:
            xi, yi, zi = self.xi_term, self.yi_term, self.zi_term
            thetai, phii, gammai = self.thetai_term, self.phii_term, self.gammai_term
            x, y, z, x_dot, y_dot, z_dot, theta, phi, gamma, theta_dot, phi_dot, gamma_dot = self.get_state(
                self.quad_identifier)
        dest_x, dest_y, dest_z = self.target
        x_error = dest_x - x
        y_error = dest_y - y
        z_error = dest_z - z
        xi += self.LINEAR_I[0] * x_error
        yi += self.LINEAR_I[1] * y_error
        zi += self.LINEAR_I[2] * z_error
        dest_x_dot = self.LINEAR_P[0] * x_error - self.LINEAR_D[0] * x_dot + xi
        dest_y_dot = self.LINEAR_P[1] * y_error - self.LINEAR_D[1] * y_dot + yi
        dest_z_dot = self.LINEAR_P[2] * z_error - self.LINEAR_D[2] * z_dot + zi
        throttle = np.clip(dest_z_dot, self.Z_LIMITS[0], self.Z_LIMITS[1])
        dest_theta = self.LINEAR_TO_ANGULAR_SCALER[0] * (dest_x_dot * math.sin(gamma) - dest_y_dot * math.cos(gamma))
        dest_phi = self.LINEAR_TO_ANGULAR_SCALER[1] * (dest_x_dot * math.cos(gamma) + dest_y_dot * math.sin(gamma))
        dest_gamma = self.yaw_target
        dest_theta, dest_phi = np.clip(dest_theta, self.TILT_LIMITS[0], self.TILT_LIMITS[1]), np.clip(dest_phi,
                                                                                                      self.TILT_LIMITS[
                                                                                                          0],
                                                                                                      self.TILT_LIMITS[
                                                                                                          1])
        theta_error = dest_theta - theta
        phi_error = dest_phi - phi
        gamma_dot_error = (self.YAW_RATE_SCALER * utils.wrap_angle(dest_gamma - gamma)) - gamma_dot
        thetai += self.ANGULAR_I[0] * theta_error
        phii += self.ANGULAR_I[1] * phi_error
        gammai += self.ANGULAR_I[2] * gamma_dot_error
        x_val = self.ANGULAR_P[0] * theta_error + self.ANGULAR_D[0] * (-theta_dot) + thetai
        y_val = self.ANGULAR_P[1] * phi_error + self.ANGULAR_D[1] * (-phi_dot) + phii
        z_val = self.ANGULAR_P[2] * gamma_dot_error + gammai
        z_val = np.clip(z_val, self.YAW_CONTROL_LIMITS[0], self.YAW_CONTROL_LIMITS[1])
        m1 = throttle + x_val + z_val
        m2 = throttle + y_val - z_val
        m3 = throttle - x_val + z_val
        m4 = throttle - y_val - z_val
        M = np.clip([m1, m2, m3, m4], self.MOTOR_LIMITS[0], self.MOTOR_LIMITS[1])
        if not sim:
            self.actuate_motors(self.quad_identifier, M)
        return M

    def update_target(self, target):
        self.target = target

    def update_yaw_target(self, target):
        self.yaw_target = utils.wrap_angle(target)

    def thread_run(self, update_rate, time_scaling):
        update_rate = update_rate * time_scaling
        last_update = self.get_time()
        while self.run:
            time.sleep(0)
            self.time = self.get_time()
            if (self.time - last_update).total_seconds() > update_rate:
                self.update()
                last_update = self.time

    def start_thread(self, update_rate=0.005, time_scaling=1):
        self.thread_object = threading.Thread(target=self.thread_run, args=(update_rate, time_scaling))
        self.thread_object.start()

    def stop_thread(self):
        self.run = False

    def reset_sim(self):
        self.xi_term_sim = self.xi_term
        self.yi_term_sim = self.yi_term
        self.zi_term_sim = self.zi_term
        self.thetai_term_sim = self.thetai_term
        self.phii_term_sim = self.phii_term
        self.gammai_term_sim = self.gammai_term


class Controller_PID_Velocity(Controller_PID_Point2Point):
    def update(self):
        [dest_x, dest_y, dest_z] = self.target
        [x, y, z, x_dot, y_dot, z_dot, theta, phi, gamma, theta_dot, phi_dot, gamma_dot] = self.get_state(
            self.quad_identifier)
        x_error = dest_x - x_dot
        y_error = dest_y - y_dot
        z_error = dest_z - z
        self.xi_term += self.LINEAR_I[0] * x_error
        self.yi_term += self.LINEAR_I[1] * y_error
        self.zi_term += self.LINEAR_I[2] * z_error
        dest_x_dot = self.LINEAR_P[0] * (x_error) + self.LINEAR_D[0] * (-x_dot) + self.xi_term
        dest_y_dot = self.LINEAR_P[1] * (y_error) + self.LINEAR_D[1] * (-y_dot) + self.yi_term
        dest_z_dot = self.LINEAR_P[2] * (z_error) + self.LINEAR_D[2] * (-z_dot) + self.zi_term
        throttle = np.clip(dest_z_dot, self.Z_LIMITS[0], self.Z_LIMITS[1])
        dest_theta = self.LINEAR_TO_ANGULAR_SCALER[0] * (dest_x_dot * math.sin(gamma) - dest_y_dot * math.cos(gamma))
        dest_phi = self.LINEAR_TO_ANGULAR_SCALER[1] * (dest_x_dot * math.cos(gamma) + dest_y_dot * math.sin(gamma))
        dest_gamma = self.yaw_target
        dest_theta, dest_phi = np.clip(dest_theta, self.TILT_LIMITS[0], self.TILT_LIMITS[1]), np.clip(dest_phi,
                                                                                                      self.TILT_LIMITS[
                                                                                                          0],
                                                                                                      self.TILT_LIMITS[
                                                                                                          1])
        theta_error = dest_theta - theta
        phi_error = dest_phi - phi
        gamma_dot_error = (self.YAW_RATE_SCALER * utils.wrap_angle(dest_gamma - gamma)) - gamma_dot
        self.thetai_term += self.ANGULAR_I[0] * theta_error
        self.phii_term += self.ANGULAR_I[1] * phi_error
        self.gammai_term += self.ANGULAR_I[2] * gamma_dot_error
        x_val = self.ANGULAR_P[0] * (theta_error) + self.ANGULAR_D[0] * (-theta_dot) + self.thetai_term
        y_val = self.ANGULAR_P[1] * (phi_error) + self.ANGULAR_D[1] * (-phi_dot) + self.phii_term
        z_val = self.ANGULAR_P[2] * (gamma_dot_error) + self.gammai_term
        z_val = np.clip(z_val, self.YAW_CONTROL_LIMITS[0], self.YAW_CONTROL_LIMITS[1])
        m1 = throttle + x_val + z_val
        m2 = throttle + y_val - z_val
        m3 = throttle - x_val + z_val
        m4 = throttle - y_val - z_val
        M = np.clip([m1, m2, m3, m4], self.MOTOR_LIMITS[0], self.MOTOR_LIMITS[1])
        self.actuate_motors(self.quad_identifier, M)
