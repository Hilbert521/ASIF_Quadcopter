import numpy as np
import time
from quads.quadcopter import Quadcopter


class YZQuad(Quadcopter):
    # State = [y, z, vy, vz, phi, omega_x]
    def __init__(self, initial_state=np.zeros(6)):
        self.Ix = 0.01
        self.m = 1
        self.g = 9.81
        self.state = initial_state

    def state_dot(self, t, state, inputs):
        y, z, vy, vz, phi, omega_x = state
        tau, Mx = inputs

        state_dot = np.zeros(6)

        state_dot[0] = vy
        state_dot[1] = vz
        state_dot[2] = (tau * np.sin(phi)) / self.m
        state_dot[3] = (self.m * self.g - tau * np.cos(phi))
        state_dot[4] = omega_x
        state_dot[5] = Mx / self.Ix
        return state_dot

    def update(self, new_state):
        time.sleep(1e-9)
        self.state = new_state

    def get_position(self):
        return np.array([0, self.state[0], self.state[1]])

    def get_attitude(self):
        return np.array([self.state[-1], 0, 0])

    def get_state(self):
        return self.state
