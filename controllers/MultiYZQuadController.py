from controllers.controller import MultiAgentQuadController

import numpy as np
import time

class MultiYZQuadController(MultiAgentQuadController):

    def __init__(self):
        self.Ks = 3
        self.Kr = 5
        self.Kw = 1
        self.sigma = 0.03
        self.ds = 2
        self.b = 0.7
        self.tau1 = 0
        self.tau2 = 0
        self.Mx1 = 0
        self.Mx2 = 0

    def generate_inputs(self, states):
        s1, s2 = states
        y1, z1, vy1, vz1, phi1, omega_x1 = s1
        y2, z2, vy2, vz2, phi2, omega_x2 = s2
        alpha = y2 - y1
        beta = z2 - z1

        Fy1 = self.Ks*(np.tanh(self.sigma*(alpha - self.ds*(alpha/np.sqrt(alpha**2 + beta**2))))) - self.b*vy1
        Fy2 = -self.Ks*(np.tanh(self.sigma*(alpha - self.ds*(alpha/np.sqrt(alpha**2 + beta**2))))) - self.b*vy2
        Fz1 = self.Ks * (np.tanh(self.sigma * (beta - self.ds * (beta / np.sqrt(alpha ** 2 + beta ** 2))))) - 1*9.81 - self.b * vz1
        Fz2 = -self.Ks * (np.tanh(self.sigma * (beta - self.ds * (beta / np.sqrt(alpha ** 2 + beta ** 2))))) - 1*9.81 - self.b * vz2

        self.tau1 = Fy1 * np.sin(phi1) - Fz1 * np.cos(phi1)
        self.tau2 = Fy2 * np.sin(phi2) - Fz2 * np.cos(phi2)

        phi_des1 = np.arctan2(Fy1, -Fz1)
        phi_des2 = np.arctan2(Fy2, -Fz2)

        self.Mx1 = -self.Kr * (phi1 - phi_des1) - self.Kw*(omega_x1)
        self.Mx2 = -self.Kr * (phi2 - phi_des2) - self.Kw*(omega_x2)

        return [[self.tau1, self.Mx1], [self.tau2, self.Mx2]]