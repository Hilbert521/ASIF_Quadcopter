import numpy as np


def Feedback_Linearization(t, v, state, m, Ix, Iy, Iz):
    '''Credit to https://www.kth.se/polopoly_fs/1.588039.1550155544!/Thesis%20KTH%20-%20Francesco%20Sabatino.pdf'''
    p, q, r = state[11:14]
    cs, ct, cp = np.cos(state[3:6])
    ss, st, sp = np.sin(state[3:6])
    ts, tt, tp = np.tan(state[3:6])

    zeta = state[9]
    xi = state[10]
    # d_i,j = L_gj * L_f^(r_i-1) (h_i(x))
    delta = np.array([[-(sp * ss + st * cp * cs)/m,
                       zeta * (sp * st * cs - ss * cp) / (Ix * m),
                       -(zeta * cs * ct) / (Iy * m),
                       0],
                      [(-sp * cs + ss * st * cp) / m,
                       -zeta * (sp * ss * st + cp * cs) / (Ix * m),
                       zeta * ss * ct / (Iy * m),
                       0],
                      [-cp * ct / m,
                       zeta * sp * ct / (Ix * m),
                       zeta * st / (Iy * m),
                       0],
                      [0,
                       0,
                       sp/(Iy*ct),
                       cp/(Iz*ct)]])

    b = np.array([xi*(-(q*cp - r*sp)*cp*cs*ct/m - (sp*cs - ss*st*cp)*(q*sp/ct + r*cp/ct)/m - (-sp*st*cs + ss*cp)*(p + q*sp*tt + r*cp*tt)/m) + (q*cp - r*sp)*(-xi*cp*cs*ct/m + zeta*(q*cp - r*sp)*st*cp*cs/m - zeta*(sp*cs - ss*st*cp)*(q*sp*st/ct**2 + r*st*cp/ct**2)/m - zeta*(q*(tt**2 + 1)*sp + r*(tt**2 + 1)*cp)*(-sp*st*cs + ss*cp)/m + zeta*(q*sp/ct + r*cp/ct)*ss*cp*ct/m + zeta*(p + q*sp*tt + r*cp*tt)*sp*cs*ct/m) + (q*sp/ct + r*cp/ct)*(-xi*(sp*cs - ss*st*cp)/m + zeta*(q*cp - r*sp)*ss*cp*ct/m - zeta*(-sp*ss - st*cp*cs)*(q*sp/ct + r*cp/ct)/m - zeta*(sp*ss*st + cp*cs)*(p + q*sp*tt + r*cp*tt)/m) + (p + q*sp*tt + r*cp*tt)*(-xi*(-sp*st*cs + ss*cp)/m - zeta*(-q*sp - r*cp)*cp*cs*ct/m + zeta*(q*cp - r*sp)*sp*cs*ct/m - zeta*(-sp*ss - st*cp*cs)*(p + q*sp*tt + r*cp*tt)/m - zeta*(sp*cs - ss*st*cp)*(q*cp/ct - r*sp/ct)/m - zeta*(q*sp/ct + r*cp/ct)*(sp*ss*st + cp*cs)/m - zeta*(q*cp*tt - r*sp*tt)*(-sp*st*cs + ss*cp)/m) + p*r*(-Ix + Iz)*(-zeta*(sp*cs - ss*st*cp)*sp/(m*ct) - zeta*(-sp*st*cs + ss*cp)*sp*tt/m - zeta*cp**2*cs*ct/m)/Iy - q*r*zeta*(Iy - Iz)*(-sp*st*cs + ss*cp)/(Ix*m),
                  xi*((q*cp - r*sp)*ss*cp*ct/m - (-sp*ss - st*cp*cs)*(q*sp/ct + r*cp/ct)/m - (sp*ss*st + cp*cs)*(p + q*sp*tt + r*cp*tt)/m) + (q*cp - r*sp)*(xi*ss*cp*ct/m - zeta*(q*cp - r*sp)*ss*st*cp/m - zeta*(-sp*ss - st*cp*cs)*(q*sp*st/ct**2 + r*st*cp/ct**2)/m - zeta*(q*(tt**2 + 1)*sp + r*(tt**2 + 1)*cp)*(sp*ss*st + cp*cs)/m + zeta*(q*sp/ct + r*cp/ct)*cp*cs*ct/m - zeta*(p + q*sp*tt + r*cp*tt)*sp*ss*ct/m) + (q*sp/ct + r*cp/ct)*(-xi*(-sp*ss - st*cp*cs)/m + zeta*(q*cp - r*sp)*cp*cs*ct/m - zeta*(-sp*cs + ss*st*cp)*(q*sp/ct + r*cp/ct)/m - zeta*(sp*st*cs - ss*cp)*(p + q*sp*tt + r*cp*tt)/m) + (p + q*sp*tt + r*cp*tt)*(-xi*(sp*ss*st + cp*cs)/m + zeta*(-q*sp - r*cp)*ss*cp*ct/m - zeta*(q*cp - r*sp)*sp*ss*ct/m - zeta*(-sp*ss - st*cp*cs)*(q*cp/ct - r*sp/ct)/m - zeta*(-sp*cs + ss*st*cp)*(p + q*sp*tt + r*cp*tt)/m - zeta*(q*sp/ct + r*cp/ct)*(sp*st*cs - ss*cp)/m - zeta*(q*cp*tt - r*sp*tt)*(sp*ss*st + cp*cs)/m) + p*r*(-Ix + Iz)*(-zeta*(-sp*ss - st*cp*cs)*sp/(m*ct) - zeta*(sp*ss*st + cp*cs)*sp*tt/m + zeta*ss*cp**2*ct/m)/Iy - q*r*zeta*(Iy - Iz)*(sp*ss*st + cp*cs)/(Ix*m),
                  xi*((q*cp - r*sp)*st*cp/m + (p + q*sp*tt + r*cp*tt)*sp*ct/m) + (q*cp - r*sp)*(xi*st*cp/m + zeta*(q*cp - r*sp)*cp*ct/m + zeta*(q*(tt**2 + 1)*sp + r*(tt**2 + 1)*cp)*sp*ct/m - zeta*(p + q*sp*tt + r*cp*tt)*sp*st/m) + (p + q*sp*tt + r*cp*tt)*(xi*sp*ct/m + zeta*(-q*sp - r*cp)*st*cp/m - zeta*(q*cp - r*sp)*sp*st/m + zeta*(q*cp*tt - r*sp*tt)*sp*ct/m + zeta*(p + q*sp*tt + r*cp*tt)*cp*ct/m) + p*r*(-Ix + Iz)*(zeta*sp**2*ct*tt/m + zeta*st*cp**2/m)/Iy + q*r*zeta*(Iy - Iz)*sp*ct/(Ix*m),
                  (q*cp - r*sp)*(q*sp*st/ct**2 + r*st*cp/ct**2) + (q*cp/ct - r*sp/ct)*(p + q*sp*tt + r*cp*tt) + p*r*(-Ix + Iz)*sp/(Iy*ct)])
    # u = alpha + beta * v
    try:
        d_inv = np.linalg.inv(delta)
        alpha = -d_inv @ b
        beta = d_inv
        u_bar = alpha + beta @ v
        return u_bar
    except np.linalg.LinAlgError:
        print("Matrix Non-Invertible!")
        return [0,0,0,0]

def Feedback_Linearization_Zero_Dynamics(v, x_state, m, Ix, Iy, Iz):
    x, y, z, psi, theta, phi, x_dot, y_dot, z_dot, p, q, r = x_state
    g = 9.81

    cs, ct, cp = np.cos(x_state[3:6])

    delta = np.array([[-ct*cp/m, 0, 0, 0],
                      [0, 1/Ix, 0, 0],
                      [0, 0, 1/Iy, 0],
                      [0, 0, 0, 1/Iz]])

    b = np.array([g,
                  q*r*(Iy-Iz)/Ix,
                  p*r*(Iz-Ix)/Iy,
                  p*q*(Ix-Iy)/Iz])

    try:
        d_inv = np.linalg.inv(delta)
        alpha = -d_inv @ b
        beta = d_inv
        u = alpha + beta @ v
        return u
    except np.linalg.LinAlgError:
        print("Matrix Non-Invertible!")
        return [0,0,0,0]
