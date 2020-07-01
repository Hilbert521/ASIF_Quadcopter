import numpy as np
import math


def rotation_matrix(angles):
    """
    Calculates rotation matrix in ZYX form

    :param angles: [x_axis (theta), y_axis (phi), z_axis (gamma)]
    :return: Rotation matrix
    """

    ct, cp, cg = np.cos(angles)
    st, sp, sg = np.sin(angles)

    R_x = np.array([[1, 0, 0],
                    [0, ct, -st],
                    [0, st, ct]])
    R_y = np.array([[cp, 0, sp],
                    [0, 1, 0],
                    [-sp, 0, cp]])
    R_z = np.array([[cg, -sg, 0],
                    [sg, cg, 0],
                    [0, 0, 1]])
    return R_z @ R_y @ R_x


def wrap_angle(angle):
    """
    Constrain angle between [-pi, pi]
    :param angle: angle, in radians
    :return: Constrained angle
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def thrust(speed, pitch, diameter):
    """
    From http://www.electricrcaircraftguy.com/2013/09/propeller-static-dynamic-thrust-equation.html
    :param speed: RPM of the motor
    :param pitch: Propeller pitch (tilt), in inches
    :param diameter: Propeller diameter, in inches
    :return: Thrust of the propeller, in N
    """
    return 4.392e-8 * speed * math.pow(diameter, 3.5) / (math.sqrt(pitch)) * 4.23e-4 * speed * pitch
