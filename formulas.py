from sympy import symbols, simplify, Matrix
import sympy

solve_delta = False
solve_b = True

x, y, z, psi, theta, phi, xdot, ydot, zdot, zeta, xi, p, q, r, m, Ix, Iy, Iz = symbols('x, y, z, psi, theta, '
                                                                        'phi, xdot, ydot, zdot, '
                                                                        'zeta, xi, p, q, r, m, Ix, Iy, Iz', real=True)
variables = [x, y, z, psi, theta, phi, xdot, ydot, zdot, zeta, xi, p, q, r]
sp, ss, st = sympy.sin(phi), sympy.sin(psi), sympy.sin(theta)
cp, cs, ct = sympy.cos(phi), sympy.cos(psi), sympy.cos(theta)
tp, ts, tt = sympy.tan(phi), sympy.tan(psi), sympy.tan(theta)

x_bar = Matrix([[x, y, z, psi, theta, phi, xdot, ydot, zdot, zeta, xi, p, q]])
f_bar = Matrix([[xdot,
                 ydot,
                 zdot,
                 (q * sp / ct + r * cp / ct),
                 (q * cp - r * sp),
                 (p + q * sp * tt + r * cp * tt),
                 (-1/m * (sp*ss + cp*cs*st) * zeta),
                 (-1/m * (cs*sp - cp*ss*st) * zeta),
                 (-1/m * (cs*ct) * zeta),
                 xi,
                 0,
                 (q * r * (Iy - Iz)/Ix),
                 (p * r * (Iz - Ix)/Iy),
                 0]]).transpose() #(q * p * (Ix - Iy)/Iz) = 0
g_bar_1 = Matrix([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]).transpose()
g_bar_2 = Matrix([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1/Ix, 0, 0]]).transpose()
g_bar_3 = Matrix([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1/Iy, 0]]).transpose()
g_bar_4 = Matrix([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1/Iz]]).transpose()

if solve_delta:
    m = []
    for var in [x, y, z, psi]:
        row = []
        x_bar_grad = Matrix([[var.diff(v) for v in variables]])
        Lf_1 = (x_bar_grad * f_bar)
        Lf_2 = Matrix([[Lf_1.diff(v) for v in variables]]) * f_bar
        Lf_3 = Matrix([[Lf_2.diff(v) for v in variables]]) * f_bar
        for g_bar in [g_bar_1, g_bar_2, g_bar_3, g_bar_4]:
            Lg = Matrix([[Lf_3.diff(v) for v in variables]]) * g_bar
            row.append(simplify(Lg))
            print(row[-1])
        m.append(row)

    delta = Matrix(m)
print("----------------------------------------")
if solve_b:
    vec = []
    for var in [x, y, z, psi]:
        x_bar_grad = Matrix([[var.diff(v) for v in variables]])
        Lf_1 = (x_bar_grad * f_bar)
        Lf_2 = Matrix([[Lf_1.diff(v) for v in variables]]) * f_bar
        Lf_3 = Matrix([[Lf_2.diff(v) for v in variables]]) * f_bar
        Lf_4 = Matrix([[Lf_3.diff(v) for v in variables]]) * f_bar
        vec.append((Lf_4))
        print(vec[-1])