from quads.YZQuad import YZQuad
from controllers.MultiYZQuadController import MultiYZQuadController
from simulator import Simulator
import multiprocessing

'''
Simulation Parameters
'''
dt = 0.005

'''
Quadcopter Parameters
'''



# Make objects for quadcopter
sim = Simulator(dt=dt)

quad1 = YZQuad(initial_state=[0, 0, 0, 0, 0, 0])
quad2 = YZQuad(initial_state=[4, 4, 0, 0, 0, 0])
controller = MultiYZQuadController()

sim.add_quad_group(quads=[quad1, quad2], controller=controller)
sim.begin()

"""# Make objects for quadcopter

sim = Simulator(dt=dt)
for i in range(2):
    quad1 = YZQuad(initial_state=[0, 0, 0, 0, 0, 0])
    quad2 = YZQuad(initial_state=[4, 4, 0, 0, 0, 0])
    controller = MultiYZQuadController()
    sim.add_quad_group(quads=[quad1, quad2], controller=controller)
sim.begin()"""
