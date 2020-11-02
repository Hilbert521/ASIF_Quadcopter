from gui import GUI
import threading
import scipy.integrate
import multiprocessing
import time

class Simulator():
    def __init__(self, dt):
        self.quads = []
        self.t = time.time()
        self.quad_groups=[]
        self.quad_list = []
        self.threads = []
        self.GUI = None
        self.dt = 0.005
        self.queue = multiprocessing.Queue()
        self.run = True

    def add_quad(self, quad, controller=None):
        integrator = scipy.integrate.ode(self.quads[-1].state_dot).set_integrator('dopri5', atol=1e-3, rtol=1e-6)
        self.quad_list.append(quad)
        self.quads.append({"quad": quad, "integrator": integrator, "controller": controller})

    def add_quad_group(self, quads, controller):
        integrators = [scipy.integrate.ode(quad.state_dot).set_integrator('dopri5', atol=1e-3, rtol=1e-6) for quad in quads]
        for quad in quads:
            self.quad_list.append(quad)
        self.quad_groups.append({"quads": quads, "integrators": integrators, "controller": controller})

    def begin(self):
        # Get controller inputs from current states
        # Run integrator threads, update quadcopter states
        # Wait until maximum of threads finished or dt hit
        # Update gui
        # Repeat
        self.run = True
        for i in range(0, len(self.quads)):
            self.threads.append(threading.Thread(target=self.simulate_quad, args=(i,)))
        for i in range(0, len(self.quad_groups)):
            self.threads.append(threading.Thread(target=self.simulate_quad_group, args=(i,)))
        for thread in self.threads:
            thread.start()
        self.GUI = GUI(quads=self.quad_list, queue=self.queue)

    def stop(self):
        self.run = False

    def simulate_quad(self, i):
        self.quads[i]["integrator"].set_initial_value(self.quads[i].get_state(), 0)
        while self.run:
            self.quads[i]["quad"].update(self.quads[i]["integrator"].integrate(self.quads[i]["integrator"].t + self.dt))
            time.sleep(0)

    def simulate_quad_group(self, i):
        for index, integrator in enumerate(self.quad_groups[i]["integrators"]):
            integrator.set_initial_value(self.quad_groups[i]['quads'][index].get_state(), 0)
        while self.run:
            if time.time()-self.t >= self.dt:
                inputs = self.quad_groups[i]['controller'].generate_inputs([quad.get_state() for quad in self.quad_groups[i]['quads']])
                for index, integrator in enumerate(self.quad_groups[i]["integrators"]):
                    integrator.set_f_params(inputs[index],)
                    self.quad_groups[i]['quads'][index].update(integrator.integrate(integrator.t + self.dt))
                print(time.time()-self.t)
                self.t = time.time()
            time.sleep(0)
