from abc import ABC

'''
Abstract class/interface for a quadcopter model
'''
class Quadcopter(ABC):

    # Return x' for the simulator to integrate
    def state_dot(self, t, state):
        pass

    # Retrieve updated state from simulator
    def update(self, new_state):
        pass

    # Return position [x, y, z]
    def get_position(self):
        pass

    # Return attiude [phi, theta psi]
    def get_attitude(self):
        pass

    def get_state(self):
        pass