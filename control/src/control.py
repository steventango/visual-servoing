import numpy as np
from wam import WAM

class ControlMethod:
    def __init__(self, args):
        pass

    def initialize(self, wam: WAM, control_node):
        if self.initialized:
            return

    def get_action(self, state: np.ndarray, position: np.ndarray):
      raise NotImplementedError()
