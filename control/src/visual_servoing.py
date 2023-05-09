from typing import List
from control import ControlMethod
import numpy as np


class VisualServoing(ControlMethod):
    def __init__(self, args: List[str]):
        super().__init__(args)  


    def get_action(self, state: np.ndarray):
        return np.zeros(7)
