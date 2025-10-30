# bodies.py
from collections import deque
from typing import Tuple, List
import numpy as np


TRAIL_LENGTH = 150

class Body:
    """Representing body and its properties"""

    def __init__(self,
                 mass: float,
                 position: List[float] | np.ndarray,
                 velocity: List[float] | np.ndarray,
                 color: Tuple[int, int, int],
                 radius: float,
                 name: str = ""):
        """Creates body with initial property"""
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.color = color
        self.radius = radius
        self.name = name
        
        # Internal state variables for the simulation
        self.acceleration = np.array([0.0, 0.0], dtype=float)
        self.trail = deque(maxlen=TRAIL_LENGTH)
        self.merge_flash_timer = 0