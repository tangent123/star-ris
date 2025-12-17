
import numpy as np
from config import SimConfig as C
from geometry import project_speed

class UAV:
    def __init__(self, start_pos, goal_pos):
        self.start = np.array(start_pos, dtype=float)
        self.goal = np.array(goal_pos, dtype=float)
        self.pos = np.array(start_pos, dtype=float)
        self.energy = 0.6 * C.E_CAP
        self.traj = [self.pos.copy()]

    def step_to(self, target_pos):
        new_pos = project_speed(self.pos, target_pos, C.VMAX, C.DT)
        self.pos = new_pos
        self.traj.append(self.pos.copy())

    def consume_energy(self, speed_factor=1.0):
        cons = (C.P_FLY * speed_factor + C.P_RIS) * C.DT
        self.energy = max(0.0, self.energy - cons)

    def harvest_energy(self, harvested_J):
        self.energy = min(C.E_CAP, self.energy + harvested_J)

    def feasible(self):
        return self.energy >= C.E_SAFE
