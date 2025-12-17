
import numpy as np
from config import SimConfig as C

def pathloss_gain(d):
    d = max(d, 1.0)
    return C.PL_REF * (d ** (-C.PL_EXP))

def cascaded_gain(bs, ris, user):
    g1 = pathloss_gain(np.linalg.norm(bs - ris))
    g2 = pathloss_gain(np.linalg.norm(ris - user))
    return g1 * g2

def bs_to_ris_gain(bs, ris):
    return pathloss_gain(np.linalg.norm(bs - ris))

def ris_to_uav_gain(ris, uav):
    return pathloss_gain(np.linalg.norm(ris - uav))

def bs_to_ris_element_gains(bs, ris, N):
    base = pathloss_gain(np.linalg.norm(bs - ris))
    return np.full(N, base / max(N,1))

def ris_to_point_element_gains(ris, point, N):
    base = pathloss_gain(np.linalg.norm(ris - point))
    return np.full(N, base / max(N,1))
