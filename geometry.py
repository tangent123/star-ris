
import numpy as np

def dist(a, b):
    return float(np.linalg.norm(a - b))

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def project_speed(prev, nxt, vmax, dt):
    delta = nxt - prev
    L = np.linalg.norm(delta)
    maxL = vmax*dt
    if L <= 1e-9:
        return nxt
    if L <= maxL:
        return nxt
    return prev + delta * (maxL / L)
