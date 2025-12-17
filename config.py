
import numpy as np

class SimConfig:
    # Geometry
    AREA_SIZE = (60.0, 60.0)  # meters (X,Y) ground area
    UAV_ALT = 80.0            # meters (fixed altitude)
    BS_POS = np.array([30.0, 0.0, 0.0])  # ground RF source (x,y,0)
    # Users configuration
    NUM_R_USERS = 3  # reflection-side users
    NUM_T_USERS = 3  # transmission-side users
    # Time/slots
    T = 4            # time slots
    DT = 1.0         # seconds per slot
    VMAX = 15.0      # m/s
    # RIS/STAR-RIS
    N_RIS = 20       # default number of programmable elements
    XI_INIT = 0.5    # initial fraction for TZ (energy transmission zone)
    PHASE_INIT = 0.0 # initial phase angle (radians)
    # Power / noise
    P_TX_DBM = 40.0      # dBm
    BW = 10e6            # 10 MHz
    NOISE_DBM = -80.0    # dBm
    # Path loss model
    PL_REF = 1e-3        # reference channel gain at 1m (linear)
    PL_EXP = 2.4         # path loss exponent (alpha)
    # Energy
    ETA_EH = 0.6         # RF-DC efficiency
    P_FLY = 120.0        # W (flight power baseline)
    P_RIS = 10.0         # W (RIS control + electronics)
    E_CAP = 2.0e4        # Joules (battery capacity ~ 5.5 Wh)
    E_SAFE = 2.0e3        # Joules (reserve)
    LAMBDA_E = 1e-4       # energy penalty weight for joint objective
    # Optimizer
    MAX_OUTER_ITERS = 8
    PHASE_INNER_ITERS = 6
    TRAJ_STEP = 2.0      # meters gradient step per slot
    XI_STEP0 = 0.15
    PHASE_STEP0 = 0.5
    CONV_TOL = 1e-3
    RNG_SEED = 42

def dbm_to_w(d):
    return 10**((d-30)/10.0)

def w_to_dbm(w):
    return 10*np.log10(w)+30
