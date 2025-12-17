
import numpy as np
from config import SimConfig as C, dbm_to_w
from geometry import dist, project_speed
from channels import cascaded_gain
from ris import StarRIS
from uav import UAV

def random_users(num_R, num_T, area, rng):
    ax, ay = area
    R = []
    T = []
    for _ in range(num_R):
        R.append(np.array([rng.uniform(5, ax-5), rng.uniform(+10, ay/2), 0.0]))
    for _ in range(num_T):
        T.append(np.array([rng.uniform(5, ax-5), rng.uniform(-ay/2, -10), 0.0]))
    return np.array(R), np.array(T)

def noise_power_W():
    return dbm_to_w(C.NOISE_DBM) * (C.BW / 1.0)

def weighted_sum_rate(ris, bs_pos, uav_pos, users_R, users_T, p_tx_W, weights):
    terms = ris.effective_snr_terms(bs_pos, uav_pos, users_R, users_T, p_tx_W)
    N0 = noise_power_W()
    rates = [w * np.log2(1.0 + s / N0) for w, s in zip(weights, terms)]
    return float(np.sum(rates))

def objective_with_energy(ris, bs_pos, uav, users_R, users_T, p_tx_W, weights):
    N0 = noise_power_W()
    terms = ris.effective_snr_terms(bs_pos, uav.pos, users_R, users_T, p_tx_W)
    rate = np.sum([w * np.log2(1.0 + s/N0) for w, s in zip(weights, terms)])
    margin = uav.energy - C.E_SAFE
    penalty = 0.0 if margin >= 0 else abs(margin) * C.LAMBDA_E
    return float(rate - penalty)

def fixed_phase_alignment(ris):
    if ris.N_hz > 0:
        ris.phase = np.zeros_like(ris.phase)

def optimize(sim_seed=0, N_elems=None, P_tx_dbm=None, num_R=None, num_T=None, scheme="joint"):
    rng = np.random.default_rng(C.RNG_SEED + sim_seed)
    N_elems = int(N_elems if N_elems is not None else C.N_RIS)
    P_tx_dbm = float(P_tx_dbm if P_tx_dbm is not None else C.P_TX_DBM)
    num_R = int(num_R if num_R is not None else C.NUM_R_USERS)
    num_T = int(num_T if num_T is not None else C.NUM_T_USERS)

    bs = C.BS_POS.copy()
    start = np.array([10.0, +20.0, C.UAV_ALT])
    goal  = np.array([50.0, -20.0, C.UAV_ALT])
    uav = UAV(start, goal)

    users_R, users_T = random_users(num_R, num_T, C.AREA_SIZE, rng)
    weights = np.ones(num_R + num_T, dtype=float)

    p_tx_W = dbm_to_w(P_tx_dbm)
    ris = StarRIS(N=N_elems, xi=C.XI_INIT)
    fixed_phase_alignment(ris)

    optimize_xi = (scheme == "joint")
    optimize_traj = (scheme in ["joint", "fixed_xi_traj"])
    optimize_phase = (scheme in ["joint", "fixed_xi_traj"])
    static_ris = (scheme == "static_ris")

    if static_ris:
        uav.pos = (start + goal) / 2.0
        uav.traj = [uav.pos.copy()] * (C.T + 1)

    prev_val = -1e9
    history = []
    for it in range(C.MAX_OUTER_ITERS):
        # 1) Trajectory update
        if optimize_traj:
            cand_dirs = np.array([[+1,0,0], [-1,0,0], [0,+1,0], [0,-1,0], [+1,+1,0], [+1,-1,0], [-1,+1,0], [-1,-1,0], [0,0,0]], float)
            best_pos = uav.pos.copy()
            best_val = -1e9
            for d in cand_dirs:
                target = uav.pos + C.TRAJ_STEP * d / (np.linalg.norm(d)+1e-9)
                target[2] = C.UAV_ALT
                if scheme == "joint":
                    val = objective_with_energy(ris, bs, uav, users_R, users_T, p_tx_W, weights)
                else:
                    val = weighted_sum_rate(ris, bs, target, users_R, users_T, p_tx_W, weights)
                if val > best_val:
                    best_val, best_pos = val, target
            uav.step_to(best_pos)

        # 2) Phase optimization
        if optimize_phase and ris.N_hz > 0:
            step = C.PHASE_STEP0
            for _ in range(C.PHASE_INNER_ITERS):
                _ = ris.gradient_phase_step(bs, uav.pos, users_R, users_T, p_tx_W, weights, noise_power_W(), step)
                step *= 0.7

        # 3) Xi (partition) optimization
        if optimize_xi:
            if scheme == "joint":
                candidates = [0.1, 0.3, 0.5, 0.7, 0.9]
                best_xi = ris.xi
                best_val = objective_with_energy(ris, bs, uav, users_R, users_T, p_tx_W, weights)
                for xi_try in candidates:
                    ris.repartition(float(np.clip(xi_try, 0.0, 1.0)))
                    val = objective_with_energy(ris, bs, uav, users_R, users_T, p_tx_W, weights)
                    if val > best_val:
                        best_val = val
                        best_xi = ris.xi
                ris.repartition(best_xi)
            else:
                step = C.XI_STEP0
                for _ in range(3):
                    base = weighted_sum_rate(ris, bs, uav.pos, users_R, users_T, p_tx_W, weights)
                    xi_try = float(np.clip(ris.xi + (np.random.uniform(-1,1))*step, 0.0, 1.0))
                    old_xi = ris.xi
                    ris.repartition(xi_try)
                    val = weighted_sum_rate(ris, bs, uav.pos, users_R, users_T, p_tx_W, weights)
                    if val < base:
                        ris.repartition(old_xi)
                    step *= 0.6

        # 4) Energy accounting
        harvested = ris.energy_harvested(bs, uav.pos, uav.pos, p_tx_W) * C.DT
        uav.harvest_energy(harvested)
        if len(uav.traj) >= 2:
            spd = np.linalg.norm(uav.traj[-1] - uav.traj[-2]) / max(C.DT,1e-9)
        else:
            spd = 0.0
        uav.consume_energy(speed_factor=1.0 + 0.02*spd)

        # 5) Objective value (pure rate for reporting)
        val_rate = weighted_sum_rate(ris, bs, uav.pos, users_R, users_T, p_tx_W, weights)
        history.append(val_rate)

        # 6) Convergence / feasibility checks
        if not uav.feasible():
            break
        if abs(val_rate - prev_val) <= C.CONV_TOL:
            break
        prev_val = val_rate

    final_val = history[-1] if history else -1e9
    return dict(rate=final_val, ris=ris, uav=uav, users_R=users_R, users_T=users_T, hist=np.array(history))

def theoretical_upper_bound(N_elems, P_tx_dbm, num_R, num_T):
    rng = np.random.default_rng(C.RNG_SEED + 123)
    bs = C.BS_POS.copy()
    uav = np.array([30.0, 0.0, C.UAV_ALT])
    users_R, users_T = random_users(num_R, num_T, C.AREA_SIZE, rng)
    p_tx_W = dbm_to_w(P_tx_dbm)
    N0 = noise_power_W()
    def pl(d): return C.PL_REF * (max(d,1.0) ** (-C.PL_EXP))
    avg_g = 0.0
    all_users = np.vstack([users_R, users_T])
    for u in all_users:
        g1 = pl(np.linalg.norm(bs - uav))
        g2 = pl(np.linalg.norm(uav - u))
        avg_g += g1 * g2
    avg_g /= len(all_users)
    eff_power = p_tx_W * (N_elems ** 2) * avg_g
    rates = [np.log2(1.0 + eff_power/N0) for _ in range(len(all_users))]
    return float(np.sum(rates))
