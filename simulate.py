
import numpy as np
import matplotlib.pyplot as plt
from config import SimConfig as C
from optimizer import optimize, theoretical_upper_bound

def run_vs_ris_elements(elems_list):
    res = {"joint": [], "fixed_xi_traj": [], "static_ris": [], "upper": []}
    for N in elems_list:
        r_joint = optimize(sim_seed=0, N_elems=N, scheme="joint")["rate"]
        r_fix   = optimize(sim_seed=0, N_elems=N, scheme="fixed_xi_traj")["rate"]
        r_stat  = optimize(sim_seed=0, N_elems=N, scheme="static_ris")["rate"]
        r_up    = theoretical_upper_bound(N, C.P_TX_DBM, C.NUM_R_USERS, C.NUM_T_USERS)
        res["joint"].append(r_joint)
        res["fixed_xi_traj"].append(r_fix)
        res["static_ris"].append(r_stat)
        res["upper"].append(r_up)
    return res

def run_vs_tx_power(p_dbm_list):
    res = {"joint": [], "fixed_xi_traj": [], "static_ris": [], "upper": []}
    for p in p_dbm_list:
        r_joint = optimize(sim_seed=1, P_tx_dbm=p, scheme="joint")["rate"]
        r_fix   = optimize(sim_seed=1, P_tx_dbm=p, scheme="fixed_xi_traj")["rate"]
        r_stat  = optimize(sim_seed=1, P_tx_dbm=p, scheme="static_ris")["rate"]
        r_up    = theoretical_upper_bound(C.N_RIS, p, C.NUM_R_USERS, C.NUM_T_USERS)
        res["joint"].append(r_joint)
        res["fixed_xi_traj"].append(r_fix)
        res["static_ris"].append(r_stat)
        res["upper"].append(r_up)
    return res

def run_vs_user_count(user_counts):
    res = {"joint": [], "fixed_xi_traj": [], "static_ris": [], "upper": []}
    for m in user_counts:
        num_R = m//2
        num_T = m - num_R
        r_joint = optimize(sim_seed=2, num_R=num_R, num_T=num_T, scheme="joint")["rate"]
        r_fix   = optimize(sim_seed=2, num_R=num_R, num_T=num_T, scheme="fixed_xi_traj")["rate"]
        r_stat  = optimize(sim_seed=2, num_R=num_R, num_T=num_T, scheme="static_ris")["rate"]
        r_up    = theoretical_upper_bound(C.N_RIS, C.P_TX_DBM, num_R, num_T)
        res["joint"].append(r_joint)
        res["fixed_xi_traj"].append(r_fix)
        res["static_ris"].append(r_stat)
        res["upper"].append(r_up)
    return res

def plot_curve(x, res, xlabel, title, outfile):
    plt.figure()
    plt.plot(x, res["upper"], marker="^", label="理论上界")
    plt.plot(x, res["joint"], marker="o", label="动态分区+联合优化")
    plt.plot(x, res["fixed_xi_traj"], marker="s", label="固定分区+轨迹/相移优化")
    plt.plot(x, res["static_ris"], marker="d", label="静态RIS+固定轨迹")
    plt.xlabel(xlabel)
    plt.ylabel("系统加权和速率 (bit/s/Hz)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()
