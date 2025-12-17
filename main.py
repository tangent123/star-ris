
import os, csv, json
import numpy as np
from config import SimConfig as C
from simulate import run_vs_ris_elements, run_vs_tx_power, run_vs_user_count, plot_curve

outdir = "C:\\Users\唐剑\Downloads\star_ris_uav_sim\out"
os.makedirs(outdir, exist_ok=True)

elems = [10, 20, 30, 40, 55, 70]
res1 = run_vs_ris_elements(elems)
plot_curve(elems, res1, xlabel="STAR-RIS可编程单元数 N", title="系统和速率 vs STAR-RIS单元数", outfile=os.path.join(outdir,"fig_vs_N.png"))

p_dbm = [30, 35, 40, 45, 50]
res2 = run_vs_tx_power(p_dbm)
plot_curve(p_dbm, res2, xlabel="射频源发射功率 P_tx (dBm)", title="系统和速率 vs 发射功率", outfile=os.path.join(outdir,"fig_vs_P.png"))

users = [10, 20, 30, 40, 50]
res3 = run_vs_user_count(users)
plot_curve(users, res3, xlabel="用户数量 M", title="系统和速率 vs 用户数量", outfile=os.path.join(outdir,"fig_vs_M.png"))

def save_csv(x, res, name):
    csvfile = os.path.join(outdir, f"{name}.csv")
    with open(csvfile, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x", "理论上界", "动态分区+联合优化", "固定分区+轨迹/相移优化", "静态RIS+固定轨迹"])
        for i, xv in enumerate(x):
            w.writerow([xv, res["upper"][i], res["joint"][i], res["fixed_xi_traj"][i], res["static_ris"][i]])

save_csv(elems, res1, "vs_N")
save_csv(p_dbm, res2, "vs_P")
save_csv(users, res3, "vs_M")

with open(os.path.join(outdir, "RESULTS.json"), "w", encoding="utf-8") as f:
    json.dump({
        "figures": ["fig_vs_N.png", "fig_vs_P.png", "fig_vs_M.png"],
        "tables": ["vs_N.csv", "vs_P.csv", "vs_M.csv"]
    }, f, ensure_ascii=False, indent=2)

print("Done. Outputs in:", outdir)
