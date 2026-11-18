"""c6_backstress.py - largest resolved back stress max_alpha |tau_kin,alpha| along the grain boundary ahead of the tip (elements touching the boundary), model R = 4 mm.
Grain 1: elements above, grain 2: below the grain boundary.  KH: tau_kin = Omega (saturation c/d); LKH: resolved from the back stress tensor.
Usage: python c6_backstress.py <folder with the C6_run* directories> <output png>"""
import os, sys
import numpy as np
import c6_style
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
B, OUT = sys.argv[1], sys.argv[2]
cases = [("KH", "C6_run3_R4mm_KH", "-"), ("LKH", "C6_run4_R4mm_LKH", "--")]
loads = [(3, 8.0, "#e9a3c9"), (6, 16.0, "#c51b7d"), (8, 21.333, "#4d004b")]
n = 5184
def rp_gb(B_, d_, N_):
    """length of the plastic zone along the grain boundary r_p (criterion: max slip > 1e-4), from analysis_max1e-4/summary.csv"""
    s_ = np.genfromtxt(os.path.join(B_, d_, "results", "analysis_max1e-4", "summary.csv"), delimiter=",", names=True)
    return float(s_["r_p_GB"][N_ - 1])


fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
res = {}
for N, K, col in loads:
    for lab, d, st in cases:
        colm = c6_style.shade(lab, K)
        a = np.genfromtxt(os.path.join(B, d, "results", "export", "elements_frame%02d.csv" % N), delimiter=",", names=True)
        c = np.column_stack([a["x%d" % i][:n] if j == 0 else a["y%d" % i][:n] for i in range(1, 5) for j in (0, 1)])
        gb = (np.abs(c[:, 1::2]).min(axis=1) < 1e-9) & (a["xc"][:n] > 0)
        BS = np.abs(np.array([a["bs%d" % i][:n] for i in range(1, 13)]))
        for g, (s, axx) in enumerate(((1, ax[0]), (-1, ax[1]))):
            m = gb & (s * a["yc"][:n] > 0); o = np.argsort(a["xc"][:n][m]); idx = np.where(m)[0][o]
            x = a["xc"][:n][idx]; v = BS[:, idx].max(axis=0)
            k = (x >= 0.004) & (x <= 0.1)
            axx.plot(x[k], v[k], st, color=colm, lw=1.4)
            axx.plot([rp_gb(B, d, N)], [0], marker="^", ms=4.5, mfc=colm if lab == "KH" else "white", mec=colm, ls="none", transform=axx.get_xaxis_transform(), clip_on=False, zorder=5)
            res[(K, lab, g + 1)] = tuple(np.interp([0.005, 0.01, 0.02, 0.05], x, v))
for a, t in zip(ax, ("(a) grain 1", "(b) grain 2")):
    a.set_xlim(0, 0.1); a.set_xlabel(r"$x_1$ [mm]"); a.grid(alpha=.3)
    a.text(0.97, 0.95, t, transform=a.transAxes, va="top", ha="right")
ax[0].set_ylabel(r"$\max_\alpha|\tau^{\mathrm{kin}}_\alpha|$ [MPa]")
c6_style.k_legend(ax[1], loc="upper right", bbox_to_anchor=(1.0, 0.86), fontsize=8, columnspacing=0.7, handlelength=1.4, labelspacing=0.25)
fig.tight_layout(); fig.savefig(OUT, dpi=300)
print("max_a |tau_kin| [MPa] at x = 0.005, 0.01, 0.02, 0.05 mm")
for (K, lab, g), v in sorted(res.items()):
    print("K=%5.2f %-3s grain %d:" % (K, lab, g), np.round(v, 0))
