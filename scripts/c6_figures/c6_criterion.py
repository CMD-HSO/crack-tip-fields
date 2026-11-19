"""c6_criterion.py - effect of the size of the plastic zone relative to the model radius, r_max/R, on the accuracy of the small model (case C6, R = 1 mm).
(a) deviation of the stress at the model boundary (r = 1 mm) from the prescribed K-field (maximum over the nodes, relative to the rms of the K-field),
(b) difference of the near-tip results of the model R = 1 mm and the model R = 4 mm (same mesh at the tip): area of the plastic zone and crack-tip opening displacement.
Usage: python c6_criterion.py <folder with the C6_run* directories> <repository root> <output pdf>"""
import os, sys, csv
import numpy as np
B, REPO, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.argv = ["x", B, REPO, os.path.dirname(os.path.abspath(OUT))]
sys.path.insert(0, os.path.join(REPO, "scripts", "c6_figures"))
import c6_kfield as kf
import c6_style
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = {"KH": ("C6_run1_R1mm_KH", "C6_run3_R4mm_KH"), "LKH": ("C6_run2_R1mm_LKH", "C6_run4_R4mm_LKH")}
STYLE = {"KH": "-", "LKH": "--"}


def summary(d):
    return np.genfromtxt(os.path.join(B, d, "results", "analysis_max1e-4", "summary.csv"), delimiter=",", names=True)


def boundary_deviation(d, N, Kmp):
    A_ = os.path.join(B, d, "results", "export")
    nd = kf.boundary_nodes(A_, N, Kmp * np.sqrt(1000.0), 1.0)
    fe = np.array([n[1] for n in nd]); an = np.array([n[2] for n in nd])
    return 100 * np.sqrt(((fe - an) ** 2).sum(axis=1)).max() / kf.rms(an)


fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
res = {}
for lab, (d1, d4) in MODELS.items():
    s1, s4 = summary(d1), summary(d4)
    x = s1["r_max_over_R"]
    dev = np.array([boundary_deviation(d1, i + 1, s1["K_I"][i]) for i in range(len(s1))])
    col = c6_style.COL[lab]
    ax[0].plot(x, dev, STYLE[lab], color=col, lw=1.3, marker="o", ms=3.5, label=lab)
    n = min(len(s1), len(s4))
    for key, mk, nm in (("area", "s", "area $A$"), ("CTOD_fix", "^", "CTOD")):
        y = 100 * (s1[key][:n] / s4[key][:n] - 1)
        ax[1].plot(x[:n], y, STYLE[lab], color=col, lw=1.1, marker=mk, ms=3.5, mfc=col if key == "area" else "white", mew=0.8)
    res[lab] = (x, dev)
for t in (2.0, 5.0):
    ax[0].axhline(t, color="0.5", lw=0.6, ls=":")
ax[1].axhline(0, color="k", lw=0.5)
ax[0].set_ylabel(r"$\delta_\sigma$ at the boundary [%]")
ax[1].set_ylabel(r"difference to $R=4$ mm [%]")
for a, t in zip(ax, ("(a)", "(b)")):
    a.set_xlabel(r"$r_{\max}/R$ [--]"); a.grid(alpha=.3); a.set_xlim(0, 0.56)
    a.text(0.04 if t == "(a)" else 0.96, 0.95, t, transform=a.transAxes, va="top", ha="left" if t == "(a)" else "right")
ax[0].legend(loc="lower right", frameon=False, fontsize=8.5, bbox_to_anchor=(1.0, 0.0))
from matplotlib.lines import Line2D
ax[1].legend(handles=[Line2D([], [], color="0.35", ls="none", marker="s", ms=3.5, label="area $A$"),
                      Line2D([], [], color="0.35", ls="none", marker="^", ms=3.5, mfc="white", mew=0.8, label="CTOD")],
             loc="lower left", frameon=False, fontsize=8.5)
fig.tight_layout(); fig.savefig(OUT, dpi=300)
for lab in res:
    x, dev = res[lab]
    for tol in (2.0, 5.0):
        i = np.searchsorted(dev, tol)
        if 0 < i < len(dev):
            xi = x[i - 1] + (tol - dev[i - 1]) * (x[i] - x[i - 1]) / (dev[i] - dev[i - 1])
            print("%-3s deviation of the stress at the boundary reaches %.0f %% at r_max/R = %.2f" % (lab, tol, xi))
print("saved", OUT)
