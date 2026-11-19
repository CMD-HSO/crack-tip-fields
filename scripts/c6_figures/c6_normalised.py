"""c6_normalised.py - plastic zone of KH and LKH normalised with (K_I/tau_0)^2 and (K_I/tau_0)^4 (case C6, model R = 4 mm).
The size of the zone scales with (K_I/tau_0)^2 (length) and (K_I/tau_0)^4 (area), so that the models KH and LKH, which differ in tau_0, can be compared.
Three thresholds of the total slip (plastic if the sum over the systems of |gamma| exceeds the threshold in at least one integration point).
Needs results/analysis_max1e-4, analysis_max1e-3 and analysis_max1e-2 (summary.csv) of the runs, written by
    python c6_analyze.py --dir <run>/results --crit max --thr <thr> --analysis-dir analysis_max<thr> --no-plots
Usage: python c6_normalised.py <folder with the C6_run* directories> <output pdf>"""
import os, sys
import numpy as np
import c6_style
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

B, OUT = sys.argv[1], sys.argv[2]
RUNS = {"KH": "C6_run3_R4mm_KH", "LKH": "C6_run4_R4mm_LKH"}
TAU0 = {"KH": 168.89, "LKH": 254.96}          # MPa, first constant of the slip rate law in the material cards
STYLE = {"KH": "-", "LKH": "--"}
THR = (("1e-4", "o", r"$10^{-4}$"), ("1e-3", "s", r"$10^{-3}$"), ("1e-2", "^", r"$10^{-2}$"))
FRAMES = {"KH": slice(2, 8), "LKH": slice(2, 12)}  # KH: frames 3..8 (K_I = 8 ... 21.3); LKH: frames 3..12 (up to 32), so that both cover K_I/tau_0 = 1.5 ... 4 sqrt(mm);
                                                   # the zone of the first frames consists of too few elements
SQ = np.sqrt(1000.0)                             # K_I in MPa sqrt(m) -> MPa sqrt(mm), so that K_I/tau_0 is in sqrt(mm) and the ratios are dimensionless

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
for lab, run in RUNS.items():
    for thr, mk, _ in THR:
        s = np.genfromtxt(os.path.join(B, run, "results", "analysis_max" + thr, "summary.csv"), delimiter=",", names=True)
        x = s["K_I"][FRAMES[lab]] * SQ / TAU0[lab]
        kw = dict(color=c6_style.COL[lab], ls=STYLE[lab], lw=1.1, marker=mk, ms=4.0, mfc=c6_style.COL[lab] if thr == "1e-4" else "white", mew=0.8)
        ax[0].plot(x, s["r_max"][FRAMES[lab]] / x ** 2, **kw)
        ax[1].plot(x, s["area"][FRAMES[lab]] / x ** 4, **kw)
for a, t, yl in zip(ax, ("(a)", "(b)"), (r"$r_{\max}\,/\,(K_I/\tau_0)^2$ [--]", r"$A\,/\,(K_I/\tau_0)^4$ [--]")):
    a.set_yscale("log"); a.set_xlim(0.5, 4.3); a.grid(alpha=.3, which="both")
    a.set_xlabel(r"$K_I/\tau_0$ [$\sqrt{\mathrm{mm}}$]"); a.set_ylabel(yl)
    a.text(0.04, 0.95, t, transform=a.transAxes, va="top")
ax[0].set_ylim(2.2e-3, 4.5e-2); ax[1].set_ylim(3.0e-6, 1.5e-3)
ax[0].set_yticks([0.004, 0.006, 0.01, 0.02, 0.04]); ax[0].set_yticklabels(["0.004", "0.006", "0.01", "0.02", "0.04"]); ax[0].minorticks_off()
h = [Line2D([], [], color=c6_style.COL[m], ls=STYLE[m], lw=1.4, label=m) for m in ("KH", "LKH")]
h += [Line2D([], [], color="0.35", ls="none", marker=mk, ms=4.0, mfc="0.35" if thr == "1e-4" else "white", mew=0.8, label=lab) for thr, mk, lab in THR]
ax[0].legend(handles=h[:2], loc="lower right", frameon=False, fontsize=8.5)
ax[1].legend(handles=h[2:], title="threshold", loc="lower left", frameon=False, fontsize=8.5, title_fontsize=8.5, ncol=3, columnspacing=1.0, handletextpad=0.3)
fig.tight_layout(); fig.savefig(OUT, dpi=300)
print("saved", OUT)
