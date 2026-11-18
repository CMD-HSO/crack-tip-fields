"""c6_active_systems.py - latent hardening (6.3): number of active slip systems and accumulated slip per system and grain, model R = 4 mm, K_I = 21.3.
Active: slip rate > rate_thr (set in the extraction) in at least one integration point.  Slip: |mean over the integration points of the net accumulated slip|.
Grain 1: y > 0, grain 2: y < 0.  Elements with 0.01 mm < r < 0.1 mm.
Usage: python c6_active_systems.py <folder with the C6_run* directories> <output png>"""
import os, sys
import numpy as np
import c6_style
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
B, OUT = sys.argv[1], sys.argv[2]
cases = [("KH", "C6_run3_R4mm_KH"), ("LKH", "C6_run4_R4mm_LKH")]
FRAME, R1, R2, n, COL = 8, 0.01, 0.1, 5184, "#d98200"
E = {l: np.genfromtxt(os.path.join(B, d, "results", "export", "elements_frame%02d.csv" % FRAME), delimiter=",", names=True) for l, d in cases}
fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.9))
w = 0.38
for col, (gname, sgn) in enumerate((("grain 1", 1), ("grain 2", -1))):
    ah, ab = axs[0, col], axs[1, col]
    top = 0.0
    for k, (lab, d) in enumerate(cases):
        a = E[lab]; r = np.hypot(a["xc"], a["yc"])[:n]
        m = (sgn * a["yc"][:n] > 0) & (r > R1) & (r < R2)
        na = a["n_act_max"][:n][m]
        h = np.array([(na == i).sum() for i in range(13)]) / m.sum() * 100
        kw = dict(facecolor=c6_style.COL[lab] if lab == "KH" else "white", edgecolor=c6_style.COL[lab], hatch="" if lab == "KH" else "////", linewidth=0.8)
        nz = h > 0          # no zero-height bars: the hatch of an empty bar produces stray lines in some PDF viewers
        ah.bar((np.arange(13) + (k - 0.5) * w)[nz], h[nz], w, label="%s (mean %.1f)" % (lab, na.mean()), **kw)
        g = np.array([np.abs(a["g%d" % i][:n][m]).mean() for i in range(1, 13)]) * 1e3
        nz = g > 1e-9
        ab.bar((np.arange(1, 13) + (k - 0.5) * w)[nz], g[nz], w, label="%s" % lab, **kw)
        top = max(top, g.max())
    ah.set_xticks(range(13)); ah.set_xlabel("number of active slip systems"); ah.grid(alpha=.3, axis="y")
    ah.set_ylabel("share of the elements [%]" if col == 0 else ""); ah.set_ylim(0, 1.35 * ah.get_ylim()[1] / 1.0)
    ah.text(0.04, 0.95, "(%s) %s" % ("ab"[col], gname), transform=ah.transAxes, va="top")
    ah.legend(loc="upper right", frameon=False, fontsize=8.5)
    ab.set_xticks(range(1, 13)); ab.set_xlabel(r"slip system $\alpha$"); ab.grid(alpha=.3, axis="y"); ab.set_ylim(0, 1.3 * top)
    ab.set_ylabel(r"$|\gamma_\alpha|$ [$10^{-3}$]" if col == 0 else "")
    ab.text(0.04, 0.95, "(%s) %s" % ("cd"[col], gname), transform=ab.transAxes, va="top")
    if col == 0:
        ab.legend(loc="upper left", bbox_to_anchor=(0.0, 0.84), frameon=False, fontsize=8.5)
fig.tight_layout(); fig.savefig(OUT, dpi=300)
