"""c6_kfield_paper.py - paper figure: deviation of the FE stress and displacement at r = 1 mm from the K-field (data: c6_kfield.csv).
The deviation is the maximum (default) or the root mean square (rms) over the nodes of the ring, relative to the rms of the K-field.
Usage: python c6_kfield_paper.py <c6_kfield.csv> <output pdf or png> [max|rms]"""
import sys, csv
import numpy as np
import c6_style
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
rows = list(csv.DictReader(open(sys.argv[1])))
STAT = sys.argv[3] if len(sys.argv) > 3 else "max"
STATNAME = "maximum" if STAT == "max" else "rms"
runs =[("R = 1 mm", "#1b6ca8", "o"), ("R = 4 mm", "#d98200", "s"), ("R = 10 mm", "#2e8b57", "^")]
SC = 0.62
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
for lab, c, mk in runs:
    for tag, st in (("KH", "-"), ("LKH", "--")):
        sel = [r for r in rows if r["R"] == lab and r["variant"] == tag and float(r["K_I"]) <= 21.4]
        k = [float(r["K_I"]) for r in sel]
        cc = c6_style.COL[tag]
        kw = dict(color=cc, lw=2.6 * SC, marker=mk, ms=3.2, alpha=.7, mfc="none" if tag == "LKH" else cc)
        ax[0].plot(k, [100 * float(r["boundary_stress_dev_" + STAT]) for r in sel], st, **kw)
        if lab != "R = 1 mm":
            ax[1].plot(k, [100 * float(r["ring_u_dev_" + STAT]) for r in sel], st, **kw)
ax[0].set_ylabel("stress, %s deviation [%%]" % STATNAME); ax[1].set_ylabel("displacement, %s deviation [%%]" % STATNAME)
for a, t in zip(ax, ("(a)", "(b)")):
    a.set_xlabel(r"prescribed $K_I$ [MPa$\sqrt{\mathrm{m}}$]"); a.grid(alpha=.3)
    a.text(0.04, 0.93, t, transform=a.transAxes, va="top")
from matplotlib.lines import Line2D
hh = [Line2D([], [], color=c6_style.COL["KH"], lw=1.6, label="KH"), Line2D([], [], color=c6_style.COL["LKH"], lw=1.6, ls="--", label="LKH")]
hh += [Line2D([], [], color="0.3", marker=m_, ls="none", ms=4, label="$R=%s$ mm" % l_.split()[2]) for l_, c_, m_ in runs]
ax[0].legend(handles=hh, loc="upper left", bbox_to_anchor=(0.12, 0.98), frameon=False)
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=300)
