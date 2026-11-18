"""c6_zone_map.py - accumulated slip and outlines of the plastic zone (KH | LKH) for the model of R = 4 mm.
Plastic: max over the integration points of sum |gamma_s| > THR (net slip).
Usage: python c6_zone_map.py <folder with the C6_run* directories> <output png>"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.colors import LogNorm
B, OUT = sys.argv[1], sys.argv[2]
import c6_style
THR = 1e-4
cases = [("KH", "C6_run3_R4mm_KH"), ("LKH", "C6_run4_R4mm_LKH")]
loads = [(3, 8.0, "white", ":"), (6, 16.0, "white", "--"), (8, 21.33, "black", "-")]       # frame, K_I, outline colour; last = shown as map
LIM, ZOOM = 0.6, 0.06


def read(d, f):
    return np.genfromtxt(os.path.join(B, d, "results", "export", "elements_frame%02d.csv" % f), delimiter=",", names=True)


def outline(d, plastic):
    """edges that separate plastic from non-plastic elements (or lie at the model edge)"""
    cnt, seg = {}, {}
    for k in range(len(d)):
        c = [(d["x%d" % i][k], d["y%d" % i][k]) for i in (1, 2, 3, 4)]
        for a, b in zip(c, c[1:] + c[:1]):
            key = tuple(sorted((tuple(np.round(a, 7)), tuple(np.round(b, 7)))))
            cnt.setdefault(key, []).append(bool(plastic[k]))
            seg[key] = (a, b)
    return [seg[k] for k, v in cnt.items() if len(v) == 1 and v[0] or (len(v) == 2 and v[0] != v[1])]


fig = plt.figure(figsize=(7.0, 3.3), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.045])
axs = np.array([fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])])
cax = fig.add_subplot(gs[0, 2])
vmax = 1.0
for ax, (lab, d) in zip(axs, cases):
    E = read(d, 8)
    inside = (np.abs(E["xc"]) < 1.3) & (np.abs(E["yc"]) < 1.3)
    polys = [[(E["x%d" % i][k], E["y%d" % i][k]) for i in (1, 2, 3, 4)] for k in np.where(inside)[0]]
    g = np.abs(E["g_max"][inside]); pl = g > THR
    vmax = max(vmax, g.max())
    pc = PolyCollection(polys, array=np.where(pl, g, np.nan), cmap="viridis", norm=LogNorm(vmin=THR, vmax=10.0),
                        edgecolors="0.8", linewidths=0.15, rasterized=True)
    ax.add_collection(pc)
    ins = ax.inset_axes([0.04, 0.04, 0.34, 0.34])
    pc2 = PolyCollection(polys, array=np.where(pl, g, np.nan), cmap="viridis", norm=LogNorm(vmin=THR, vmax=10.0), edgecolors="0.6", linewidths=0.2, rasterized=True)
    ins.add_collection(pc2)
    for f, K, col, lsty in loads:
        Ef = read(d, f)
        segs = outline(Ef, np.abs(Ef["g_max"]) > THR)
        for a_ in (ax, ins):
            a_.add_collection(LineCollection(segs, colors=col, linestyles=lsty, linewidths=0.9 if a_ is ax else 0.7, label="$K_I=%.1f$" % K if a_ is ax else None))
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$ [mm]"); ax.set_ylabel(r"$x_2$ [mm]"); ax.text(0.04, 0.95, "(%s) %s" % ("ab"[axs.tolist().index(ax)], lab), transform=ax.transAxes, va="top", bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
    ins.set_xlim(-ZOOM, ZOOM); ins.set_ylim(-ZOOM, ZOOM); ins.set_aspect("equal"); ins.set_xticks([]); ins.set_yticks([])
    ax.indicate_inset_zoom(ins, edgecolor="k")
    if lab == "LKH":
        ax.set_ylabel(""); ax.set_yticklabels([])
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([], [], color="0.2", ls=ls_, lw=1.0, label="$K_I=%.1f$" % K_) for K_, ls_ in ((8.0, ":"), (16.0, "--"), (21.3, "-"))], loc="lower right", frameon=True, fontsize=8.5, title="outline of the zone", title_fontsize=8.5)
cb = fig.colorbar(pc, cax=cax)
cb.set_label("accumulated slip [-]")
# the colour bar has the same height as the (square) maps: after the layout, take over the vertical extent of the right map
fig.canvas.draw()
p, c = axs[1].get_position(), cax.get_position()
cax.set_position([c.x0, p.y0, c.width, p.height])
fig.set_layout_engine("none")
fig.savefig(OUT, dpi=300)
