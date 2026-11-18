"""plot_s22_field.py - sigma_22 field around the crack tip from export_s22_field.py (normal Python 3)

    python plot_s22_field.py <prefix>_s22_field_nodes.csv <prefix>_s22_field_quads.csv --out <name> [options]

Colour scale symmetric about zero (default +-14000 MPa, values beyond are clipped); small negative values on
the traction-free crack faces are therefore (nearly) white. The elements are not drawn across the crack,
because the crack faces are separate node copies.

Options: --vmax <MPa>   half range of the colour scale (default 14000)
         --cmap <name>  matplotlib colour map (default RdBu_r; use the one of the analytical figure)
         --half <mm>    half edge of the plotted window (default 0.005)
         --title <text> title of the panel
         --out <name>   output name without extension (PNG and PDF are written)
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("nodes")
    ap.add_argument("quads")
    ap.add_argument("--out", default="s22_field")
    ap.add_argument("--vmax", type=float, default=14000.0)
    ap.add_argument("--cmap", default="RdBu_r")
    ap.add_argument("--half", type=float, default=0.005)
    ap.add_argument("--title", default="")
    a = ap.parse_args()

    nd = np.genfromtxt(a.nodes, delimiter=",", names=True)
    q = np.genfromtxt(a.quads, delimiter=",", names=True, dtype=int)
    q = np.atleast_1d(q)
    row = {int(l): k for k, l in enumerate(np.atleast_1d(nd["label"]))}
    x, y, s = np.atleast_1d(nd["x"]), np.atleast_1d(nd["y"]), np.atleast_1d(nd["s22"])

    tris = []
    for r in q:
        ids = [row[int(r[k])] for k in range(4)]
        for t in ((ids[0], ids[1], ids[2]), (ids[0], ids[2], ids[3])):
            if len(set(t)) == 3:                                  # collapsed tip elements: skip degenerate triangles
                tris.append(t)
    tris = np.array(tris)
    bad = np.isnan(s)
    mask = bad[tris].any(axis=1)                                   # triangles at the tip node (NaN)
    sv = np.where(bad, 0.0, np.clip(s, -a.vmax, a.vmax))
    tri = Triangulation(x, y, tris, mask=mask)

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    im = ax.tripcolor(tri, sv, shading="gouraud", cmap=a.cmap, vmin=-a.vmax, vmax=a.vmax, rasterized=True)
    ax.set_xlim(-a.half, a.half)
    ax.set_ylim(-a.half, a.half)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$ [mm]")
    ax.set_ylabel(r"$x_2$ [mm]")
    if a.title:
        ax.set_title(a.title, fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$\sigma_{22}$ [MPa]")
    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=300)
    fig.savefig(a.out + ".pdf")
    print("written: %s.png, %s.pdf  (%d triangles, %d masked at the tip)" % (a.out, a.out, len(tris), int(mask.sum())))


if __name__ == "__main__":
    main()
