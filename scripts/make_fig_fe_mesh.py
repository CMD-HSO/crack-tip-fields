# -*- coding: utf-8 -*-
"""
make_fig_fe_mesh.py - figure of the boundary layer finite element model (vector graphic from the .inp file)

Figure (written to ./figs/fe_mesh.{pdf,png}):
    (a) circular domain with focused mesh, crack (seam) along the negative x1 axis, crack tip, axes x1/x2,
        radius R, the prescribed K-field displacements on the outer boundary, zoom window
    (b) mesh around the crack tip in a square window with scale bar

The mesh is read from the 2D input file of the case study with crack_tip_fields.parse_abaqus_mesh
(CPE8 elements; the edges are drawn through the mid-side nodes).

Usage:   python make_fig_fe_mesh.py [--inp file.inp] [--half 0.03] [--grains] [--R 1.0]
Options: --half    half edge of the zoom window in (b) [mm] (default 0.03)
         --grains  tint the region above the crack (grain 1) and below it (grain 2), for the interface cases
         --inp     default: examples/fe_case_study/2D_mode_I/Fe_27_15_10_Cu_5_40_0/Fe_27_15_10_Cu_5_40_0_mode_I.inp
Requires crack_tip_fields.py in ../src and matplotlib.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle, ConnectionPatch, Circle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import crack_tip_fields as ctf

DEFAULT_INP = os.path.join(HERE, "..", "examples", "fe_case_study", "2D_mode_I", "Fe_27_15_10_Cu_5_40_0",
                           "Fe_27_15_10_Cu_5_40_0_mode_I.inp")
GRAIN1, GRAIN2 = "#dbe9f6", "#fbe5d0"          # light tints of grain 1 (above) and grain 2 (below)
LINE = "#555555"


def mesh_segments(mesh):
    """Unique element edges as polylines through the mid-side nodes: list of arrays (3, 2)."""
    nodes = mesh["nodes"]
    seen, segs = set(), []
    for conn in mesh["elements"].values():
        if len(conn) < 8:                                   # linear / other element: edges between corners only
            corners = conn[:4]
            for i in range(4):
                a, b = corners[i], corners[(i + 1) % 4]
                key = (min(a, b), max(a, b))
                if key not in seen:
                    seen.add(key)
                    segs.append(np.array([nodes[a][:2], nodes[b][:2]]))
            continue
        corners, mids = conn[:4], conn[4:8]
        for i in range(4):
            a, m, b = corners[i], mids[i], corners[(i + 1) % 4]
            key = (min(a, b), max(a, b), m)
            if key not in seen:
                seen.add(key)
                segs.append(np.array([nodes[a][:2], nodes[m][:2], nodes[b][:2]]))
    return segs


def axes_arrows(ax, origin, length, fs, color="k"):
    """Coordinate axes x1 (right) and x2 (up) at origin."""
    ox, oy = origin
    ax.annotate("", xy=(ox + length, oy), xytext=(ox, oy),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color=color, shrinkA=0, shrinkB=0), zorder=6)
    ax.annotate("", xy=(ox, oy + length), xytext=(ox, oy),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color=color, shrinkA=0, shrinkB=0), zorder=6)
    ax.text(ox + length * 1.08, oy - length * 0.04, r"$x_1$", fontsize=fs, ha="left", va="center", zorder=6)
    ax.text(ox, oy + length * 1.1, r"$x_2$", fontsize=fs, ha="center", va="bottom", zorder=6)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--inp", default=DEFAULT_INP)
    ap.add_argument("--half", type=float, default=0.03, help="half edge of the zoom window [mm]")
    ap.add_argument("--R", type=float, default=1.0, help="radius of the model [mm]")
    ap.add_argument("--grains", action="store_true", help="tint grain 1 (above) and grain 2 (below the crack)")
    ap.add_argument("--out", default=os.path.join(HERE, "figs", "fe_mesh"), help="output prefix")
    args = ap.parse_args()

    mesh = ctf.parse_abaqus_mesh(args.inp)
    segs = mesh_segments(mesh)
    R, h = args.R, args.half
    fs = 8.5
    plt.rcParams.update({"font.size": fs, "figure.dpi": 150,
                         # Computer Modern like the text (lmodern)
                         "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True, "axes.unicode_minus": False})

    fig = plt.figure(figsize=(6.6, 3.3))
    ax_a = fig.add_axes([0.01, 0.06, 0.46, 0.86])
    ax_b = fig.add_axes([0.55, 0.06, 0.43, 0.86])        # 0.43 * 6.6 in = 0.86 * 3.3 in: square window

    # ---------------- (a) full model
    if args.grains:
        for sgn, col in ((1, GRAIN1), (-1, GRAIN2)):
            th = np.linspace(0, np.pi, 200) if sgn > 0 else np.linspace(0, -np.pi, 200)
            ax_a.fill(np.r_[R * np.cos(th), 0], np.r_[R * np.sin(th), 0], color=col, lw=0, zorder=0)
    ax_a.add_collection(LineCollection(segs, colors=LINE, linewidths=0.12, zorder=1))
    ax_a.add_patch(Circle((0, 0), R, fill=False, ec="k", lw=0.8, zorder=3))
    ax_a.plot([-R, 0], [0, 0], color="#c0392b", lw=1.6, solid_capstyle="butt", zorder=4)       # crack (seam)
    ax_a.plot(0, 0, "o", ms=3.2, color="k", zorder=5)
    # radius (towards the upper left, away from the lines to the zoom panel); the label runs parallel to the
    # arrow on its upper-right side, so that it does not cross it
    ang_r = np.radians(135)
    ax_a.annotate("", xy=(R * np.cos(ang_r), R * np.sin(ang_r)), xytext=(0, 0),
                  arrowprops=dict(arrowstyle="<->", lw=0.8, color="k", shrinkA=0, shrinkB=0), zorder=5)
    ax_a.text(0.5 * R * np.cos(ang_r) + 0.025, 0.5 * R * np.sin(ang_r) + 0.025, r"$R = %g$ mm" % R,
              fontsize=fs, ha="center", va="bottom", rotation=-45, rotation_mode="anchor", zorder=6,
              bbox=dict(fc="white", ec="none", pad=0.3, alpha=0.85))
    ax_a.text(-0.5 * R, 0.04, r"crack", fontsize=fs, color="#c0392b", ha="center", va="bottom",
              bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.8), zorder=6)
    axes_arrows(ax_a, (-1.12, -1.0), 0.22, fs)                     # coordinate axes next to the model
    if args.grains:
        ax_a.text(0.0, 0.62, "grain 1", fontsize=fs, ha="center", va="center", color="#34495e", zorder=6)
        ax_a.text(0.0, -0.62, "grain 2", fontsize=fs, ha="center", va="center", color="#34495e", zorder=6)
    # zoom window
    ax_a.add_patch(Rectangle((-h, -h), 2 * h, 2 * h, fill=False, ec="k", lw=0.7, ls="--", zorder=6))
    ax_a.set_xlim(-1.3, 1.3)
    ax_a.set_ylim(-1.3, 1.3)
    ax_a.set_aspect("equal")
    ax_a.axis("off")
    ax_a.text(1.29, 1.29, "(a)", fontsize=fs + 0.5, ha="right", va="top")     # as the panel labels of the other figures

    # ---------------- (b) zoom
    ax_b.clear()
    if args.grains:
        ax_b.add_patch(Rectangle((-h, 0), 2 * h, h, color=GRAIN1, lw=0, zorder=0))
        ax_b.add_patch(Rectangle((-h, -h), 2 * h, h, color=GRAIN2, lw=0, zorder=0))
    ax_b.add_collection(LineCollection(segs, colors="#8a8a8a", linewidths=0.3, zorder=1))   # light, so that the contours stand out
    ax_b.plot([-h, 0], [0, 0], color="#c0392b", lw=1.8, solid_capstyle="butt", zorder=4)
    ax_b.plot(0, 0, "o", ms=3.4, color="k", zorder=5)
    ax_b.text(-0.5 * h, 0.04 * h, "crack", fontsize=fs, color="#c0392b", ha="center", va="bottom",
              bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.8), zorder=6)
    # contours 1 to 6 of the J-integral: contour k = k-th ring of corner nodes (radius k * element size at the tip)
    xs = sorted(v[0] for v in mesh["nodes"].values() if abs(v[1]) < 1e-9 and v[0] > 1e-9)
    s = xs[1]                                             # first ring of corner nodes (xs[0] is the quarter point)
    phi = np.linspace(0, 2 * np.pi, 400)
    for k in range(1, 7):
        ax_b.plot(k * s * np.cos(phi), k * s * np.sin(phi), color="#0b3d91", lw=1.3, ls=(0, (4, 2)), zorder=3)
        ang = np.radians(62)
        ax_b.text(k * s * np.cos(ang), k * s * np.sin(ang), str(k), fontsize=fs, color="#0b3d91", ha="center",
                  va="center", zorder=7, bbox=dict(fc="white", ec="none", pad=0.25, alpha=0.95))
    print(f"contour radii: {', '.join(f'{k * s:.4f}' for k in range(1, 7))} mm")
    # scale bar of 0.01 mm
    bar = 0.01 if h >= 0.02 else 0.005
    x0, y0 = 0.55 * h, -0.88 * h
    ax_b.add_patch(Rectangle((x0 - 0.12 * h, y0 - 0.08 * h), bar + 0.24 * h, 0.34 * h, fc="white", ec="none",
                             alpha=0.92, zorder=5))
    ax_b.plot([x0, x0 + bar], [y0, y0], color="k", lw=1.6, solid_capstyle="butt", zorder=6)
    ax_b.text(x0 + bar / 2, y0 + 0.05 * h, f"{bar:g} mm", fontsize=fs - 0.5, ha="center",
              va="bottom", zorder=6)
    ax_b.set_xlim(-h, h)
    ax_b.set_ylim(-h, h)
    ax_b.set_aspect("equal")
    ax_b.set_xticks([])
    ax_b.set_yticks([])
    for sp in ax_b.spines.values():
        sp.set_linewidth(0.7)
        sp.set_linestyle("--")
    ax_b.text(0.97 * h, 0.97 * h, "(b)", fontsize=fs + 0.5, ha="right", va="top", zorder=8,
              bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9))

    # connection lines between zoom window and (b)
    for (xa, ya), (xb, yb) in (((h, h), (-h, h)), ((h, -h), (-h, -h))):
        fig.add_artist(ConnectionPatch(xyA=(xa, ya), coordsA=ax_a.transData, xyB=(xb, yb), coordsB=ax_b.transData,
                                       color="k", lw=0.5, ls=":", zorder=0))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"{len(mesh['elements'])} elements, {len(mesh['nodes'])} nodes -> {args.out}.pdf / .png")


if __name__ == "__main__":
    main()
