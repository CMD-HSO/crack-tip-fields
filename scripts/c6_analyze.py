# -*- coding: utf-8 -*-
"""
c6_analyze.py - local analysis (Python 3) of the CSV files written by c6_extract_abaqus.py and evaluate_kfield_abaqus.py

One run per call:
    python c6_analyze.py --dir <folder with export/ and ssy_f*_J.csv> --label "R=10 mm, KH" [--thr 1e-3] [--crit mean|max]
    (criterion of Section 5: --crit max --thr 1e-4 --analysis-dir analysis_max1e-4)

Output in <dir>/analysis/ : summary.csv (one row per frame), zone_frameNN.png (outline of the plastic zone), summary.png

Definitions (all lengths in mm, K_I in MPa sqrt(m)):
  plastic element : --crit mean (default): mean over the integration points of sum_s |gamma_s| > thr     (default 1e-3)
                    --crit max:            the sum_s |gamma_s| exceeds thr in at least one integration point
                    (gamma_s: accumulated slip of the system s as stored in the ODB, i.e. the net slip; the total slip
                    sum_s int |d gamma_s/dt| dt is the same as long as no system slips back)
  r_max           : largest distance of a corner of a plastic element from the tip
  r_p             : largest x of a plastic element touching the grain boundary (y = 0, x > 0)
  area            : area of the plastic elements in the x-y plane
  slip_max_ring>=3: largest slip (maximum over the integration points of sum |gamma_s|) outside the first two element rings at the tip
  n_act           : largest number of active slip systems (slip rate > rate_thr) in an element outside the first two rings
  K deviation     : sqrt(mean J_mid of the contours 3-15 / prescribed K-field J) - 1, J = J40 (K/40)^2
  CTOD_fix        : opening of the crack faces 0.01 mm (undeformed) behind the tip; the 45 degree intersection (Rice) lies inside
                    the first element ring and cannot be resolved with this mesh
"""
import argparse
import csv
import glob
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--label", default="")
ap.add_argument("--thr", type=float, default=1.0e-3, help="slip threshold for 'plastic'")
ap.add_argument("--crit", choices=("mean", "max"), default="mean",
                help="plastic if the mean (default) or the maximum over the integration points of the slip exceeds --thr")
ap.add_argument("--slip", choices=("net", "total"), default="net",
                help="slip measure for the plastic zone: net = sum over the systems of |accumulated slip per system| (g_max / g_mean); "
                     "total = total accumulated slip written by c6_extract_abaqus.py --sdv-total (g_tot_max / g_tot_mean)")
ap.add_argument("--analysis-dir", default="analysis", help="name of the output folder inside --dir")
ap.add_argument("--ctod-dist", type=float, default=0.01, help="distance behind the tip for CTOD_fix [mm]")
ap.add_argument("--J40", type=float, default=9.4090, help="J of the prescribed K-field at K_I = 40 MPa sqrt(m) [N/mm] (In718 pair)")
ap.add_argument("--ringskip", type=int, default=2, help="number of element rings at the tip excluded for max. slip / active systems")
ap.add_argument("--no-plots", action="store_true")
a = ap.parse_args()

A = os.path.join(a.dir, "export")
OUT = os.path.join(a.dir, a.analysis_dir)
os.makedirs(OUT, exist_ok=True)

meta = dict(row[:2] for row in csv.reader(open(os.path.join(A, "meta.csv"))) if len(row) >= 2)
R = float(meta["R_model"])
frames = np.genfromtxt(os.path.join(A, "frames.csv"), delimiter=",", names=True)
frames = np.atleast_1d(frames)


def area(c):
    x, y = c[:, 0::2], c[:, 1::2]
    return 0.5 * np.abs(np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1))


def interp(xs, ys, x0):
    o = np.argsort(xs)
    return float(np.interp(x0, np.asarray(xs)[o], np.asarray(ys)[o]))


def ctod(fn, dist):
    d = np.genfromtxt(fn, delimiter=",", names=True)
    x, y, side = d["x"], d["y"], d["side"]
    tip = np.abs(x) < 1e-9
    xt = np.mean(x[tip] + d["u1"][tip])
    yt = np.mean(d["u2"][tip])
    up, lo = (side > 0) & ~tip, (side < 0) & ~tip
    # one value per x on each flank
    def flank(m):
        xx = np.round(x[m], 9)
        ux = d["u1"][m]
        uy = d["u2"][m]
        keys = np.unique(xx)
        return keys, np.array([ux[xx == k].mean() for k in keys]), np.array([uy[xx == k].mean() for k in keys])
    ku, uxu, uyu = flank(up)
    kl, uxl, uyl = flank(lo)
    # fixed distance (undeformed distance behind the tip)
    fix = interp(ku, uyu, -dist) - interp(kl, uyl, -dist)
    return fix


rows = []
last = None
for fr in frames:
    N = int(fr["frame"])
    K = float(fr["K_I_MPa_sqrt_m"])
    d = np.genfromtxt(os.path.join(A, "elements_frame%02d.csv" % N), delimiter=",", names=True)
    c = np.column_stack([d["x%d" % i] if j == 0 else d["y%d" % i] for i in range(1, 5) for j in (0, 1)])
    rc = np.hypot(c[:, 0::2], c[:, 1::2])
    rmin = rc.min(axis=1)
    # element rings at the tip: ring index from the smallest corner radius
    radii = np.unique(np.round(rmin[rmin < 0.05], 7))
    ring = np.searchsorted(radii, np.round(rmin, 7)) + 1
    ring[rmin >= 0.05] = 999
    pre = "g_tot_" if a.slip == "total" else "g_"
    plastic = d[pre + ("max" if a.crit == "max" else "mean")] > a.thr
    r_max = rc.max(axis=1)[plastic].max() if plastic.any() else 0.0
    gb = (np.abs(c[:, 1::2]).min(axis=1) < 1e-9) & (d["xc"] > 0)
    r_p = c[:, 0::2].max(axis=1)[plastic & gb].max() if (plastic & gb).any() else 0.0
    ar = float(area(c)[plastic].sum())
    out = ring > a.ringskip
    inner = np.isfinite(d["n_act_max"])
    smax = float(d[pre + "max"][out].max())
    nact = float(np.nanmax(d["n_act_max"][out & inner])) if (out & inner).any() else float("nan")
    # J
    J = np.genfromtxt(os.path.join(a.dir, "ssy_f%d_J.csv" % N), delimiter=",", names=True)
    Jm = float(np.mean(J["J_mid"][2:]))
    Jp = a.J40 * (K / 40.0) ** 2
    try:
        cf = ctod(os.path.join(A, "crack_faces_frame%02d.csv" % N), a.ctod_dist)
    except Exception as err:
        cf = float("nan")
        print("CTOD frame %d: %s" % (N, err))
    rows.append([N, K, r_max, r_max / R, r_p, r_p / R, ar, ar / (np.pi * R ** 2 / 2.0), smax, nact,
                 100.0 * (np.sqrt(Jm / Jp) - 1.0), cf])
    last = (N, c, plastic, d)
    print("frame %d  K=%.2f  r_max=%.4f (%.4f R)  r_p=%.4f  area=%.4g  slip_max(ring>%d)=%.3f  n_act=%.0f  dK=%+.2f %%  CTOD_fix=%.3g"
          % (N, K, r_max, r_max / R, r_p, ar, a.ringskip, smax, nact, rows[-1][10], cf))

head = ["frame", "K_I", "r_max", "r_max_over_R", "r_p_GB", "r_p_GB_over_R", "area", "area_over_half_disc",
        "slip_max_ring>%d" % a.ringskip, "n_act_max", "K_dev_percent", "CTOD_fix"]
with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(head)
    w.writerows(rows)

if not a.no_plots:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    S = np.array(rows)
    fig, ax = plt.subplots(2, 2, figsize=(9, 7))
    ax[0, 0].plot(S[:, 1], S[:, 3], "o-", label="$r_\\mathrm{max}/R$")
    ax[0, 0].plot(S[:, 1], S[:, 5], "s-", label="$r_p/R$ (grain boundary)")
    ax[0, 0].set_xlabel("$K_I$ [MPa$\\sqrt{m}$]"); ax[0, 0].set_ylabel("-"); ax[0, 0].legend()
    ax[0, 1].plot(S[:, 1], S[:, 10], "o-"); ax[0, 1].set_xlabel("$K_I$ [MPa$\\sqrt{m}$]"); ax[0, 1].set_ylabel("K deviation from J [%]")
    ax[1, 0].plot(S[:, 1], S[:, 11], "o-", label="%g mm behind the tip" % a.ctod_dist)
    ax[1, 0].set_xlabel("$K_I$ [MPa$\\sqrt{m}$]"); ax[1, 0].set_ylabel("CTOD [mm]"); ax[1, 0].legend()
    ax[1, 1].plot(S[:, 1], S[:, 8], "o-", label="max. slip (rings > %d)" % a.ringskip)
    ax[1, 1].set_xlabel("$K_I$ [MPa$\\sqrt{m}$]"); ax[1, 1].set_ylabel("$\\sum|\\gamma_s|$"); ax[1, 1].legend()
    fig.suptitle(a.label)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "summary.png"), dpi=150); plt.close(fig)
    N, c, plastic, d = last
    fig, ax = plt.subplots(figsize=(6, 6))
    polys = [c[i].reshape(4, 2) for i in np.where(plastic)[0]]
    ax.add_collection(PolyCollection(polys, facecolors="tab:red", edgecolors="none", alpha=0.5))
    ax.axhline(0, color="k", lw=0.5)
    ext = max(rows[-1][2] * 1.3, 1e-3)
    ax.set_xlim(-ext, ext); ax.set_ylim(-ext, ext); ax.set_aspect("equal")
    ax.set_title("%s, frame %d, K=%.1f, plastic elements" % (a.label, N, rows[-1][1]))
    fig.savefig(os.path.join(OUT, "zone_frame%02d.png" % N), dpi=150); plt.close(fig)
print("written", OUT)
