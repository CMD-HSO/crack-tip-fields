"""
make_fig_s22_comparison.py - opening stress sigma_22 around the crack tip, analytical vs. finite elements

Figure (2 x 3 panels, written to ./figs/fig_s22_comparison.{pdf,png}):

    row 1  Fe single crystal (27,15,10), case C3          row 2  Fe/Cu interface (grain boundary), case C4
    column 1  analytical sigma_22 (Stroh / oscillating Stroh)
    column 2  sigma_22 from Abaqus (3D slice, generalised plane strain, mid-plane nodes)
    column 3  normalised angular distribution on the node ring r = 0.0974 mm:
              analytical (line), FE (markers), isotropic Williams field as reference
              (row 1 additionally contains case C2, Fe (27,0,0))

Columns 1 and 2 use the same window, the same colour scale and the same contour levels
(banded, clipped; values outside the range are drawn grey).

Input (written by the FE case study, see examples/fe_case_study/3D_mode_I/):
    ../../Paper/Bilder/C3_S22.png, C4_S22.png                   Abaqus screenshots of S22 (column 2), see below
    Fe_27_15_10/sc_3D_s22_field_{nodes,quads}.csv               fallback if no screenshot exists:
    Fe_27_15_10_Cu_5_40_0/bc_3D_s22_field_{nodes,quads}.csv     export_s22_field.py --mode single_crystal / bicrystal
    Fe_27_0_0/sc_3D_circle.csv, Fe_27_15_10/sc_3D_circle.csv,
    Fe_27_15_10_Cu_5_40_0/bc_3D_circle.csv                      compare_stress_circle.py

Abaqus screenshots (column 2): contour limits -250 ... 5250 MPa in 11 bands of 500 MPa (the colours of the
legend are used for the analytical column as well), zoomed on the tip, legend, title bar and frame switched off.
Window and scale of the screenshot are found from the mesh itself: the element rings of the focused mesh have
a radial spacing of 0.05/14 mm (14 elements in the refined partition of radius 0.05 mm) and are fitted in the image.
If the legend is still in the screenshot, its rectangle is greyed out.

Usage:   python make_fig_s22_comparison.py [--shot-c3 C3_S22.png] [--shot-c4 C4_S22.png] [--fe-source auto|shots|csv]
                                           [--vmin -250] [--vmax 5250] [--bands 11]
(Abaqus: Options -> Contour -> Limits: Min -250, Max 5250; Basic: Contour Intervals 11. The lowest band
 (-250 ... 250 MPa) contains the stress-free zone (sigma_22 = 0 on the crack faces), so that it is not split by a band
 limit. The band colours are those of the Abaqus spectrum for the chosen number of bands.)
Requires crack_tip_fields.py in ../src (like reproduce_manuscript_results.py).
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import crack_tip_fields as ctf

PI = np.pi
A_HALF = 20.0                       # reference length a [mm] of the FE case study (2a = 40 mm)
K0 = 27.5 * np.sqrt(1000.)          # K_I = 27.5 MPa*sqrt(m) in MPa*sqrt(mm)
CASES = os.path.join(HERE, "..", "examples", "fe_case_study", "3D_mode_I")
SC_DIR, SC0_DIR, BC_DIR = "Fe_27_15_10", "Fe_27_0_0", "Fe_27_15_10_Cu_5_40_0"
FE_MAT = "Fe (cubic, example)"
CU_MAT = "Cu (cubic, example)"
BILDER = os.path.join(HERE, "..", "..", "Paper", "Bilder")
RING_SPACING = 0.05 / 14.0          # radial element size in the refined partition around the tip [mm]
ABAQUS_OVER, ABAQUS_UNDER = (204, 204, 204), (51, 51, 51)         # above the maximum / below the minimum


def abaqus_bands(n):
    """Colours of an Abaqus contour legend with n bands (default spectrum), highest band first: n equally spaced points
    on red -> yellow -> green -> cyan -> blue (checked against legends with 11 and 12 intervals)."""
    out = []
    for t in np.linspace(0.0, 1020.0, n):
        if t <= 255:
            c = (255, t, 0)
        elif t <= 510:
            c = (255 - (t - 255), 255, 0)
        elif t <= 765:
            c = (0, 255, t - 510)
        else:
            c = (0, 255 - (t - 765), 255)
        out.append(tuple(int(round(v)) for v in c))
    return out


ABAQUS_BANDS = abaqus_bands(11)
LEGEND_BOX = (19, 39, 218, 408)     # x0, y0, x1, y1 of the Abaqus legend in the screenshots (pixels)


# ---------------------------------------------------------------------------------------------
def analytic_field(model, kind, half, n):
    """sigma_22 on an n x n grid (n even: the crack plane y = 0 is not on the grid)."""
    x = np.linspace(-half, half, n)
    s22 = np.empty((n, n))
    for j, yy in enumerate(x):
        for i, xx in enumerate(x):
            if kind == "interface":
                s22[j, i] = model.stress(xx, yy, K0, 0., 0., A_HALF, "top" if yy > 0 else "bot")[1]
            else:
                s22[j, i] = model.stress(xx, yy, K0, 0., 0., A_HALF)[1]
    return x, s22


def analytic_circle(model, kind, theta_deg, r):
    out = []
    for th in np.radians(theta_deg):
        xx, yy = r * np.cos(th), r * np.sin(th)
        if kind == "interface":
            side = "top" if (yy > 0 or (yy == 0 and th == 0)) else "bot"
            out.append(model.stress(xx, yy, K0, 0., 0., A_HALF, side)[1])
        else:
            out.append(model.stress(xx, yy, K0, 0., 0., A_HALF)[1])
    return np.array(out) * np.sqrt(2 * PI * r) / K0


def read_fe_field(folder, prefix):
    from matplotlib.tri import Triangulation
    nd = np.genfromtxt(os.path.join(CASES, folder, prefix + "_s22_field_nodes.csv"), delimiter=",", names=True)
    q = np.atleast_1d(np.genfromtxt(os.path.join(CASES, folder, prefix + "_s22_field_quads.csv"),
                                    delimiter=",", names=True, dtype=int))
    row = {int(l): k for k, l in enumerate(np.atleast_1d(nd["label"]))}
    tris = []
    for r in q:
        ids = [row[int(r[k])] for k in range(4)]
        for t in ((ids[0], ids[1], ids[2]), (ids[0], ids[2], ids[3])):
            if len(set(t)) == 3:                                  # collapsed tip elements
                tris.append(t)
    tris = np.array(tris)
    s = np.atleast_1d(nd["s22"]).astype(float)
    bad = np.isnan(s)
    tri = Triangulation(np.atleast_1d(nd["x"]), np.atleast_1d(nd["y"]), tris, mask=bad[tris].any(axis=1))
    return tri, np.where(bad, 0.0, s)


def load_shot(path):
    """Abaqus screenshot -> (image, tip pixel (cx, cy), pixels per mm); title bar and frame are cropped later."""
    from PIL import Image
    im = np.array(Image.open(path).convert("RGB"))
    h, w, _ = im.shape
    legend = (im[100, 30] > 250).all() and (im[400, 30] > 250).all() and (im[41, 30] > 250).all()
    if legend:
        lx0, ly0, lx1, ly1 = LEGEND_BOX
        im[ly0:ly1, lx0:lx1] = ABAQUS_OVER
        print("  warning: %s still contains the Abaqus legend; its rectangle is greyed out "
              "(switch the legend off in Abaqus and save the screenshot again)" % os.path.basename(path))
    dark = im.astype(int).sum(axis=2) < 90
    ys, xs = np.nonzero(dark)
    sel = (xs > 230) & (ys > 60) & (ys < h - 30) & (xs < w - 10)
    xs, ys = xs[sel], ys[sel]

    def hist(cx, cy):
        return np.bincount(np.hypot(xs - cx, ys - cy).astype(int), minlength=420)[:420].astype(float)
    # rough position of the tip: the densest accumulation of mesh lines (all element edges converge there);
    # the ring fit below is then only refined within a few pixels, so that it cannot lock on a wrong centre
    from scipy.ndimage import uniform_filter
    dens = uniform_filter(dark.astype(float), size=15)
    y0, x0 = int(h / 2 - 60), int(w / 2 - 60)
    iy, ix = np.unravel_index(np.argmax(dens[y0:y0 + 120, x0:x0 + 120]), (120, 120))
    bx, by = x0 + ix, y0 + iy
    best = max(((hist(cx, cy) ** 2).sum(), cx, cy) for cx in np.arange(bx - 6, bx + 7, 1.0)
               for cy in np.arange(by - 6, by + 7, 1.0))
    _, cx, cy = best
    hh = hist(cx, cy)
    # ring spacing d [px]: comb of radii k*d (k = 1..5) with the strongest accumulation of mesh-line pixels
    hm = np.maximum.reduce([np.roll(hh, -1), hh, np.roll(hh, 1)])
    score, d = max((sum(hm[int(round(k * dd))] for k in range(1, 6)), dd) for dd in np.arange(40.0, 83.0, 0.05))
    if score < 0.5 * hh[40:].max() * 5:
        raise SystemExit("%s: mesh rings not found, cannot calibrate the screenshot" % path)
    return im, (cx, cy), d / RING_SPACING


def frame_of(im):
    """(left, top, right, bottom) border of a window grab [px]: Abaqus title bar (dark blue) and frame; 0 for a clean export."""
    return (6, 29, 6, 6) if tuple(im[0, im.shape[1] // 2]) == (0, 0, 116) else (0, 0, 0, 0)


def crop_shot(im, tip, scale, half):
    cx, cy = tip
    x0, x1 = int(round(cx - half * scale)), int(round(cx + half * scale)) + 1
    y0, y1 = int(round(cy - half * scale)), int(round(cy + half * scale)) + 1
    ext = [(x0 - 0.5 - cx) / scale, (x1 - 0.5 - cx) / scale, (cy - (y1 - 0.5)) / scale, (cy - (y0 - 0.5)) / scale]
    return im[y0:y1, x0:x1], ext


def read_fe_circle(folder, name):
    d = np.genfromtxt(os.path.join(CASES, folder, name), delimiter=",", names=True)
    r = float(np.mean(d["r"]))
    return d["theta_deg"], d["s22"] * np.sqrt(2 * PI * d["r"]) / K0, r


# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--half", type=float, default=None,
                    help="half edge of the plotted window [mm] (default: largest window that fits into the screenshots, "
                         "0.022 for the csv fallback)")
    ap.add_argument("--shot-c3", default=os.path.join(BILDER, "C3_S22.png"), help="Abaqus screenshot of C3 (single crystal)")
    ap.add_argument("--shot-c4", default=os.path.join(BILDER, "C4_S22.png"), help="Abaqus screenshot of C4 (interface)")
    ap.add_argument("--fe-source", choices=("auto", "shots", "csv"), default="auto",
                    help="FE panels from the screenshots or from the exported nodal values (auto: screenshots if present)")
    ap.add_argument("--vmin", type=float, default=-250.0, help="lower limit of the colour scale [MPa]")
    ap.add_argument("--vmax", type=float, default=5250.0, help="upper limit of the colour scale [MPa]")
    ap.add_argument("--bands", type=int, default=11, help="number of equal contour bands between vmin and vmax")
    ap.add_argument("--n", type=int, default=160, help="grid points per direction of the analytical field (even)")
    ap.add_argument("--out", default=os.path.join(HERE, "figs", "fig_s22_comparison"))
    a = ap.parse_args()
    n = a.n + (a.n % 2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "figure.dpi": 150,
                         # Computer Modern like the text (lmodern)
                         "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True, "axes.unicode_minus": False})

    levels = np.linspace(a.vmin, a.vmax, a.bands + 1)
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([tuple(c / 255 for c in col) for col in abaqus_bands(a.bands)[::-1]])   # same as the legend
    cmap.set_over(tuple(c / 255 for c in ABAQUS_OVER))
    cmap.set_under(tuple(c / 255 for c in ABAQUS_UNDER))

    iso = ctf.IsotropicCrackModel("In718 (isotropic)")
    sc3 = ctf.AnisotropicCrackModel(FE_MAT, 27, 15, 10)
    sc2 = ctf.AnisotropicCrackModel(FE_MAT, 27, 0, 0)
    gb = ctf.InterfaceCrackModel(FE_MAT, (27, 15, 10), CU_MAT, (5, 40, 0))

    rows = [
        dict(label="C3 (single crystal)", model=sc3, kind="aniso", field=(SC_DIR, "sc_3D"), shot=a.shot_c3),
        dict(label="C4 (interface)", model=gb, kind="interface", field=(BC_DIR, "bc_3D"), shot=a.shot_c4),
    ]
    use_shots = a.fe_source == "shots" or (a.fe_source == "auto" and all(os.path.exists(r["shot"]) for r in rows))
    if use_shots:
        print("FE panels: Abaqus screenshots")
        for r in rows:
            r["im"], r["tip"], r["scale"] = load_shot(r["shot"])
            print("  %s: tip at pixel (%.0f, %.0f), %.0f px/mm" % (os.path.basename(r["shot"]), r["tip"][0], r["tip"][1], r["scale"]))
        fit = min(min(r["tip"][0] - frame_of(r["im"])[0], r["im"].shape[1] - frame_of(r["im"])[2] - r["tip"][0],
                      r["tip"][1] - frame_of(r["im"])[1], r["im"].shape[0] - frame_of(r["im"])[3] - r["tip"][1])
                  / r["scale"] for r in rows)
        half = a.half if a.half else float(np.floor(fit * 1e4) / 1e4)
    else:
        print("FE panels: nodal values exported from the .odb (csv)")
        half = a.half if a.half else 0.022

    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 1.25], height_ratios=[1, 1, .06],
                          left=.085, right=.985, bottom=.10, top=.94, wspace=.36, hspace=.42)
    tk = 0.005 * np.floor(half / 0.005 + 1e-9)
    ticks = [-tk, 0, tk]
    cs = None
    for k, row in enumerate(rows):
        # --- column 1: analytical field ---
        ax = fig.add_subplot(gs[k, 0])
        x, s = analytic_field(row["model"], row["kind"], half, n)
        cs = ax.contourf(x, x, s, levels=levels, cmap=cmap, extend="both")
        # --- column 2: FE field ---
        bx = fig.add_subplot(gs[k, 1])
        if use_shots:
            crop, ext = crop_shot(row["im"], row["tip"], row["scale"], half)
            bx.imshow(crop, extent=ext, interpolation="lanczos")
        else:
            tri, sfe = read_fe_field(*row["field"])
            bx.tricontourf(tri, sfe, levels=levels, cmap=cmap, extend="both")
        for axx in (ax, bx):
            axx.set_xticks(ticks); axx.set_yticks(ticks)
            axx.set_xlim(-half, half); axx.set_ylim(-half, half); axx.set_aspect("equal")
            axx.set_xticklabels([f"{t:g}" for t in ticks]); axx.set_yticklabels([f"{t:g}" for t in ticks])
            axx.plot([-half, 0], [0, 0], color="k", lw=1.2)          # crack along the negative x1 axis
        bx.set_yticklabels([])
        ax.set_ylabel(r"$x_2$ [mm]")
        if k == 1:
            ax.set_xlabel(r"$x_1$ [mm]"); bx.set_xlabel(r"$x_1$ [mm]")
        else:
            ax.set_xticklabels([]); bx.set_xticklabels([])
        ax.text(0.04, 0.96, row["label"], transform=ax.transAxes, fontsize=7.5, fontweight="bold", va="top",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
        ax.set_facecolor("0.8"); bx.set_facecolor("0.8")           # tip element (no value) and clipped region
        ax.set_title("(%s) analytical" % "abcdef"[3 * k], fontsize=8)          # panel letters are part of the titles
        bx.set_title("(%s) finite elements (Abaqus)" % "abcdef"[3 * k + 1], fontsize=8)

        # --- column 3: angular distribution on the node ring ---
        cx = fig.add_subplot(gs[k, 2])
        th = np.linspace(-179.5, 179.5, 359)
        if k == 0:
            th_fe3, y_fe3, r_ring = read_fe_circle(SC_DIR, "sc_3D_circle.csv")
            th_fe2, y_fe2, _ = read_fe_circle(SC0_DIR, "sc_3D_circle.csv")
        else:
            th_fe3, y_fe3, r_ring = read_fe_circle(BC_DIR, "bc_3D_circle.csv")
        cx.plot(th, analytic_circle(iso, "iso", th, r_ring), "k-", lw=1.0, label="isotropic (Williams)")
        if k == 0:
            cx.plot(th, analytic_circle(sc2, "aniso", th, r_ring), "-", color="C0", lw=1.0, label=r"C2 analytical")
            cx.plot(th_fe2[::4], y_fe2[::4], "o", color="C0", ms=2.5, mfc="none", label="C2 FE")
            cx.plot(th, analytic_circle(sc3, "aniso", th, r_ring), "-", color="C1", lw=1.0, label=r"C3 analytical")
            cx.plot(th_fe3[::4], y_fe3[::4], "s", color="C1", ms=2.5, mfc="none", label="C3 FE")
        else:
            cx.plot(th, analytic_circle(gb, "interface", th, r_ring), "-", color="C3", lw=1.0, label="C4 analytical")
            cx.plot(th_fe3[::4], y_fe3[::4], "s", color="C3", ms=2.5, mfc="none", label="C4 FE")
        cx.set_xlim(-180, 180); cx.set_xticks([-180, -90, 0, 90, 180])
        cx.set_ylim(-0.05, 2.0)
        cx.grid(alpha=.3)
        cx.set_ylabel(r"$\sigma_{22}\sqrt{2\pi r}/K_I$")
        cx.legend(frameon=False, fontsize=6, loc="upper center", ncol=2, columnspacing=0.8, handlelength=1.6)
        if k == 1:
            cx.set_xlabel(r"$\theta$ [deg]")
        else:
            cx.set_xticklabels([])
        cx.set_title(r"(%s) ring $r=%.4f$ mm" % ("abcdef"[3 * k + 2], r_ring), fontsize=8)

    cb = fig.colorbar(cs, cax=fig.add_subplot(gs[2, :2]), orientation="horizontal", ticks=levels)
    cb.ax.set_xticklabels(["%.0f" % v for v in levels], fontsize=6.5)
    cb.set_label(r"$\sigma_{22}$ [MPa]")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out + ".pdf")
    fig.savefig(a.out + ".png", dpi=300)
    print("written: %s.pdf, %s.png" % (a.out, a.out))


if __name__ == "__main__":
    main()
