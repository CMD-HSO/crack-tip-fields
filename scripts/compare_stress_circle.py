"""
compare_stress_circle.py - FE stresses on a circle around the crack tip vs. the analytical field
==============================================================================================

Compares the stresses written by evaluate_kfield_abaqus.py (<prefix>_circle.csv: all stress components
on a node ring around the tip, averaged per node separately for the elements above and below the crack
plane) with the closed-form crack-tip field of crack_tip_fields at the same points.

The deviation of each component is given relative to the reference stress K_I / sqrt(2 pi r):
    dev_ij = max over theta of |sigma_ij^FE - sigma_ij^analytical| * sqrt(2 pi r) / K_I .

Usage (Python 3, from any folder):
  # isotropic
  python compare_stress_circle.py iso_circle.csv --isotropic "In718 (isotropic)"
  # single crystal
  python compare_stress_circle.py sc_3D_circle.csv --material "Fe (cubic, example)" --euler 27 15 10
  # bicrystal (grain 1 above, grain 2 below the crack plane)
  python compare_stress_circle.py bc_3D_circle.csv --material "Fe (cubic, example)" --euler 27 15 10 \\
                                  --material2 "Cu (cubic, example)" --euler2 5 40 0
Options: --KI --KII --KIII [MPa*sqrt(mm)] (default K_I = 27.5*sqrt(1000)), --a [mm] (default 20), --out.

Output (next to the input file):
  <prefix>_compare.csv   theta_deg,side and FE / analytical stresses normalised with K_I / sqrt(2 pi r)
  <prefix>_compare.png   six panels sigma_ij(theta), FE (markers) and analytical field (lines)
Crack tip at the origin, crack along the negative x axis (alpha = 0), as in the FE case study.
"""
import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _add_module_path():
    """Find crack_tip_fields.py: environment variable CRACK_TIP_FIELDS_SRC, the folder of this script, or a
    folder 'src' in this folder or one of its parents (also starting from the current working directory),
    so that the script also works when it is copied next to the Abaqus results."""
    starts = [os.path.dirname(os.path.abspath(__file__)), os.getcwd()]
    cands = [os.environ.get("CRACK_TIP_FIELDS_SRC", "")]
    for d in starts:
        while True:
            cands += [d, os.path.join(d, "src")]
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "crack_tip_fields.py")):
            sys.path.insert(0, c)
            return
    raise SystemExit("crack_tip_fields.py not found: run the script inside the repository, copy crack_tip_fields.py "
                     "next to it, or set the environment variable CRACK_TIP_FIELDS_SRC to the folder 'src'.")


_add_module_path()
import crack_tip_fields as ctf

COMPONENTS = ("s11", "s22", "s33", "s23", "s13", "s12")
LABELS = (r"$\sigma_{11}$", r"$\sigma_{22}$", r"$\sigma_{33}$", r"$\sigma_{23}$", r"$\sigma_{13}$", r"$\sigma_{12}$")


def build_model(args):
    if args.isotropic:
        return ctf.IsotropicCrackModel(args.isotropic), "isotropic, %s" % args.isotropic
    if args.material2:
        return (ctf.InterfaceCrackModel(args.material, tuple(args.euler), args.material2, tuple(args.euler2)),
                "bicrystal, grain 1 %s %s / grain 2 %s %s" % (args.material, tuple(args.euler),
                                                              args.material2, tuple(args.euler2)))
    return ctf.AnisotropicCrackModel(args.material, *args.euler), "single crystal, %s %s" % (args.material,
                                                                                           tuple(args.euler))


def analytical(model, x, y, side, K, a):
    """Analytical stress (11,22,33,23,13,12) at (x, y); on the crack plane the side decides the face/grain."""
    x2 = np.copysign(0.0, side) if abs(y) < 1e-9 else y
    if isinstance(model, ctf.InterfaceCrackModel):
        return np.array(model.stress(x, x2, *K, a, "top" if side > 0 else "bot"), dtype=float)
    return np.array(model.stress(x, x2, *K, a), dtype=float)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="<prefix>_circle.csv written by evaluate_kfield_abaqus.py")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--isotropic", help="isotropic material (key of MATERIALS)")
    g.add_argument("--material", help="material of the crystal / grain 1 (key of MATERIALS)")
    ap.add_argument("--euler", nargs=3, type=float, default=(0.0, 0.0, 0.0), metavar=("PHI1", "PHI", "PHI2"))
    ap.add_argument("--material2", help="material of grain 2 (below the crack plane) -> bicrystal")
    ap.add_argument("--euler2", nargs=3, type=float, default=(0.0, 0.0, 0.0), metavar=("PHI1", "PHI", "PHI2"))
    ap.add_argument("--KI", type=float, default=27.5 * 1000 ** 0.5, help="K_I [MPa*sqrt(mm)]")
    ap.add_argument("--KII", type=float, default=0.0, help="K_II [MPa*sqrt(mm)]")
    ap.add_argument("--KIII", type=float, default=0.0, help="K_III [MPa*sqrt(mm)]")
    ap.add_argument("--a", type=float, default=20.0, help="reference length a (crack half-length) [mm]")
    ap.add_argument("--out", default=None, help="output prefix (default: input name without '_circle.csv')")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    if not rows:
        raise SystemExit("no data in %s" % args.csv)
    model, case = build_model(args)
    K = (args.KI, args.KII, args.KIII)
    r = float(rows[0]["r"])
    norm = args.KI / np.sqrt(2.0 * np.pi * r)

    th = np.array([float(row["theta_deg"]) for row in rows])
    side = np.array([int(row["side"]) for row in rows])
    fe = np.array([[float(row[c]) for c in COMPONENTS] for row in rows]) / norm
    ana = np.array([analytical(model, float(row["x"]), float(row["y"]), int(row["side"]), K, args.a)
                    for row in rows]) / norm
    dev = np.abs(fe - ana)
    is_2d = bool(np.all(fe[:, 3:5] == 0.0))          # written as exact zeros for 2D (CPE) models

    folder = os.path.dirname(os.path.abspath(args.csv))
    prefix = args.out or os.path.basename(args.csv).replace("_circle.csv", "").replace(".csv", "")
    with open(os.path.join(folder, prefix + "_compare.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["theta_deg", "side"] + ["fe_" + c for c in COMPONENTS] + ["ana_" + c for c in COMPONENTS])
        for k in range(len(th)):
            w.writerow(["%.6f" % th[k], side[k]] + ["%.8e" % v for v in fe[k]] + ["%.8e" % v for v in ana[k]])

    print("Stress circle: %s" % args.csv)
    print("  case: %s" % case)
    print("  r = %.6f mm, K_I = %.3f MPa*sqrt(mm), reference stress K_I/sqrt(2 pi r) = %.1f MPa, %d points"
          % (r, args.KI, norm, len(th)))
    print("  max |FE - analytical| / (K_I/sqrt(2 pi r)) per component%s:" % (
        " (2D plane-strain model: sigma_23 = sigma_13 = 0 in the FE model)" if is_2d else ""))
    for c, name in enumerate(COMPONENTS):
        k = int(np.argmax(dev[:, c]))
        print("    %-4s %.4f  (at theta = %7.2f deg, side %+d; max |analytical| = %.3f)"
              % (name, dev[k, c], th[k], side[k], np.abs(ana[:, c]).max()))
    in_plane = [0, 1, 2, 5] if is_2d else list(range(6))
    print("  overall max (%s): %.4f" % ("in-plane components and s33" if is_2d else "all components",
                                       dev[:, in_plane].max()))

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for c, a in enumerate(axes.T.flatten()):
        bi = isinstance(model, ctf.InterfaceCrackModel)
        for sd, mk, col, lab in ((1, "o", "C0", "grain 1 (above)" if bi else "above crack plane"),
                                 (-1, "s", "C1", "grain 2 (below)" if bi else "below crack plane")):
            m = side == sd
            order = np.argsort(th[m])
            a.plot(th[m][order], ana[m, c][order], "-", color=col, lw=1.2, label="analytical, " + lab)
            a.plot(th[m], fe[m, c], mk, color=col, ms=3, mfc="none", label="FE, " + lab)
        a.set_title("%s  (max dev. %.3f)" % (LABELS[c], dev[:, c].max()))
        if max(np.abs(fe[:, c]).max(), np.abs(ana[:, c]).max()) < 1e-6:
            # component vanishes (e.g. anti-plane stresses for decoupled orientations): avoid scaling up round-off
            a.set_ylim(-0.05, 0.05)
            a.set_title(r"%s $\equiv 0$  (max dev. %.3f)" % (LABELS[c], dev[:, c].max()))
        a.set(xlim=(-180, 180))
        a.set_xticks(range(-180, 181, 90))
        a.grid(True, ls="--", alpha=0.5)
    for a in axes[1]:
        a.set_xlabel(r"$\theta$ [deg]")
    for a in axes[:, 0]:
        a.set_ylabel(r"$\sigma_{ij}\sqrt{2\pi r}/K_I$ [-]")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Stresses on the circle r = %.4f mm: %s" % (r, case), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(folder, prefix + "_compare.png"), dpi=200)
    plt.close(fig)
    print("-> written: %s_compare.csv, %s_compare.png" % (prefix, prefix))


if __name__ == "__main__":
    main()
