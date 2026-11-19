"""c6_overview.py - four key diagrams of the C6 domain-size study (R = 1, 4, 10 mm; KH and LKH).
Usage: python c6_overview.py <folder with the C6_run* directories> <output png> [K_max]"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B, OUT = sys.argv[1], sys.argv[2]
PAPER = os.path.basename(OUT).startswith("fe_")
if PAPER:
    import c6_style
SC = 0.62 if PAPER else 1.0
KMAX = float(sys.argv[3]) if len(sys.argv) > 3 else 21.4
SUB = sys.argv[4] if len(sys.argv) > 4 else "analysis_max1e-4"      # folder with summary.csv (criterion: slip > 1e-4 in at least one integration point)
ZNOTE = "  (slip > 1e-4)" if "max" in SUB else ""
CRIT = "plastic: max over the integration points > 1e-4" if "max" in SUB else "plastic: mean over the integration points > 1e-3"
runs = [("R = 1 mm", "C6_run1_R1mm_KH", "C6_run2_R1mm_LKH", "#1b6ca8", "o", 2.6, 6.5),
        ("R = 4 mm", "C6_run3_R4mm_KH", "C6_run4_R4mm_LKH", "#d98200", "s", 2.6, 6.5),
        ("R = 10 mm", "C6_run6_R10mm_KH", "C6_run5_R10mm_LKH", "#2e8b57", "^", 2.6, 6.5)]
SLIM = len(sys.argv) > 5 and sys.argv[5] == "slim"      # only R = 1 and 4 mm (runs 1-4)
if SLIM:
    runs = runs[:2]
D = {}
for lab, kh, lkh, *_ in runs:
    for tag, d in (("KH", kh), ("LKH", lkh)):
        D[(lab, tag)] = np.genfromtxt(os.path.join(B, d, "results", SUB, "summary.csv"), delimiter=",", names=True)

KD = {}      # K deviation from J on contour 15 (outermost), J_mid, J40 = 9.4090 N/mm
for (lab_, tag_), d_ in D.items():
    run_dir = [r for r in runs if r[0] == lab_][0][1 if tag_ == "KH" else 2]
    kd = []
    for N_, K_ in zip(d_["frame"].astype(int), d_["K_I"]):
        J_ = np.genfromtxt(os.path.join(B, run_dir, "results", "ssy_f%d_J.csv" % N_), delimiter=",", names=True)
        kd.append(100.0 * (np.sqrt(J_["J_mid"][14] / (9.4090 * (K_ / 40.0) ** 2)) - 1.0))
    KD[(lab_, tag_)] = np.array(kd)
panels = [("area", (r"area of the zone $A$ [mm$^2$]" if PAPER else "area of the plastic zone [mm$^2$]"), "(a) plastic zone area" + ZNOTE),
          ("r_max_over_R", (r"$r_{\max}/R$ [-]" if PAPER else "$r_{max}$ / R [-]"), "(b) zone size relative to the model radius" + ZNOTE),
          ("K_dev_percent", (r"deviation $\delta_K$ [%]" if PAPER else "K deviation from J [%]"), "(c) K from $J$ (contour 15) minus prescribed K"),
          ("CTOD_fix", ("CTOD [mm]" if PAPER else "CTOD at 0.01 mm behind the tip [mm]"), "(d) crack-tip opening")]
fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.3) if PAPER else (11, 8))
for a, (key, yl, ttl) in zip(ax.ravel(), panels):
    for lab, kh, lkh, c, mk, lw, ms in runs:
        for tag, st in (("KH", "-"), ("LKH", "--")):
            d = D[(lab, tag)]; m = d["K_I"] <= KMAX
            a.plot(d["K_I"][m], (KD[(lab, tag)] if key == "K_dev_percent" else d[key])[m], st, color=(c6_style.COL[tag] if PAPER else c), lw=lw * SC, marker=mk, ms=ms * (0.5 if PAPER else 1.0) * SC / (SC if PAPER else 1.0), alpha=0.7 if PAPER else 0.85, mfc="none" if tag == "LKH" else (c6_style.COL[tag] if PAPER else c),
                   label=None if PAPER else "%s, %s" % (lab, tag), zorder=4 - runs.index((lab, kh, lkh, c, mk, lw, ms)))
    a.set_xlabel(r"prescribed $K_I$ [MPa$\sqrt{\mathrm{m}}$]"); a.set_ylabel(yl); (a.text(0.04, 0.07 if ttl[:3] == "(c)" else 0.93, ttl[:3], transform=a.transAxes, va="bottom" if ttl[:3] == "(c)" else "top") if PAPER else a.set_title(ttl, fontsize=10)); a.grid(alpha=.3)
if PAPER:
    from matplotlib.lines import Line2D
    hh = [Line2D([], [], color=c6_style.COL["KH"], lw=1.6, label="KH"), Line2D([], [], color=c6_style.COL["LKH"], lw=1.6, ls="--", label="LKH")]
    hh += [Line2D([], [], color="0.3", marker=r_[4], ls="none", ms=4, label="$R=%s$ mm" % r_[0].split()[2]) for r_ in runs]
    ax[0, 0].legend(handles=hh, fontsize=9, loc="upper left", bbox_to_anchor=(0.12, 0.98), frameon=False)
else:
    ax[0, 0].legend(fontsize=8, ncol=1, loc="upper left")
if not os.path.basename(OUT).startswith("fe_"):      # paper version: no title in the image
    fig.suptitle(("C6 bicrystal (solid: KH, dashed: LKH)" if SLIM else "C6 bicrystal, domain size study (solid: KH, dashed/open: LKH)") , fontsize=10)
fig.tight_layout(); fig.savefig(OUT, dpi=300 if PAPER else 140)
