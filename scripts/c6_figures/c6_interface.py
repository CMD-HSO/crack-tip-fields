"""c6_interface.py - opening stress sigma22 and shear stress sigma12 on the interface ahead of the tip (mid-plane nodes), FE minus K-field.
Model R = 4 mm, KH and LKH.  Deviation of sigma22 relative to the K-field; sigma12: difference relative to K_I / sqrt(2 pi x).
Usage: python c6_interface.py <folder with the C6_run* directories> <repository root> <output png>"""
import os, sys
B, REPO, OUTPNG = sys.argv[1], sys.argv[2], sys.argv[3]
sys.argv = ["c6_kfield.py", B, REPO, "."]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c6_kfield import *            # model, ana_stress, np, plt, os, csv (module part only)
import c6_style
cases = [("KH", "C6_run3_R4mm_KH", "-"), ("LKH", "C6_run4_R4mm_LKH", "--")]
loads = [(3, 8.0, "#e9a3c9"), (6, 16.0, "#c51b7d"), (8, 21.333, "#4d004b")]
X0 = np.array([0.0071, 0.0107, 0.02, 0.05, 0.1, 0.3, 1.0])


def line(d, N):
    s = np.genfromtxt(os.path.join(B, d, "results", "export", "nodal_stress_frame%02d.csv" % N), delimiter=",", names=True, dtype=None, encoding=None)
    l = s[s["group"] == "line"]
    xs = np.unique(np.round(l["x"], 8))
    return xs, {c: np.array([l[c][np.round(l["x"], 8) == x].mean() for x in xs]) for c in ("s22", "s12")}


def rp_gb(B_, d_, N_):
    """length of the plastic zone along the grain boundary r_p (criterion: max slip > 1e-4), from analysis_max1e-4/summary.csv"""
    s_ = np.genfromtxt(os.path.join(B_, d_, "results", "analysis_max1e-4", "summary.csv"), delimiter=",", names=True)
    return float(s_["r_p_GB"][N_ - 1])


res = {}
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
for N, K, col in loads:
    Kmm = K * np.sqrt(1000.0)
    for lab, d, st in cases:
        colm = c6_style.shade(lab, K)
        x, s = line(d, N)
        m = (x >= 0.0107) & (x <= 1.0)
        ana = np.array([ana_stress(xx, 0.0, 1, Kmm) for xx in x[m]])
        ref = Kmm / np.sqrt(2 * np.pi * x[m])
        d22 = 100 * (s["s22"][m] / ana[:, 1] - 1)
        d12 = 100 * (s["s12"][m] - ana[:, 5]) / ref
        ax[0].plot(x[m], d22, st, color=colm, lw=1.4)
        ax[1].plot(x[m], d12, st, color=colm, lw=1.4)
        for a_ in ax:
            a_.plot([rp_gb(B, d, N)], [0], marker="^", ms=4.5, mfc=colm if lab == "KH" else "white", mec=colm, ls="none", transform=a_.get_xaxis_transform(), clip_on=False, zorder=5)
        res[(K, lab)] = (np.interp(X0, x[m], d22), np.interp(X0, x[m], d12))
for a, t, yl in zip(ax, ("(a)", "(b)"), (r"$\sigma_{22}$: FE / $K$-field $-1$ [%]", r"$\sigma_{12}$: (FE $-$ $K$-field) / ($K_I/\sqrt{2\pi x_1}$) [%]")):
    a.set_xlim(0, 0.2); a.set_xlabel(r"$x_1$ [mm]"); a.set_ylabel(yl); a.grid(alpha=.3); a.axhline(0, color="k", lw=.5)
    a.text(0.96, 0.93, t, transform=a.transAxes, va="top", ha="right")
c6_style.k_legend(ax[0], loc="lower right", fontsize=8, columnspacing=0.7, handlelength=1.4, labelspacing=0.25)
fig.tight_layout(); fig.savefig(OUTPNG, dpi=300)
print("x =", X0)
for (K, lab), (d22, d12) in res.items():
    print("K=%5.2f %-3s s22 dev [%%]:" % (K, lab), np.round(d22, 1), " s12 dev [%% of ref]:", np.round(d12, 1))
