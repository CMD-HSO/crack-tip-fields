"""c6_numbers.py - numbers of Section 5.3 and 5.4 that are given in the text only (case C6, model R = 4 mm, K_I = 21.3 MPa sqrt(m)).

(1) comparison of KH and LKH at the same ratio K_I/tau_0 (KH at K_I = 14.1, interpolated between the load levels 13.3 and 16.0, LKH at 21.3):
    plastic zone for the thresholds 1e-4, 1e-3, 1e-2 of the total slip, mean number of active slip systems, back stress on the grain boundary,
    maximum of the opening stress on the grain boundary (normalised with K_I/sqrt(2 pi x_1); c6_interface.py, which uses the interface field as
    reference, gives values that differ by up to 0.2 percentage points), and the ratio of the normalised zones over K_I/tau_0
(2) reduction of the total slip of LKH against KH at the same K_I in the ring 0.01 mm < r < 0.1 mm and in three radial bins
(3) correlation of the resolved shear stress of the elastic K-field (generalised Schmid factor, c6_schmid.py) with the slip per system and element
Needs results/analysis_max1e-4, analysis_max1e-3 and analysis_max1e-2 (c6_analyze.py) of the runs C6_run3_R4mm_KH and C6_run4_R4mm_LKH.
Usage: python c6_numbers.py <folder with the C6_run* directories>"""
import os, sys
import numpy as np
from scipy.stats import spearmanr
import c6_schmid

B = sys.argv[1]
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = {"KH": "C6_run3_R4mm_KH", "LKH": "C6_run4_R4mm_LKH"}
TAU0 = c6_schmid.TAU0
N_EL = 5184                                   # elements of the mid-plane layer
KF = {f: 40.0 * f / 15.0 for f in range(1, 16)}   # K_I of the load level f [MPa sqrt(m)]
K_EQ = 21.3333 * TAU0["KH"] / TAU0["LKH"]      # K_I of KH at the same K_I/tau_0 as LKH at 21.3
W = (K_EQ - KF[5]) / (KF[6] - KF[5])           # linear weight between the load levels 5 and 6
SQ = np.sqrt(1000.0)


def summary(lab, thr):
    return np.genfromtxt(os.path.join(B, RUNS[lab], "results", "analysis_max" + thr, "summary.csv"), delimiter=",", names=True)


def elements(lab, f):
    return np.genfromtxt(os.path.join(B, RUNS[lab], "results", "export", "elements_frame%02d.csv" % f), delimiter=",", names=True)


def log_interp(s, col, K):
    return np.exp(np.interp(np.log(K), np.log(s["K_I"]), np.log(s[col])))


def ring(e, sgn, r1=0.01, r2=0.1):
    r = np.hypot(e["xc"], e["yc"])[:N_EL]
    return (sgn * e["yc"][:N_EL] > 0) & (r > r1) & (r < r2)


def slip(e, m):
    return np.abs(np.array([e["g%d" % i][:N_EL][m] for i in range(1, 13)]))      # 12 x elements


def backstress_gb(e, x_eval=0.005):
    c = np.column_stack([e["x%d" % i][:N_EL] if j == 0 else e["y%d" % i][:N_EL] for i in range(1, 5) for j in (0, 1)])
    gb = (np.abs(c[:, 1::2]).min(axis=1) < 1e-9) & (e["xc"][:N_EL] > 0)
    bs = np.abs(np.array([e["bs%d" % i][:N_EL] for i in range(1, 13)]))
    out = []
    for sgn in (1, -1):
        m = gb & (sgn * e["yc"][:N_EL] > 0); o = np.argsort(e["xc"][:N_EL][m]); idx = np.where(m)[0][o]
        out.append(np.interp(x_eval, e["xc"][:N_EL][idx], bs[:, idx].max(axis=0)))
    return np.array(out)


def s22_peak(lab, f, x_min=0.0075):    # without the first two nodes (x_1 <= 0.007 mm), as in the figure
    s = np.genfromtxt(os.path.join(B, RUNS[lab], "results", "ssy_f%d_s22.csv" % f), delimiter=",", names=True)
    m = s["x"] > x_min
    v = s["s22"][m] * np.sqrt(2 * np.pi * s["x"][m]) / (KF[f] * SQ)
    return 100 * (v.max() - 1)


print("(1) comparison at the same K_I/tau_0: KH at K_I = %.1f, LKH at 21.3 MPa sqrt(m)" % K_EQ)
for thr in ("1e-4", "1e-3", "1e-2"):
    kh, lkh = summary("KH", thr), summary("LKH", thr)
    print("  threshold %s:" % thr, ", ".join("%s LKH/KH = %.2f" % (c, log_interp(lkh, c, 21.3333) / log_interp(kh, c, K_EQ)) for c in ("r_max", "r_p_GB", "area")))
    x_k, x_l = kh["K_I"] * SQ / TAU0["KH"], lkh["K_I"] * SQ / TAU0["LKH"]
    m = (x_l >= x_k[2]) & (x_l <= x_k[7])
    rr = (lkh["r_max"][m] / x_l[m] ** 2) / np.interp(x_l[m], x_k[2:8], kh["r_max"][2:8] / x_k[2:8] ** 2)
    ra = (lkh["area"][m] / x_l[m] ** 4) / np.interp(x_l[m], x_k[2:8], kh["area"][2:8] / x_k[2:8] ** 4)
    print("     normalised zone LKH/KH for K_I/tau_0 = %.1f ... %.1f sqrt(mm): r_max %.2f ... %.2f, area %.2f ... %.2f"
          % (x_l[m].min(), x_l[m].max(), rr.min(), rr.max(), ra.min(), ra.max()))
e5, e6, e8k, e8l = elements("KH", 5), elements("KH", 6), elements("KH", 8), elements("LKH", 8)
for g, sgn in ((1, 1), (2, -1)):
    n_kh = (1 - W) * e5["n_act_max"][:N_EL][ring(e5, sgn)].mean() + W * e6["n_act_max"][:N_EL][ring(e6, sgn)].mean()
    print("  grain %d: mean number of active systems KH %.1f, LKH %.1f" % (g, n_kh, e8l["n_act_max"][:N_EL][ring(e8l, sgn)].mean()))
bs_kh = (1 - W) * backstress_gb(e5) + W * backstress_gb(e6)
print("  back stress on the grain boundary at x_1 = 0.005 mm [MPa]: KH %.0f / %.0f, LKH %.0f / %.0f (grain 1 / 2); KH at the same K_I: %.0f / %.0f"
      % (*bs_kh, *backstress_gb(e8l), *backstress_gb(e8k)))
print("  peak of the opening stress on the grain boundary above the K-field [%%]: KH %.1f (K_I = 8), %.1f (13.3), %.1f (16), %.1f (21.3); LKH %.1f (8), %.1f (21.3)"
      % (s22_peak("KH", 3), s22_peak("KH", 5), s22_peak("KH", 6), s22_peak("KH", 8), s22_peak("LKH", 3), s22_peak("LKH", 8)))

print("(2) total slip at the same K_I = 21.3 MPa sqrt(m), reduction of LKH against KH (mean over the elements)")
for g, sgn in ((1, 1), (2, -1)):
    parts = []
    for r1, r2 in ((0.01, 0.1), (0.01, 0.02), (0.02, 0.05), (0.05, 0.1)):
        a = slip(e8k, ring(e8k, sgn, r1, r2)).sum(axis=0).mean(); b = slip(e8l, ring(e8l, sgn, r1, r2)).sum(axis=0).mean()
        parts.append("%.2f-%.2f mm: %.0f %%" % (r1, r2, 100 * (1 - b / a)))
    print("  grain %d: " % g + ", ".join(parts))

print("(3) resolved shear stress of the elastic K-field against the slip per system (plastic elements, 0.02 mm < r < 0.3 mm)")
cache = {}
for lab in ("KH", "LKH"):
    for f in (3, 5, 8):
        e = elements(lab, f); x, y = e["xc"][:N_EL], e["yc"][:N_EL]; r = np.hypot(x, y)
        for g, sgn in ((1, 1), (2, -1)):
            idx = np.where((sgn * y > 0) & (r > 0.02) & (r < 0.3))[0]
            gam = slip(e, idx).T
            act = gam.sum(axis=1) > 1e-4
            if act.sum() < 5:
                print("  %-3s K_I = %4.1f grain %d: fewer than 5 plastic elements" % (lab, KF[f], g)); continue
            th = np.degrees(np.arctan2(y[idx], x[idx]))
            M = np.array([cache.setdefault((g, round(t, 2)), c6_schmid.m_alpha(REPO, g, t)) for t in th])
            tau = M * KF[f] * SQ / np.sqrt(2 * np.pi * r[idx])[:, None]
            hit = 100 * (tau.argmax(axis=1) == gam.argmax(axis=1))[act].mean()
            rho = spearmanr(tau[act].ravel(), gam[act].ravel()).correlation
            sel = (gam > 1e-6) & act[:, None]
            slope = np.polyfit(np.log(tau[sel]), np.log(gam[sel]), 1)[0]
            print("  %-3s K_I = %4.1f grain %d: %4d elements, largest tau = largest slip in %.0f %%, rank correlation %.2f, slope log(slip)/log(tau) %.1f"
                  % (lab, KF[f], g, act.sum(), hit, rho, slope))
