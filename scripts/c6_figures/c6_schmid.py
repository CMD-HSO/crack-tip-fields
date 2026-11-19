"""c6_schmid.py - resolved shear stress of the 12 slip systems in the mode I K-field of the grain boundary crack (case C6) and the elastic estimate of the plastic zone.

m_alpha(theta) = tau_alpha(r, theta) * sqrt(2 pi r) / K_I : generalisation of the Schmid factor to the crack-tip field (tau_alpha = n_alpha . sigma . m_alpha, sigma from the
interface field of crack_tip_fields, crystal orientation of grain 1 (105, 60, 30) deg and grain 2 (0, 0, 0) deg, slip systems as in Table 'tab:slip-systems' of the manuscript).
Elastic estimate of the zone: the slip rate of the power law gamma_dot = gamma_dot_0 (tau/tau_0)^n with tau = m K(t) / sqrt(2 pi r) and K(t) = K_dot t gives, integrated in time,
gamma = gamma_dot_0 t (tau/tau_0)^n / (n + 1); the boundary of the zone is where gamma = thr :
    r(theta) = 1/(2 pi) (K m_max(theta) / tau_0)^2 * [gamma_dot_0 t / (thr (n + 1))]^(2/n),   t = K / K_dot,   m_max(theta) = max_alpha |m_alpha(theta)|.
Usage: python c6_schmid.py <folder with the C6_run* directories> <repository root> <output pdf>     (figure: estimate against the FE zone at K_I = 21.3)"""
import os, sys
import numpy as np
from scipy.spatial.transform import Rotation

SYSTEMS = [((1, 1, 1), (1, 0, -1)), ((1, 1, 1), (-1, 1, 0)), ((1, 1, 1), (0, -1, 1)),
           ((-1, 1, 1), (0, 1, -1)), ((-1, 1, 1), (-1, -1, 0)), ((-1, 1, 1), (1, 0, 1)),
           ((-1, -1, 1), (-1, 0, -1)), ((-1, -1, 1), (1, -1, 0)), ((-1, -1, 1), (0, 1, 1)),
           ((1, -1, 1), (0, -1, -1)), ((1, -1, 1), (1, 1, 0)), ((1, -1, 1), (-1, 0, 1))]
ORIENTATION = {1: (105.0, 60.0, 30.0), 2: (0.0, 0.0, 0.0)}      # Bunge angles of grain 1 (above the crack plane) and grain 2 (below)
TAU0 = {"KH": 168.89, "LKH": 254.96}                             # MPa
N_EXP, GDOT0, KDOT, THR = 10.0, 7.5e-4, 40.0, 1.0e-4            # rate exponent, reference rate [1/s], K rate [MPa sqrt(m)/s], threshold of the total slip
A_REF = 20.0                                                      # reference length of the interface field [mm]
_model = {}


def _get_model(repo):
    if "m" not in _model:
        sys.path.insert(0, os.path.join(repo, "src"))
        import crack_tip_fields as ctf
        _model["m"] = ctf.InterfaceCrackModel("In718 (anisotropic)", ORIENTATION[1], "In718 (anisotropic)", ORIENTATION[2])
    return _model["m"]


def m_alpha(repo, grain, theta_deg, r=0.1):
    """|m_alpha| of the 12 systems at the angle theta (degrees; grain 1: 0..180, grain 2: -180..0), K_I = 1 (the result is independent of r)"""
    model = _get_model(repo)
    side = "top" if grain == 1 else "bot"
    x1, x2 = r * np.cos(np.radians(theta_deg)), (0.0 if abs(theta_deg) < 1e-9 else r * np.sin(np.radians(theta_deg)))
    s11, s22, s33, s23, s13, s12 = model.stress(x1, x2, 1.0, 0.0, 0.0, A_REF, side)
    S = np.array([[s11, s12, s13], [s12, s22, s23], [s13, s23, s33]])
    R = Rotation.from_euler("ZXZ", ORIENTATION[grain], degrees=True).as_matrix()      # crystal -> global, as in rotate_stiffness_euler of the toolkit
    out = []
    for n, m in SYSTEMS:
        n_g = R @ (np.array(n, float) / np.linalg.norm(n)); m_g = R @ (np.array(m, float) / np.linalg.norm(m))
        out.append(n_g @ S @ m_g)
    return np.abs(np.array(out)) * np.sqrt(2 * np.pi * r)


def angles(grain, n=361):
    return np.linspace(0.0, 180.0, n) if grain == 1 else np.linspace(-180.0, 0.0, n)


def m_table(repo):
    """{grain: (m at theta = 0 (grain boundary), max over theta, max_alpha m(theta), theta array)}"""
    out = {}
    for g in (1, 2):
        th = angles(g); M = np.array([m_alpha(repo, g, t) for t in th])
        out[g] = (M[0 if g == 1 else -1], M.max(axis=0), M.max(axis=1), th)
    return out


def r_estimate(m_max, K, model):
    """estimated boundary of the plastic zone r(theta) [mm] at K_I [MPa sqrt(m)] for the hardening model KH / LKH"""
    Kmm = K * np.sqrt(1000.0); t = K / KDOT
    return (Kmm * m_max / TAU0[model]) ** 2 / (2 * np.pi) * (GDOT0 * t / (THR * (N_EXP + 1))) ** (2.0 / N_EXP)


if __name__ == "__main__":
    B, REPO, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
    import c6_style
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = m_table(REPO)
    print("m_alpha at theta = 0 / maximum over theta:")
    for g in (1, 2):
        print(" grain %d  m(0)  :" % g, np.round(T[g][0], 3)); print("          m_max :", np.round(T[g][1], 3))
    K = 21.3333; n = 5184
    runs = {"KH": "C6_run3_R4mm_KH", "LKH": "C6_run4_R4mm_LKH"}
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)
    for a, (lab, d) in zip(ax, runs.items()):
        s = np.genfromtxt(os.path.join(B, d, "results", "analysis_max1e-4", "summary.csv"), delimiter=",", names=True)
        e = np.genfromtxt(os.path.join(B, d, "results", "export", "elements_frame08.csv"), delimiter=",", names=True)
        pl = e["g_max"][:n] > THR
        a.scatter(e["xc"][:n][pl], e["yc"][:n][pl], s=1.5, color="0.65", lw=0, rasterized=True, label="FE: plastic elements")
        for g in (1, 2):
            th = T[g][3]; r = r_estimate(T[g][2], K, lab)
            a.plot(r * np.cos(np.radians(th)), r * np.sin(np.radians(th)), color=c6_style.COL[lab], lw=1.4, label="estimate" if g == 1 else None)
        a.set_xlim(-0.15, 0.3); a.set_ylim(-0.45, 0.45); a.set_aspect("equal"); a.grid(alpha=.3)
        a.set_xlabel(r"$x_1$ [mm]"); a.text(0.04, 0.95, "(%s) %s" % ("ab"[list(runs).index(lab)], lab), transform=a.transAxes, va="top")
    ax[0].set_ylabel(r"$x_2$ [mm]")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color="0.65", ls="none", marker="o", ms=3, label="FE: plastic elements"), Line2D([], [], color="k", lw=1.4, label="elastic estimate")], loc="lower center", ncol=2, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(OUT, dpi=300)
    print("estimate r_max / r_p(theta=0) at K = %.1f [mm]  (FE: r_max = 0.436 / 0.200, r_p = 0.069 / 0.036)" % K)
    for lab in runs:
        rm = max(r_estimate(T[g][2], K, lab).max() for g in (1, 2)); rp = max(r_estimate(T[g][2][0 if g == 1 else -1], K, lab) for g in (1, 2))
        print("  %-3s  r_max %.3f   r_p %.3f" % (lab, rm, rp))
    print("saved", OUT)
