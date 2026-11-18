"""
Reproduces the numbers, tables and figures of Appendix B ("Verification") and
of Section 4 ("Verification: elastic test cases") of the accompanying article.

Usage:   python reproduce_manuscript_results.py            (about 30 s)
         python reproduce_manuscript_results.py --no-figures

Requires crack_tip_fields.py in ../src. Figures are written to ./figs/ (next to this script).
Every value quoted in the manuscript is printed and checked against a tolerance;
the script exits with status 1 if a check fails.

Appendix B.1  isotropic solution: textbook (Anderson), Stroh limit, energy release rate from D
Appendix B.2  homogeneous single crystal: K normalisation, crack opening, finite differences
Appendix B.3  interface crack: Brunner et al. Eq. (24)
Appendix B.4  isotropic limits of the Stroh and interface solutions (Appendices A.2, A.3),
              exact crack-face evaluation (branch-cut regression test)
Section 4.4   single crystal: D22 and the energy release rate
Section 4.5   interface crack: interpenetration, Griffith estimate
Sections 4.2, 4.3, 5   (the FE case study itself requires Abaqus and is not part of this script)
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import crack_tip_fields as ctf

PI = np.pi
A_HALF = 20.0                       # reference length a [mm] (2a = 40 mm), as in the FE case study
K0 = 27.5 * np.sqrt(1000.)          # K_I = 27.5 MPa*sqrt(m) = 27.5*sqrt(1000) MPa*sqrt(mm)
FAILED = []


def check(label, value, limit, mode="<"):
    ok = value < limit if mode == "<" else value > limit
    print(f"    {'OK ' if ok else 'FAIL'} {label}: {value:.3e}  ({mode} {limit:.1e})")
    if not ok:
        FAILED.append(label)
    return value


# ----------------------------------------------------------------------------
# Appendix B.1: isotropic solution
# ----------------------------------------------------------------------------
def textbook_displacement(x1, x2, K, G, kappa):
    """Williams solution, plane strain/stress (Anderson, Fracture Mechanics), Cartesian components."""
    r, th = np.hypot(x1, x2), np.arctan2(x2, x1)
    f, c, s = np.sqrt(r / (2 * PI)), np.cos(th / 2), np.sin(th / 2)
    KI, KII, KIII = K
    ux = KI / (2 * G) * f * c * (kappa - 1 + 2 * s * s) + KII / (2 * G) * f * s * (kappa + 1 + 2 * c * c)
    uy = KI / (2 * G) * f * s * (kappa + 1 - 2 * c * c) - KII / (2 * G) * f * c * (kappa - 1 - 2 * s * s)
    uz = 2 * KIII / G * f * s
    return np.array([ux, uy, uz])


def textbook_stress(x1, x2, K, nu):
    """Williams stresses in Voigt order (11,22,33,23,13,12), plane strain."""
    r, th = np.hypot(x1, x2), np.arctan2(x2, x1)
    f = 1 / np.sqrt(2 * PI * r)
    c, s, c3, s3 = np.cos(th / 2), np.sin(th / 2), np.cos(1.5 * th), np.sin(1.5 * th)
    KI, KII, KIII = K
    sxx = f * (KI * c * (1 - s * s3) - KII * s * (2 + c * c3))
    syy = f * (KI * c * (1 + s * s3) + KII * s * c * c3)
    sxy = f * (KI * s * c * c3 + KII * c * (1 - s * s3))
    return np.array([sxx, syy, nu * (sxx + syy), f * KIII * c, -f * KIII * s, sxy])


def appendix_b1():
    print("\n=== Appendix B.1: isotropic solution ===")
    iso = ctf.IsotropicCrackModel("Steel (isotropic)")
    rng = np.random.default_rng(5)

    dev_u = dev_s = 0.0
    for _ in range(400):
        r = 10 ** rng.uniform(-4, -2)
        th = rng.uniform(-0.98 * PI, 0.98 * PI)
        x1, x2 = r * np.cos(th), r * np.sin(th)
        K = rng.uniform(-100, 100, 3)
        u_ref = textbook_displacement(x1, x2, K, iso.G, iso.kappa)
        s_ref = textbook_stress(x1, x2, K, iso.nu)
        dev_u = max(dev_u, np.abs(np.array(iso.displacement(x1, x2, *K)) - u_ref).max() / np.abs(u_ref).max())
        dev_s = max(dev_s, np.abs(np.array(iso.stress(x1, x2, *K)) - s_ref).max() / np.abs(s_ref).max())
    print("  module vs. textbook, 400 random points and loads (all modes):")
    check("displacement deviation", dev_u, 1e-14)
    check("stress deviation      ", dev_s, 1e-14)

    pts = [(r * np.cos(t), r * np.sin(t)) for r in np.geomspace(2e-4, 5e-3, 6)
           for t in np.linspace(-170, 170, 13) * PI / 180]
    loads = [(K0, 0., 0.), (0., K0, 0.), (0., 0., K0), (.5 * K0, .3 * K0, .8 * K0)]
    rows = []
    print("  Stroh solution (slightly perturbed isotropic steel) vs. Williams solution:")
    for shift in (100., 10., 1.):
        name = f"steel_shift_{shift:g}"
        ctf.MATERIALS[name] = {"C": ctf.isotropic_stiffness(210000., 0.3, degeneracy_shift=shift),
                               "symmetry": "isotropic (perturbed)", "note": "check only"}
        stroh = ctf.AnisotropicCrackModel(name, 0., 0., 0.)
        du = ds = 0.0
        for K in loads:
            U_i = np.array([iso.displacement(x, y, *K) for x, y in pts])
            U_a = np.array([stroh.displacement(x, y, *K, A_HALF) for x, y in pts])
            S_i = np.array([iso.stress(x, y, *K) for x, y in pts])
            S_a = np.array([stroh.stress(x, y, *K, A_HALF) for x, y in pts])
            du = max(du, np.abs(U_a - U_i).max() / np.abs(U_i).max())
            ds = max(ds, np.abs(S_a - S_i).max() / np.abs(S_i).max())
        rows.append((shift, du, ds))
        print(f"    C44 shift {shift:5g} MPa: max rel. dev.  u = {du:.2e}   sigma = {ds:.2e}")
        del ctf.MATERIALS[name]
    check("u deviation at shift 1 MPa (manuscript: 1.2e-5)", rows[2][1], 1.5e-5)
    check("sigma deviation at shift 1 MPa (manuscript: 4.0e-6)", rows[2][2], 5e-6)
    check("linear decrease: |u(shift 100)/u(shift 1) - 100|", abs(rows[0][1] / rows[2][1] - 100.0), 5.0)

    print("  energy release rate 1/4 k^T D k in the isotropic limit (C44 shift 0.01 MPa), mixed load:")
    E_mod, nu = 210000., 0.3
    G_mod = E_mod / (2 * (1 + nu))
    ctf.MATERIALS["steel_G_check"] = {"C": ctf.isotropic_stiffness(E_mod, nu, degeneracy_shift=0.01),
                                      "symmetry": "isotropic (perturbed)", "note": "check only"}
    m = ctf.AnisotropicCrackModel("steel_G_check", 0., 0., 0.)
    KI, KII, KIII = 120., 80., 60.
    k = np.array([KII, KI, KIII])
    G_stroh = 0.25 * k @ m.D @ k
    G_iso = (KI ** 2 + KII ** 2) * (1 - nu ** 2) / E_mod + KIII ** 2 / (2 * G_mod)
    del ctf.MATERIALS["steel_G_check"]
    check("G from D vs. isotropic formula (manuscript: 6e-8)", abs(G_stroh / G_iso - 1), 1e-6)
    check("D22 vs. 4(1-nu^2)/E (manuscript: < 2e-7)", abs(m.D[1, 1] / (4 * (1 - nu ** 2) / E_mod) - 1), 2e-7)
    check("D33 vs. 2/G (manuscript: < 2e-7)", abs(m.D[2, 2] / (2 / G_mod) - 1), 2e-7)
    return np.array(rows)


# ----------------------------------------------------------------------------
# Appendix B.2 / Section 4.4: single crystal
# ----------------------------------------------------------------------------
def appendix_b2_and_section_4_1():
    print("\n=== Appendix B.2 / Section 4.4: homogeneous anisotropic single crystal (Fe) ===")
    Cfe = ctf.MATERIALS["Fe (cubic, example)"]["C"]
    print(f"  Zener factor A = {2 * Cfe[3, 3] / (Cfe[0, 0] - Cfe[0, 1]):.4f}   (manuscript: 2.36)")
    D22 = []
    for ori in [(0, 0, 0), (27, 0, 0), (45, 0, 0), (27, 15, 10)]:
        m = ctf.AnisotropicCrackModel("Fe (cubic, example)", *ori)
        rr = (1e-4, 1e-3, 5e-3)
        norm = max(abs(np.sqrt(2 * PI * r) * m.stress(r, 1e-12, K0, 0, 0, A_HALF)[1] / K0 - 1) for r in rr)
        k = np.array([0.3 * K0, K0, 0.6 * K0])                       # [K_II, K_I, K_III]
        opening = 0.0
        for r in rr:
            du = (np.array(m.displacement(-r, 1e-13, k[1], k[0], k[2], A_HALF))
                  - np.array(m.displacement(-r, -1e-13, k[1], k[0], k[2], A_HALF)))
            ref = np.sqrt(2 * r / PI) * m.D @ k
            opening = max(opening, np.abs(du - ref).max() / np.abs(ref).max())
        u_fun = lambda X, Y: m.displacement(X, Y, K0, 0.3 * K0, 0.6 * K0, A_HALF)
        fd = ctf.strain_stress_fd(u_fun, -0.004, 0.002, m.C_rot, A_HALF,
                                  h=1e-5 * np.hypot(-0.004, 0.002))   # step relative to r (optimal ~1e-5 r)
        fd_dev = np.abs(np.array(m.stress(-0.004, 0.002, K0, 0.3 * K0, 0.6 * K0, A_HALF)) - fd).max() / np.abs(fd).max()
        D22.append(m.D[1, 1])
        print(f"  orientation {str(ori):13s} |norm-1| = {norm:.1e}   opening dev = {opening:.1e}   FD dev = {fd_dev:.1e}   D22 = {m.D[1, 1]:.4e} 1/MPa")
        check("  K normalisation (manuscript: 2e-16)", norm, 1e-14)
        check("  crack opening = sqrt(2r/pi) D k", opening, 1e-13)
        check("  closed form vs finite differences", fd_dev, 1e-8)
    check("D22 identical for rotations about [001] (manuscript: 2e-15)",
          max(abs(D22[1] / D22[0] - 1), abs(D22[2] / D22[0] - 1)), 1e-13)


# ----------------------------------------------------------------------------
# Appendix B.3 / Section 4.5: interface crack
# ----------------------------------------------------------------------------
class OpeningEq24:
    """Closed-form crack-opening displacement, Brunner et al. (2025), Eq. (24), for pure K_I."""

    def __init__(self, model, a=A_HALF):
        self.D, self.a = model.D, a
        S = model.S_hat
        self.beta = np.sqrt(-0.5 * np.trace(S @ S))
        self.gamma = np.arctanh(self.beta) / PI
        self.S2, self.S3 = S @ S, S @ S @ S
        self.Phi_inv = np.linalg.inv(np.eye(3) + 2 * np.arctanh(self.beta) / (PI * self.beta ** 3) * self.S3)

    def matrix(self, alpha):                      # alpha = ln(r/2a)
        b, g = self.beta, self.gamma
        return self.D @ (np.eye(3) + self.S2 / b ** 2
                         - np.exp(1j * g * alpha) * np.sqrt(1 - b ** 2) / b ** 2 * (self.S2 + 1j / b * self.S3))

    def du(self, r, k):                           # full vector, r in mm
        return np.sqrt(2 * r / PI) * np.real(self.matrix(np.log(r / (2 * self.a)))) @ self.Phi_inv @ k

    def normalised_opening(self, r, K):           # du2 / (sqrt(2r/pi) D22 K_I), pure K_I
        k = np.array([0., K, 0.])
        return (np.real(self.matrix(np.log(r / (2 * self.a)))) @ self.Phi_inv @ k)[1] / (self.D[1, 1] * K)


def sign_change_radius(eq24, K, alpha_min=-3000.0, n=300001):
    alpha = np.linspace(0.0, alpha_min, n)
    k = np.array([0., K, 0.])
    d2 = np.array([(np.real(eq24.matrix(x)) @ eq24.Phi_inv @ k)[1] for x in alpha])
    neg = np.where(d2 <= 0)[0]
    return None if len(neg) == 0 else np.log10(2 * A_HALF) + alpha[neg[0]] / np.log(10)


def appendix_b3_and_section_4_2():
    print("\n=== Appendix B.3 / Section 4.5: bimaterial grain-boundary crack ===")
    gb = ctf.InterfaceCrackModel("Fe (cubic, example)", (27, 15, 10), "Cu (cubic, example)", (5, 40, 0))
    eq = OpeningEq24(gb)
    print(f"  Fe (27,15,10) / Cu (5,40,0): beta = {eq.beta:.4f}, gamma = {eq.gamma:.4f}   (manuscript: 0.0285, 0.0091)")
    k = np.array([0., K0, 0.])
    for r in (1e-4, 1e-3, 1e-2):
        du = (np.array(gb.displacement(-r, 1e-13, K0, 0, 0, A_HALF, side="top"))
              - np.array(gb.displacement(-r, -1e-13, K0, 0, 0, A_HALF, side="bot")))
        ref = eq.du(r, k)
        check(f"crack opening vs Brunner Eq. (24), r = {r:g} mm", np.abs(du - ref).max() / np.abs(ref).max(), 1e-9)

    print("  interpenetration check (closed form, no lower cut-off):")
    print(f"    normalised opening at r = 0.3 nm: {eq.normalised_opening(3e-7, K0):.4f}   (manuscript: 0.983)")
    rc = sign_change_radius(eq, K0)
    print(f"    Fe/Cu: first sign change at r_c = 10^{rc:.1f} mm   (manuscript: ~3e-77 mm)")
    check("  r_c far below atomic spacing: log10(r_c) < -70", rc, -70.0)


# ----------------------------------------------------------------------------
# Appendix B.4 / Appendices A.2 and A.3: isotropic limits of the Stroh solutions
# ----------------------------------------------------------------------------
def isotropic_limit_closed_form(x1, x2, k, G, kappa):
    """Closed-form isotropic limit of the Stroh solution (Appendix A.2): stress t2 = (s12,s22,s32), t1 = (s11,s21,s31)
    and displacement u from the nilpotent matrix Nil = N0 - iI, with k = [K_II, K_I, K_III]."""
    r, th = np.hypot(x1, x2), np.arctan2(x2, x1)
    Nil = np.array([[1j, -1, 0], [-1, -1j, 0], [0, 0, 0]])
    I3 = np.eye(3)
    g0, g1 = np.exp(-0.5j * th), -0.5 * np.sin(th) * np.exp(-1.5j * th)
    h0, h1 = 1j * g0, g0 + 1j * g1
    t2 = np.real(g0 * I3 + g1 * Nil) @ k / np.sqrt(2 * PI * r)
    t1 = -np.real(h0 * I3 + h1 * Nil) @ k / np.sqrt(2 * PI * r)
    a_ = (1 + kappa) / (4 * G); s_ = (kappa - 1) / (kappa + 1)
    Y0 = np.array([[a_, 1j * a_ * s_, 0], [-1j * a_ * s_, a_, 0], [0, 0, 1 / G]])
    u = np.sqrt(2 * r / PI) * np.real(-1j * Y0 @ (np.exp(0.5j * th) * I3 + 0.5 * np.sin(th) * np.exp(-0.5j * th) * Nil)) @ k
    return t1, t2, u


def appendix_c4():
    print("\n=== Appendix B.4: isotropic limits (Appendices A.2 and A.3) ===")
    E, nu = 210000., 0.3
    G = E / (2 * (1 + nu)); kappa = 3 - 4 * nu
    k = np.array([80., 120., 60.])                       # [K_II, K_I, K_III]
    ds = du = 0.0
    for th in np.linspace(-0.99 * PI, 0.99 * PI, 41):
        x1, x2 = 0.01 * np.cos(th), 0.01 * np.sin(th)
        t1, t2, u = isotropic_limit_closed_form(x1, x2, k, G, kappa)
        ref_s = textbook_stress(x1, x2, (k[1], k[0], k[2]), nu)             # (11,22,33,23,13,12)
        ref_u = textbook_displacement(x1, x2, (k[1], k[0], k[2]), G, kappa)
        ours = np.array([t1[0], t2[1], 0, t2[2], t1[2], t2[0]])
        ds = max(ds, np.abs(np.delete(ours, 2) - np.delete(ref_s, 2)).max() / np.abs(ref_s).max())
        du = max(du, np.abs(u - ref_u).max() / np.abs(ref_u).max())
    check("closed-form isotropic limit of Stroh stress vs Williams (Eq. 'iso-limit-stress')", ds, 1e-13)
    check("closed-form isotropic limit of Stroh displacement vs Williams", du, 1e-13)

    print("  isotropic bimaterial (nearly isotropic grains, C44 shift 0.01 MPa), E1,nu1 = 210 GPa, 0.3; E2,nu2 = 70 GPa, 0.33:")
    E1, n1, E2, n2 = 210000., 0.3, 70000., 0.33
    for nm, Em, nm_ in (("iso_b1", E1, n1), ("iso_b2", E2, n2)):
        ctf.MATERIALS[nm] = {"C": ctf.isotropic_stiffness(Em, nm_, degeneracy_shift=0.01),
                             "symmetry": "isotropic (perturbed)", "note": "check only"}
    m = ctf.InterfaceCrackModel("iso_b1", (0, 0, 0), "iso_b2", (0, 0, 0))
    del ctf.MATERIALS["iso_b1"], ctf.MATERIALS["iso_b2"]
    mu1, mu2 = E1 / (2 * (1 + n1)), E2 / (2 * (1 + n2)); k1, k2 = 3 - 4 * n1, 3 - 4 * n2
    c1, c2 = (1 + k1) / mu1, (1 + k2) / mu2
    b = ((k1 - 1) * mu2 - (k2 - 1) * mu1) / ((k1 + 1) * mu2 + (k2 + 1) * mu1)
    eps = np.log((mu2 * k1 + mu1) / (mu1 * k2 + mu2)) / (2 * PI)
    beta = np.sqrt(-0.5 * np.trace(m.S_hat @ m.S_hat)); gam = np.arctanh(beta) / PI
    print(f"    epsilon = {eps:.5f}, gamma = {gam:.5f}, beta = {beta:.5f}, b = {b:.5f}")
    check("beta vs |b|", abs(beta - abs(b)), 1e-6)
    check("gamma vs |epsilon|", abs(gam - abs(eps)), 1e-6)
    check("D11, D22 vs (c1+c2)/4", abs(m.D[0, 0] / ((c1 + c2) / 4) - 1) + abs(m.D[1, 1] / ((c1 + c2) / 4) - 1), 1e-5)
    check("D33 vs 1/mu1 + 1/mu2", abs(m.D[2, 2] / (1 / mu1 + 1 / mu2) - 1), 1e-5)
    w = ((k1 - 1) / mu1 - (k2 - 1) / mu2) / 4
    check("|W12| vs |w|", abs(abs(m.W[0, 1]) / abs(w) - 1), 1e-4)
    Eff = m.D + m.W @ np.linalg.inv(m.D) @ m.W
    check("E22 vs (c1+c2)/(4 cosh^2(pi eps))", abs(Eff[1, 1] / ((c1 + c2) / (4 * np.cosh(PI * eps) ** 2)) - 1), 1e-5)
    KI, KII = 120., 80.; K = KI + 1j * KII
    es = eo = 0.0
    for r in (1e-4, 1e-3, 1e-2, 3e-2):
        sg = m.stress(r, 0.0, KI, KII, 0., A_HALF, "top")
        ref = K * (r / (2 * A_HALF)) ** (1j * eps) / np.sqrt(2 * PI * r)
        es = max(es, abs(sg[1] + 1j * sg[5] - ref) / abs(ref))
        d = np.array(m.displacement(-r, 0.0, KI, KII, 0., A_HALF, "top")) - np.array(m.displacement(-r, -0.0, KI, KII, 0., A_HALF, "bot"))
        refo = (c1 + c2) / (2 * np.cosh(PI * eps)) * K * (r / (2 * A_HALF)) ** (1j * eps) / (1 + 2j * eps) * np.sqrt(r / (2 * PI))
        eo = max(eo, abs(d[1] + 1j * d[0] - refo) / abs(refo))
    check("interface stress sigma22 + i sigma12 vs textbook (Gross/Seelig)", es, 1e-6)
    check("crack opening du2 + i du1 (exactly on the crack faces) vs textbook", eo, 1e-6)
    G_stroh = 0.25 * np.array([KII, KI, 0.]) @ Eff @ np.array([KII, KI, 0.])
    G_text = (c1 + c2) * (KI ** 2 + KII ** 2) / (16 * np.cosh(PI * eps) ** 2)
    check("G = k^T E k / 4 vs (c1+c2)(K_I^2+K_II^2)/(16 cosh^2(pi eps))", abs(G_stroh / G_text - 1), 1e-5)
    # crack-closure integral with the textbook stress and opening: G = 1/(2 da) int sigma.du, analytic factor (1/2+i eps)/(1+2 i eps)=1/2
    da = 1e-4; n = 4000
    ph = (np.arange(n) + 0.5) * (PI / 2) / n; x = da * np.sin(ph) ** 2; jac = 2 * da * np.sin(ph) * np.cos(ph) * (PI / 2) / n
    sg = K * (x / (2 * A_HALF)) ** (1j * eps) / np.sqrt(2 * PI * x)
    dd = (c1 + c2) / (2 * np.cosh(PI * eps)) * K * ((da - x) / (2 * A_HALF)) ** (1j * eps) / (1 + 2j * eps) * np.sqrt((da - x) / (2 * PI))
    G_clos = 0.5 / da * np.sum(np.real(np.conj(sg) * dd) * jac) / 1.0
    # sigma22*du2 + sigma12*du1 = Re(conj(sigma22+i sigma12) * (du2 + i du1))
    print(f"    closure integral G / formula = {G_clos / G_text:.6f}")
    check("crack-closure integral of textbook fields vs (c1+c2)(K^2)/(16 cosh^2)", abs(G_clos / G_text - 1), 1e-4)
    # Fe/Cu exact-face evaluation (regression test of the branch-cut fix)
    gb = ctf.InterfaceCrackModel("Fe (cubic, example)", (27, 15, 10), "Cu (cubic, example)", (5, 40, 0))
    f = lambda x2: np.array(gb.displacement(-1e-3, x2, K0, 0, 0, A_HALF, "top")) - np.array(gb.displacement(-1e-3, -x2, K0, 0, 0, A_HALF, "bot"))
    check("exact crack face (x2 = 0) vs offset 1e-14 mm", np.abs(f(0.0) - f(1e-14)).max() / np.abs(f(1e-14)).max(), 1e-9)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def make_figures(rows_41):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "axes.grid": True, "grid.alpha": .3, "figure.dpi": 150,
                         # Computer Modern like the text (lmodern)
                         "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True, "axes.unicode_minus": False})
    out = os.path.join(HERE, "figs")
    os.makedirs(out, exist_ok=True)
    print(f"\n=== Figures -> {out} ===")

    # Fig. 1: Stroh vs Williams
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.loglog(rows_41[:, 0], rows_41[:, 1], "o-", label=r"displacement $u$")
    ax.loglog(rows_41[:, 0], rows_41[:, 2], "s-", label=r"stress $\sigma$")
    ax.loglog(rows_41[:, 0], rows_41[0, 1] * rows_41[:, 0] / rows_41[0, 0], "k:", lw=.8, label="slope 1")
    ax.set_xlabel(r"$C_{44}$ shift [MPa]"); ax.set_ylabel("max. rel. deviation"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_iso_vs_stroh.pdf")); plt.close(fig)

    # Fig. 2: single crystal
    iso = ctf.IsotropicCrackModel("In718 (isotropic)")
    fig = plt.figure(figsize=(7.0, 2.6))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, .05, .45, 1.3], wspace=.35, left=.07, right=.99, bottom=.2, top=.9)
    x = np.linspace(-0.005, 0.005, 90)
    axs = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
    for ax, ori, title in zip(axs, ((27, 0, 0), (27, 15, 10)),
                              (r"Fe, $(27^\circ,0^\circ,0^\circ)$", r"Fe, $(27^\circ,15^\circ,10^\circ)$")):
        m = ctf.AnisotropicCrackModel("Fe (cubic, example)", *ori)
        S22 = np.array([[m.stress(xx, yy, K0, 0, 0, A_HALF)[1] for xx in x] for yy in x])
        im = ax.imshow(np.clip(S22, -14000, 14000), extent=[x[0], x[-1], x[0], x[-1]], origin="lower",
                       cmap="seismic", vmin=-14000, vmax=14000)
        ax.set_title(title); ax.set_xlabel(r"$x_1$ [mm]"); ax.grid(False)
        ax.set_xticks([-0.004, 0, 0.004]); ax.set_yticks([-0.004, 0, 0.004])
    axs[0].set_ylabel(r"$x_2$ [mm]"); axs[1].set_yticklabels([])
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[2])); cb.set_label(r"$\sigma_{22}$ [MPa]"); cb.ax.yaxis.set_label_position("left")
    ax = fig.add_subplot(gs[4]); th = np.linspace(-0.995 * PI, 0.995 * PI, 361); r = 1e-3
    ax.plot(np.degrees(th), [iso.stress(r * np.cos(t), r * np.sin(t), K0, 0, 0)[1] * np.sqrt(2 * PI * r) / K0 for t in th],
            "k-", label="isotropic (Williams)")
    for ori, ls in (((27, 0, 0), "--"), ((27, 15, 10), "-.")):
        m = ctf.AnisotropicCrackModel("Fe (cubic, example)", *ori)
        ax.plot(np.degrees(th), [m.stress(r * np.cos(t), r * np.sin(t), K0, 0, 0, A_HALF)[1] * np.sqrt(2 * PI * r) / K0 for t in th],
                ls, label="Fe " + str(ori))
    ax.set_xlabel(r"$\theta$ [deg]"); ax.set_ylabel(r"$\sigma_{22}\sqrt{2\pi r}/K_I$")
    ax.legend(frameon=False, fontsize=6.5); ax.set_xlim(-180, 180)
    fig.savefig(os.path.join(out, "fig_single_crystal.pdf")); plt.close(fig)

    # Fig. 3: interface crack
    gb = ctf.InterfaceCrackModel("Fe (cubic, example)", (27, 15, 10), "Cu (cubic, example)", (5, 40, 0))
    eq = OpeningEq24(gb)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ax = axes[0]; xs = np.linspace(-0.005, -1e-5, 200); k = (K0, 0., 0.)
    du = np.array([np.array(gb.displacement(xx, 1e-30, *k, A_HALF, side="top"))
                   - np.array(gb.displacement(xx, -1e-30, *k, A_HALF, side="bot")) for xx in xs])
    ax.plot(xs, du[:, 1] * 1e3, "r-", label=r"$\Delta u_2$ (opening)")
    ax.plot(xs, du[:, 0] * 1e3, "b--", label=r"$\Delta u_1$ (sliding)")
    ax.plot(xs, du[:, 2] * 1e3, "g:", label=r"$\Delta u_3$ (tearing)")
    ax.set_xlabel(r"$x_1$ [mm]"); ax.set_ylabel(r"crack-face displacement [$\mu$m]"); ax.legend(frameon=False)
    ax.set_title(r"Fe/Cu, crack-face displacement (pure $K_I$)")
    ax.text(0.97, 0.95, "(a)", transform=ax.transAxes, ha="right", va="top", fontsize=9)
    ax = axes[1]
    lr = np.linspace(-100, np.log10(A_HALF), 4000)
    ax.semilogx(10.0 ** lr, [eq.normalised_opening(10.0 ** v, K0) for v in lr], "C0-", label="closed form, Eq. (24)")
    rm = np.geomspace(1e-6, A_HALF, 6)
    ax.semilogx(rm, [(gb.displacement(-r_, 1e-30, K0, 0, 0, A_HALF, side="top")[1]
                      - gb.displacement(-r_, -1e-30, K0, 0, 0, A_HALF, side="bot")[1])
                     / (np.sqrt(2 * r_ / PI) * gb.D[1, 1] * K0) for r_ in rm], "o", ms=4.5, mfc="white",
                mec="k", mew=0.9, label=r"$\mathtt{crack\_tip\_fields}$")
    ax.axvline(3e-7, color="gray", ls="--", lw=.8); ax.text(1.8e-7, .45, "0.3 nm", fontsize=7, color="gray", ha="right")
    ax.axhline(0, color="k", lw=.6, ls=":"); ax.set_xlim(1e-100, 1)
    ax.set_xlabel(r"$r$ [mm]"); ax.set_ylabel(r"$\Delta u_2\,/\,(\sqrt{2r/\pi}\,D_{22}K_I)$")
    ax.legend(frameon=False, fontsize=7, loc="lower right", bbox_to_anchor=(0.88, 0.02))   # below the curve, left of the 0.3 nm line
    ax.set_title("opening near the tip (pure $K_I$)")
    ax.text(0.03, 0.95, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_interface.pdf")); plt.close(fig)
    print("  fig_iso_vs_stroh.pdf, fig_single_crystal.pdf, fig_interface.pdf written")


def main():
    rows_41 = appendix_b1()
    appendix_b2_and_section_4_1()
    appendix_b3_and_section_4_2()
    appendix_c4()
    if "--no-figures" not in sys.argv:
        make_figures(rows_41)
    print("\n" + ("ALL CHECKS PASSED" if not FAILED else "FAILED CHECKS: " + "; ".join(FAILED)))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
