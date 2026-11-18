"""
Validation of the closed-form stress solution of crack_tip_fields.py
(homogeneous anisotropic crystal and interface crack with general misorientation).

Checks, for several grain pairs and loading modes (K_I, K_II, K_III, mixed):

  (a) Hwu (2010), Eqs. 7.112/7.113: sigma_i2 on the interface ahead of the tip (theta = 0)
  (b) finite-difference stress of the displacement field (independent numerical reference)
      at generic points in both grains
  (c) continuity of sigma_i2 and of u across the bonded interface (x1 > 0, x2 -> 0)
  (d) equilibrium  d(sigma_i1)/dx1 + d(sigma_i2)/dx2 = 0

Usage:  python validate_stress_interface.py
Expected: deviations (a), (b), (d) < 1e-6 and (c) < 1e-5 (the interface continuity is evaluated at
a finite offset |x2| = 1e-9 mm from the interface, which limits its accuracy); the script
exits with status 1 otherwise.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import crack_tip_fields as ctf

A_HALF = 20.0                 # reference length a [mm] (2a = 40 mm), as in the FE case study
K0 = 27.5 * np.sqrt(1000.)    # reference stress intensity, 27.5 MPa*sqrt(m) [MPa*sqrt(mm)]
TOL = 1.0e-6              # checks (a), (b), (d)
TOL_CONTINUITY = 1.0e-5   # check (c): limited by the finite offset from the interface
I3 = np.eye(3)

LOADS = {
    "K_I":   np.array([0.0, K0, 0.0]),          # k = [K_II, K_I, K_III]
    "K_II":  np.array([K0, 0.0, 0.0]),
    "K_III": np.array([0.0, 0.0, K0]),
    "mixed": np.array([0.4 * K0, K0, 0.7 * K0]),
}

# (material grain 1, Euler angles grain 1, material grain 2, Euler angles grain 2)
GRAIN_PAIRS = {
    # the pairs of the FE case study (C4/C5: bimaterial interface Fe/Cu; C6: bicrystal In718)
    "Fe/Cu (C4, C5)":   ("Fe (cubic, example)", (27, 15, 10), "Cu (cubic, example)", (5, 40, 0)),
    "In718 (C6)":       ("In718 (anisotropic)", (105, 60, 30), "In718 (anisotropic)", (0, 0, 0)),
    # two further pairs with general misorientation and higher contrast
    "Cu/Fe":            ("Cu (cubic, example)", (60, 30, 20), "Fe (cubic, example)", (10, 70, 40)),
    "Fe/In718":         ("Fe (cubic, example)", (145, 80, 120), "In718 (anisotropic)", (45, 60, 275)),
}


def setup_pair(mat1, ori1, mat2, ori2):
    C1 = ctf.rotate_stiffness_euler(ctf.MATERIALS[mat1]["C"], *ori1)
    C2 = ctf.rotate_stiffness_euler(ctf.MATERIALS[mat2]["C"], *ori2)
    g1, g2 = ctf.solve_stroh(C1), ctf.solve_stroh(C2)          # (p, A, B) of each grain
    S_hat, D, W = ctf.compute_D_W_S(g1[1], g1[2], g2[1], g2[2])
    return C1, C2, g1, g2, S_hat


def check_hwu_theta0(g1, S_hat, k):
    """(a) sigma_i2(r, theta = 0) against Hwu (2010), Eqs. 7.112/7.113 (reference length 2a)."""
    beta = np.sqrt(-0.5 * np.trace(S_hat @ S_hat))
    gamma = np.arctanh(beta) / np.pi
    r = 0.01
    c = (r / (2 * A_HALF))**(1j * gamma)
    M = I3 + (1 - c.real) / beta**2 * (S_hat @ S_hat) - (c.imag / beta) * S_hat
    hwu = M @ k / np.sqrt(2 * np.pi * r)
    s = ctf.stress_field_anisoElast(g1[0], g1[2], S_hat, r, 1e-9, k, A_HALF, 1)
    ours = np.array([s[2], s[1], s[4]])                          # sigma_12, sigma_22, sigma_23
    return np.max(np.abs(ours - hwu)) / np.max(np.abs(hwu))


def check_fd(C1, C2, g1, g2, S_hat, k):
    """(b) closed-form stress against finite differences of u, generic points in both grains."""
    worst = 0.0
    for mid, (p, A, B), C in ((1, g1, C1), (2, g2, C2)):
        for (x1, y) in [(-0.01, 0.004), (-0.02, 0.002), (0.008, 0.006), (0.015, 0.012), (-0.005, 0.0007)]:
            y = abs(y) if mid == 1 else -abs(y)
            fd = ctf.strain_stress_fd(
                lambda X, Y: ctf.displacement_field_anisoElast(p, A, B, S_hat, X, Y, k, A_HALF, mid),
                x1, y, C, A_HALF, h=1e-5 * np.hypot(x1, y))   # step relative to r (optimal ~1e-5 r)
            s = ctf.stress_field_anisoElast(p, B, S_hat, x1, y, k, A_HALF, mid)
            s33 = ctf.sigma33_from_others(C, s[0], s[1], s[4], s[3], s[2])
            an = np.array([s[0], s[1], s33, s[4], s[3], s[2]])
            worst = max(worst, np.max(np.abs(an - fd)) / np.max(np.abs(fd)))
    return worst


def check_continuity(g1, g2, S_hat, k):
    """(c) continuity of sigma_i2 (relative to K0) and of u across the bonded interface."""
    err_s, err_u = 0.0, 0.0
    for x1 in (0.06, 0.1, 0.2):
        st = ctf.stress_field_anisoElast(g1[0], g1[2], S_hat, x1, 1e-9, k, A_HALF, 1)
        sb = ctf.stress_field_anisoElast(g2[0], g2[2], S_hat, x1, -1e-9, k, A_HALF, 2)
        err_s = max(err_s, np.max(np.abs(np.array([st[2], st[1], st[4]]) - np.array([sb[2], sb[1], sb[4]]))) / K0)
        ut = np.array(ctf.displacement_field_anisoElast(g1[0], g1[1], g1[2], S_hat, x1, 1e-9, k, A_HALF, 1))
        ub = np.array(ctf.displacement_field_anisoElast(g2[0], g2[1], g2[2], S_hat, x1, -1e-9, k, A_HALF, 2))
        err_u = max(err_u, np.max(np.abs(ut - ub)) / np.max(np.abs(ut)))
    return err_s, err_u


def check_equilibrium(g1, S_hat, k):
    """(d) div(sigma) = 0 in the upper grain (central differences of the closed-form stress)."""
    h, x1, y = 1e-7, -0.012, 0.0035
    f = lambda X, Y: np.array(ctf.stress_field_anisoElast(g1[0], g1[2], S_hat, X, Y, k, A_HALF, 1))  # s11,s22,s12,s13,s23
    dx = (f(x1 + h, y) - f(x1 - h, y)) / (2 * h)
    dy = (f(x1, y + h) - f(x1, y - h)) / (2 * h)
    residual = np.array([dx[0] + dy[2], dx[2] + dy[1], dx[3] + dy[4]])
    return np.max(np.abs(residual)) / np.max(np.abs(np.r_[dx, dy]))


def main():
    worst = {"(a)": 0.0, "(b)": 0.0, "(c)": 0.0, "(d)": 0.0}
    for pair_name, (m1, o1, m2, o2) in GRAIN_PAIRS.items():
        C1, C2, g1, g2, S_hat = setup_pair(m1, o1, m2, o2)
        beta = np.sqrt(-0.5 * np.trace(S_hat @ S_hat))
        print(f"\n##### {pair_name}: beta = {beta:.4f}, oscillation index = {np.arctanh(beta) / np.pi:.4f}")
        for load_name, k in LOADS.items():
            ea = check_hwu_theta0(g1, S_hat, k)
            eb = check_fd(C1, C2, g1, g2, S_hat, k)
            ec, ecu = check_continuity(g1, g2, S_hat, k)
            ed = check_equilibrium(g1, S_hat, k)
            worst["(a)"] = max(worst["(a)"], ea); worst["(b)"] = max(worst["(b)"], eb)
            worst["(c)"] = max(worst["(c)"], ec, ecu); worst["(d)"] = max(worst["(d)"], ed)
            print(f"  {load_name:6s} (a) Hwu {ea:.1e} | (b) FD {eb:.1e} | (c) continuity sigma_i2 {ec:.1e}, u {ecu:.1e}"
                  f" | (d) equilibrium {ed:.1e}")
    print("\nLargest deviations:  " + "   ".join(f"{k} {v:.1e}" for k, v in worst.items()))
    ok = all(worst[k] < TOL for k in ("(a)", "(b)", "(d)")) and worst["(c)"] < TOL_CONTINUITY
    print(f"Tolerances: (a),(b),(d) < {TOL:.0e}; (c) < {TOL_CONTINUITY:.0e}  ->  {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
