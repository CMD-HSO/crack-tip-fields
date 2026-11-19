"""
crack_tip_fields.py
===================

Core library for crack-tip fields in isotropic and anisotropic linear-elastic
materials and for cracks along the interface between two anisotropic grains.

Contents
--------
  - Material database (MATERIALS) and helpers to build stiffness matrices
  - Rotation tools (Euler angles -> global coordinate system), crystal-orientation helpers
  - Three crack models with a common interface:
      IsotropicCrackModel, AnisotropicCrackModel, InterfaceCrackModel
    (displacement and, for the anisotropic models, closed-form stress fields)
  - Interface-fracture quantities: effective compliance, Griffith critical K,
    directional Young's modulus, effective interface stiffness
  - A simple Abaqus .inp parser (nodes + node sets)
  - An exporter for Abaqus *BOUNDARY displacement boundary conditions

Physics and derivation: see the accompanying article, the notebook
"crack_tip_fields_validation_and_examples.ipynb" and
  - C. Hwu, Anisotropic Elastic Plates, Springer, 2010 (Ch. 3, 5, 7, 11)
  - F. Brunner, P. Andric, F. Maresca, An atomistic K-test framework for general grain
    boundaries and triclinic single crystals, Modelling Simul. Mater. Sci. Eng. 33 (2025) 035004

Units in this module: mm, N, s, MPa (K therefore has the unit MPa*sqrt(mm)).
Voigt order of stiffness matrices and stress vectors: 11, 22, 33, 23, 13, 12.
"""

__version__ = "1.1.0"

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig, inv
from scipy.spatial.transform import Rotation

np.set_printoptions(precision=4, suppress=True)


# =============================================================================
# 1. Material database
# =============================================================================

def cubic_stiffness(C11, C12, C44):
    """Build a 6x6 stiffness matrix (Voigt) for cubic symmetry."""
    return np.array([
        [C11, C12, C12, 0,   0,   0  ],
        [C12, C11, C12, 0,   0,   0  ],
        [C12, C12, C11, 0,   0,   0  ],
        [0,   0,   0,   C44, 0,   0  ],
        [0,   0,   0,   0,   C44, 0  ],
        [0,   0,   0,   0,   0,   C44],
    ])


def isotropic_stiffness(E, nu, degeneracy_shift=0.0):
    """
    Build a 6x6 stiffness matrix (Voigt) for isotropic material behaviour.

    degeneracy_shift: optional small addition to C44 if the material is to be used in
    the anisotropic Stroh formalism (exactly isotropic constants lead to degenerate
    eigenvalues there, which raises an exception; a small shift, e.g. 10 MPa, avoids
    this without noticeably changing the results).
    """
    C11 = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
    C12 = E * nu / ((1 + nu) * (1 - 2 * nu))
    C44 = E / (2 * (1 + nu)) + degeneracy_shift
    return cubic_stiffness(C11, C12, C44)


MATERIALS = {
    "Steel (isotropic)": {
        "C": isotropic_stiffness(E=210000., nu=0.3),
        "symmetry": "isotropic",
        "note": "E = 210 GPa, nu = 0.3 (example values for steel)",
    },
    "Steel (isotropic, Stroh-compatible)": {
        "C": isotropic_stiffness(E=210000., nu=0.3, degeneracy_shift=10.),
        "symmetry": "isotropic (slightly perturbed)",
        "note": "As above, but with C44 + 10 MPa to avoid degeneracy in the Stroh solver.",
    },
    "Fe (cubic, example)": {
        "C": cubic_stiffness(C11=243360., C12=145010., C44=116040.),
        "symmetry": "cubic",
        "note": "Example values for cubic alpha-Fe [MPa]",
    },
    "Cu (cubic, example)": {
        "C": cubic_stiffness(C11=180850., C12=120020., C44=83560.),
        "symmetry": "cubic",
        "note": "Example values for cubic Cu [MPa]",
    },
    "Generic cubic (example)": {
        "C": cubic_stiffness(C11=244.2857, C12=145.3217, C44=116.3485),
        "symmetry": "cubic",
        "note": "Generic example data set [MPa]",
    },
    "In718 (isotropic)": {
        "C": isotropic_stiffness(E=162000., nu=0.3),
        "symmetry": "isotropic",
        "note": "E = 162 GPa, nu = 0.3 (In718 at 650 C, VDM data sheet)",
    },
    "In718 (anisotropic)": {
        "C": cubic_stiffness(C11=193500.0, C12=127800.0, C44=89400.0),
        "symmetry": "cubic",
        "note": "Alloy 718 single crystal at room temperature [MPa]: constants C11/C12/C44 = "
                "193.5/127.8/89.4 GPa (Schilli et al., Mater. Sci. Eng. A 830 (2022) 142030)",
    },
}


def print_engineering_constants(C, label=""):
    """Print E1, E2, E3, nu12, nu23, nu13 and G (from C[3,3]) for a stiffness matrix C."""
    S = inv(C)
    E1, E2, E3 = 1 / S[0, 0], 1 / S[1, 1], 1 / S[2, 2]
    nu12 = -S[0, 1] / S[0, 0]
    nu23 = -S[1, 2] / S[1, 1]
    nu13 = -S[0, 2] / S[0, 0]
    G = 1 / (2 * S[3, 3])
    print(f"--- {label} ---")
    print(f"E1, E2, E3   = {E1:10.1f}, {E2:10.1f}, {E3:10.1f} MPa")
    print(f"nu12,23,13   = {nu12:10.3f}, {nu23:10.3f}, {nu13:10.3f}")
    print(f"G (from C44) = {G:10.1f} MPa")


# =============================================================================
# 2. Rotation of the stiffness matrix (crystal -> global coordinate system)
# =============================================================================

def voigt_to_tensor(C_voigt):
    """Convert a 6x6 stiffness matrix (Voigt) into the 3x3x3x3 elasticity tensor."""
    C = np.zeros((3, 3, 3, 3))
    voigt_pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    for I in range(6):
        for J in range(6):
            i, j = voigt_pairs[I]
            k, l = voigt_pairs[J]
            C[i, j, k, l] = C_voigt[I, J]
            C[j, i, k, l] = C_voigt[I, J]
            C[i, j, l, k] = C_voigt[I, J]
            C[j, i, l, k] = C_voigt[I, J]
    return C


def rotate_stiffness(C_voigt, R):
    """Rotate a 6x6 stiffness matrix (Voigt) with the 3x3 rotation matrix R."""
    C_tensor = voigt_to_tensor(C_voigt)
    index = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    C_out = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            I, J = index[i], index[j]
            val = 0.0
            for k in range(3):
                for l in range(3):
                    for m in range(3):
                        for n in range(3):
                            val += (R[I[0], k] * R[I[1], l] *
                                    R[J[0], m] * R[J[1], n] *
                                    C_tensor[k, l, m, n])
            C_out[i, j] = val
    return C_out


def rotate_stiffness_euler(C_voigt, phi1_deg, Phi_deg, phi2_deg):
    """Rotate C from the crystal to the global coordinate system via Bunge Euler angles (z-x-z)."""
    R = Rotation.from_euler('ZXZ', [phi1_deg, Phi_deg, phi2_deg], degrees=True).as_matrix()
    return rotate_stiffness(C_voigt, R)


# =============================================================================
# 3. Isotropic crack-tip field (closed form)
# =============================================================================

def displacement_field_isoElast(x1, x2, G, kappa, K_I, K_II, K_III):
    """
    Displacement vector [u1,u2,u3] for isotropic linear-elastic behaviour, modes I+II+III
    superposed (Williams solution, first term; crack along x1 < 0, tip at the origin).

    G: shear modulus; kappa = 3 - 4*nu (plane strain) or (3 - nu)/(1 + nu) (plane stress).
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    r = np.sqrt(x1**2 + x2**2)
    r = np.where(r == 0, 1e-6, r)
    theta = np.arctan2(x2, x1)

    factor_I = (K_I / (2 * G)) * np.sqrt(r / (2 * np.pi))
    ur_I = factor_I * np.cos(theta / 2) * (kappa - 1 + 2 * np.sin(theta / 2)**2)
    ut_I = factor_I * np.sin(theta / 2) * (kappa + 1 - 2 * np.cos(theta / 2)**2)

    factor_II = (K_II / (2 * G)) * np.sqrt(r / (2 * np.pi))
    ur_II = factor_II * np.sin(theta / 2) * (kappa + 1 + 2 * np.cos(theta / 2)**2)
    ut_II = -factor_II * np.cos(theta / 2) * (kappa - 1 - 2 * np.sin(theta / 2)**2)

    # (names ur/ut are historical: all four quantities are already Cartesian components)
    ux = ur_I + ur_II
    uy = ut_I + ut_II
    u3 = (2 * K_III / G) * np.sqrt(r / (2 * np.pi)) * np.sin(theta / 2)
    return ux, uy, u3


def stress_field_isoElast(K_I, K_II, K_III, x, y, nu_plane_strain=None):
    """
    Stress components (sxx,syy,szz,sxy,sxz,syz) for isotropic linear-elastic behaviour
    (modes I+II+III superposed, Williams solution, first term).

    nu_plane_strain: if given, the out-of-plane normal stress is szz = nu*(sxx + syy)
    (plane strain, consistent with kappa = 3 - 4*nu in displacement_field_isoElast);
    if None, szz = 0 (plane stress).
    Note the component order (sxx,syy,szz,sxy,sxz,syz) differs from the Voigt order
    (11,22,33,23,13,12) used by the anisotropic models; IsotropicCrackModel.stress
    returns the Voigt order.
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    r = np.where(r == 0, 1e-9, r)
    factor = 1 / np.sqrt(2 * np.pi * r)
    cos_t2, sin_t2 = np.cos(theta / 2), np.sin(theta / 2)
    sin_3t2, cos_3t2 = np.sin(3 * theta / 2), np.cos(3 * theta / 2)

    sigma_xx_I = K_I * factor * cos_t2 * (1 - sin_t2 * sin_3t2)
    sigma_yy_I = K_I * factor * cos_t2 * (1 + sin_t2 * sin_3t2)
    sigma_xy_I = K_I * factor * sin_t2 * cos_t2 * cos_3t2
    sigma_xx_II = -K_II * factor * sin_t2 * (2 + cos_t2 * cos_3t2)
    sigma_yy_II = K_II * factor * sin_t2 * cos_t2 * cos_3t2
    sigma_xy_II = K_II * factor * cos_t2 * (1 - sin_t2 * sin_3t2)
    sigma_xz = -K_III * factor * np.sin(theta / 2)
    sigma_yz = K_III * factor * np.cos(theta / 2)

    sigma_xx = sigma_xx_I + sigma_xx_II
    sigma_yy = sigma_yy_I + sigma_yy_II
    sigma_zz = 0.0 if nu_plane_strain is None else nu_plane_strain * (sigma_xx + sigma_yy)

    return (sigma_xx, sigma_yy, sigma_zz, sigma_xy_I + sigma_xy_II, sigma_xz, sigma_yz)


# =============================================================================
# 4. Anisotropic crack-tip field (Stroh formalism)
# =============================================================================

def extract_QRT(C):
    """Extract the 3x3 blocks Q, R, T from the 6x6 stiffness matrix C (Stroh formalism)."""
    Q = np.array([[C[0][0], C[0][5], C[0][4]],
                  [C[0][5], C[5][5], C[4][5]],
                  [C[0][4], C[4][5], C[4][4]]])
    R = np.array([[C[0][5], C[0][1], C[0][3]],
                  [C[5][5], C[1][5], C[3][5]],
                  [C[4][5], C[1][4], C[3][4]]])
    T = np.array([[C[5][5], C[1][5], C[3][5]],
                  [C[1][5], C[1][1], C[1][3]],
                  [C[3][5], C[1][3], C[3][3]]])
    return Q, R, T


def solve_stroh_eigenproblem(Q, R, T):
    """Solve the Stroh eigenvalue problem N*xi = p*xi; returns (p, A, B) (Hwu 2010)."""
    n = Q.shape[0]
    N = np.zeros((2 * n, 2 * n))
    T_inv = inv(T)
    N[n:, n:] = -R @ T_inv.T
    N[:n, :n] = -T_inv @ R.T
    N[:n, n:] = T_inv
    N[n:, :n] = R @ T_inv @ R.T - Q

    eigvals, eigvecs = eig(N)
    mask = np.imag(eigvals) > 0
    p = eigvals[mask]
    xi = eigvecs[:, mask]

    eps2 = 1.0e-6
    if (np.abs(p[0] - p[1]) < eps2 or np.abs(p[1] - p[2]) < eps2 or np.abs(p[0] - p[2]) < eps2):
        raise ValueError(
            "The given stiffness matrix is degenerate in the Stroh formalism "
            "(repeated eigenvalues, e.g. exactly isotropic material). See "
            "MATERIALS['Steel (isotropic, Stroh-compatible)'] for a workaround "
            "via a small C44 shift."
        )

    I3 = np.eye(3)
    I6 = np.array([[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1],
                   [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
    eta = I6 @ xi
    normf = np.sqrt(eta.T @ xi)
    xi_n = np.array([xi[:, 0] / normf[0, 0], xi[:, 1] / normf[1, 1], xi[:, 2] / normf[2, 2]]).T
    eta_n = I6 @ xi_n

    eps1 = 1.0e-12
    normcheck = eta_n.T @ xi_n
    ok = (np.real(normcheck[0, 0]) - 1.0 < eps1 and np.real(normcheck[1, 1]) - 1.0 < eps1 and
          np.real(normcheck[2, 2]) - 1.0 < eps1 and np.imag(normcheck[0, 0]) < eps1 and
          np.imag(normcheck[1, 1]) < eps1 and np.imag(normcheck[2, 2]) < eps1 and
          np.abs(np.linalg.norm(normcheck) - np.linalg.norm(I3)) < eps1)
    if not ok:
        raise ValueError("Stroh eigenvectors are not correctly normalised!")

    A = xi_n[0:3, :]
    B = xi_n[3:6, :]
    return p, A, B


def solve_stroh(C):
    """Convenience function: extract Q, R, T from C and solve the eigenvalue problem."""
    Q, R, T = extract_QRT(C)
    return solve_stroh_eigenproblem(Q, R, T)


def compute_D_W_S(A1, B1, A2, B2):
    """Bimaterial matrices D, W and S_hat = D^-1 W (Hwu 2010, Ch. 7)."""
    L1 = np.real(-2j * B1 @ B1.T)
    S1 = np.real(1j * (2 * A1 @ B1.T - np.eye(3)))
    L2 = np.real(-2j * B2 @ B2.T)
    S2 = np.real(1j * (2 * A2 @ B2.T - np.eye(3)))
    L1_inv, L2_inv = inv(L1), inv(L2)
    D = L1_inv + L2_inv
    W = S1 @ L1_inv - S2 @ L2_inv
    eps = 1.0e-12
    if not (np.linalg.norm(D - D.T) < eps and np.linalg.norm(W + W.T) < eps):
        raise ValueError("D is not symmetric or W is not antisymmetric!")
    S_hat = inv(D) @ W
    return S_hat, D, W


def _rho(p, theta):
    return np.cos(theta) + p * np.sin(theta)


def _z_alpha(p, x1, x2):
    """z_alpha = x1 + p_alpha*x2 with the sign of Im(z_alpha) kept for x2 = +-0.

    Python adds 0.0 + (-0.0) = 0.0, so for x2 = -0.0 (lower crack face) the plain sum would lose the sign of the
    imaginary part and put z_alpha on the wrong side of the branch cut of z**(+-i*gamma) (arg = +pi instead of -pi).
    """
    z = x1 + p * x2
    if x2 == 0.0:
        sgn = -1.0 if np.signbit(x2) else 1.0
        z = np.array([complex(zz.real, np.copysign(abs(zz.imag), sgn)) for zz in z])
    return z


def _zpow(z, L, e):
    """(z/L)**e on the principal branch, evaluated as exp(e*(ln|z/L| + i*arg z)).

    np.angle keeps the sign of a negative zero in Im z; the division z/L (complex/float in NumPy) does not,
    which would put the lower crack face (arg z = -pi) on the wrong branch.
    """
    return np.exp(e * (np.log(np.abs(z) / L) + 1j * np.angle(z)))


def displacement_field_anisoElast(p, A, B, S_hat, x1, x2, k, a, material_id=1):
    """
    Displacement vector [u1,u2,u3] for anisotropic elastic behaviour (Stroh formalism).

    For S_hat = 0 (homogeneous single crystal) this reduces to the classical anisotropic
    crack-tip field. material_id = 1 (upper grain) or 2 (lower grain) sets the sign of
    the oscillating terms. k = [K_II, K_I, K_III]; a = crack half-length.
    """
    pi = np.pi
    I = np.eye(3)
    tr_S2 = np.trace(S_hat @ S_hat)
    eps = 1.0e-12
    beta = eps if np.abs(tr_S2) <= eps else np.sqrt(-0.5 * tr_S2)
    if not (np.isreal(beta) and 0.0 <= beta < 1.0):
        raise ValueError("Parameter beta was computed incorrectly!")
    gamma = (1 / pi) * np.arctanh(beta)

    S2 = S_hat @ S_hat
    S3 = S2 @ S_hat
    Phi = I + (2 * np.arctanh(beta)) / (pi * beta**3) * S3
    Phi_inv = inv(Phi)

    r = np.sqrt(x1**2 + x2**2)
    theta = np.arctan2(x2, x1)
    at_tip = r <= 1.0e-9
    if at_tip:
        r = 1.0e-9   # numerical regularisation directly at the crack tip

    rho_j = np.array([_rho(pj, theta) for pj in p])
    sqrt_rho = np.sqrt(rho_j)
    # IMPORTANT: complex (holomorphic) z_j, not |z_j|. Only then is the solution away from
    # the interface (theta != 0) consistent (sigma12 = sigma21, equilibrium).
    z_nuj = r * rho_j if at_tip else _z_alpha(p, x1, x2)   # at the tip: avoid 0**(i*gamma) = NaN

    sign = +1 if material_id == 1 else -1
    exp1 = np.exp(sign * gamma * pi) * _zpow(z_nuj, 2 * a, 1j * gamma)
    exp2 = np.exp(-sign * gamma * pi) * _zpow(z_nuj, 2 * a, -1j * gamma)
    kappa = 0.5 * (exp1 + exp2)
    lam = 0.5 * (exp1 - exp2)

    sqrt_rho_diag = np.diag(sqrt_rho)
    sqrt_rho_kappa_diag = np.diag(sqrt_rho * kappa)
    sqrt_rho_lambda_diag = np.diag(sqrt_rho * lam)

    u = np.sqrt(2 * r / np.pi) * np.real(
        A @ sqrt_rho_diag @ inv(B) @ (np.eye(3) + (1 / beta**2) * S2)
        - np.sqrt(1 - beta**2) / beta**2 * (
            A @ sqrt_rho_kappa_diag @ inv(B) @ S2
            + 1j / beta * A @ sqrt_rho_lambda_diag @ inv(B) @ S3
        )
    ) @ Phi_inv @ k
    return u[0], u[1], u[2]


def stress_field_anisoElast(p, B, S_hat, x1, x2, k, a, material_id=1):
    """
    Closed-form near-tip stresses for the homogeneous anisotropic crack (S_hat = 0)
    and the interface crack (S_hat != 0), Stroh formalism.

    Derivation: sigma_i2 = d(phi_i)/dx1 (=t2), sigma_i1 = -d(phi_i)/dx2 (=t1) with the
    stress function phi (same as u, but outer matrix B instead of A). Since kappa and
    lambda are holomorphic in z_j = x1 + p_j x2, dz_j/dx1 = 1, dz_j/dx2 = p_j and
        d[sqrt(2/pi) sqrt(z) kappa]/dx1 = (kappa + 2i*gamma*lambda) / sqrt(2 pi r rho),
    i.e. the structure of Brunner et al. (2025), Eq. (49), with 1/sqrt(rho_j) throughout
    and an additional factor p_j in t1. The prefactors sqrt(1-beta^2)/beta^2 of the
    oscillating terms are the same as in the displacement formula.

    Validation: sigma12 = sigma21 (symmetry), equilibrium, continuity of u and sigma_i2
    across the bonded interface, comparison with Hwu (2010) Eqs. 7.112/7.113 and with
    the finite-difference stress of the displacement field.

    Returns sigma11, sigma22, sigma12, sigma13, sigma23 (for sigma33 see
    sigma33_from_others).
    """
    pi = np.pi
    I = np.eye(3)
    tr_S2 = np.trace(S_hat @ S_hat)
    eps = 1.0e-12
    beta = eps if np.abs(tr_S2) <= eps else np.sqrt(-0.5 * tr_S2)
    gamma = (1 / pi) * np.arctanh(beta)

    S2 = S_hat @ S_hat
    S3 = S2 @ S_hat
    Phi = I + (2 * np.arctanh(beta)) / (pi * beta**3) * S3
    Phi_inv = inv(Phi)

    r = np.sqrt(x1**2 + x2**2)
    theta = np.arctan2(x2, x1)
    at_tip = r <= 1.0e-9
    if at_tip:
        r = 1.0e-9   # numerical regularisation directly at the crack tip

    rho_j = np.array([_rho(pj, theta) for pj in p])
    sqrt_rho = np.sqrt(rho_j)

    z_nuj = r * rho_j if at_tip else _z_alpha(p, x1, x2)   # at the tip: avoid 0**(i*gamma) = NaN
    sign = +1 if material_id == 1 else -1
    exp1 = np.exp(sign * gamma * pi) * _zpow(z_nuj, 2 * a, 1j * gamma)
    exp2 = np.exp(-sign * gamma * pi) * _zpow(z_nuj, 2 * a, -1j * gamma)
    kappa = 0.5 * (exp1 + exp2)
    lam = 0.5 * (exp1 - exp2)

    B_inv = inv(B)
    c = np.sqrt(1 - beta**2) / beta**2
    pref = 1.0 / np.sqrt(2 * pi * r)

    o1 = (kappa + 2j * gamma * lam) / sqrt_rho
    o2 = (lam + 2j * gamma * kappa) / sqrt_rho

    def assemble(lead, f1, f2):
        return np.real(
            B @ np.diag(lead) @ B_inv @ (I + S2 / beta**2)
            - c * (B @ np.diag(f1) @ B_inv @ S2 + (1j / beta) * B @ np.diag(f2) @ B_inv @ S3)
        ) @ Phi_inv @ k

    t1 = -pref * assemble(p / sqrt_rho, p * o1, p * o2)   # [s11, s21, s31]
    t2 = pref * assemble(1.0 / sqrt_rho, o1, o2)          # [s12, s22, s32]

    sigma11, sigma13 = t1[0], t1[2]
    sigma12 = t2[0]
    sigma22, sigma23 = t2[1], t2[2]
    return sigma11, sigma22, sigma12, sigma13, sigma23


def sigma33_from_others(C, sigma11, sigma22, sigma23, sigma13, sigma12):
    """
    Complete sigma33 (generalised plane strain, eps33 = 0) from the other five stress
    components - determined analytically or numerically - via the reduced Hooke's law.
    Exact, no approximation.
    """
    idx = [0, 1, 3, 4, 5]
    C_sub = C[np.ix_(idx, idx)]
    sigma_sub = np.array([sigma11, sigma22, sigma23, sigma13, sigma12])
    strain_sub = inv(C_sub) @ sigma_sub   # [eps11, eps22, 2eps23, 2eps13, 2eps12]
    return C[2, idx] @ strain_sub


def strain_stress_fd(u_func, x1, x2, C, a, h=None):
    """
    Strains/stresses at (x1,x2) via central finite differences of u_func
    (generalised plane strain, eps33 = 0). Serves as an independent numerical reference
    for the analytical solution stress_field_anisoElast; the two agree to ~1e-9 for the
    single crystal and the interface crack. Not usable directly at the crack tip or on
    the interface line (the step crosses the interface there).
    """
    r = np.hypot(x1, x2)
    if h is None:
        h = max(1e-6 * a, 1e-6 * r)
    u1p, u2p, u3p = u_func(x1 + h, x2)
    u1m, u2m, u3m = u_func(x1 - h, x2)
    du1_dx1, du2_dx1, du3_dx1 = (u1p - u1m) / (2 * h), (u2p - u2m) / (2 * h), (u3p - u3m) / (2 * h)
    u1p, u2p, u3p = u_func(x1, x2 + h)
    u1m, u2m, u3m = u_func(x1, x2 - h)
    du1_dx2, du2_dx2, du3_dx2 = (u1p - u1m) / (2 * h), (u2p - u2m) / (2 * h), (u3p - u3m) / (2 * h)
    eps11, eps22, eps33 = du1_dx1, du2_dx2, 0.0
    eps12 = 0.5 * (du1_dx2 + du2_dx1)
    eps13 = 0.5 * du3_dx1
    eps23 = 0.5 * du3_dx2
    eps_voigt = np.array([eps11, eps22, eps33, 2 * eps23, 2 * eps13, 2 * eps12])
    return C @ eps_voigt


def compute_field_grid(u_func, C, x_vals, y_vals, a, stress_func=None):
    """
    Evaluate displacements and stresses of an anisotropic crack-tip field on a grid.
    If stress_func(x, y) -> (s11,s22,s33,s23,s13,s12) is given (e.g. model.stress with
    fixed K), the analytical stress solution is used, otherwise finite differences of
    the displacement (strain_stress_fd). The Stroh formalism is not vectorised ->
    Python double loop.

    Returns U1, U2, U3 (each ny x nx) and SIG (ny x nx x 6, Voigt order 11,22,33,23,13,12).
    """
    ny, nx = len(y_vals), len(x_vals)
    U1 = np.zeros((ny, nx)); U2 = np.zeros((ny, nx)); U3 = np.zeros((ny, nx))
    SIG = np.zeros((ny, nx, 6))
    for iy, y in enumerate(y_vals):
        for ix, x in enumerate(x_vals):
            u1, u2, u3 = u_func(x, y)
            U1[iy, ix], U2[iy, ix], U3[iy, ix] = u1, u2, u3
            if stress_func is not None:
                SIG[iy, ix, :] = stress_func(x, y)
            else:
                SIG[iy, ix, :] = strain_stress_fd(u_func, x, y, C, a)
    return U1, U2, U3, SIG


def plot_heatmap(field, X, Y, title, label, vmin=None, vmax=None, cmap="seismic",
                  points=None, points_label=None, ax=None):
    """
    Heatmap of a scalar field field(X,Y). Optionally, points (e.g. the FE nodes that
    are actually evaluated) can be overlaid: points = (px, py) - coordinates in the
    same (local) coordinate system as X, Y.
    """
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(6, 5))
    extent = [X.min(), X.max(), Y.min(), Y.max()]
    im = ax.imshow(field, extent=extent, origin="lower", cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label=label)
    if points is not None and len(points[0]) > 0:
        px, py = points
        ax.scatter(px, py, s=22, facecolor="none", edgecolor="black", linewidth=1.1,
                   label=points_label or "FE nodes")
        ax.legend(loc="upper right", fontsize=8)
    # Prevents overlaid points outside the grid from stretching the axes
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("x1 [mm]")
    ax.set_ylabel("x2 [mm]")
    ax.set_title(title)
    if created_fig:
        plt.tight_layout()
        plt.show()
        plt.close(fig)


# =============================================================================
# 5. Model layer: common interface for the app and the notebooks
# =============================================================================

class IsotropicCrackModel:
    """Crack-tip field for isotropic linear-elastic material behaviour."""

    kind = "isotropic"

    def __init__(self, material_name, plane="strain"):
        self.material_name = material_name
        C = MATERIALS[material_name]["C"]
        C11, C12, C44 = C[0, 0], C[0, 1], C[3, 3]
        self.E = (C11 - C12) * (C11 + 2 * C12) / (C11 + C12)
        self.nu = C12 / (C11 + C12)
        self.G = C44
        self.plane = plane
        self.kappa = (3 - 4 * self.nu) if plane == "strain" else (3 - self.nu) / (1 + self.nu)

    def displacement(self, x1, x2, K_I, K_II, K_III, a=None):
        """Displacement [u1,u2,u3] in the local crack-tip coordinate system."""
        return displacement_field_isoElast(x1, x2, self.G, self.kappa, K_I, K_II, K_III)

    def stress(self, x1, x2, K_I, K_II, K_III, a=None):
        """
        Stress (s11,s22,s33,s23,s13,s12) in the local crack-tip coordinate system (Voigt order,
        like the anisotropic models). s33 = nu*(s11 + s22) for plane strain, 0 for plane stress.
        """
        nu_ps = self.nu if self.plane == "strain" else None
        sxx, syy, szz, sxy, sxz, syz = stress_field_isoElast(K_I, K_II, K_III, x1, x2, nu_plane_strain=nu_ps)
        return sxx, syy, szz, syz, sxz, sxy

    def __repr__(self):
        return f"IsotropicCrackModel({self.material_name!r}, E={self.E:.0f} MPa, nu={self.nu:.3f})"


class AnisotropicCrackModel:
    """Crack-tip field for anisotropic elastic behaviour in a homogeneous single crystal."""

    kind = "anisotropic"

    def __init__(self, material_name, phi1=0.0, Phi=0.0, phi2=0.0):
        self.material_name = material_name
        self.orientation = (phi1, Phi, phi2)
        C = MATERIALS[material_name]["C"]
        self.C_rot = rotate_stiffness_euler(C, phi1, Phi, phi2)
        self.p, self.A, self.B = solve_stroh(self.C_rot)
        self.S_hat, self.D, self.W = compute_D_W_S(self.A, self.B, self.A, self.B)

    def displacement(self, x1, x2, K_I, K_II, K_III, a):
        """Displacement [u1,u2,u3] in the local crack-tip coordinate system."""
        k = np.array([K_II, K_I, K_III])
        return displacement_field_anisoElast(self.p, self.A, self.B, self.S_hat, x1, x2, k, a, 1)

    def stress(self, x1, x2, K_I, K_II, K_III, a):
        """Stress (s11,s22,s33,s23,s13,s12) in the local crack-tip coordinate system, analytical (closed form)."""
        k = np.array([K_II, K_I, K_III])
        s11, s22, s12, s13, s23 = stress_field_anisoElast(self.p, self.B, self.S_hat, x1, x2, k, a, 1)
        s33 = sigma33_from_others(self.C_rot, s11, s22, s23, s13, s12)
        return s11, s22, s33, s23, s13, s12

    def __repr__(self):
        return f"AnisotropicCrackModel({self.material_name!r}, orientation={self.orientation})"


class InterfaceCrackModel:
    """
    Crack-tip field for an interface crack between two anisotropic grains.
    Grain 1 = above the interface (x2 >= 0), grain 2 = below (x2 <= 0).
    """

    kind = "interface"

    def __init__(self, material_top, orientation_top, material_bot, orientation_bot):
        self.material_top, self.orientation_top = material_top, orientation_top
        self.material_bot, self.orientation_bot = material_bot, orientation_bot
        C_t = MATERIALS[material_top]["C"]
        C_b = MATERIALS[material_bot]["C"]
        self.C_rot_top = rotate_stiffness_euler(C_t, *orientation_top)
        self.C_rot_bot = rotate_stiffness_euler(C_b, *orientation_bot)
        self.p1, self.A1, self.B1 = solve_stroh(self.C_rot_top)
        self.p2, self.A2, self.B2 = solve_stroh(self.C_rot_bot)
        self.S_hat, self.D, self.W = compute_D_W_S(self.A1, self.B1, self.A2, self.B2)

    def displacement(self, x1, x2, K_I, K_II, K_III, a, side):
        """
        Displacement [u1,u2,u3] in the local crack-tip coordinate system.
        side: 'top' (grain 1) or 'bot' (grain 2) - determines which grain/material is
        used for the node (independent of the sign of x2).
        """
        k = np.array([K_II, K_I, K_III])
        if side == "top":
            return displacement_field_anisoElast(self.p1, self.A1, self.B1, self.S_hat, x1, abs(x2), k, a, 1)
        elif side == "bot":
            return displacement_field_anisoElast(self.p2, self.A2, self.B2, self.S_hat, x1, -abs(x2), k, a, 2)
        else:
            raise ValueError("side must be 'top' or 'bot'")

    def stress(self, x1, x2, K_I, K_II, K_III, a, side):
        """Stress (s11,s22,s33,s23,s13,s12) in the local crack-tip coordinate system, analytical (closed form)."""
        k = np.array([K_II, K_I, K_III])
        if side == "top":
            s11, s22, s12, s13, s23 = stress_field_anisoElast(self.p1, self.B1, self.S_hat, x1, abs(x2), k, a, 1)
            C_ = self.C_rot_top
        elif side == "bot":
            s11, s22, s12, s13, s23 = stress_field_anisoElast(self.p2, self.B2, self.S_hat, x1, -abs(x2), k, a, 2)
            C_ = self.C_rot_bot
        else:
            raise ValueError("side must be 'top' or 'bot'")
        s33 = sigma33_from_others(C_, s11, s22, s23, s13, s12)
        return s11, s22, s33, s23, s13, s12

    def __repr__(self):
        return (f"InterfaceCrackModel(top={self.material_top!r}@{self.orientation_top}, "
                f"bot={self.material_bot!r}@{self.orientation_bot})")


# =============================================================================
# 6. Coordinate transformation: global Abaqus system <-> local crack-tip system
# =============================================================================

def global_to_local(X, Y, origin, alpha_deg):
    """
    Transform global coordinates (X,Y) into the local crack-tip coordinate system.

    origin   : (X0, Y0) - global coordinates of the crack tip
    alpha_deg: angle [deg] between the global X axis and the local x1 axis
               (crack-propagation direction), counter-clockwise positive
    """
    X0, Y0 = origin
    alpha = np.deg2rad(alpha_deg)
    dx, dy = X - X0, Y - Y0
    x1 = np.cos(alpha) * dx + np.sin(alpha) * dy
    x2 = -np.sin(alpha) * dx + np.cos(alpha) * dy
    return x1, x2


def local_to_global_vector(u1, u2, alpha_deg):
    """Transform a displacement vector (u1,u2) from the local back to the global coordinate system."""
    alpha = np.deg2rad(alpha_deg)
    Ux = np.cos(alpha) * u1 - np.sin(alpha) * u2
    Uy = np.sin(alpha) * u1 + np.cos(alpha) * u2
    return Ux, Uy


# =============================================================================
# 7. Abaqus .inp parser (nodes + node sets)
# =============================================================================

def parse_abaqus_mesh(path):
    """
    Read an Abaqus .inp file: nodes, elements, node sets and instance information.

    Supported: *NODE, *ELEMENT (records continued over several lines with a trailing comma),
    *NSET blocks (explicit lists and GENERATE), *INSTANCE, NAME=..., PART=... and the
    INSTANCE= parameter of assembly-level *NSET blocks. Node sets that reference other node
    sets by name are not supported. For assembly-based files a single part (one instance) is
    assumed, as written by Abaqus/CAE for a single-part model.

    Returns a dict with
      nodes         : {node_id: (x, y, z)}
      elements      : {element_id: [node_id, ...]}
      node_sets     : {set_name: [node_id, ...]}
      instances     : list of instance names (empty for a flat input file / orphan mesh)
      nset_instance : {set_name: instance name} for assembly-level node sets
    """
    nodes, elements, node_sets = {}, {}, {}
    instances, nset_instance = [], {}
    section, current_nset, nset_generate = None, None, False
    elem_buf = []

    def flush_element():
        if elem_buf:
            vals = [int(float(v)) for v in elem_buf]
            elements[vals[0]] = vals[1:]
            elem_buf.clear()

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("**"):
            continue

        if line.startswith("*"):
            flush_element()
            parts = [p.strip() for p in line.split(",")]
            keyword = parts[0].upper()
            params = {}
            for p in parts[1:]:
                if "=" in p:
                    key, val = (s.strip() for s in p.split("=", 1))
                    params[key.upper()] = val
                else:
                    params[p.upper()] = True
            if keyword == "*NODE":
                section = "NODE"
            elif keyword == "*ELEMENT":
                section = "ELEMENT"
            elif keyword == "*NSET":
                section = "NSET"
                nset_generate = "GENERATE" in params
                current_nset = params.get("NSET")
                if current_nset:
                    node_sets.setdefault(current_nset, [])
                    if "INSTANCE" in params:
                        nset_instance[current_nset] = params["INSTANCE"]
            elif keyword == "*INSTANCE":
                section = None
                if "NAME" in params:
                    instances.append(params["NAME"])
            else:
                section = None
            continue

        if section == "NODE":
            vals = [v for v in line.split(",") if v.strip() != ""]
            try:
                nid = int(float(vals[0]))
                x, y = float(vals[1]), float(vals[2])
                z = float(vals[3]) if len(vals) > 3 else 0.0
                nodes[nid] = (x, y, z)
            except (ValueError, IndexError):
                pass
        elif section == "ELEMENT":
            elem_buf.extend(v for v in line.split(",") if v.strip() != "")
            if not line.endswith(","):              # record complete (no continuation line)
                flush_element()
        elif section == "NSET" and current_nset:
            vals = [v.strip() for v in line.split(",") if v.strip() != ""]
            if nset_generate:
                try:
                    start, stop, step = (int(float(v)) for v in vals[:3])
                    node_sets[current_nset].extend(range(start, stop + 1, step))
                except ValueError:
                    pass
            else:
                for v in vals:
                    try:
                        node_sets[current_nset].append(int(float(v)))
                    except ValueError:
                        pass  # e.g. reference to another set (not supported)
    flush_element()

    return {"nodes": nodes, "elements": elements, "node_sets": node_sets,
            "instances": instances, "nset_instance": nset_instance}


def parse_abaqus_inp(path):
    """
    Read an Abaqus .inp file and return (nodes, node_sets), see parse_abaqus_mesh.

      nodes     : dict {node_id (int): (x, y, z)}
      node_sets : dict {set_name (str): list[int]}
    """
    mesh = parse_abaqus_mesh(path)
    return mesh["nodes"], mesh["node_sets"]


def abaqus_instance_name(mesh, set_name=None):
    """
    Instance name to be used as prefix ('<instance>.<node>') in an assembly-level *BOUNDARY.

    Abaqus/CAE writes assembly-based input files, in which node labels belong to a part
    instance; a boundary condition in the step must then address the nodes as
    '<instance>.<node label>'. Flat input files (orphan meshes without *INSTANCE) use the plain
    node labels. Returns the instance of the node set if it is defined at assembly level, the
    only instance of the model otherwise, or '' for a flat input file.
    """
    if set_name and set_name in mesh["nset_instance"]:
        return mesh["nset_instance"][set_name]
    if len(mesh["instances"]) == 1:
        return mesh["instances"][0]
    if not mesh["instances"]:
        return ""
    raise ValueError("Several part instances found - please specify the instance name explicitly.")


def crack_face_signs(node_ids, nodes, elements, origin, alpha_deg, tol=None):
    """
    Assign the crack faces of nodes that lie on the crack line (x1 < 0, x2 = 0 in the local
    crack-tip system).

    The crack is modelled as a seam: every node on the crack line exists twice with identical
    coordinates, one on the upper (theta = +pi) and one on the lower (theta = -pi) crack face.
    The displacement field is discontinuous there, so the face cannot be inferred from the
    coordinates; it is taken from the elements attached to the node (element centroid above
    or below the crack plane).

    Returns {node_id: +1 (upper face) or -1 (lower face)} for all crack-line nodes in node_ids.
    The sign is passed to the field functions as x2 = +0.0 or -0.0 (atan2 then gives +pi or -pi).
    """
    if tol is None:
        coords = np.array(list(nodes.values()))
        tol = 1e-9 * max(np.ptp(coords[:, 0]), np.ptp(coords[:, 1]), 1.0)
    wanted = set(node_ids)
    on_line = {}
    for nid in wanted:
        if nid in nodes:
            X, Y, _ = nodes[nid]
            x1, x2 = global_to_local(X, Y, origin, alpha_deg)
            if x1 < 0.0 and abs(x2) < tol:
                on_line[nid] = []
    if not on_line:
        return {}
    for conn in elements.values():
        hit = [n for n in conn if n in on_line]
        if not hit:
            continue
        cx = np.mean([nodes[n][0] for n in conn])
        cy = np.mean([nodes[n][1] for n in conn])
        _, c2 = global_to_local(cx, cy, origin, alpha_deg)
        for n in hit:
            on_line[n].append(c2)
    signs = {}
    for nid, c2s in on_line.items():
        if not c2s:
            raise ValueError(f"Node {nid} lies on the crack line but belongs to no element.")
        if max(c2s) < 0.0:
            signs[nid] = -1
        elif min(c2s) > 0.0:
            signs[nid] = +1
        else:
            raise ValueError(f"Node {nid} lies on the crack line but is shared by elements above and "
                             "below it - is the crack modelled as a seam (duplicate nodes)?")
    return signs


# =============================================================================
# 8. Export: Abaqus displacement boundary condition (*BOUNDARY)
# =============================================================================

def export_abaqus_boundary(filename, entries, dof_list=(1, 2), header_comment=None,
                            value_format="{:.6e}", step_name=None, instance=""):
    """
    Write a standalone .inp file with a *BOUNDARY definition that can be included in an
    Abaqus main model via *INCLUDE, INPUT=<filename>.

    entries  : list of (node_id, [u_dof1, u_dof2, ...]) - order matching dof_list
    dof_list : Abaqus degrees of freedom, e.g. (1,2) for u1,u2 or (1,2,3) including u3
    instance : part-instance name for assembly-based models (nodes written as
               '<instance>.<node>'), '' for flat input files (plain node labels);
               see abaqus_instance_name
    """
    prefix = f"{instance}." if instance else ""
    with open(filename, "w") as f:
        if header_comment:
            for line in header_comment.splitlines():
                f.write(f"** {line}\n")
        if step_name:
            f.write(f"** Include with: *INCLUDE, INPUT={filename}\n")
        f.write("*BOUNDARY\n")
        for node_id, values in entries:
            for dof, val in zip(dof_list, values):
                f.write(f"{prefix}{node_id}, {dof}, {dof}, {value_format.format(val)}\n")


# =============================================================================
# 9. Interface-crack evaluation helpers
# =============================================================================

def displacement_field_anisoElast_GB(p1, A1, B1, p2, A2, B2, S_hat, x1, x2, k, a):
    """
    Displacements above and below the interface at (x1, x2).
    Returns (u_top, u_bottom), each [u1, u2, u3].
    """
    u_top = list(displacement_field_anisoElast(p1, A1, B1, S_hat, x1, abs(x2), k, a, 1))
    u_bottom = list(displacement_field_anisoElast(p2, A2, B2, S_hat, x1, -abs(x2), k, a, 2))
    return u_top, u_bottom


def von_mises_3d(sxx, syy, szz, sxy, sxz, syz):
    """von Mises equivalent stress from the six stress components."""
    return np.sqrt(0.5 * ((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2 + 6 * (sxy**2 + sxz**2 + syz**2)))


def hydro_stress_3d(sxx, syy, szz, sxy, sxz, syz):
    """Hydrostatic stress (mean normal stress)."""
    return (sxx + syy + szz) / 3.


# =============================================================================
# 10. Fracture quantities of the interface (Griffith criterion)
# =============================================================================

def compute_eff_compliance(D, W):
    """Effective compliance matrix E = D + W D^-1 W (Hwu 2010, Ch. 7)."""
    D_inv = inv(D)
    return D + W @ D_inv @ W


def compute_critical_K(E22, gamma_s1, gamma_s2, gamma_gb):
    """
    Critical mode-I stress intensity factor K_Ic [MPa*sqrt(mm)] for crack propagation
    along the interface (Griffith criterion, G_c = gamma_s1 + gamma_s2 - gamma_gb).

    E22                          : E[1,1] of the effective compliance matrix
                                   (from compute_eff_compliance)
    gamma_s1, gamma_s2, gamma_gb : surface and grain-boundary energies [mJ/mm^2]
    """
    G_c = gamma_s1 + gamma_s2 - gamma_gb
    return 2 * np.sqrt(G_c / E22)


# =============================================================================
# 11. Directional elastic properties and effective interface stiffness
# =============================================================================

def calculate_E(orientation, C11, C12, C44):
    """
    Directional Young's modulus for cubic symmetry.

    orientation   : direction vector [x, y, z] (normalised internally)
    C11, C12, C44 : elastic constants in the crystal coordinate system
    """
    orientation = np.asarray(orientation, dtype=float)
    orientation = orientation / np.linalg.norm(orientation)
    alpha, beta, gamma = orientation
    den = (C11 - C12) * (C11 + 2 * C12)
    S11 = (C11 + C12) / den
    S12 = -C12 / den
    S44 = 1.0 / C44
    inv_E = S11 - (2 * (S11 - S12) - S44) * (alpha**2 * beta**2 + alpha**2 * gamma**2 + beta**2 * gamma**2)
    return 1.0 / inv_E


def calculate_Eeff(R, C1_unrot, A2, B2):
    """4/E22 for grain 1 (rotated with R) against the fixed grain 2 (A2, B2)."""
    C1_rot = rotate_stiffness(C1_unrot, R)
    Q1, R1, T1 = extract_QRT(C1_rot)
    p1_, A1_, B1_ = solve_stroh_eigenproblem(Q1, R1, T1)
    S_hat_, D_, W_ = compute_D_W_S(A1_, B1_, A2, B2)
    E_ = compute_eff_compliance(D_, W_)
    return 4 / E_[1, 1]


# =============================================================================
# 12. Crystal-orientation helpers
# =============================================================================

def EulerAnglesDegFromDirection(uvw, ref):
    """Bunge Euler angles [deg] (z-x-z) of the rotation that maps ref onto the direction uvw."""
    uvw = np.asarray(uvw, dtype=float)
    uvw = uvw / np.linalg.norm(uvw)
    rot, _ = Rotation.align_vectors([uvw], [ref])
    return rot.as_euler('ZXZ', degrees=True)


def direction_from_EulerAnglesDeg(phi1, Phi, phi2, ref):
    """Inverse of EulerAnglesDegFromDirection: direction from Euler angles."""
    rot = Rotation.from_euler('ZXZ', [phi1, Phi, phi2], degrees=True)
    v = rot.apply(ref)
    return v / np.linalg.norm(v)


def vector_to_miller(v):
    """Round a (normalised) direction vector to integer Miller indices [h,k,l]."""
    v = np.asarray(v, dtype=float)
    v = v / np.linalg.norm(v)
    v_abs = np.abs(v)
    min_nonzero = np.min(v_abs[np.nonzero(v_abs)])
    return np.rint(v / min_nonzero).astype(int)
