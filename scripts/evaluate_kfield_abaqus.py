# -*- coding: utf-8 -*-
"""
evaluate_kfield_abaqus.py - post-processing of K-field boundary-layer models in Abaqus
=====================================================================================

Evaluates an Abaqus result file (.odb) of a boundary-layer model loaded with the K-field
boundary condition of crack_tip_fields (two-dimensional plane-strain model or three-dimensional
slice in generalised plane strain written by make_gpe_model.py):

  - J-integral on all contours (*CONTOUR INTEGRAL output); for 3D slices for every node of the
    crack front (bottom face, mid-plane, top face) and their mean,
  - stress intensity factor recovered from J for pure mode I,
        K_I = 2 sqrt(J / E22),   E22 = 4/E' (isotropic), D22 (single crystal), [D + W D^-1 W]_22 (bicrystal),
    compared with the prescribed value,
  - opening stress sigma_22 at the nodes ahead of the tip (x2 = 0, x1 > 0; 3D: mid-plane), compared
    with K_I / sqrt(2 pi x1),
  - all stress components on a circle around the tip (node ring closest to R_CIRCLE; 3D: mid-plane),
    averaged per node separately for the elements above and below the crack plane (grain 1 / grain 2),
    for the comparison with the analytical field by compare_stress_circle.py (Python 3, crack_tip_fields).

For 3D slices the mid-plane node of the crack front is decisive: at the two end nodes the tractions
transmitted through the coupled faces enter the domain integral; this contribution cancels between
the two ends (their mean equals the mid-plane value).

Output (in the folder of the .odb):
  <prefix>_J.csv       2D: contour,J            3D: contour,J_mid,J_bottom,J_top,J_mean   [N/mm]
  <prefix>_s22.csv     x,s22                    [mm, MPa]
  <prefix>_circle.csv  theta_deg,r,x,y,side,s11,s22,s33,s23,s13,s12   [deg, mm, -, MPa]
                       (side = +1: elements above the crack plane, -1: below; 2D: s23 = s13 = 0)
  <prefix>_J_K.png, <prefix>_s22.png, <prefix>_circle.png
The CSV files are the data of the plots in Sections 4 and 5 of the accompanying paper.

Coordinate system: crack tip at the origin, crack along the negative x axis, y normal to the crack
(grain 1 above, y > 0), z along the crack front. Units: mm, MPa, K in MPa*sqrt(m) for the output.
The material model below is self-contained: Abaqus 2023 runs Python 2.7 with numpy, so the module
crack_tip_fields (Python 3) cannot be imported here; the comparison of the stresses on the circle with
the analytical field is therefore done afterwards with compare_stress_circle.py.

Usage (in the folder with the .odb):
  abaqus python evaluate_kfield_abaqus.py [--mode isotropic|single_crystal|bicrystal] [--odb file.odb]
                                          [--euler1 PHI1 PHI PHI2] [--euler2 PHI1 PHI PHI2] [--prefix name]
                                          [--KI K] [--KII K] [--KIII K]      (K in MPa*sqrt(m))
                                          [--frame N] [--step-time T]
                                          [--elastic1 C11 C12 C44] [--elastic2 C11 C12 C44]   (MPa)
--frame N evaluates frame N of the last step instead of the last frame (frame 0 = start, 1 = first increment).
J_prescribed is then scaled with the load fraction squared (frame time / step time, linear ramp of the
boundary condition): an early frame is still elastic and checks boundary condition, material and orientation.
--step-time T gives the total time of the load ramp (default: time period of the step); --elastic1/--elastic2 set the cubic
elastic constants of the grains (default: Fe / Cu of the examples).
The options override the settings below. For mixed-mode loading (K_II, K_III != 0) J is compared with
J = k^T E k / 4 (full compliance matrix E); J alone cannot separate the modes, so the reported deviation is
that of the magnitude of k, sqrt(J / J_prescribed) - 1.
"""
import glob
import math
import os
import sys

import numpy as np

try:                                  # cluster without libpng12: matplotlib cannot be imported, then only the CSV files are written
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from odbAccess import openOdb
from abaqusConstants import ELEMENT_NODAL, CARTESIAN


# ==============================================================================
# SETTINGS
# ==============================================================================
MODE = "bicrystal"          # "isotropic" | "single_crystal" | "bicrystal"
ODB = None                  # file name of the .odb; None = most recent .odb in the folder
PREFIX = None               # file prefix of the output; None = derived from MODE and model type

# isotropic material (plane strain)
E_MODULUS = 162000.0        # MPa
NU = 0.3

# cubic crystals: C11, C12, C44 [MPa] in the crystal frame, Bunge Euler angles (phi1, Phi, phi2) [deg]
GRAIN1 = {"C11": 243360.0, "C12": 145010.0, "C44": 116040.0, "euler": (27.0, 15.0, 10.0)}  # Fe, crystal / grain 1 (y > 0)
GRAIN2 = {"C11": 180850.0, "C12": 120020.0, "C44": 83560.0, "euler": (5.0, 40.0, 0.0)}     # Cu, grain 2 (y < 0)

# prescribed stress intensity factor
STEP_TIME = None            # total time of the load ramp [s]; None = time period of the step (see --step-time)
FRAME = None                # None = last frame; otherwise index of the frame (see --frame)
K_I_PRESCRIBED = 27.5       # MPa*sqrt(m)
K_II_PRESCRIBED = 0.0       # MPa*sqrt(m); mixed mode: J is compared with k^T E k / 4 (full compliance matrix E)
K_III_PRESCRIBED = 0.0      # MPa*sqrt(m); K_III requires a 3D model (u3)

# evaluation
FIRST_CONTOUR_FOR_MEAN = 3  # mean over contours >= this one (contour 1 is affected by the tip discretisation)
Y_TOL = 1e-5                # tolerance for "node lies on y = 0" [mm]
Z_TOL = 1e-6                # tolerance for the z level of the mid-plane [mm] (3D)
TRANSFORM_TO_GLOBAL = True  # transform stresses to the global system (anisotropic material orientations)
R_CIRCLE = 0.1              # radius of the stress circle [mm]; the node ring closest to it is used
R_TOL = 1e-5                # tolerance for "node lies on the ring" [mm]


def parse_options(argv):
    """Command-line options override the settings above."""
    global MODE, ODB, PREFIX, K_I_PRESCRIBED, K_II_PRESCRIBED, K_III_PRESCRIBED, FRAME, STEP_TIME
    args = list(argv)
    while args:
        a = args.pop(0)
        if a == "--mode":
            MODE = args.pop(0)
        elif a == "--KI":
            K_I_PRESCRIBED = float(args.pop(0))
        elif a == "--KII":
            K_II_PRESCRIBED = float(args.pop(0))
        elif a == "--KIII":
            K_III_PRESCRIBED = float(args.pop(0))
        elif a == "--odb":
            ODB = args.pop(0)
        elif a == "--frame":
            FRAME = int(args.pop(0))
        elif a == "--prefix":
            PREFIX = args.pop(0)
        elif a == "--step-time":
            STEP_TIME = float(args.pop(0))
        elif a in ("--elastic1", "--elastic2"):
            c = [float(args.pop(0)) for _ in range(3)]
            (GRAIN1 if a == "--elastic1" else GRAIN2).update({"C11": c[0], "C12": c[1], "C44": c[2]})
        elif a in ("--euler1", "--euler2"):
            eul = tuple(float(args.pop(0)) for _ in range(3))
            (GRAIN1 if a == "--euler1" else GRAIN2)["euler"] = eul
        else:
            raise SystemExit("unknown option: %s" % a)


parse_options(sys.argv[1:])


def log(text):
    print(text)
    if sys.stdout is not sys.__stdout__:          # Abaqus/CAE: also to the command window
        try:
            sys.__stdout__.write(str(text) + "\n")
            sys.__stdout__.flush()
        except Exception:
            pass


def write_csv(path, header, rows):
    with open(path, "w") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(",".join("%.10e" % v if isinstance(v, float) else str(v) for v in row) + "\n")
    log("-> CSV written: %s (%d rows)" % (path, len(rows)))


# ==============================================================================
# ANISOTROPIC ELASTICITY (Stroh formalism, Sections 2.3 and 2.4 of the paper)
# ==============================================================================
def cubic_stiffness(c11, c12, c44):
    C = np.zeros((6, 6))
    C[:3, :3] = c12
    for i in range(3):
        C[i, i] = c11
        C[i + 3, i + 3] = c44
    return C


def _rz(deg):
    t = math.radians(deg)
    return np.array([[math.cos(t), -math.sin(t), 0.0], [math.sin(t), math.cos(t), 0.0], [0.0, 0.0, 1.0]])


def _rx(deg):
    t = math.radians(deg)
    return np.array([[1.0, 0.0, 0.0], [0.0, math.cos(t), -math.sin(t)], [0.0, math.sin(t), math.cos(t)]])


def bunge_rotation(phi1, Phi, phi2):
    """Rotation crystal -> global for Bunge Euler angles: R = Rz(phi1) Rx(Phi) Rz(phi2)."""
    return _rz(phi1).dot(_rx(Phi)).dot(_rz(phi2))


def bond_matrix(Q):
    """6x6 transformation of stresses in Voigt notation (11,22,33,23,13,12): sigma_global = K sigma_crystal."""
    K = np.zeros((6, 6))
    for i in range(3):
        K[i, :3] = Q[i, :] ** 2
        K[i, 3] = 2.0 * Q[i, 1] * Q[i, 2]
        K[i, 4] = 2.0 * Q[i, 2] * Q[i, 0]
        K[i, 5] = 2.0 * Q[i, 0] * Q[i, 1]
    for r, (i, j) in enumerate([(1, 2), (0, 2), (0, 1)]):
        K[3 + r, :3] = Q[i, :] * Q[j, :]
        K[3 + r, 3] = Q[i, 1] * Q[j, 2] + Q[i, 2] * Q[j, 1]
        K[3 + r, 4] = Q[i, 0] * Q[j, 2] + Q[i, 2] * Q[j, 0]
        K[3 + r, 5] = Q[i, 0] * Q[j, 1] + Q[i, 1] * Q[j, 0]
    return K


def rotated_stiffness(grain):
    K = bond_matrix(bunge_rotation(*grain["euler"]))
    return K.dot(cubic_stiffness(grain["C11"], grain["C12"], grain["C44"])).dot(K.T)


def stroh_AB(C):
    """Stroh eigenvalues p and normalised eigenvector matrices A, B (Im p > 0)."""
    def build(Cm):
        Q = np.array([[Cm[0, 0], Cm[0, 5], Cm[0, 4]], [Cm[0, 5], Cm[5, 5], Cm[4, 5]], [Cm[0, 4], Cm[4, 5], Cm[4, 4]]])
        R = np.array([[Cm[0, 5], Cm[0, 1], Cm[0, 3]], [Cm[5, 5], Cm[1, 5], Cm[3, 5]], [Cm[4, 5], Cm[1, 4], Cm[3, 4]]])
        T = np.array([[Cm[5, 5], Cm[1, 5], Cm[3, 5]], [Cm[1, 5], Cm[1, 1], Cm[1, 3]], [Cm[3, 5], Cm[1, 3], Cm[3, 3]]])
        Ti = np.linalg.inv(T)
        N = np.zeros((6, 6), dtype=complex)
        N[:3, 3:] = np.eye(3)
        N[3:, :3] = -Ti.dot(Q)
        N[3:, 3:] = -Ti.dot(R + R.T)
        w, v = np.linalg.eig(N)
        idx = sorted([k for k in range(6) if w[k].imag > 1e-9], key=lambda k: w[k].imag)
        return R, T, w, v, idx

    Cm = C.copy()
    for attempt in range(3):
        R, T, w, v, idx = build(Cm)
        p = np.array([w[k] for k in idx])
        if len(idx) == 3 and min(abs(p[i] - p[j]) for i in range(3) for j in range(i + 1, 3)) > 1e-6:
            break
        Cm = C.copy()
        Cm[1, 1] *= 1.0 + 1e-7 * (attempt + 1)      # separate repeated roots (exactly isotropic input)
    else:
        raise RuntimeError("Stroh eigenvalue problem: no three distinct roots with Im(p) > 0")
    A = np.zeros((3, 3), dtype=complex)
    B = np.zeros((3, 3), dtype=complex)
    for j in range(3):
        a = v[:3, idx[j]]
        b = (R.T + p[j] * T).dot(a)
        s = np.sqrt(2.0 * a.dot(b))
        A[:, j], B[:, j] = a / s, b / s
    return A, B


def interface_compliance(C1, C2):
    """Effective compliance E = D + W D^-1 W [1/MPa], oscillation parameters beta, gamma."""
    LS = []
    for C in (C1, C2):
        A, B = stroh_AB(C)
        LS.append((np.real(-2j * B.dot(B.T)), np.real(1j * (2.0 * A.dot(B.T) - np.eye(3)))))
    (L1, S1), (L2, S2) = LS
    L1i, L2i = np.linalg.inv(L1), np.linalg.inv(L2)
    D = L1i + L2i
    W = S1.dot(L1i) - S2.dot(L2i)
    Di = np.linalg.inv(D)
    Sh = Di.dot(W)
    beta = math.sqrt(max(-0.5 * np.trace(Sh.dot(Sh)), 0.0))
    return D + W.dot(Di).dot(W), beta, math.atanh(beta) / math.pi


def compliance_of_case():
    """Compliance matrix E [1/MPa] with J = k^T E k / 4, k = [K_II, K_I, K_III], and a description of the case."""
    if MODE == "isotropic":
        e = 4.0 * (1.0 - NU ** 2) / E_MODULUS
        E = np.diag([e, e, 4.0 * (1.0 + NU) / E_MODULUS])       # E33 = 2/G
        return E, "isotropic: E = %g MPa, nu = %g, E22 = 4(1-nu^2)/E = %.6e 1/MPa" % (E_MODULUS, NU, e)
    C1 = rotated_stiffness(GRAIN1)
    if MODE == "single_crystal":
        E, _, _ = interface_compliance(C1, C1)
        return E, "single crystal: C = (%g, %g, %g) MPa, Bunge %s, E22 = D22 = %.6e 1/MPa" % (
            GRAIN1["C11"], GRAIN1["C12"], GRAIN1["C44"], GRAIN1["euler"], E[1, 1])
    if MODE == "bicrystal":
        E, beta, gamma = interface_compliance(C1, rotated_stiffness(GRAIN2))
        return E, ("bicrystal: grain 1 C = (%g, %g, %g) MPa, Bunge %s; grain 2 C = (%g, %g, %g) MPa, Bunge %s;\n"
                   "           E22 = [D + W D^-1 W]_22 = %.6e 1/MPa, beta = %.4f, gamma = %.4f") % (
            GRAIN1["C11"], GRAIN1["C12"], GRAIN1["C44"], GRAIN1["euler"],
            GRAIN2["C11"], GRAIN2["C12"], GRAIN2["C44"], GRAIN2["euler"], E[1, 1], beta, gamma)
    raise ValueError("MODE must be 'isotropic', 'single_crystal' or 'bicrystal'")


E_MAT, CASE_INFO = compliance_of_case()
E22 = E_MAT[1, 1]
K_MM = K_I_PRESCRIBED * math.sqrt(1000.0)          # MPa*sqrt(mm)
K_VEC = np.array([K_II_PRESCRIBED, K_I_PRESCRIBED, K_III_PRESCRIBED]) * math.sqrt(1000.0)   # [K_II, K_I, K_III]
J_PRESCRIBED = 0.25 * K_VEC.dot(E_MAT).dot(K_VEC)   # N/mm
MIXED = K_II_PRESCRIBED != 0.0 or K_III_PRESCRIBED != 0.0


def k_from_j(j):
    """K_I [MPa*sqrt(m)] from J [N/mm] (pure mode I)."""
    return 2.0 * math.sqrt(abs(j) / E22) / math.sqrt(1000.0)


def dev_j(j):
    """Deviation [%] of the stress intensity recovered from J: sqrt(J/J_prescribed) - 1.
    For pure mode I this is K_I^FE/K_I - 1 with K_I^FE = 2 sqrt(J/E22); for mixed mode it is the
    deviation of the magnitude of k (all K scaled by the same factor) - J alone cannot separate the modes."""
    return 100.0 * (math.sqrt(abs(j) / J_PRESCRIBED) - 1.0)


# ==============================================================================
# EVALUATION
# ==============================================================================
def read_j(odb, step, t_target=None):
    """J values from the history output.

    Abaqus names the output 'J at <crack name>_<crack-front node set>_Contour_<n>' (for crack-front
    nodes given as node numbers: '..._<node>_Contour_<n>'). Returns {contour: {front: J}} and
    {front: z coordinate of the crack-front node or None}.
    """
    out, fronts = {}, {}
    for h_key, h_reg in step.historyRegions.items():
        for name, data in h_reg.historyOutputs.items():
            if not name.upper().startswith("J AT ") or "_CONTOUR_" not in name.upper() or not data.data:
                continue
            k = name.upper().rindex("_CONTOUR_")
            front, tail = name[5:k], name[k + 9:]
            if not tail.strip().isdigit():
                continue
            if t_target is None:
                val = data.data[-1][1]
            else:                                           # entry closest in time to the chosen frame
                val = min(data.data, key=lambda d: abs(d[0] - t_target))[1]
            out.setdefault(int(tail), {})[front] = val
            fronts[front] = None
    # z of each crack front: the (assembly or instance) node set whose name ends the front name
    sets = []
    for nm, ns in odb.rootAssembly.nodeSets.items():
        sets.append((nm, ns))
    for inst in odb.rootAssembly.instances.values():
        for nm, ns in inst.nodeSets.items():
            sets.append((nm, ns))
    for front in fronts:
        best = None
        for nm, ns in sets:
            if front.upper().endswith(nm.upper()) and (best is None or len(nm) > len(best[0])):
                best = (nm, ns)
        try:
            nodes = best[1].nodes
            nodes = nodes[0] if nodes and not hasattr(nodes[0], "coordinates") else nodes
            c = nodes[0].coordinates
            fronts[front] = c[2] if len(c) > 2 else 0.0
        except Exception:
            pass
    return out, fronts


def main():
    odb_path = ODB or sorted(glob.glob("*.odb"), key=os.path.getmtime)[-1]
    log("Reading %s ..." % odb_path)
    odb = openOdb(path=str(odb_path))
    step = odb.steps[list(odb.steps.keys())[-1]]
    inst = list(odb.rootAssembly.instances.values())
    node_z = {}
    for i in inst:
        for n in i.nodes:
            node_z[n.label] = n.coordinates[2] if len(n.coordinates) > 2 else 0.0
    is_3d = (max(node_z.values()) - min(node_z.values())) > 1e-12
    z_mid = 0.5 * (max(node_z.values()) + min(node_z.values()))
    prefix = PREFIX or {"isotropic": "iso", "single_crystal": "sc", "bicrystal": "bc"}[MODE] + ("_3D" if is_3d else "")

    global J_PRESCRIBED
    frame_obj = step.frames[-1 if FRAME is None else FRAME]
    t_frame = frame_obj.frameValue
    if FRAME is not None:
        t_total = STEP_TIME or (step.timePeriod if getattr(step, "timePeriod", None) else step.frames[-1].frameValue)
        frac = t_frame / float(t_total)
        J_PRESCRIBED *= frac ** 2
        log("   frame %d: step time %.5g of %.5g (load fraction %.4g), J_prescribed scaled by %.4g -> %.6g N/mm"
            % (FRAME, t_frame, t_total, frac, frac ** 2, J_PRESCRIBED))
    jmap, fronts = read_j(odb, step, None if FRAME is None else t_frame)
    if not jmap:
        log("[!] No J-integral found in the history output.")
        odb.close()
        return
    contours = sorted(jmap)
    if is_3d:
        log("   crack fronts: " + ", ".join("%s (z = %s)" % (f, "%.4g" % z if z is not None else "?")
                                             for f, z in sorted(fronts.items(), key=lambda t: (t[1] is None, t[1]))))
        if len(fronts) > 1 and any(z is None for z in fronts.values()):
            log("[!] The z position of a crack front could not be determined; the fronts are taken in the order"
                " of their names.")
            for k, f in enumerate(sorted(fronts)):
                fronts[f] = float(k)
            z_mid_front = 0.5 * (len(fronts) - 1)
        else:
            z_mid_front = z_mid
    elif len(fronts) > 1:
        log("[!] 2D model with %d crack fronts; the first one is used: %s" % (len(fronts), sorted(fronts)[0]))

    # --- J per contour (2D: single value; 3D: per crack-front node) ---------------------------------
    rows, j_ref = [], []
    for c in contours:
        vals = jmap[c]
        if is_3d and len(vals) > 1:
            pts = sorted((fronts[f], v) for f, v in vals.items())
            j_bot, j_top = pts[0][1], pts[-1][1]
            j_mid = min(pts, key=lambda p: abs(p[0] - z_mid_front))[1]
            j_mean = sum(p[1] for p in pts) / float(len(pts))
            rows.append((c, j_mid, j_bot, j_top, j_mean))
            j_ref.append(j_mid)
        else:
            j = vals[sorted(vals)[0]]
            rows.append((c, j))
            j_ref.append(j)
    used = [k for k, c in enumerate(contours) if c >= FIRST_CONTOUR_FOR_MEAN] or list(range(len(contours)))
    j_mean_c = sum(j_ref[k] for k in used) / float(len(used))
    k_mean_c = k_from_j(j_mean_c)

    lines = ["", "=" * 78, "K-FIELD CHECK (%s, %s model%s)" % (MODE, "3D" if is_3d else "2D", ", mixed mode" if MIXED else ""),
             "  " + CASE_INFO]
    kname = "|k|" if MIXED else "K_I"
    for row in rows:
        c = row[0]
        if len(row) == 2:
            if MIXED:
                lines.append("  contour %d: J = %.6f N/mm (J/J_presc %+.3f %%, |k| %+.3f %%)" % (
                    c, row[1], 100.0 * (row[1] / J_PRESCRIBED - 1.0), dev_j(row[1])))
            else:
                lines.append("  contour %d: J = %.6f N/mm -> K_I = %.4f MPa*sqrt(m) (%+.3f %%)" % (
                    c, row[1], k_from_j(row[1]), dev_j(row[1])))
        else:
            lines.append("  contour %d: J_mid = %.6f (%s %+.3f %%), J_bottom = %.6f, J_top = %.6f, mean = %.6f N/mm" % (
                c, row[1], kname, dev_j(row[1]), row[2], row[3], row[4]))
    if MIXED:
        lines += ["  mean of contours %d-%d%s: J = %.6f N/mm, J/J_presc %+.3f %%, |k| deviation %+.3f %%"
                  % (contours[used[0]], contours[used[-1]], " (mid-plane node)" if is_3d else "",
                     j_mean_c, 100.0 * (j_mean_c / J_PRESCRIBED - 1.0), dev_j(j_mean_c)),
                  "  prescribed: K_I, K_II, K_III = %.4f, %.4f, %.4f MPa*sqrt(m) = %.3f, %.3f, %.3f MPa*sqrt(mm)" % (
                      K_I_PRESCRIBED, K_II_PRESCRIBED, K_III_PRESCRIBED, K_VEC[1], K_VEC[0], K_VEC[2]),
                  "              J = k^T E k / 4 = %.6f N/mm (E: full compliance matrix)" % J_PRESCRIBED, "=" * 78]
    else:
        lines += ["  mean of contours %d-%d%s: J = %.6f N/mm, K_I = %.4f MPa*sqrt(m) = %.3f MPa*sqrt(mm), deviation %+.3f %%"
                  % (contours[used[0]], contours[used[-1]], " (mid-plane node)" if is_3d else "",
                     j_mean_c, k_mean_c, k_mean_c * math.sqrt(1000.0), dev_j(j_mean_c)),
                  "  prescribed: K_I = %.4f MPa*sqrt(m) = %.3f MPa*sqrt(mm), J = %.6f N/mm" % (
                      K_I_PRESCRIBED, K_MM, J_PRESCRIBED), "=" * 78]
    log("\n".join(lines))
    per_front = len(rows[0]) > 2
    write_csv(prefix + "_J.csv", "contour,J" if not per_front else "contour,J_mid,J_bottom,J_top,J_mean",
              [tuple([r[0]] + [float(v) for v in r[1:]]) for r in rows])

    # --- opening stress ahead of the tip ------------------------------------------------------------
    field = frame_obj.fieldOutputs["S"]
    if TRANSFORM_TO_GLOBAL and MODE != "isotropic":
        try:
            csys = odb.rootAssembly.DatumCsysByThreePoints(name="GLOBAL_XY_EVAL", coordSysType=CARTESIAN,
                                                           origin=(0.0, 0.0, 0.0), point1=(1.0, 0.0, 0.0),
                                                           point2=(0.0, 1.0, 0.0))
            field = field.getTransformedField(datumCsys=csys)
        except Exception as e:
            log("[!] Transformation to the global system failed (%s); raw values are used." % e)
    # nodes of interest (3D: mid-plane): on the line ahead of the tip and on the node ring closest to R_CIRCLE
    coords, centroid_y, node_xy = {}, {}, {}
    for i in inst:
        for n in i.nodes:
            c = n.coordinates
            node_xy[(i.name, n.label)] = (float(c[0]), float(c[1]))
            z = c[2] if len(c) > 2 else 0.0
            if not is_3d or abs(z - z_mid) < Z_TOL:
                coords[(i.name, n.label)] = (float(c[0]), float(c[1]))
    # ring radius from the element corner nodes (2D and 3D use the same rings; in 3D the mid-plane
    # nodes lie above the corner nodes)
    corner_r = set()
    for i in inst:
        for e in i.elements:
            conn = e.connectivity
            nc = 8 if (is_3d and len(conn) in (8, 20)) else (4 if len(conn) in (4, 8) else len(conn))
            for m in conn[:nc]:
                cx, cy = node_xy[(i.name, m)]
                corner_r.add(round(math.hypot(cx, cy), 7))
    r_ring = min(corner_r, key=lambda r: abs(r - R_CIRCLE))
    line_nodes = set(k for k, (x, y) in coords.items() if abs(y) < Y_TOL and x > 1e-9)
    ring_nodes = set(k for k, (x, y) in coords.items() if abs(math.hypot(x, y) - r_ring) < R_TOL)
    wanted = line_nodes | ring_nodes
    # element centroids (y) decide the side of the crack plane / the grain of each nodal value
    node_y = {}
    for i in inst:
        for n in i.nodes:
            node_y[(i.name, n.label)] = n.coordinates[1]
    for i in inst:
        for e in i.elements:
            centroid_y[(i.name, e.label)] = sum(node_y[(i.name, m)] for m in e.connectivity) / float(len(e.connectivity))

    def voigt(d):
        """Abaqus order (11,22,33,12[,13,23]) -> (11,22,33,23,13,12)."""
        if len(d) == 4:
            return (d[0], d[1], d[2], 0.0, 0.0, d[3])
        return (d[0], d[1], d[2], d[5], d[4], d[3])

    acc = {}                                     # (instance, node, side) -> list of stress vectors
    for v in field.getSubset(position=ELEMENT_NODAL).values:
        name = v.instance.name if getattr(v, "instance", None) is not None else ""
        if (name, v.nodeLabel) not in wanted:
            continue
        side = 1 if centroid_y[(name, v.elementLabel)] > 0.0 else -1
        acc.setdefault((name, v.nodeLabel, side), []).append(voigt(v.data))
    odb.close()

    def mean_stress(keys):
        vals = [s for k in keys for s in acc.get(k, [])]
        return [sum(s[c] for s in vals) / float(len(vals)) for c in range(6)] if vals else None

    pts = []
    for k in line_nodes:
        s = mean_stress([k + (1,), k + (-1,)])   # sigma_22 is continuous: average over both sides
        if s is not None:
            pts.append((coords[k][0], s[1]))
    pts.sort()
    if pts:
        # s22: mean over both sides; s22_above / s22_below: extrapolated separately in the elements above
        # (grain 1, y > 0) and below (grain 2, y < 0) the crack plane - they coincide (sigma_22 is continuous)
        side_rows = []
        for k in line_nodes:
            s, s_up, s_lo = mean_stress([k + (1,), k + (-1,)]), mean_stress([k + (1,)]), mean_stress([k + (-1,)])
            if s is not None:
                side_rows.append((float(coords[k][0]), float(s[1]),
                                  float(s_up[1]) if s_up else float("nan"), float(s_lo[1]) if s_lo else float("nan")))
        side_rows.sort()
        write_csv(prefix + "_s22.csv", "x,s22,s22_above,s22_below", side_rows)
    else:
        log("[!] No nodes found on y = 0, x > 0%s (check Y_TOL / Z_TOL)." % (" in the mid-plane" if is_3d else ""))

    circle = []
    for k in ring_nodes:
        x, y = coords[k]
        for side in (1, -1):
            s = mean_stress([k + (side,)])
            if s is None:
                continue
            th = math.degrees(math.atan2(y, x))
            if abs(y) < Y_TOL:                   # crack faces (+-180 deg) and interface ahead of the tip (0 deg)
                th = 180.0 * side if x < 0.0 else 0.0
            circle.append((th, math.hypot(x, y), x, y, side) + tuple(s))
    circle.sort()
    if circle:
        write_csv(prefix + "_circle.csv", "theta_deg,r,x,y,side,s11,s22,s33,s23,s13,s12",
                  [tuple(float(v) if j != 4 else int(v) for j, v in enumerate(row)) for row in circle])
        log("   stress circle: node ring r = %.6f mm (closest to R_CIRCLE = %g mm), %d nodes, %d values" % (
            r_ring, R_CIRCLE, len(ring_nodes), len(circle)))
        log("   compare with the analytical field:  python compare_stress_circle.py %s_circle.csv <material options>"
            % prefix)
    else:
        log("[!] No stresses found on the node ring r = %.6f mm." % r_ring)

    # --- plots ------------------------------------------------------------------------------------------
    if plt is None:
        log("[!] matplotlib not available: no PNG files, the CSV files are written.")
        odb.close()
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    if per_front:
        for col, lab, st in ((1, "mid-plane node", "o-"), (2, "bottom face", "x:"), (3, "top face", "v:"),
                             (4, "mean over front", "^--")):
            a1.plot(contours, [r[col] for r in rows], st, label=lab)
            a2.plot(contours, [dev_j(r[col]) for r in rows], st, label=lab)
    else:
        a1.plot(contours, j_ref, "o-", label="FE")
        a2.plot(contours, [dev_j(j) for j in j_ref], "s-", label="FE")
    a1.axhline(J_PRESCRIBED, color="k", ls="--", label="prescribed")
    a2.axhline(0.0, color="k", ls="--", label="prescribed")
    a1.set(xlabel="contour [-]", ylabel="J [N/mm]", title="J-integral")
    if MIXED:
        a2.set(xlabel="contour [-]", ylabel="sqrt(J/J_presc) - 1 [%]", title="deviation of |k| (J = k^T E k / 4)")
    else:
        a2.set(xlabel="contour [-]", ylabel="K_I^FE / K_I - 1 [%]", title="K_I = 2 sqrt(J/E22)")
    for a in (a1, a2):
        a.set_xticks(contours)
        a.grid(True, ls="--", alpha=0.5)
        a.legend()
    fig.tight_layout()
    fig.savefig(prefix + "_J_K.png", dpi=200)
    plt.close(fig)
    if pts:
        fig, a = plt.subplots(figsize=(7, 4.5))
        a.plot([p[0] for p in pts], [p[1] for p in pts], "ro", ms=3, label="FE")
        xs = np.linspace(max(pts[0][0], 1e-4), pts[-1][0], 300)
        a.plot(xs, K_MM / np.sqrt(2.0 * np.pi * xs), "k--", label="K_I / sqrt(2 pi x1)")
        a.set(xlabel="x1 [mm]", ylabel="sigma_22 [MPa]", title="Opening stress ahead of the tip")
        a.grid(True, ls="--", alpha=0.5)
        a.legend()
        fig.tight_layout()
        fig.savefig(prefix + "_s22.png", dpi=200)
        plt.close(fig)
    if circle:
        # FE stresses on the circle, normalised with K_I / sqrt(2 pi r)
        names = ("sigma_11", "sigma_22", "sigma_33", "sigma_23", "sigma_13", "sigma_12")
        norm = K_MM / math.sqrt(2.0 * math.pi * circle[0][1])
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
        for c, a in enumerate(axes.T.flatten()):
            for side, mk, lab in ((1, "o", "above crack plane (grain 1)"), (-1, "s", "below (grain 2)")):
                rows_s = [row for row in circle if row[4] == side]
                a.plot([row[0] for row in rows_s], [row[5 + c] / norm for row in rows_s], mk, ms=3, label=lab)
            a.set(title=names[c], xlim=(-180, 180))
            if max(abs(row[5 + c]) for row in circle) / norm < 1e-6:
                a.set(ylim=(-0.05, 0.05), title=names[c] + " = 0")   # vanishing component: no round-off scaling
            a.set_xticks(range(-180, 181, 90))
            a.grid(True, ls="--", alpha=0.5)
        for a in axes[1]:
            a.set_xlabel("theta [deg]")
        for a in axes[:, 0]:
            a.set_ylabel("sigma_ij sqrt(2 pi r) / K_I [-]")
        axes[0, 0].legend(fontsize=8)
        fig.suptitle("FE stresses on the circle r = %.4f mm%s" % (circle[0][1], " (mid-plane)" if is_3d else ""))
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(prefix + "_circle.png", dpi=200)
        plt.close(fig)
    log("-> plots written: %s_J_K.png%s%s" % (prefix, ", %s_s22.png" % prefix if pts else "",
                                              ", %s_circle.png" % prefix if circle else ""))


if __name__ == "__main__":
    main()
