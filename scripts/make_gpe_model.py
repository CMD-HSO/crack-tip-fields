# -*- coding: utf-8 -*-
"""
make_gpe_model.py - three-dimensional K-field slice in generalised plane strain for Abaqus
=========================================================================================

The analytical crack-tip fields of crack_tip_fields (Stroh solutions for single crystals and
bicrystals) are generalised plane-strain fields: u_i = u_i(x1, x2), i.e. eps33 = 0 but u3 != 0
and transverse shear strains eps13, eps23 != 0 whenever in-plane and anti-plane deformation are
coupled (crystal orientations for which the crack front is not a twofold symmetry axis).
Two-dimensional plane-strain elements suppress u3 and are therefore too stiff for such
orientations (Section 4.2 of the accompanying paper). This script turns a thin, meshed 3D slice
of the boundary-layer model into an input file with exactly the kinematics of the analytical
solution:

  1. Crystal orientations: the elements are assigned to one crystal (single crystal) or to two
     grains (bicrystal; grain 1 above, grain 2 below the crack plane, from the element centroid),
     each with the unrotated stiffness of the material from crack_tip_fields.MATERIALS and an
     *ORIENTATION from its Bunge Euler angles (axes = columns of R = Rz(phi1) Rx(Phi) Rz(phi2)).
  2. Boundary condition: u1, u2, u3 of the analytical field at all nodes of the node columns that
     belong to the outer boundary (all node levels through the thickness). The duplicate nodes
     of the crack faces are assigned to the upper/lower face from the attached elements.
  3. Generalised plane strain: all other nodes of a node column (same x, y) are tied to its lowest
     node by *EQUATION, u_i(z) = u_i(z_min), i = 1, 2, 3 - the field is independent of z
     (eps33 = 0) while u3(x1, x2) is free.

Input: a 3D slice meshed in Abaqus/CAE by extruding the 2D boundary-layer mesh in the crack-front
direction (z = x3), one or more element layers of C3D8(R) or C3D20(R) elements, the crack as a
seam (duplicate nodes on the crack faces), a node set containing the boundary nodes (at least one
node level is sufficient - the whole node columns are taken), the *CONTOUR INTEGRAL output and a
static step. Mesh, crack definition, contour integral and output requests are kept unchanged;
the following parts of the input file are replaced:
  - *ORIENTATION and *SOLID SECTION definitions inside the part,
  - all *MATERIAL definitions,
  - *BOUNDARY definitions and boundary-condition *INCLUDEs inside the step.

Output (next to the input file, prefix --out, default <input>_gpe):
  <out>.inp            modified model (includes the two files below)
  <out>_bc.inp         *BOUNDARY u1, u2, u3 of the analytical field
  <out>_equations.inp  *EQUATION constraints (generalised plane strain)

Examples
--------
  Single crystal:
    python make_gpe_model.py slice.inp --material "Fe (cubic, example)" --euler 27 15 10 \
                             --boundary-set DISPLACEMENT
  Bicrystal (crack along the grain boundary):
    python make_gpe_model.py slice.inp --material "Fe (cubic, example)" --euler 27 15 10 \
                             --material2 "Cu (cubic, example)" --euler2 5 40 0 \
                             --boundary-set DISPLACEMENT

Units as in crack_tip_fields: mm, MPa, K in MPa*sqrt(mm) (default K_I = 27.5*sqrt(1000)).
"""
import argparse
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation


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

# Abaqus *ELASTIC, TYPE=ANISOTROPIC order (21 values) and position in the Voigt matrix (11,22,33,23,13,12)
_VOIGT_POS = {"11": 0, "22": 1, "33": 2, "23": 3, "13": 4, "12": 5}
_ABQ_ORDER = ["1111", "1122", "2222", "1133", "2233", "3333", "1112", "2212", "3312", "1212",
              "1113", "2213", "3313", "1213", "1313", "1123", "2223", "3323", "1223", "1323", "2323"]
_MATERIAL_OPTIONS = {"*ELASTIC", "*DENSITY", "*EXPANSION", "*PLASTIC", "*DEPVAR", "*USER MATERIAL",
                     "*CONDUCTIVITY", "*SPECIFIC HEAT", "*DAMPING", "*HYPERELASTIC", "*CREEP"}


# ---------------------------------------------------------------------------------------------
# Mesh topology
# ---------------------------------------------------------------------------------------------
def vertical_pairs(conn):
    """Node pairs of a hexahedron that lie on top of each other (edges in the extrusion direction)."""
    if len(conn) == 8:                                   # C3D8(R): face 1-4, face 5-8
        return [(conn[i], conn[i + 4]) for i in range(4)]
    if len(conn) == 20:                                  # C3D20(R): + mid-side nodes 9-12 / 13-16, 17-20
        return ([(conn[i], conn[i + 4]) for i in range(4)]
                + [(conn[8 + i], conn[12 + i]) for i in range(4)]
                + [(conn[i], conn[16 + i]) for i in range(4)])
    raise ValueError(f"unsupported element with {len(conn)} nodes (C3D8(R) or C3D20(R) expected)")


def node_columns(nodes, elements, tol):
    """
    Group the nodes into columns (same x, y, different z) via the vertical element edges.
    Duplicate seam nodes on the crack faces form separate columns. Returns {node: root}, where
    root is the lowest node of the column, and {root: [nodes sorted by z]}.
    """
    parent = {}

    def find(n):
        parent.setdefault(n, n)
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    for conn in elements.values():
        for a, b in vertical_pairs(conn):
            xa, xb = nodes[a], nodes[b]
            if abs(xa[0] - xb[0]) > tol or abs(xa[1] - xb[1]) > tol or abs(xa[2] - xb[2]) <= tol:
                raise ValueError(f"nodes {a} and {b} are not on top of each other - the slice must be an "
                                 "extrusion of the 2D mesh in z with the element faces 1-4 / 5-8 in z")
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    columns = {}
    for conn in elements.values():
        for n in conn:
            columns.setdefault(find(n), set()).add(n)
    out_root, out_cols = {}, {}
    for members in columns.values():
        col = sorted(members, key=lambda n: nodes[n][2])
        out_cols[col[0]] = col
        for n in col:
            out_root[n] = col[0]
    return out_root, out_cols


# ---------------------------------------------------------------------------------------------
# Input-file editing
# ---------------------------------------------------------------------------------------------
def keyword(line):
    return line.split(",")[0].strip().upper() if line.startswith("*") and not line.startswith("**") else None


def remove_blocks(lines, start, stop, names):
    """Remove keyword blocks (keyword line + data lines) whose keyword is in names, within [start, stop)."""
    out, removed, skipping = [], [], False
    for k, l in enumerate(lines):
        kw = keyword(l)
        if start <= k < stop and kw in names:
            skipping = True
            removed.append(l.strip())
            continue
        if skipping and (kw is not None or l.startswith("**")):
            skipping = False
        if skipping:
            continue
        out.append(l)
    return out, removed


def anisotropic_card(C):
    v = [C[_VOIGT_POS[k[:2]], _VOIGT_POS[k[2:]]] for k in _ABQ_ORDER]
    vmax = max(abs(x) for x in v)
    v = [0.0 if abs(x) < 1e-9 * vmax else x for x in v]
    rows = [v[0:8], v[8:16], v[16:21]]
    return [", ".join(f"{x:.7g}" for x in r) for r in rows]


def orientation_axes(euler):
    R = Rotation.from_euler("ZXZ", euler, degrees=True).as_matrix()   # Bunge, crystal -> global
    return ", ".join(f"{x:.9f}" for x in np.concatenate([R[:, 0], R[:, 1]]))


def id_lines(ids):
    return [", ".join(str(v) for v in ids[k:k + 16]) for k in range(0, len(ids), 16)]


# ---------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inp", help="Abaqus input file of the meshed 3D slice")
    ap.add_argument("--material", required=True, help="material of the crystal / grain 1 (key of MATERIALS)")
    ap.add_argument("--euler", nargs=3, type=float, default=(0.0, 0.0, 0.0), metavar=("PHI1", "PHI", "PHI2"),
                    help="Bunge Euler angles [deg] of the crystal / grain 1")
    ap.add_argument("--material2", help="material of grain 2 (below the crack plane) -> bicrystal")
    ap.add_argument("--euler2", nargs=3, type=float, default=(0.0, 0.0, 0.0), metavar=("PHI1", "PHI", "PHI2"),
                    help="Bunge Euler angles [deg] of grain 2")
    ap.add_argument("--boundary-set", action="append", required=True,
                    help="node set(s) with the boundary nodes (repeat for several sets)")
    ap.add_argument("--KI", type=float, default=27.5 * 1000 ** 0.5, help="K_I [MPa*sqrt(mm)]")
    ap.add_argument("--KII", type=float, default=0.0, help="K_II [MPa*sqrt(mm)]")
    ap.add_argument("--KIII", type=float, default=0.0, help="K_III [MPa*sqrt(mm)]")
    ap.add_argument("--a", type=float, default=20.0, help="reference length a (crack half-length) [mm]")
    ap.add_argument("--tip", nargs=2, type=float, default=(0.0, 0.0), metavar=("X0", "Y0"),
                    help="crack-tip position in the global system [mm]")
    ap.add_argument("--alpha", type=float, default=0.0, help="crack direction against global X [deg]")
    ap.add_argument("--instance", default=None, help="instance name used as node prefix (default: automatic)")
    ap.add_argument("--out", default=None, help="output prefix, relative to the current folder (default: <input>_gpe next to the input file)")
    args = ap.parse_args(argv)

    for m in (args.material, args.material2):
        if m is not None and m not in ctf.MATERIALS:
            ap.error(f"unknown material {m!r}; available: {', '.join(ctf.MATERIALS)}")
    bicrystal = args.material2 is not None
    if args.out:                       # --out: relative to the current working directory
        out = os.path.abspath(args.out)
    else:                              # default: <input>_gpe next to the input file
        out = os.path.splitext(os.path.abspath(args.inp))[0] + "_gpe"
    f_inp, f_bc, f_eq = (out + s for s in (".inp", "_bc.inp", "_equations.inp"))
    tip, alpha = tuple(args.tip), args.alpha
    K = (args.KI, args.KII, args.KIII)

    mesh = ctf.parse_abaqus_mesh(args.inp)
    nodes, elements = mesh["nodes"], mesh["elements"]
    instance = args.instance if args.instance is not None else ctf.abaqus_instance_name(mesh)
    prefix = f"{instance}." if instance else ""
    coords = np.array(list(nodes.values()))
    tol = 1e-6 * max(np.ptp(coords[:, 0]), np.ptp(coords[:, 1]), 1.0)

    # --- columns, grains --------------------------------------------------------------------------
    root, columns = node_columns(nodes, elements, tol)
    n_levels = sorted({len(c) for c in columns.values()})

    def local_y(n_list):
        c = np.mean([nodes[n][:2] for n in n_list], axis=0)
        return ctf.global_to_local(c[0], c[1], tip, alpha)[1]

    elem_grain = {e: (1 if (not bicrystal or local_y(conn) > 0.0) else 2) for e, conn in elements.items()}
    node_grains = {}
    for e, conn in elements.items():
        for n in conn:
            node_grains.setdefault(n, set()).add(elem_grain[e])

    # --- boundary nodes: complete node columns of the given node sets ---------------------------------
    bset = set()
    for name in args.boundary_set:
        if name not in mesh["node_sets"]:
            ap.error(f"node set {name!r} not found; available: {', '.join(sorted(mesh['node_sets']))}")
        bset.update(n for n in mesh["node_sets"][name] if n in root)
    b_roots = {root[n] for n in bset}
    bc_nodes = [n for r in sorted(b_roots) for n in columns[r]]

    # --- analytical boundary condition --------------------------------------------------------------
    if bicrystal:
        model = ctf.InterfaceCrackModel(args.material, tuple(args.euler), args.material2, tuple(args.euler2))
    else:
        model = ctf.AnisotropicCrackModel(args.material, *args.euler)
    face = ctf.crack_face_signs(bc_nodes, nodes, elements, tip, alpha)     # crack-face nodes: +1 upper, -1 lower
    with open(f_bc, "w") as f:
        f.write("** Crack-tip field boundary condition (generalised plane strain), written by make_gpe_model.py\n")
        if bicrystal:
            f.write(f"** grain 1 (x2 > 0): {args.material}, Bunge {tuple(args.euler)}; "
                    f"grain 2 (x2 < 0): {args.material2}, Bunge {tuple(args.euler2)}\n")
        else:
            f.write(f"** single crystal: {args.material}, Bunge {tuple(args.euler)}\n")
        f.write(f"** K_I={K[0]:.3f}, K_II={K[1]:.3f}, K_III={K[2]:.3f} [MPa*sqrt(mm)], a={args.a} mm, "
                f"crack tip={tip} mm, alpha={alpha} deg\n*BOUNDARY\n")
        for n in bc_nodes:
            X, Y, _ = nodes[n]
            x1, x2 = (float(v) for v in ctf.global_to_local(X, Y, tip, alpha))
            if n in face:
                x2 = np.copysign(0.0, face[n])
            if bicrystal:
                g = node_grains[n]
                side = "bot" if (g == {2} or (n in face and face[n] < 0)) else "top"
                u1, u2, u3 = model.displacement(x1, x2, *K, args.a, side=side)
            else:
                u1, u2, u3 = model.displacement(x1, x2, *K, args.a)
            Ux, Uy = ctf.local_to_global_vector(u1, u2, alpha)
            for dof, val in ((1, Ux), (2, Uy), (3, u3)):
                f.write(f"{prefix}{n}, {dof}, {dof}, {val:.6e}\n")

    # --- constraint equations (generalised plane strain) -----------------------------------------------
    n_eq = 0
    with open(f_eq, "w") as f:
        f.write("** Generalised plane strain: u_i(x, y, z) = u_i(x, y, z_min), i = 1, 2, 3 (make_gpe_model.py)\n")
        for r, col in sorted(columns.items()):
            if r in b_roots:
                continue
            for n in col[1:]:
                for dof in (1, 2, 3):
                    f.write(f"*Equation\n2\n{prefix}{n}, {dof}, 1., {prefix}{r}, {dof}, -1.\n")
                    n_eq += 1

    # --- modified input file -------------------------------------------------------------------------
    lines = open(args.inp, encoding="utf-8", errors="ignore").read().split("\n")
    kws = [keyword(l) for l in lines]
    end_part = kws.index("*END PART") if ("*END PART" in kws and "*ELEMENT" in kws[:kws.index("*END PART")]) \
        else kws.index("*END INSTANCE")
    part_start = max(k for k in range(end_part) if kws[k] in ("*PART", "*INSTANCE"))
    lines, removed_sec = remove_blocks(lines, part_start, end_part, {"*ORIENTATION", "*SOLID SECTION"})
    kws = [keyword(l) for l in lines]
    lines, removed_mat = remove_blocks(lines, 0, len(lines), {"*MATERIAL"} | _MATERIAL_OPTIONS)
    kws = [keyword(l) for l in lines]
    step0, step1 = kws.index("*STEP"), kws.index("*END STEP")
    lines, removed_bc = remove_blocks(lines, step0, step1, {"*BOUNDARY"})
    removed_inc = [l.strip() for k, l in enumerate(lines)
                   if (keyword(l) == "*INCLUDE" and step0 < k
                       and any(s in l.lower() for s in ("bc", "boundary")))]
    lines = [l for l in lines if l.strip() not in removed_inc]

    grains = [(1, args.material, tuple(args.euler))] + ([(2, args.material2, tuple(args.euler2))] if bicrystal else [])
    part_block = []
    for g, mat, eul in grains:
        ids = sorted(e for e, gg in elem_grain.items() if gg == g)
        part_block += [f"*Elset, elset=GPE_GRAIN{g}"] + id_lines(ids)
        part_block += [f"*Orientation, name=GPE_ORI{g}", orientation_axes(eul), "1, 0."]
        part_block += [f"** grain {g}: {mat}, Bunge {eul}",
                       f"*Solid Section, elset=GPE_GRAIN{g}, orientation=GPE_ORI{g}, material=GPE_MAT{g}", ","]
    mat_block = ["**", "** MATERIALS (unrotated crystal stiffness, rotated by the orientations; make_gpe_model.py)"]
    for g, mat, eul in grains:
        mat_block += [f"*Material, name=GPE_MAT{g}", "*Elastic, type=ANISOTROPIC"]
        mat_block += anisotropic_card(ctf.MATERIALS[mat]["C"])

    kws = [keyword(l) for l in lines]
    end_part = kws.index("*END PART") if ("*END PART" in kws and "*ELEMENT" in kws[:kws.index("*END PART")]) \
        else kws.index("*END INSTANCE")
    lines = lines[:end_part] + part_block + lines[end_part:]
    kws = [keyword(l) for l in lines]
    lines = lines[:kws.index("*END ASSEMBLY")] + [f"*INCLUDE, INPUT={os.path.basename(f_eq)}"] + \
        lines[kws.index("*END ASSEMBLY"):]
    kws = [keyword(l) for l in lines]
    k_mat = kws.index("*STEP")
    while k_mat > 0 and lines[k_mat - 1].startswith("**"):      # before the comment lines of the step
        k_mat -= 1
    lines = lines[:k_mat] + mat_block + lines[k_mat:]
    kws = [keyword(l) for l in lines]
    step0, step1 = kws.index("*STEP"), kws.index("*END STEP")
    # insert directly after the procedure keyword of the step (e.g. *STATIC) and its data line(s)
    k_proc = next(k for k in range(step0 + 1, step1 + 1) if kws[k] is not None)
    k_ins = next(k for k in range(k_proc + 1, step1 + 1) if lines[k].startswith("*"))
    lines = (lines[:k_ins] + ["** BOUNDARY CONDITION (make_gpe_model.py)", f"*INCLUDE, INPUT={os.path.basename(f_bc)}"]
             + lines[k_ins:])
    desc = "; ".join(f"grain {g}: {mat}, Bunge {eul}" if bicrystal else f"{mat}, Bunge {eul}"
                     for g, mat, eul in grains)
    kws = [keyword(l) for l in lines]
    if "*HEADING" in kws:
        k = kws.index("*HEADING") + 1
        if k < len(lines) and not lines[k].startswith("*"):
            lines[k] = f"Generalised plane-strain K-field model: {desc}"
    header = ["** Generalised plane-strain K-field model, written by make_gpe_model.py from "
              + os.path.basename(args.inp),
              f"** {'bicrystal' if bicrystal else 'single crystal'}: {desc}",
              f"** K_I={args.KI:.3f}, K_II={args.KII:.3f}, K_III={args.KIII:.3f} [MPa*sqrt(mm)], a={args.a} mm, "
              f"crack tip=({args.tip[0]}, {args.tip[1]}) mm, alpha={args.alpha} deg",
              f"** {os.path.basename(f_bc)}: u1, u2, u3 of the analytical field on all nodes of the boundary node columns",
              f"** {os.path.basename(f_eq)}: *EQUATION u_i(z) = u_i(z_min) for all other node columns",
              "**"]
    lines = [l for k, l in enumerate(lines) if not (l.strip() == "**" and k > 0 and lines[k - 1].strip() == "**")]
    open(f_inp, "w").write("\n".join(header + lines))

    print(f"input            : {args.inp}")
    print(f"model            : {'bicrystal' if bicrystal else 'single crystal'}; instance prefix: {instance or '(none)'}")
    print(f"node columns     : {len(columns)} (node levels per column: {n_levels})")
    for g, mat, eul in grains:
        print(f"grain {g}          : {sum(1 for v in elem_grain.values() if v == g)} elements, {mat}, Bunge {eul}, "
              f"*Orientation axes {orientation_axes(eul)}")
    print(f"boundary         : {len(bc_nodes)} nodes ({len(b_roots)} columns), {len(face)} crack-face nodes")
    print(f"equations        : {n_eq}")
    print("removed          : " + "; ".join(removed_sec + removed_mat + removed_bc + removed_inc))
    print(f"written          : {f_inp}, {f_bc}, {f_eq}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
