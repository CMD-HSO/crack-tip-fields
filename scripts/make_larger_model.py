# -*- coding: utf-8 -*-
"""
make_larger_model.py - extend the C6 boundary layer model by an outer ring of elements (R = 1 mm -> R_NEW)

The mesh inside r = 1 mm stays unchanged (all node and element numbers, the crack tip with duplicate nodes, the
sets and the equations). N_LAYERS rings of C3D20 elements are added from r = 1 mm to r = R_NEW with the same
angular division (64 sectors), the same element type and the same radial growth (r_k = R_NEW**(k / N_LAYERS)), or with --graded a growth that increases outwards. The
seam crack is continued along the negative x axis (duplicate nodes of the upper and lower face), the grains are
continued along the interface (y = 0, x > 0).

Input : the model folder with In718_bicrystal_ssy_dup.inp and In718_bicrystal_ssy_dup_equations.inp (R = 1 mm)
Output: new model files with the same names in --out (input file, equations; the boundary condition is written
        separately with the analytical field at the new outer boundary).

Usage:  python make_larger_model.py --src <folder> --out <folder> [--R 2.0] [--layers 19 | --graded [--q0 1.04 --dq 0.01 --qmax 1.10]]
"""
import argparse
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mesh_lib import parse  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--R", type=float, default=2.0)
ap.add_argument("--layers", type=int, default=19)
ap.add_argument("--graded", action="store_true", help="coarser outside: ratio of the radii grows from --q0 by --dq per ring up to --qmax (the number of rings follows from R)")
ap.add_argument("--q0", type=float, default=1.04)
ap.add_argument("--dq", type=float, default=0.01)
ap.add_argument("--qmax", type=float, default=1.10)
ap.add_argument("--name", default="In718_bicrystal_ssy_dup")
a = ap.parse_args()

SRC_INP = os.path.join(a.src, a.name + ".inp")
SRC_EQ = os.path.join(a.src, a.name + "_equations.inp")
lines, nodes, elems, sets = parse(SRC_INP)
B = set(sets[("nset", "BOUNDARY")])
print("old mesh:", len(nodes), "nodes,", len(elems), "elements, boundary nodes", len(B))

if a.graded:
    q, r_, k_ = [], 1.0, 0
    while r_ < a.R * (1 - 1e-9):
        q.append(min(a.q0 + a.dq * k_, a.qmax))
        r_ *= q[-1]
        k_ += 1
    sc = (a.R / np.prod(q)) ** (1.0 / len(q))                   # scale all ratios so that the last radius is exactly R
    RADII = list(np.cumprod([x * sc for x in q]))
    NL = len(RADII)
else:
    NL = a.layers
    RADII = [a.R ** (k / NL) for k in range(1, NL + 1)]      # 1 < r_1 < ... < r_NL = R
print("radii:", np.round(RADII[:3], 5), "...", np.round(RADII[-3:], 5), " ratio", RADII[0])

# ------------------------------------------------------------------ node bookkeeping
new_nodes = {}                  # id -> (x, y, z)
next_id = max(nodes) + 1
cols = {}                       # key -> (bottom, top, mid or None); mid only for corner columns
col_kind = {}                   # bottom id -> "corner" | "mid"
col_ring = {}                   # bottom id -> ring index k


def xy(n):
    return nodes[n][:2] if n in nodes else new_nodes[n][:2]


def zc(n):
    return nodes[n][2] if n in nodes else new_nodes[n][2]


def new_column(key, x, y, kind, k, with_mid, zb, zt):
    global next_id
    if key in cols:
        return cols[key]
    b = next_id
    t = next_id + 1
    m = None
    new_nodes[b] = (x, y, zb)
    new_nodes[t] = (x, y, zt)
    next_id += 2
    if with_mid:
        m = next_id
        new_nodes[m] = (x, y, 0.5 * (zb + zt))
        next_id += 1
    cols[key] = (b, t, m)
    col_kind[b] = kind
    col_ring[b] = k
    return cols[key]


# ------------------------------------------------------------------ outer elements of the old mesh
# element description: dict with 20 node ids (C3D20 numbering) in counter-clockwise order
def outer_edge_index(c):
    """corner positions i with corners i and i+1 on the boundary (bottom face)"""
    res = []
    for i in range(4):
        if c[i] in B and c[(i + 1) % 4] in B:
            res.append(i)
    return res


sectors = []                    # one entry per sector: (element ids chain), list of 20-node lists
for e, c in elems.items():
    oi = outer_edge_index(c)
    if len(oi) == 1 and c[8 + oi[0]] in B:
        sectors.append((e, c, oi[0]))
print("outer elements:", len(sectors))
assert len(sectors) == 64

# vertical mid nodes of the old corner columns are in the element (idx 16..19); bottom id -> (top, mid) for old nodes
z_bot, z_top = 0.0, 0.01
for n in nodes:
    z = nodes[n][2]
    z_top = max(z_top, z) if z > 0 else z_top
z_top = max(v[2] for v in nodes.values())
print("thickness:", z_top)

new_elems = {}
next_eid = max(elems) + 1
sector_grain = {}               # new element id -> 1 or 2
old_grain = {}
for g in (1, 2):
    for e in sets[("elset", "GPE_GRAIN%d" % g)]:
        old_grain[e] = g


def build_ring(prev, i, k, r_k, last):
    """prev: 20 node ids (CCW), outer edge = corners i, i+1; returns the new 20-node list and its outer edge index (=2)"""
    ci, cj = prev[i], prev[(i + 1) % 4]                    # bottom corners i, i+1
    ti, tj = prev[4 + i], prev[4 + (i + 1) % 4]            # top corners
    vi, vj = prev[16 + i], prev[16 + (i + 1) % 4]          # vertical mids
    mb, mt = prev[8 + i], prev[12 + i]                     # tangential mids (bottom, top) of the outer edge
    zb, zt = zc(ci), zc(ti)
    Pi, Pj = np.array(xy(ci)), np.array(xy(cj))
    ui, uj = Pi / np.linalg.norm(Pi), Pj / np.linalg.norm(Pj)
    Qi, Qj = ui * r_k, uj * r_k
    # corner columns (keyed by the bottom node of the inner corner), radial mid columns, tangential mid column
    cq_i = new_column(("c", ci), Qi[0], Qi[1], "corner", k, True, zb, zt)
    cq_j = new_column(("c", cj), Qj[0], Qj[1], "corner", k, True, zb, zt)
    Ri = 0.5 * (Pi + Qi)
    Rj = 0.5 * (Pj + Qj)
    cr_i = new_column(("r", ci), Ri[0], Ri[1], "mid", k, False, zb, zt)
    cr_j = new_column(("r", cj), Rj[0], Rj[1], "mid", k, False, zb, zt)
    if last:                                              # boundary: arc
        mid = Qi + Qj
        mid = mid / np.linalg.norm(mid) * r_k
    else:                                                 # interior: straight edge
        mid = 0.5 * (Qi + Qj)
    ct = new_column(("t", mb), mid[0], mid[1], "mid", k, False, zb, zt)
    # CCW corner order of the new element: (j, i, Qi, Qj)
    N = [cj, ci, cq_i[0], cq_j[0], tj, ti, cq_i[1], cq_j[1],
         mb, cr_i[0], ct[0], cr_j[0],                      # 9..12  bottom: mid(1-2), mid(2-3), mid(3-4), mid(4-1)
         mt, cr_i[1], ct[1], cr_j[1],                      # 13..16 top
         vj, vi, cq_i[2], cq_j[2]]                         # 17..20 vertical mids of corners 1..4
    return N


# order: mid(1-2) is the old tangential edge (corners j,i) -> mb; mid(2-3) = radial at i; mid(3-4) = outer tangential; mid(4-1) = radial at j
for (e, c, i) in sectors:
    prev, pi = list(c), i
    for k in range(1, NL + 1):
        N = build_ring(prev, pi, k, RADII[k - 1], k == NL)
        eid = next_eid
        next_eid += 1
        new_elems[eid] = N
        sector_grain[eid] = old_grain[e]
        prev, pi = N, 2                                   # outer edge of the new element: corners 3 and 4 (idx 2, 3)
print("new nodes:", len(new_nodes), "new elements:", len(new_elems))

# ------------------------------------------------------------------ checks of the geometry of the new elements
allnodes = dict(nodes)
allnodes.update(new_nodes)


def shape_derivs_c3d20():
    """natural coordinates of the 20 nodes (Abaqus C3D20 numbering)"""
    c = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
         (0, -1, -1), (1, 0, -1), (0, 1, -1), (-1, 0, -1), (0, -1, 1), (1, 0, 1), (0, 1, 1), (-1, 0, 1),
         (-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
    return np.array(c, float)


NAT = shape_derivs_c3d20()


def dN(xi, eta, ze):
    """derivatives of the 20 shape functions"""
    d = np.zeros((20, 3))
    for n in range(20):
        a_, b_, c_ = NAT[n]
        if n < 8:
            f1, f2, f3 = (1 + a_ * xi), (1 + b_ * eta), (1 + c_ * ze)
            s = a_ * xi + b_ * eta + c_ * ze - 2
            d[n, 0] = 0.125 * a_ * f2 * f3 * (s + f1)
            d[n, 1] = 0.125 * b_ * f1 * f3 * (s + f2)
            d[n, 2] = 0.125 * c_ * f1 * f2 * (s + f3)
        elif a_ == 0:
            f2, f3 = (1 + b_ * eta), (1 + c_ * ze)
            d[n, 0] = -0.5 * xi * f2 * f3
            d[n, 1] = 0.25 * (1 - xi ** 2) * b_ * f3
            d[n, 2] = 0.25 * (1 - xi ** 2) * f2 * c_
        elif b_ == 0:
            f1, f3 = (1 + a_ * xi), (1 + c_ * ze)
            d[n, 0] = 0.25 * (1 - eta ** 2) * a_ * f3
            d[n, 1] = -0.5 * eta * f1 * f3
            d[n, 2] = 0.25 * (1 - eta ** 2) * f1 * c_
        else:
            f1, f2 = (1 + a_ * xi), (1 + b_ * eta)
            d[n, 0] = 0.25 * (1 - ze ** 2) * a_ * f2
            d[n, 1] = 0.25 * (1 - ze ** 2) * f1 * b_
            d[n, 2] = -0.5 * ze * f1 * f2
    return d


def min_jacobian(conn):
    X = np.array([allnodes[n] for n in conn])
    worst = 1e300
    for pt in NAT:
        J = dN(*pt).T @ X
        worst = min(worst, np.linalg.det(J))
    for g in (-0.57735, 0.57735):
        for h in (-0.57735, 0.57735):
            for p in (-0.57735, 0.57735):
                J = dN(g, h, p).T @ X
                worst = min(worst, np.linalg.det(J))
    return worst


# reference: positive determinants of the old outer elements, same test for the new ones
old_min = min(min_jacobian(elems[e]) for e, c, i in sectors)
new_min = min(min_jacobian(c) for c in new_elems.values())
vol = sum(np.linalg.det(dN(0, 0, 0).T @ np.array([allnodes[n] for n in c])) * 8 for c in new_elems.values())
print("min det J: old outer elements %.3e, new elements %.3e" % (old_min, new_min))
assert new_min > 0, "negative Jacobian in the new elements"
area = np.pi * (a.R ** 2 - 1.0) * 0.01
print("volume of the new elements (centre-point estimate) %.6f, ring volume %.6f" % (vol, area))

# edges used once (bottom face, corner-corner) in the whole model: must lie at r = R or on the crack faces
allel = dict(elems)
allel.update(new_elems)
edge_count = {}
for e, c in allel.items():
    for i in range(4):
        key = tuple(sorted((c[i], c[(i + 1) % 4])))
        edge_count[key] = edge_count.get(key, 0) + 1
bad = 0
n_outer = 0
n_crack = 0
for (p, q), cnt in edge_count.items():
    if cnt == 1:
        P, Q = np.array(allnodes[p][:2]), np.array(allnodes[q][:2])
        if abs(np.linalg.norm(P) - a.R) < 1e-5 and abs(np.linalg.norm(Q) - a.R) < 1e-5:
            n_outer += 1
        elif abs(P[1]) < 1e-9 and abs(Q[1]) < 1e-9 and P[0] <= 1e-9 and Q[0] <= 1e-9:
            n_crack += 1
        else:
            bad += 1
    elif cnt > 2:
        bad += 1
print("free edges: outer boundary", n_outer, " crack faces", n_crack, " other (must be 0):", bad)
assert bad == 0 and n_outer == 64

# ------------------------------------------------------------------ new boundary sets, grain sets
outer_nodes = []                 # all nodes of the new outer boundary (all levels)
for key, (b, t, m) in cols.items():
    if col_ring[b] == NL and col_kind[b] in ("corner", "mid") and key[0] in ("c", "t"):
        outer_nodes += [b, t] + ([m] if m is not None else [])
print("outer boundary nodes:", len(outer_nodes))
assert len(outer_nodes) == 323
grain_of_node = {}
for eid, c in new_elems.items():
    for n in c:
        grain_of_node.setdefault(n, set()).add(sector_grain[eid])
g1 = sorted(n for n in outer_nodes if 1 in grain_of_node[n])
g2 = sorted(n for n in outer_nodes if grain_of_node[n] == {2})
print("BOUNDARY_GRAIN1/2:", len(g1), len(g2))

# ------------------------------------------------------------------ equations (generalised plane strain) for the new interior columns
eq = []


def add_eq(n, b):
    for d in (1, 2, 3):
        eq.append("*Equation\n2\nPart-1-1.%d, %d, 1., Part-1-1.%d, %d, -1." % (n, d, b, d))


# (a) the old boundary columns become interior columns
for n in sorted(B):
    if abs(nodes[n][2]) < 1e-9:                          # bottom node of a column
        top = n + 15715
        add_eq(top, n)
        if (n + 31430) in B:
            add_eq(n + 31430, n)
n_old = len(eq) // 3
# (b) new columns except the outer boundary
outer_set = set(outer_nodes)
for key, (b, t, m) in cols.items():
    if b in outer_set:
        continue
    add_eq(t, b)
    if m is not None:
        add_eq(m, b)
print("new equations:", len(eq) // 3, "(old boundary columns:", n_old, ")")

# coverage: every non-bottom node is constrained, on the outer boundary or a tip node
constrained = set()
for s in eq:
    constrained.add(int(re.search(r"Part-1-1\.(\d+),", s).group(1)))
old_eq_nodes = set()
for l in open(SRC_EQ, encoding="utf-8", errors="ignore"):
    m_ = re.match(r"Part-1-1\.(\d+),", l)
    if m_:
        old_eq_nodes.add(int(m_.group(1)))
free = [n for n in new_nodes if new_nodes[n][2] > 1e-9 and n not in constrained and n not in outer_set]
print("unconstrained non-bottom new nodes that are not on the outer boundary (must be 0):", len(free))
assert not free

# ------------------------------------------------------------------ write the new input file
def fmt_node(n, v):
    return "%7d, %13.9g, %13.9g, %13.9g" % (n, v[0], v[1], v[2])


def fmt_ids(ids, per=16):
    out = []
    for i in range(0, len(ids), per):
        out.append(", ".join("%6d" % x for x in ids[i:i + per]) + ("," if i + per < len(ids) else ""))
    return out


def fmt_elem(e, c):
    s = "%5d, " % e
    txt = ", ".join("%6d" % x for x in c[:15]) + ",\n      " + ", ".join("%6d" % x for x in c[15:])
    return s + txt


out = []
i = 0
n_l = len(lines)
# find positions
i_elem = next(k for k, l in enumerate(lines) if l.lower().startswith("*element, type=c3d20"))
i_bulk = next(k for k, l in enumerate(lines) if l.startswith("*Elset, elset=BULK"))
i_bnd = next(k for k, l in enumerate(lines) if l.startswith("*Nset, nset=BOUNDARY") and "GRAIN" not in l)
i_bg1 = next(k for k, l in enumerate(lines) if l.startswith("*Nset, nset=BOUNDARY_GRAIN1"))
i_bg2 = next(k for k, l in enumerate(lines) if l.startswith("*Nset, nset=BOUNDARY_GRAIN2"))
i_eg1 = next(k for k, l in enumerate(lines) if l.startswith("*Elset, elset=GPE_GRAIN1"))
i_eg2 = next(k for k, l in enumerate(lines) if l.startswith("*Elset, elset=GPE_GRAIN2"))


def block_end(start):
    k = start + 1
    while k < n_l and (not lines[k].startswith("*") or lines[k].startswith("**")):
        if lines[k].startswith("**") and lines[k].strip() != "**" and not re.match(r"\*\* (grain|crack tip)", lines[k]):
            pass
        k += 1
    return k


# the end of a set block: first line starting with "*" but not "**", or a "** grain" comment line (before *Solid Section)
def set_end(start):
    k = start + 1
    while k < n_l and not (lines[k].startswith("*") and not lines[k].startswith("**")) and not lines[k].startswith("** grain"):
        k += 1
    return k


end_bnd, end_bg1, end_bg2 = set_end(i_bnd), set_end(i_bg1), set_end(i_bg2)
end_eg1, end_eg2 = set_end(i_eg1), set_end(i_eg2)
# node block ends before "*Element"; the element block ends before the first "*Elset"
node_end = i_elem
elem_end = i_bulk

g1_new = [e for e in new_elems if sector_grain[e] == 1]
g2_new = [e for e in new_elems if sector_grain[e] == 2]
n_el_total = max(new_elems)

for k in range(n_l):
    l = lines[k]
    if k == node_end:
        out.append("** outer ring of elements r = 1 mm ... %g mm (make_larger_model.py): %d new nodes" % (a.R, len(new_nodes)))
        for n in sorted(new_nodes):
            out.append(fmt_node(n, new_nodes[n]))
    if k == elem_end:
        for e in sorted(new_elems):
            out.append(fmt_elem(e, new_elems[e]))
    if k == i_bulk:
        out.append("*Elset, elset=BULK, generate")
        out.append("     1,   %d, 1" % n_el_total)
        continue
    if k == i_bulk + 1:
        continue
    if i_bnd <= k < end_bnd:
        if k == i_bnd:
            out.append("*Nset, nset=BOUNDARY")
            out += fmt_ids(sorted(outer_nodes))
        continue
    if i_bg1 <= k < end_bg1:
        if k == i_bg1:
            out.append("*Nset, nset=BOUNDARY_GRAIN1")
            out += fmt_ids(g1)
        continue
    if i_bg2 <= k < end_bg2:
        if k == i_bg2:
            out.append("*Nset, nset=BOUNDARY_GRAIN2")
            out += fmt_ids(g2)
        continue
    if k == end_eg1:
        out += fmt_ids(sorted(g1_new), 16)
    if k == end_eg2:
        out += fmt_ids(sorted(g2_new), 16)
    out.append(l)

# comments: heading
txt = "\n".join(out)
txt = txt.replace("outer radius R = 1 mm.", "outer radius R = %g mm (inner mesh up to r = 1 mm unchanged, outer ring added by make_larger_model.py)." % a.R)
txt = txt.replace("5184 C3D20 elements, 47145 nodes", "%d C3D20 elements, %d nodes" % (n_el_total, len(nodes) + len(new_nodes) - 320))
os.makedirs(a.out, exist_ok=True)
open(os.path.join(a.out, a.name + ".inp"), "w", encoding="utf-8", newline="").write(txt + ("" if txt.endswith("\n") else "\n"))

# equations: old ones + new ones
old = open(SRC_EQ, encoding="utf-8", errors="ignore", newline="").read().replace("\r\n", "\n")
with open(os.path.join(a.out, a.name + "_equations.inp"), "w", encoding="utf-8", newline="") as f:
    f.write(old if old.endswith("\n") else old + "\n")
    f.write("** outer ring (make_larger_model.py): old boundary columns (now interior) and the new columns\n")
    f.write("\n".join(eq) + "\n")
print("written to", a.out)
