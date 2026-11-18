# -*- coding: utf-8 -*-
"""
c6_extract_abaqus.py - extraction of the C6 plastic zone study from the ODB (Abaqus Python, runs on the cluster)

Reads the result file of one run (K_I ramp, R = 1 ... 10 mm) and writes small CSV files for the analysis on the PC
(c6_analyze.py). The ODB itself is only read. Python 2.7 / numpy of Abaqus 2023.

Usage (in the folder of the ODB; frames default: all frames of the last step):
    /opt/apps/abaqus/V6R2023x/linux_a64/code/bin/SMALauncher python c6_extract_abaqus.py --odb In718_bicrystal_ssy_dup.odb --KI 40

Output in ./export/ :
    meta.csv                      model radius, thickness, step time, number of frames, K at the end of the step
    frames.csv                    frame, time t, K_I [MPa sqrt(m)]
    elements_frame<N>.csv         per element: centre, corner coordinates of the lower face, accumulated slip
                                  (max / mean over the integration points of the sum over the 12 slip systems and per system),
                                  number of active slip systems, scalar backstress chi_s (SDV115-126), resolved
                                  backstress (SDV190-201) per system (the last two only for the elements INNER)
    nodal_stress_frame<N>.csv     stress at mid-plane nodes, per side of the crack plane (global system):
                                  group line (interface ahead of the tip), boundary (outer boundary), ring1 (r = 1 mm)
    crack_faces_frame<N>.csv      displacement of the crack-face nodes (y = 0, x <= 0), upper and lower face
    ring_u_frame<N>.csv           displacement of the nodes at r = 1 mm (lower level), for the comparison with the K-field

SDV layout of the UMAT (UMATHTH_SXGRADPlast_grDefo_v11sdv.for):
    115-126 scalar backstress chi_s, 166-177 slip rate, 178-189 resolved shear stress, 190-201 resolved backstress,
    226-237 accumulated slip per slip system.
"""
import argparse
import glob
import math
import os
import sys
import time

import numpy as np
from odbAccess import openOdb
from abaqusConstants import ELEMENT_NODAL, CARTESIAN

ap = argparse.ArgumentParser()
ap.add_argument("--odb", default=None)
ap.add_argument("--out", default="export")
ap.add_argument("--frames", type=int, nargs="+", default=None, help="frame numbers (default: all frames of the last step)")
ap.add_argument("--KI", type=float, default=40.0, help="K_I at the end of the step [MPa sqrt(m)]")
ap.add_argument("--rate-thr", type=float, default=1.0e-5, help="slip rate [1/s] above which a slip system counts as active")
ap.add_argument("--ring-radius", type=float, default=1.0, help="radius of the ring for the displacement comparison [mm]")
ap.add_argument("--tol", type=float, default=1.0e-5, help="geometric tolerance [mm]")
ap.add_argument("--no-global", action="store_true", help="do not transform the stresses to the global system")
ap.add_argument("--sdv-total", type=int, default=None,
                help="number of the state variable with the total accumulated slip (scalar, e.g. 256); written as g_tot_max / g_tot_mean")
a = ap.parse_args()

NSYS = 12
SDV_G = ["SDV%d" % k for k in range(226, 238)]          # accumulated slip
SDV_RATE = ["SDV%d" % k for k in range(166, 178)]       # slip rate
SDV_CHI = ["SDV%d" % k for k in range(115, 127)]        # scalar backstress
SDV_BS = ["SDV%d" % k for k in range(190, 202)]         # resolved backstress


def log(msg):
    print(msg)
    sys.stdout.flush()


odb_path = a.odb or sorted(glob.glob("*.odb"), key=os.path.getmtime)[-1]
log("Reading %s ..." % odb_path)
t_start = time.time()
odb = openOdb(path=str(odb_path), readOnly=True)
step = odb.steps[list(odb.steps.keys())[-1]]
inst = list(odb.rootAssembly.instances.values())[0]
if not os.path.isdir(a.out):
    os.makedirs(a.out)
T = 1.0   # total step time of the load ramp (step.timePeriod is only the reached time in an aborted run)
nfr = len(step.frames)
frames = a.frames if a.frames else list(range(1, nfr))
frames = [f for f in frames if 0 < f < nfr]
log("step %s: time %g s, %d frames in the ODB, frames to evaluate: %s" % (step.name, T, nfr, frames))

# ------------------------------------------------------------------ geometry
node_xyz = {}
for n in inst.nodes:
    c = n.coordinates
    node_xyz[n.label] = (float(c[0]), float(c[1]), float(c[2]) if len(c) > 2 else 0.0)
zs = [v[2] for v in node_xyz.values()]
z_min, z_max = min(zs), max(zs)
z_mid = 0.5 * (z_min + z_max)
R_MAX = max(math.hypot(v[0], v[1]) for v in node_xyz.values())
log("nodes %d, elements %d, model radius R = %.5g mm, thickness %.5g mm" % (len(node_xyz), len(inst.elements), R_MAX, z_max - z_min))

el_label, el_conn, el_cen, el_corner = [], {}, {}, {}
node_elems = {}
for e in inst.elements:
    c = list(e.connectivity)
    el_label.append(e.label)
    el_conn[e.label] = c
    pts = [node_xyz[n] for n in c[:8]]
    el_cen[e.label] = (sum(p[0] for p in pts) / 8.0, sum(p[1] for p in pts) / 8.0)
    el_corner[e.label] = [(node_xyz[n][0], node_xyz[n][1]) for n in c[:4]]
    for n in c:
        node_elems.setdefault(n, []).append(e.label)
el_label.sort()


def side_of_node(n):
    """+1: attached to elements above the crack plane (y > 0), -1: below, 0: both"""
    s = set(1 if el_cen[e][1] > 0.0 else -1 for e in node_elems.get(n, []))
    return 0 if len(s) != 1 else list(s)[0]


# element sets by name (INNER = elements r <= 1 mm)
inner = None
for nm, es in odb.rootAssembly.elementSets.items():
    if nm.upper() == "INNER":
        inner = set(e.label for e in es.elements[0]) if hasattr(es, "elements") and es.elements else None
if inner is None:
    inner = set(l for l in el_label if math.hypot(*el_cen[l]) <= 1.0 + 1.0e-3)
    log("element set INNER not found in the ODB, using the elements with centre r <= 1 mm (%d)" % len(inner))

with open(os.path.join(a.out, "meta.csv"), "w") as f:
    f.write("key,value\nR_model,%.8g\nthickness,%.8g\nstep_time,%.8g\nn_frames,%d\nK_end_MPa_sqrt_m,%.8g\nrate_thr,%.3g\nn_inner,%d\nn_elements,%d\n"
            % (R_MAX, z_max - z_min, T, nfr, a.KI, a.rate_thr, len(inner), len(el_label)))
with open(os.path.join(a.out, "frames.csv"), "w") as f:
    f.write("frame,t,K_I_MPa_sqrt_m\n")
    for i in frames:
        f.write("%d,%.8g,%.8g\n" % (i, step.frames[i].frameValue, a.KI * step.frames[i].frameValue / T))


# ------------------------------------------------------------------ reading helpers
def ip_scalar(fo):
    """(element labels, integration points, data) of a scalar field at integration points"""
    try:
        el, ip, d = [], [], []
        for b in fo.bulkDataBlocks:
            el.append(np.asarray(b.elementLabels).reshape(-1))
            ip.append(np.asarray(b.integrationPoints).reshape(-1))
            d.append(np.asarray(b.data).reshape(-1))
        return np.concatenate(el), np.concatenate(ip), np.concatenate(d)
    except Exception as err:                         # fallback: value objects (slower)
        log("   [bulk data not usable (%s), reading the values one by one]" % err)
        vals = fo.values
        return (np.array([v.elementLabel for v in vals]), np.array([v.integrationPoint for v in vals]),
                np.array([v.data for v in vals], dtype=float))


def read_group(frame, names):
    """keys (element*100 + integration point, sorted) and an array (n_ip x len(names)); None if a field is missing"""
    keys, cols = None, []
    for nm in names:
        if nm not in frame.fieldOutputs:
            return None, None
        el, ip, d = ip_scalar(frame.fieldOutputs[nm])
        k = el.astype(np.int64) * 100 + ip.astype(np.int64)
        order = np.argsort(k)
        k, d = k[order], d[order]
        if keys is None:
            keys = k
        elif len(k) != len(keys) or np.any(k != keys):
            raise RuntimeError("integration point layout of %s differs from %s" % (nm, names[0]))
        cols.append(d)
    return keys, np.column_stack(cols)


def per_element(keys, data, how):
    """aggregate the integration points of each element: returns (element labels, array n_el x ncol)"""
    el = keys // 100
    labels, start = np.unique(el, return_index=True)
    if how == "mean":
        n = np.diff(np.append(start, len(el))).astype(float)
        out = np.add.reduceat(data, start, axis=0) / n.reshape(-1, 1)
    else:
        out = np.maximum.reduceat(data, start, axis=0)
    return labels, out


# ------------------------------------------------------------------ nodes of interest (mid-plane for stresses, lower level for displacements)
TOL = a.tol
Y_TOL = max(TOL, 1.0e-9)
mid_nodes = dict((n, (v[0], v[1])) for n, v in node_xyz.items() if abs(v[2] - z_mid) < 1.0e-6 * max(z_max - z_min, 1.0))
low_nodes = dict((n, (v[0], v[1])) for n, v in node_xyz.items() if abs(v[2] - z_min) < 1.0e-6 * max(z_max - z_min, 1.0))
line_nodes = [n for n, (x, y) in mid_nodes.items() if abs(y) < Y_TOL and x > 1.0e-9]
bnd_nodes = [n for n, (x, y) in mid_nodes.items() if abs(math.hypot(x, y) - R_MAX) < 1.0e-4 * R_MAX]
ring_nodes_s = [n for n, (x, y) in mid_nodes.items() if abs(math.hypot(x, y) - a.ring_radius) < 1.0e-4 * a.ring_radius] \
    if R_MAX > a.ring_radius * 1.001 else []
wanted_s = {}
for grp, lst in (("line", line_nodes), ("boundary", bnd_nodes), ("ring1", ring_nodes_s)):
    for n in lst:
        wanted_s.setdefault(n, []).append(grp)
face_nodes = [n for n, (x, y) in low_nodes.items() if abs(y) < Y_TOL and x < 1.0e-9]
ring_nodes_u = [n for n, (x, y) in low_nodes.items() if abs(math.hypot(x, y) - a.ring_radius) < 1.0e-4 * a.ring_radius]
log("nodes: interface line %d, outer boundary %d, ring r=%.3g (stress) %d, crack faces %d, ring (displacement) %d"
    % (len(line_nodes), len(bnd_nodes), a.ring_radius, len(ring_nodes_s), len(face_nodes), len(ring_nodes_u)))

csys = None
if not a.no_global:
    try:
        csys = odb.rootAssembly.DatumCsysByThreePoints(name="GLOBAL_XY_EVAL", coordSysType=CARTESIAN, origin=(0.0, 0.0, 0.0),
                                                       point1=(1.0, 0.0, 0.0), point2=(0.0, 1.0, 0.0))
    except Exception as err:
        log("[!] datum system could not be created (%s); stresses in the material system of the elements." % err)


def voigt(d):
    """Abaqus order (11,22,33,12,13,23) -> (11,22,33,23,13,12)"""
    return (d[0], d[1], d[2], d[5], d[4], d[3])


# ------------------------------------------------------------------ frame loop
for fi in frames:
    fr = step.frames[fi]
    tag = "%02d" % fi
    K = a.KI * fr.frameValue / T
    log("frame %d: t = %.5g, K_I = %.4g MPa sqrt(m)" % (fi, fr.frameValue, K))

    # ---- elements: slip, active systems, backstress
    try:
        keys, g = read_group(fr, SDV_G)
        if keys is None:
            raise RuntimeError("%s not in the ODB" % SDV_G[0])
        gabs = np.abs(g).sum(axis=1).reshape(-1, 1)      # the slip of the systems is signed: sum of the absolute values
        lab, g_max = per_element(keys, gabs, "max")
        _, g_mean_sum = per_element(keys, gabs, "mean")
        _, g_sys = per_element(keys, g, "mean")
        rows = dict((int(l), [float(g_max[i, 0]), float(g_mean_sum[i, 0])] + [float(v) for v in g_sys[i]])
                    for i, l in enumerate(lab))
        extra = {}
        kr, rate = read_group(fr, SDV_RATE)
        if kr is not None:
            n_act = (np.abs(rate) > a.rate_thr).sum(axis=1).reshape(-1, 1).astype(float)
            l2, na_max = per_element(kr, n_act, "max")
            _, na_mean = per_element(kr, n_act, "mean")
            for i, l in enumerate(l2):
                extra.setdefault(int(l), {})["act"] = (float(na_max[i, 0]), float(na_mean[i, 0]))
        kc, chi = read_group(fr, SDV_CHI)
        if kc is not None:
            l3, chi_m = per_element(kc, chi, "mean")
            for i, l in enumerate(l3):
                extra.setdefault(int(l), {})["chi"] = [float(v) for v in chi_m[i]]
        kb, bs = read_group(fr, SDV_BS)
        if kb is not None:
            l4, bs_m = per_element(kb, bs, "mean")
            for i, l in enumerate(l4):
                extra.setdefault(int(l), {})["bs"] = [float(v) for v in bs_m[i]]
        nan = float("nan")
        gtot = {}
        if a.sdv_total:
            kt, gt = read_group(fr, ["SDV%d" % a.sdv_total])
            if kt is None:
                raise RuntimeError("SDV%d not in the ODB" % a.sdv_total)
            lt, t_max = per_element(kt, np.abs(gt), "max")
            _, t_mean = per_element(kt, np.abs(gt), "mean")
            for i, l in enumerate(lt):
                gtot[int(l)] = (float(t_max[i, 0]), float(t_mean[i, 0]))
        head = ["element", "xc", "yc", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "g_max", "g_mean"] \
            + ["g%d" % (k + 1) for k in range(NSYS)] + ["n_act_max", "n_act_mean"] \
            + ["chi%d" % (k + 1) for k in range(NSYS)] + ["bs%d" % (k + 1) for k in range(NSYS)]
        if a.sdv_total:
            head += ["g_tot_max", "g_tot_mean"]
        with open(os.path.join(a.out, "elements_frame%s.csv" % tag), "w") as f:
            f.write(",".join(head) + "\n")
            for l in el_label:
                c = el_corner[l]
                r = rows.get(l, [0.0] * (2 + NSYS))
                ex = extra.get(l, {})
                act = ex.get("act", (nan, nan))
                chi_l = ex.get("chi", [nan] * NSYS)
                bs_l = ex.get("bs", [nan] * NSYS)
                tail = (",%.6e,%.6e" % gtot.get(l, (0.0, 0.0))) if a.sdv_total else ""
                f.write("%d,%.8e,%.8e,%s,%s,%.6g,%.6g,%s%s\n" % (
                    l, el_cen[l][0], el_cen[l][1], ",".join("%.8e,%.8e" % p for p in c),
                    ",".join("%.6e" % v for v in r), act[0], act[1],
                    ",".join("%.6e" % v for v in chi_l + bs_l), tail))
        log("   elements: %d rows, largest accumulated slip %.4g" % (len(rows), max(v[0] for v in rows.values())))
        if gtot:
            log("   SDV%d (total slip): largest %.4g, largest net slip (SDV226-237) %.4g, elements with total > 1.01 net: %d"
                % (a.sdv_total, max(v[0] for v in gtot.values()), max(v[0] for v in rows.values()),
                   sum(1 for l, v in gtot.items() if l in rows and v[0] > 1.01 * rows[l][0] + 1.0e-9)))
    except Exception as err:
        log("   [!] elements failed: %s" % err)

    # ---- stresses at mid-plane nodes
    try:
        field = fr.fieldOutputs["S"]
        if csys is not None:
            try:
                field = field.getTransformedField(datumCsys=csys)
            except Exception as err:
                log("   [!] transformation to the global system failed (%s)" % err)
        acc = {}
        for v in field.getSubset(position=ELEMENT_NODAL).values:
            n = v.nodeLabel
            if n not in wanted_s:
                continue
            side = 1 if el_cen[v.elementLabel][1] > 0.0 else -1
            acc.setdefault((n, side), []).append(voigt(v.data))
        with open(os.path.join(a.out, "nodal_stress_frame%s.csv" % tag), "w") as f:
            f.write("group,node,x,y,side,s11,s22,s33,s23,s13,s12\n")
            for (n, side), lst in sorted(acc.items()):
                m = [sum(s[c] for s in lst) / float(len(lst)) for c in range(6)]
                x, y = mid_nodes[n]
                for grp in wanted_s[n]:
                    f.write("%s,%d,%.8e,%.8e,%d,%s\n" % (grp, n, x, y, side, ",".join("%.6e" % v for v in m)))
        log("   stresses: %d node/side values" % len(acc))
    except Exception as err:
        log("   [!] stresses failed: %s" % err)

    # ---- displacements (crack faces, ring)
    try:
        U = {}
        for v in fr.fieldOutputs["U"].values:
            U[v.nodeLabel] = [float(c) for c in v.data]
        with open(os.path.join(a.out, "crack_faces_frame%s.csv" % tag), "w") as f:
            f.write("node,x,y,side,u1,u2,u3\n")
            for n in sorted(face_nodes):
                u = U.get(n)
                if u is None:
                    continue
                f.write("%d,%.8e,%.8e,%d,%.8e,%.8e,%.8e\n" % (n, low_nodes[n][0], low_nodes[n][1], side_of_node(n), u[0], u[1], u[2]))
        with open(os.path.join(a.out, "ring_u_frame%s.csv" % tag), "w") as f:
            f.write("node,x,y,side,u1,u2,u3\n")
            for n in sorted(ring_nodes_u):
                u = U.get(n)
                if u is None:
                    continue
                f.write("%d,%.8e,%.8e,%d,%.8e,%.8e,%.8e\n" % (n, low_nodes[n][0], low_nodes[n][1], side_of_node(n), u[0], u[1], u[2]))
        log("   displacements: %d crack-face nodes, %d ring nodes" % (len(face_nodes), len(ring_nodes_u)))
    except Exception as err:
        log("   [!] displacements failed: %s" % err)

odb.close()
log("done in %.0f s; output in %s" % (time.time() - t_start, os.path.abspath(a.out)))
