# -*- coding: utf-8 -*-
"""
make_c6_runs.py - complete run folders for the study of the plastic zone at the grain boundary crack (case C6)

For one run it writes into --out the input file (R = 1 mm model, or the model with the outer ring up to --R), the equations,
the boundary condition (K-field of the interface model at the outer boundary), the material card, the UMAT files and the
evaluation scripts. All text files get Linux line endings (the Abaqus Fortran compiler of the cluster rejects CRLF).

The input file of the base model (--base, R = 1 mm, with material card and UMAT) is changed as follows:
  * output: field output at --frames equidistant stages; for the elements INNER (r <= 1 mm) the state variables for the
    latent kinematic hardening are written in addition (SDV115-126, 166-177, 178-189, 190-201), for all elements S, MISES
    and the accumulated slip SDV226-237; contour integral with --contours contours
  * assembly sets INNER (elements r <= 1 mm) and RING_R1 (nodes at r = 1 mm)
  * with --R > 1 the outer ring of elements (make_larger_model.py, --graded: coarser towards the boundary)

Usage:  python make_c6_runs.py --base <folder> --out <folder> --K 40 --a 20 [--R 4] [--frames 15] [--contours 15]
        (K in MPa*sqrt(m), a and R in mm)
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, HERE)
import crack_tip_fields as ctf              # noqa: E402
from mesh_lib import parse                  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--K", type=float, required=True, help="K_I at the end of the step [MPa*sqrt(m)]")
ap.add_argument("--a", type=float, required=True, help="reference length a [mm]")
ap.add_argument("--R", type=float, default=1.0, help="outer radius [mm] (1 = no outer ring)")
ap.add_argument("--frames", type=int, default=15)
ap.add_argument("--contours", type=int, default=15)
ap.add_argument("--name", default="In718_bicrystal_ssy_dup")
ap.add_argument("--tau0", type=float, default=None, help="new material card: tau_0 [MPa] (together with --cd and --d)")
ap.add_argument("--cd", type=float, default=None, help="new material card: c/d [MPa]")
ap.add_argument("--d", type=float, default=None, help="new material card: d [-]")
ap.add_argument("--no-umat", action="store_true", help="do not copy the UMAT files (a modified UMAT has to be put into the folder)")
ap.add_argument("--note", default="", help="free text for RUN_README.txt")
a = ap.parse_args()
a.base = os.path.abspath(a.base)
a.out = os.path.abspath(a.out)

K_MM = a.K * np.sqrt(1000.0)
os.makedirs(a.out, exist_ok=True)


def lf(b):
    return b.replace(b"\r\n", b"\n")


# ------------------------------------------------------------------ geometry: outer ring or not
if a.R > 1.0 + 1e-9:
    cmd = [sys.executable, os.path.join(HERE, "make_larger_model.py"), "--src", a.base, "--out", a.out, "--R", str(a.R), "--graded"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout[-900:])
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise SystemExit("make_larger_model.py failed")
else:
    for f in (a.name + ".inp", a.name + "_equations.inp"):
        shutil.copy(os.path.join(a.base, f), os.path.join(a.out, f))

inp_path = os.path.join(a.out, a.name + ".inp")
lines = lf(open(inp_path, "rb").read()).decode("utf-8", errors="ignore").split("\n")
_, bnodes, belems, bsets = parse(os.path.join(a.base, a.name + ".inp"))
ring_r1 = sorted(bsets[("nset", "BOUNDARY")])
n_inner = len(belems)

# ------------------------------------------------------------------ header comments
txt = "\n".join(lines)
txt = re.sub(r"K_I=[\d.]+, K_II=0\.000, K_III=0\.000 \[MPa\*sqrt\(mm\)\], a=[\d.]+ mm",
             "K_I=%.3f, K_II=0.000, K_III=0.000 [MPa*sqrt(mm)], a=%.1f mm" % (K_MM, a.a), txt, count=1)

# ------------------------------------------------------------------ assembly sets INNER and RING_R1
sdv = lambda lo, hi: ", ".join("SDV%d" % k for k in range(lo, hi + 1))
assy = ("*Elset, elset=INNER, instance=Part-1-1, generate\n     1,   %d, 1\n"
        "*Nset, nset=RING_R1, instance=Part-1-1\n" % n_inner)
ids = ring_r1
assy += "\n".join(", ".join("%d" % x for x in ids[i:i + 12]) + "," for i in range(0, len(ids), 12)).rstrip(",") + "\n"
assert txt.count("*End Instance") == 1
if "elset=INNER, instance=Part-1-1, generate" not in txt:     # the base model of a previous call already has the sets
    txt = txt.replace("*End Instance\n", "*End Instance\n" + assy, 1)

# ------------------------------------------------------------------ output requests
i0 = txt.index("** OUTPUT REQUESTS")
i1 = txt.index("*Contour Integral")
out_block = ("** OUTPUT REQUESTS\n"
             "*Output, field, number interval=%d\n*Node Output\nU, RF\n*Element Output\nS, MISES, %s\n"
             "*Output, field, number interval=%d\n*Element Output, elset=INNER\n"
             "** SDV115-126 scalar backstress, 166-177 slip rate, 178-189 resolved shear stress, 190-201 resolved backstress\n"
             "%s,\n%s,\n%s,\n%s\n"
             "*Output, history, variable=PRESELECT\n") % (a.frames, sdv(226, 237), a.frames,
                                                         sdv(115, 126), sdv(166, 177), sdv(178, 189), sdv(190, 201))
txt = txt[:i0] + out_block + txt[i1:]
txt = re.sub(r"\*Contour Integral, crack name=CRACK, contours=\d+, crack tip nodes",
             "*Contour Integral, crack name=CRACK, contours=%d, crack tip nodes" % a.contours, txt)
open(inp_path, "wb").write(txt.encode("utf-8"))

# ------------------------------------------------------------------ boundary condition
sys.path.insert(0, os.path.join(ROOT, "tests"))
from test_fe_workflow import boundary_condition, read_bc    # noqa: E402

model = ctf.InterfaceCrackModel("In718 (anisotropic)", (105, 60, 30), "In718 (anisotropic)", (0, 0, 0))
mesh = ctf.parse_abaqus_mesh(inp_path)
bc = boundary_condition(inp_path, model, ["BOUNDARY_GRAIN1", "BOUNDARY_GRAIN2"], (K_MM, 0.0, 0.0), a=a.a, dofs=(1, 2, 3))
ids = list(mesh["node_sets"]["BOUNDARY_GRAIN1"]) + list(mesh["node_sets"]["BOUNDARY_GRAIN2"])
entries = [(n, [bc[(n, 1)], bc[(n, 2)], bc[(n, 3)]]) for n in ids]
header = ("Crack-tip field boundary condition (interface)\n"
          "K_I=%.2f, K_II=0.0, K_III=0.0 [MPa*sqrt(mm)]\n"
          "Crack tip=(0.0,0.0) mm, alpha=0.0 deg, a=%.1f mm\n"
          "Node set(s): BOUNDARY_GRAIN1, BOUNDARY_GRAIN2\n"
          "Instance prefix: Part-1-1\n"
          "Outer boundary r = %g mm" % (K_MM, a.a, a.R))
bc_path = os.path.join(a.out, "In718_bicrystal_ssy_bc.inp")
ctf.export_abaqus_boundary(bc_path, entries, dof_list=(1, 2, 3), header_comment=header, instance="Part-1-1")
_b = open(bc_path, "rb").read()
open(bc_path, "wb").write(lf(_b))
print("boundary condition:", len(read_bc(bc_path)), "values, max |u| =", max(abs(v) for v in read_bc(bc_path).values()))

# ------------------------------------------------------------------ material card, UMAT, evaluation scripts
if a.tau0 is not None:
    mk = lf(open(os.path.join(a.base, "make_material_card.py"), "rb").read()).decode("utf-8")
    mk = re.sub(r"^TAU0, D = .*$", "TAU0, D = %g, %g" % (a.tau0, a.d), mk, flags=re.M)
    mk = re.sub(r"^C_D = .*$", "C_D = %g * D" % a.cd, mk, flags=re.M)
    open(os.path.join(a.out, "make_material_card.py"), "wb").write(mk.encode("utf-8"))
    r = subprocess.run([sys.executable, "-I", os.path.join(a.out, "make_material_card.py")], cwd=a.out, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(r.stderr)
else:
    for f in ("material_card.inp", "make_material_card.py"):
        shutil.copy(os.path.join(a.base, f), os.path.join(a.out, f))
if not a.no_umat:
    for f in glob.glob(os.path.join(a.base, "UMAT*.for")):
        shutil.copy(f, a.out)
for f in ("evaluate_kfield_abaqus.py", "compare_stress_circle.py", "c6_extract_abaqus.py"):
    shutil.copy(os.path.join(HERE, f), os.path.join(a.out, f))
for f in glob.glob(os.path.join(a.out, "*")):
    if f.split(".")[-1] in ("for", "inp", "py", "md", "txt"):
        b = open(f, "rb").read()
        if b"\r" in b:
            open(f, "wb").write(lf(b))

# ------------------------------------------------------------------ short description of the run
Kfr = [a.K * k / a.frames for k in range(1, a.frames + 1)]
with open(os.path.join(a.out, "RUN_README.txt"), "w", encoding="utf-8", newline="\n") as f:
    f.write("C6, plastic zone study, run folder written by make_c6_runs.py\n\n")
    f.write("model radius R = %g mm (inner mesh r <= 1 mm identical for all runs), reference length a = %g mm\n" % (a.R, a.a))
    f.write("K_I at the end of the step: %g MPa*sqrt(m) = %.3f MPa*sqrt(mm); one step of 1 s, frame N: K_I = %g*N/%d\n"
            % (a.K, K_MM, a.K, a.frames))
    f.write("frames 1..%d: K_I = %s MPa*sqrt(m)\n" % (a.frames, ", ".join("%.2f" % k for k in Kfr)))
    f.write("material: Schilli et al. (2022), Table 3; hardening model: see UMATHT_global_ini.for (iAFHKin)\n")
    if a.tau0 is not None:
        f.write("material card: tau_0 = %g MPa, c/d = %g MPa, d = %g, c = %.2f MPa" % (a.tau0, a.cd, a.d, a.cd * a.d) + chr(10))
    if a.no_umat:
        f.write("NOTE: the UMAT (user material subroutine, *.for) is not part of this repository. It is described in Zandomeni et al. (2026); see the data availability statement of the manuscript." + chr(10))
    if a.note:
        f.write(a.note + chr(10))
    f.write("output: field output at %d stages; INNER = elements r <= 1 mm, RING_R1 = nodes at r = 1 mm; %d contours\n"
            % (a.frames, a.contours))
    f.write("\nrun:      SMALauncher job=In718_bicrystal_ssy_dup user=UMATHTH_SXGRADPlast_grDefo_v11sdv cpus=8\n")
    f.write("evaluate: SMALauncher python evaluate_kfield_abaqus.py --mode bicrystal --odb In718_bicrystal_ssy_dup.odb "
            "--euler1 105 60 30 --euler2 0 0 0 --elastic1 193500 127800 89400 --elastic2 193500 127800 89400 "
            "--step-time 1 --KI %g --frame N --prefix ssy_fN   (N = 1..%d)\n" % (a.K, a.frames))
print("written", a.out)
