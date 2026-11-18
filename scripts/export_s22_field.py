"""export_s22_field.py - sigma_22 at the nodes around the crack tip for a field plot (Abaqus Python)

Reads the last frame of the .odb and writes the opening stress S22 at the nodes of the elements
around the crack tip, together with the element connectivity, so that the field can be plotted
outside Abaqus (plot_s22_field.py) with the same colour scale as the analytical field.

  2D models: corner nodes of the CPE8 elements.
  3D models: the mid-plane nodes (mid-side nodes of the vertical edges of the C3D20 elements).

The values are the element-nodal values averaged per node (all attached elements; no 75 % threshold).
The crack faces consist of two node copies, so the field is not smeared across the crack.
The crack-tip node itself gets NaN (singular, extrapolation artefact).

Run in the case folder, next to the .odb:

    abaqus python export_s22_field.py --mode isotropic
    abaqus python export_s22_field.py --mode single_crystal
    abaqus python export_s22_field.py --mode bicrystal

Options: --odb <file.odb>  --prefix <name>  --half <mm> (half edge of the window, default 0.005)

Output: <prefix>_s22_field_nodes.csv   label,x,y,s22      [mm, mm, MPa]
        <prefix>_s22_field_quads.csv   n1,n2,n3,n4         (node labels of the quadrilaterals)
"""
import glob
import math
import os
import sys

from odbAccess import openOdb
from abaqusConstants import ELEMENT_NODAL, CARTESIAN

MODE = "isotropic"       # "isotropic" | "single_crystal" | "bicrystal" (decides about the transformation)
ODB = None
PREFIX = None
HALF = 0.005             # half edge of the window around the tip [mm]
Z_TOL = 1e-6             # tolerance for the z level of the mid-plane (3D) [mm]


def parse_options(argv):
    global MODE, ODB, PREFIX, HALF
    args = list(argv)
    while args:
        a = args.pop(0)
        if a == "--mode":
            MODE = args.pop(0)
        elif a == "--odb":
            ODB = args.pop(0)
        elif a == "--prefix":
            PREFIX = args.pop(0)
        elif a == "--half":
            HALF = float(args.pop(0))
        else:
            raise SystemExit("unknown option: %s" % a)


parse_options(sys.argv[1:])


def main():
    odb_path = ODB or sorted(glob.glob("*.odb"), key=os.path.getmtime)[-1]
    print("Reading %s ..." % odb_path)
    odb = openOdb(path=str(odb_path))
    step = odb.steps[list(odb.steps.keys())[-1]]
    inst = list(odb.rootAssembly.instances.values())
    if len(inst) > 1:
        print("[!] %d instances; node labels are used without the instance name." % len(inst))

    coords = {}
    for i in inst:
        for n in i.nodes:
            c = n.coordinates
            coords[(i.name, n.label)] = (float(c[0]), float(c[1]), float(c[2]) if len(c) > 2 else 0.0)
    zs = [c[2] for c in coords.values()]
    is_3d = (max(zs) - min(zs)) > 1e-12
    z_mid = 0.5 * (max(zs) + min(zs))
    prefix = PREFIX or {"isotropic": "iso", "single_crystal": "sc", "bicrystal": "bc"}[MODE] + ("_3D" if is_3d else "")

    # quadrilaterals in the plane: 2D corner nodes, 3D the mid-plane nodes (nodes 17-20 of C3D20)
    quads, need = [], set()
    for i in inst:
        for e in i.elements:
            conn = e.connectivity
            if is_3d and len(conn) == 20:
                q = conn[16:20]
            elif (not is_3d) and len(conn) in (4, 8):
                q = conn[0:4]
            else:
                continue
            cx = sum(coords[(i.name, m)][0] for m in q) / 4.0
            cy = sum(coords[(i.name, m)][1] for m in q) / 4.0
            if abs(cx) <= HALF and abs(cy) <= HALF:
                quads.append((i.name, tuple(q)))
                need.update((i.name, m) for m in q)
    if not quads:
        print("[!] No elements found in the window (check --half and the element type: C3D20 or CPE8/CPE4).")
        odb.close()
        return

    field = step.frames[-1].fieldOutputs["S"]
    if MODE != "isotropic":
        try:
            csys = odb.rootAssembly.DatumCsysByThreePoints(name="GLOBAL_XY_EXPORT", coordSysType=CARTESIAN,
                                                           origin=(0.0, 0.0, 0.0), point1=(1.0, 0.0, 0.0),
                                                           point2=(0.0, 1.0, 0.0))
            field = field.getTransformedField(datumCsys=csys)
        except Exception as e:
            print("[!] Transformation to the global system failed (%s); raw values are used." % e)

    acc = {}
    for v in field.getSubset(position=ELEMENT_NODAL).values:
        name = v.instance.name if getattr(v, "instance", None) is not None else ""
        key = (name, v.nodeLabel)
        if key in need:
            acc.setdefault(key, []).append(float(v.data[1]))           # S22 (Abaqus order 11, 22, 33, 12, ...)
    odb.close()

    with open(prefix + "_s22_field_nodes.csv", "w") as f:
        f.write("label,x,y,s22\n")
        for key in sorted(need, key=lambda k: k[1]):
            x, y, _ = coords[key]
            vals = acc.get(key)
            if vals is None or (abs(x) < 1e-9 and abs(y) < 1e-9):      # tip node: singular -> NaN
                s = "nan"
            else:
                s = "%.10e" % (sum(vals) / float(len(vals)))
            f.write("%d,%.10e,%.10e,%s\n" % (key[1], x, y, s))
    with open(prefix + "_s22_field_quads.csv", "w") as f:
        f.write("n1,n2,n3,n4\n")
        for name, q in quads:
            f.write("%d,%d,%d,%d\n" % tuple(q))
    print("-> %s_s22_field_nodes.csv (%d nodes), %s_s22_field_quads.csv (%d elements)" % (
        prefix, len(need), prefix, len(quads)))
    print("   plot with:  python plot_s22_field.py %s_s22_field_nodes.csv %s_s22_field_quads.csv" % (prefix, prefix))


main()
