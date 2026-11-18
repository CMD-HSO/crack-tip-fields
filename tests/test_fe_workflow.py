"""Finite-element workflow: reading the Abaqus input files, assigning the crack faces, writing the boundary
condition (as the application does) and generating the generalised plane-strain models (make_gpe_model.py).
The results are compared with the files of examples/fe_case_study."""
import os
import re
import subprocess
import sys

import numpy as np
import pytest

import crack_tip_fields as ctf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = os.path.join(ROOT, "examples", "fe_case_study")
K_I = 869.626                       # MPa*sqrt(mm), as entered in the application
MIXED = (869.626, 434.813, 260.888)  # K_I, K_II, K_III of the mixed-mode case
A_REF = 20.0                        # reference length a of the case study [mm] (only the phase (r/2a)^(i gamma) of the interface field depends on it)


def read_bc(path):
    """{(node, dof): value} of a *BOUNDARY file."""
    out = {}
    with open(path) as f:
        for line in f:
            m = re.match(r"\s*[\w-]+\.(\d+),\s*(\d),\s*\d,\s*(\S+)", line)
            if m:
                out[(int(m.group(1)), int(m.group(2)))] = float(m.group(3))
    return out


def boundary_condition(inp, model, node_sets, K, a=A_REF, dofs=(1, 2)):
    """Boundary condition as computed by the application (crack tip at the origin, alpha = 0)."""
    mesh = ctf.parse_abaqus_mesh(inp)
    ids = [(n, "top") for n in mesh["node_sets"][node_sets[0]]]
    if len(node_sets) > 1:
        ids += [(n, "bot") for n in mesh["node_sets"][node_sets[1]]]
    face = ctf.crack_face_signs([n for n, _ in ids], mesh["nodes"], mesh["elements"], (0.0, 0.0), 0.0)
    out = {}
    for nid, side in ids:
        X, Y, _ = mesh["nodes"][nid]
        x1, x2 = (float(v) for v in ctf.global_to_local(np.array(X), np.array(Y), (0.0, 0.0), 0.0))
        if nid in face:
            x2 = np.copysign(0.0, face[nid])
        if isinstance(model, ctf.InterfaceCrackModel):
            u = model.displacement(x1, x2, *K, a, side=side)
        else:
            u = model.displacement(x1, x2, *K, a)
        ux, uy = ctf.local_to_global_vector(u[0], u[1], 0.0)
        for d, v in zip((1, 2, 3), (ux, uy, u[2])):
            if d in dofs:
                out[(nid, d)] = float(v)
    return out


def assert_same_bc(computed, reference):
    assert set(computed) == set(reference), "different nodes / degrees of freedom"
    for k, ref in reference.items():
        # the files contain 7 significant digits
        assert computed[k] == pytest.approx(ref, rel=2e-6, abs=1e-12), k


def fe():
    return ctf.AnisotropicCrackModel("Fe (cubic, example)", 27, 15, 10)


def fecu():
    return ctf.InterfaceCrackModel("Fe (cubic, example)", (27, 15, 10), "Cu (cubic, example)", (5, 40, 0))


CASES_2D = [
    ("isotropic_In718", lambda: ctf.IsotropicCrackModel("In718 (isotropic)"), ["BOUNDARY"]),
    ("Fe_27_0_0", lambda: ctf.AnisotropicCrackModel("Fe (cubic, example)", 27, 0, 0), ["BOUNDARY"]),
    ("Fe_27_15_10", fe, ["BOUNDARY"]),
    ("Fe_27_15_10_Cu_5_40_0", fecu, ["BOUNDARY_GRAIN1", "BOUNDARY_GRAIN2"]),
]


@pytest.mark.parametrize("case, model, sets", CASES_2D, ids=[c[0] for c in CASES_2D])
def test_2d_boundary_condition(case, model, sets):
    folder = os.path.join(EX, "2D_mode_I", case)
    computed = boundary_condition(os.path.join(folder, case + "_mode_I.inp"), model(), sets, (K_I, 0.0, 0.0))
    assert_same_bc(computed, read_bc(os.path.join(folder, case + "_mode_I_bc.inp")))


def test_ssy_boundary_condition():
    """Small-scale-yielding case C6 (In718 bicrystal with crystal plasticity, 3D slice in generalised plane strain):
    boundary condition of the analytical field. K_I and the reference length a are read from the header of the
    boundary condition file, so that the test follows the K level and a of the run stored in the folder."""
    folder = os.path.join(EX, "3D_crystal_plasticity", "C6_run1_R1mm_KH")
    bc_file = os.path.join(folder, "In718_bicrystal_ssy_bc.inp")
    with open(bc_file) as f:
        header = "".join(next(f) for _ in range(4))
    K = float(re.search(r"K_I=([\d.]+)", header).group(1))
    a = float(re.search(r"a=([\d.]+) mm", header).group(1))
    model = ctf.InterfaceCrackModel("In718 (anisotropic)", (105, 60, 30), "In718 (anisotropic)", (0, 0, 0))
    computed = boundary_condition(os.path.join(EX, "3D_slice", "slice_3D.inp"), model,
                                  ["BOUNDARY_GRAIN1", "BOUNDARY_GRAIN2"], (K, 0.0, 0.0), a=a, dofs=(1, 2, 3))
    assert_same_bc(computed, read_bc(bc_file))


def test_crack_faces_and_mesh():
    mesh = ctf.parse_abaqus_mesh(os.path.join(EX, "2D_mode_I", "Fe_27_15_10", "Fe_27_15_10_mode_I.inp"))
    assert len(mesh["nodes"]) == 15715 and len(mesh["elements"]) == 5184
    assert ctf.abaqus_instance_name(mesh, "BOUNDARY") == "Part-1-1"
    face = ctf.crack_face_signs(mesh["node_sets"]["BOUNDARY"], mesh["nodes"], mesh["elements"], (0.0, 0.0), 0.0)
    assert face == {7: -1, 5266: 1}         # duplicate crack-mouth nodes: lower and upper face


CASES_3D = [
    ("3D_mode_I", "Fe_27_15_10", "Fe_27_15_10_mode_I_gpe", fe, ["BOUNDARY"], (K_I, 0.0, 0.0)),
    ("3D_mode_I", "Fe_27_15_10_Cu_5_40_0", "Fe_27_15_10_Cu_5_40_0_mode_I_gpe", fecu,
     ["BOUNDARY_GRAIN1", "BOUNDARY_GRAIN2"], (K_I, 0.0, 0.0)),
    ("3D_mixed_mode", "Fe_27_15_10_Cu_5_40_0", "Fe_27_15_10_Cu_5_40_0_mixed_mode_gpe", fecu,
     ["BOUNDARY_GRAIN1", "BOUNDARY_GRAIN2"], MIXED),
]


@pytest.mark.parametrize("group, case, name, model, sets, K", CASES_3D, ids=[c[0] + "-" + c[1] for c in CASES_3D])
def test_3d_boundary_condition_from_slice(group, case, name, model, sets, K):
    """The application applied to the 3D slice gives the boundary condition of the 3D models (u1, u2, u3)."""
    computed = boundary_condition(os.path.join(EX, "3D_slice", "slice_3D.inp"), model(), sets, K, dofs=(1, 2, 3))
    assert_same_bc(computed, read_bc(os.path.join(EX, group, case, name + "_bc.inp")))


def equations(path):
    with open(path) as f:
        return sorted(re.sub(r"[\w-]+\.(\d+)", r"\1", line.strip()) for line in f
                      if not line.startswith("*") and "," in line)


@pytest.mark.parametrize("group, case, name, model, sets, K", CASES_3D, ids=[c[0] + "-" + c[1] for c in CASES_3D])
def test_make_gpe_model(tmp_path, group, case, name, model, sets, K):
    """make_gpe_model.py reproduces boundary condition and constraint equations of the 3D models."""
    args = ["--material", "Fe (cubic, example)", "--euler", "27", "15", "10"]
    if "Cu" in case:
        args += ["--material2", "Cu (cubic, example)", "--euler2", "5", "40", "0"]
    args += ["--KI", str(K[0]), "--KII", str(K[1]), "--KIII", str(K[2]), "--a", str(A_REF)]
    out = str(tmp_path / name)
    res = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_gpe_model.py"),
                          os.path.join(EX, "3D_slice", "slice_3D.inp"), *args, "--boundary-set", "BOUNDARY",
                          "--out", out], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    folder = os.path.join(EX, group, case)
    assert_same_bc(read_bc(out + "_bc.inp"), read_bc(os.path.join(folder, name + "_bc.inp")))
    assert equations(out + "_equations.inp") == equations(os.path.join(folder, name + "_equations.inp"))


def test_export_abaqus_boundary(tmp_path):
    path = tmp_path / "bc.inp"
    ctf.export_abaqus_boundary(str(path), [(7, [1.0, -2.0, 3.5e-4])], dof_list=(1, 2, 3), header_comment="test",
                               instance="Part-1-1")
    assert read_bc(str(path)) == {(7, 1): 1.0, (7, 2): -2.0, (7, 3): 3.5e-4}
    text = path.read_text()
    assert "*BOUNDARY" in text.upper() and "Part-1-1.7, 1, 1," in text


def test_version_consistent():
    """Version of the module, pyproject.toml and CITATION.cff."""
    with open(os.path.join(ROOT, "pyproject.toml")) as f:
        assert 'version = "%s"' % ctf.__version__ in f.read()
    with open(os.path.join(ROOT, "CITATION.cff")) as f:
        assert "version: %s" % ctf.__version__ in f.read()
