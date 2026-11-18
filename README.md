# crack_tip_fields – crack-tip fields and Abaqus boundary conditions

[![tests](https://github.com/CMD-HSO/crack-tip-fields/actions/workflows/tests.yml/badge.svg)](https://github.com/CMD-HSO/crack-tip-fields/actions/workflows/tests.yml)

Python toolkit for near-tip displacement and stress fields of cracks in

- **isotropic** linear-elastic material (classical LEFM / Williams solution),
- **anisotropic** single crystals (Stroh formalism),
- **interface cracks between two anisotropic grains**, e.g. grain boundaries in metals
  (Stroh–Ting–Hwu formalism with oscillating singularity),

and an interactive app that evaluates these fields as **displacement boundary conditions for
Abaqus FE sub-models** (boundary-layer / K-field models).

## Repository structure

| Path | Content |
|---|---|
| `src/crack_tip_fields.py` | Core library: material database, rotations, the three crack models, closed-form displacement and stress fields, interface fracture quantities, Abaqus `.inp` parser and `*BOUNDARY` exporter |
| `notebooks/abaqus_crack_tip_bc_app.ipynb` | Interactive app: loads an Abaqus model, evaluates the crack-tip field at the selected nodes and exports an Abaqus `*BOUNDARY` file |
| `notebooks/crack_tip_fields_validation_and_examples.ipynb` | Worked examples for all three cases, crack-opening plots, Griffith criterion, directional Young's modulus, validation against reference values and finite differences |
| `scripts/make_gpe_model.py` | Turns a meshed 3D slice of the boundary-layer model into an Abaqus model in **generalised plane strain** (single crystal or bicrystal) |
| `scripts/evaluate_kfield_abaqus.py` | Abaqus-Python post-processing of 2D and 3D K-field models: J on all contours, K_I recovered from J, opening stress ahead of the tip, all stress components on a circle around the tip, CSV files and plots |
| `scripts/compare_stress_circle.py` | Compares the FE stresses on the circle with the analytical field (all components, both grains) and reports the deviation |
| `scripts/export_s22_field.py`, `scripts/plot_s22_field.py` | Export of the opening stress field from the `.odb` (Abaqus Python) and plot next to the analytical field |
| `scripts/make_c6_runs.py`, `scripts/make_larger_model.py`, `scripts/mesh_lib.py` | Case C6: writes the run folders of the study (base model R = 1 mm, models of larger radius, output requests, boundary condition, material card) |
| `scripts/c6_extract_abaqus.py`, `scripts/c6_analyze.py` | Case C6 (together with `evaluate_kfield_abaqus.py`): export from the `.odb` (Abaqus Python: J, slip, back stress, stresses, crack faces) and local analysis of the plastic zone (accumulated slip above 1e-4 in at least one integration point), K deviation, CTOD |
| `scripts/c6_figures/` | Case C6: scripts of the figures and tables of Section 5 (`python c6_overview.py <folder of the C6_run* directories> <png>`, etc.) |
| `scripts/reproduce_manuscript_results.py` | Regenerates the numbers and figures of the verification (manuscript, Section 4 and Supplementary material 2, Section B) and checks each value against its tolerance |
| `tests/` | Automated tests (`pytest`): manuscript checks, interface stresses, finite-element workflow; `validate_stress_interface.py` also runs on its own |
| `pyproject.toml`, `.github/workflows/tests.yml` | Package definition (`pip install .`) and automatic tests on GitHub |
| `examples/fe_case_study/` | Abaqus input files of the FE case study of the manuscript, one folder per case (2D plane strain and 3D generalised plane strain, mode I and mixed mode, and the small-scale-yielding bicrystal of case C6 with crystal plasticity; the UMAT itself is not included), see its [README](examples/fe_case_study/README.md) |
| `docs/index.md` | Short documentation overview |

All physics lives in `src/crack_tip_fields.py`; notebooks and scripts only call the module.

## Installation

Python ≥ 3.10. In the folder of the repository:

```bash
pip install .            # module crack_tip_fields with numpy, scipy, matplotlib
pip install ".[app]"     # in addition ipywidgets, ipykernel for the interactive application
pip install ".[test]"    # in addition pytest
```

Afterwards `import crack_tip_fields` works from any folder (`crack_tip_fields.__version__` gives the
version). Installation is optional: the notebooks and scripts also find the module in `src/` without it,
and `pip install -r requirements.txt` installs only the dependencies. `scripts/evaluate_kfield_abaqus.py`
runs with the Python interpreter of Abaqus (`abaqus python`, Python 2.7 in Abaqus 2023) and needs only
`numpy` and `matplotlib`.

In VS Code: install the extensions **"Jupyter"** and **"Jupyter Notebook Renderers"** (Microsoft).

## Conventions

- Units: mm, N, s, MPa. The stress intensity factor has the unit MPa·√mm
  (MPa·√m = MPa·√mm / √1000).
- Stiffness matrices and stress vectors in Voigt order `11, 22, 33, 23, 13, 12`.
- Grain orientation by Bunge Euler angles `(phi1, Phi, phi2)` in degrees: successive rotations
  about `z`, the rotated `x'` and the rotated `z''` axis. The stiffness is rotated from the crystal
  to the global system with `R = Rz(phi1) Rx(Phi) Rz(phi2)` (the transpose of the orientation
  matrix `g` of texture analysis), so Euler angles from EBSD or the literature can be used directly.
- Local crack-tip system: `x1` = crack-propagation direction (the crack lies at `x1 < 0`),
  `x2` = normal to the crack/interface. Grain 1 is above the interface (`x2 >= 0`), grain 2
  below (`x2 <= 0`). The app transforms global Abaqus coordinates `(X, Y)` into this system
  with the crack-tip position `(X0, Y0)` and the angle `alpha` between the global `X` axis and
  `x1` (counter-clockwise positive).
- Stress intensity vector `k = [K_II, K_I, K_III]`; `a` is the crack half-length (reference length
  of the oscillating interface field).

## Abaqus boundary-condition app (`notebooks/abaqus_crack_tip_bc_app.ipynb`)

### First start

1. Open the notebook from the folder `notebooks/` and select a **Python kernel** in which the
   packages above are installed.
2. **"Run All"** once from top to bottom. The cell **"0. Setup"** creates all widgets; only
   afterwards do the buttons/dropdowns work reliably.

### Step by step

**1. Load Abaqus file** – enter the path to an `.inp` file and press **"Load file"**. The console
shows the numbers of nodes and elements, the node sets found and the part instance.

> The parser reads `*NODE`, `*ELEMENT` and `*NSET` blocks (explicit lists and `GENERATE`) of
> an Abaqus/CAE input file with **one part instance** (or of a flat orphan-mesh file). For
> assembly-based files the nodes are written as `<instance>.<node>`; the instance name is
> detected automatically and shown in the field *Instance prefix* (editable, empty = plain node
> labels). The crack has to be modelled as a **seam** (duplicate nodes on the two crack faces):
> the duplicate nodes on the outer boundary are assigned to the upper/lower crack face from the
> attached elements, so that the exported displacements reproduce the crack opening (and, for
> general crystal orientations, the sliding and tearing) of the analytical field.

**2. Model, material and orientation** – choose the model type (isotropic / anisotropic
single crystal / interface crack). Depending on the choice, the material dropdown and the
Euler angles for grain 1 (and grain 2 for the interface crack) are shown.

**3. Boundary conditions and crack geometry** – `K_I`, `K_II`, `K_III` [MPa·√mm], crack
half-length `a` [mm], crack-tip position `X0`, `Y0` and angle `alpha` in the **global**
(Abaqus) coordinate system.

**4. Select node set(s)** – the node set to which the boundary condition is applied. For the
interface crack, separate sets for grain 1 (top, `x2 >= 0`) and grain 2 (bottom, `x2 <= 0`).

**5. Export options and execution** – choose the degrees of freedom (u1/u2/u3) and the output
file name, then **"Compute & Export"**. This writes an `.inp` file with a `*BOUNDARY` block and
shows a scatter plot of the displacement magnitude at the nodes as a check. Include it in the
step of the main model with

```
*INCLUDE, INPUT=<your_file>.inp
```

**6. Visualise the crack-tip fields** – **"Show crack-tip fields"** shows displacement and
stress fields as heatmaps around the evaluated FE nodes and in a zoom on the crack tip
(adjustable radius; `0` = automatically 10 % of `a`).

> Before using exported boundary conditions in production, verify the coordinate convention
> (`alpha`, `x1` = crack direction) on a known test case, e.g. the models in `examples/fe_case_study/`.

## 2D or 3D model? Generalised plane strain (`scripts/make_gpe_model.py`)

The anisotropic and interface solutions are **generalised plane-strain** fields: the displacements
depend only on `(x1, x2)`, so `eps33 = 0`, but in general `u3 != 0` and `eps13, eps23 != 0`.

- **2D plane strain (`CPE`) is sufficient** for isotropic material and for crystal orientations in
  which in-plane and anti-plane deformation decouple, i.e. the crack front is a twofold symmetry
  axis of the crystal (e.g. rotations about a `<100>` crack front of a cubic crystal, `Phi = 0`).
  Export `u1, u2` with the app.
- **For general orientations** (single crystal or bicrystal) plane-strain elements suppress `u3`
  and are too stiff: in the case study (Fe `(27, 15, 10)`, Fe/Cu) the stress intensity factor
  recovered from J is 0.5 % too high. Use a **3D slice in generalised plane strain** instead, which
  reproduces the prescribed K to better than 0.001 %.

To build the 3D slice:

1. In Abaqus/CAE, extrude the meshed 2D boundary-layer model in the crack-front direction `z`
   (one or more layers of `C3D8(R)`/`C3D20(R)` elements, thickness small compared with the
   radius), keep the crack as a seam, define a node set with the boundary nodes (one node level
   is sufficient), the contour integral along the crack front and a static step, and write the
   input file (example: `examples/fe_case_study/3D_slice/slice_3D.inp`).
2. Run (from the folder of the slice)

   ```bash
   # single crystal
   python make_gpe_model.py slice.inp --material "Fe (cubic, example)" --euler 27 15 10 --boundary-set BOUNDARY
   # bicrystal (crack along the grain boundary; grain 1 above, grain 2 below the crack plane)
   python make_gpe_model.py slice.inp --material "Fe (cubic, example)" --euler 27 15 10 \
                            --material2 "Cu (cubic, example)" --euler2 5 40 0 --boundary-set BOUNDARY
   ```

   (with the path to `scripts/make_gpe_model.py`). Options: `--KI --KII --KIII` [MPa·√mm] (default
   `K_I = 27.5·√1000`), `--a` [mm], `--tip X0 Y0`, `--alpha`, `--instance`, `--out`. The script writes
   `<out>.inp`, `<out>_bc.inp` and `<out>_equations.inp` (`--out` relative to the current folder; default
   `<input>_gpe` next to the input file):
   - the elements are assigned to the crystal / the two grains (element centroid above or below
     the crack plane), each with the unrotated stiffness from `MATERIALS` and an `*ORIENTATION`
     from its Bunge angles,
   - `u1, u2, u3` of the analytical field are prescribed at **all** nodes of the boundary node
     columns (all node levels through the thickness),
   - all other nodes of a node column (same `x, y`) are tied to its lowest node by `*EQUATION`,
     `u_i(z) = u_i(z_min)`, so that the field does not depend on `z` while `u3(x1, x2)` is free.

   Mesh, crack, contour integral and output requests are kept; existing orientations and
   sections of the part, all materials and the boundary conditions of the step are replaced
   (the script lists what it removed).
3. Run the job and evaluate J at the **mid-plane node** of the crack front (see below). At the two
   end nodes, the tractions transmitted through the coupled faces enter the domain integral; this
   contribution cancels between the two ends (their mean equals the mid-plane value).

## Evaluating Abaqus results

**Step 1 – `scripts/evaluate_kfield_abaqus.py` (Abaqus Python).** Set `MODE` (`"isotropic"`,
`"single_crystal"`, `"bicrystal"`), the materials/orientations and the prescribed `K_I` at the top of the
script (or pass `--mode`, `--euler1`, `--euler2`, `--odb`, `--prefix`, `--KI`, `--KII`, `--KIII` with K in
MPa·√m), then run in the folder with the `.odb`:

```bash
abaqus python evaluate_kfield_abaqus.py --mode bicrystal
```

2D and 3D models are detected automatically. For mixed-mode loading (`K_II`, `K_III` != 0) J is compared with
`J = k^T E k / 4` (full compliance matrix `E`, `k = [K_II, K_I, K_III]`); since J does not separate the modes, the
deviation of the magnitude of `k`, `sqrt(J/J_prescribed) - 1`, is reported. The script prints J and `K_I = 2 sqrt(J/E22)` for all
contours (3D: mid-plane node, both end nodes and the mean along the front; the crack fronts are identified
by the node sets of the contour integral) with the deviation from the prescribed value, and writes

- `<prefix>_J.csv`, `<prefix>_s22.csv` (opening stress ahead of the tip, 3D: mid-plane),
- `<prefix>_circle.csv`: all stress components on the node ring closest to `R_CIRCLE` (default 0.1 mm;
  3D: mid-plane), averaged per node separately for the elements above and below the crack plane, so that
  the jumps of the stresses at a grain boundary are kept,
- plots `<prefix>_J_K.png`, `<prefix>_s22.png`, `<prefix>_circle.png`;

`prefix` = `iso`, `sc` or `bc` (plus `_3D`).

**Step 2 – `scripts/compare_stress_circle.py` (Python 3).** Abaqus 2023 runs Python 2.7, so the comparison
with the analytical field of `crack_tip_fields` is a separate step:

```bash
python compare_stress_circle.py bc_3D_circle.csv --material "Fe (cubic, example)" --euler 27 15 10 \
       --material2 "Cu (cubic, example)" --euler2 5 40 0
```

(`--isotropic "In718 (isotropic)"` for the isotropic case, only `--material/--euler` for a single crystal;
for mixed-mode loading add `--KII`, `--KIII` – note: in MPa·√mm, like the app and `make_gpe_model.py`).
It prints the maximum deviation of each stress component from the analytical field, relative to
`K_I/sqrt(2 pi r)`, and writes `<prefix>_compare.csv` and `<prefix>_compare.png` (FE and analytical
`sigma_ij(theta)`).

## Reproducing the results of the manuscript

| Manuscript | Content | Reproduced by |
|---|---|---|
| Section 4.4 | single crystal: σ22 fields for two orientations, angular distribution, D22 / energy release rate | `scripts/reproduce_manuscript_results.py` → `scripts/figs/fig_single_crystal.pdf` |
| Section 4.5 | grain-boundary crack: β, γ, crack opening / interpenetration, E22, Griffith estimate | `scripts/reproduce_manuscript_results.py` → `scripts/figs/fig_interface.pdf` |
| Sections 4 and 5 | FE case study in Abaqus (cases C1–C5: verification, C6: small-scale yielding of the grain boundary crack with crystal plasticity) | `examples/fe_case_study/` (input files, boundary conditions, expected results) |
| Supplementary material 2, Appendix A | isotropic limits of the Stroh and interface solutions (derivations) | checked numerically in Appendix B.4 |
| Supplementary material 2, Appendix B | verification of the isotropic, single-crystal and interface solutions, isotropic limits, exact crack-face evaluation | `scripts/reproduce_manuscript_results.py`, `tests/validate_stress_interface.py` |

`reproduce_manuscript_results.py` prints every number of Sections 4.4, 4.5 and Appendix B together with
its tolerance check.

## Verification

All checks run with one command (about 45 s):

```bash
pytest
```

The tests (`tests/test_*.py`) comprise the numbers of the verification appendix and of Sections 4.4–4.5
of the manuscript with their tolerances, the closed-form interface stresses (Hwu solution, finite
differences, continuity, equilibrium), the exact crack-face evaluation (regression test) and the
finite-element workflow: reading the Abaqus input files, assigning the crack faces, the boundary
conditions of all models in `examples/fe_case_study/` and the 3D models generated by `make_gpe_model.py`.
They run automatically on GitHub for Python 3.10–3.12 at every push (GitHub Actions,
`.github/workflows/tests.yml`). The scripts can also be run on their own:

```bash
python tests/validate_stress_interface.py
python scripts/reproduce_manuscript_results.py     # ~30 s; add --no-figures to skip the figures
```

`validate_stress_interface.py` checks the closed-form stresses for several grain pairs and all
loading modes against the interface solution of Hwu (θ = 0), finite differences, interface continuity
and equilibrium, for six grain pairs including a general grain boundary in alloy 718. Verified range:
oscillation parameter β ≤ 0.071, which covers real metallic grain boundaries (β ≲ 0.07, Brunner et al. 2025). `reproduce_manuscript_results.py` regenerates all
numbers of the verification section of the manuscript and writes its figures to `scripts/figs/`;
both scripts end with `PASSED` / `ALL CHECKS PASSED`.

The notebook `crack_tip_fields_validation_and_examples.ipynb` computes and plots the fields of the
three cases, crack-opening/sliding displacements, the Griffith critical stress intensity factor, the
directional Young's modulus and the effective interface stiffness over the misorientation, and checks
the results against finite differences and reference values (last section).

## Python API (excerpt)

```python
import sys; sys.path.insert(0, "src")
from crack_tip_fields import (MATERIALS, IsotropicCrackModel, AnisotropicCrackModel,
                              InterfaceCrackModel)

a, K_I = 1.0, 869.626                                  # mm, MPa*sqrt(mm) (27.5 MPa*sqrt(m))

crystal = AnisotropicCrackModel("Fe (cubic, example)", 27., 0., 0.)
u = crystal.displacement(-0.01, 0.002, K_I, 0., 0., a)       # (u1, u2, u3) [mm]
s = crystal.stress(-0.01, 0.002, K_I, 0., 0., a)             # (s11, s22, s33, s23, s13, s12) [MPa]

gb = InterfaceCrackModel("Fe (cubic, example)", (27., 15., 10.),
                         "Cu (cubic, example)", (5., 40., 0.))
u_top = gb.displacement(-0.01, 0.002, K_I, 0., 0., a, side="top")
s_bot = gb.stress(-0.01, -0.002, K_I, 0., 0., a, side="bot")
```

Other functions: `compute_field_grid`, `parse_abaqus_mesh`, `parse_abaqus_inp`,
`abaqus_instance_name`, `crack_face_signs`, `export_abaqus_boundary`,
`global_to_local`, `local_to_global_vector`, `compute_eff_compliance`, `compute_critical_K`,
`calculate_E`, `calculate_Eeff`, crystal-orientation helpers (`EulerAnglesDegFromDirection`,
`direction_from_EulerAnglesDeg`, `vector_to_miller`).

## Adding a material

In `src/crack_tip_fields.py`, add another entry to the dictionary `MATERIALS` (cubic via
`cubic_stiffness(C11, C12, C44)`, isotropic via `isotropic_stiffness(E, nu)`, or a complete
6×6 Voigt matrix for lower symmetries).

## Troubleshooting

- **Buttons do not react / "NameError"** → the setup cell (section 0) has not been executed
  yet. Use "Run All" once.
- **`ModuleNotFoundError: crack_tip_fields`** → the notebook was not started from the folder
  `notebooks/` (the module is found via `../src`).
- **A cell stays at `[*]` forever** → kernel problem, not a code error. Restart the kernel.
- **Stroh error for exactly isotropic material in the anisotropic/interface model** → use the
  material `"Steel (isotropic, Stroh-compatible)"` instead of `"Steel (isotropic)"`; exactly
  isotropic constants give repeated Stroh eigenvalues.

## Change log

**Crack-face evaluation of the interface solution (fixed).** The powers `(z_j/2a)^(+-i*gamma)` are now formed as
`exp(+-i*gamma*(ln|z_j/2a| + i*arg z_j))` with the sign of a signed zero preserved (`_z_alpha`, `_zpow`). Previously,
evaluating the *lower* grain exactly on the crack face (`x2 = -0.0`, `x1 < 0`) used `arg z = +pi` instead of `-pi`
(NumPy drops the sign of a negative zero in `complex / float`), which changed the displacement of the lower crack-face
node by about 1-2 % for the Fe/Cu example. Evaluations at `x2 != 0` and all homogeneous cases were unaffected. The
boundary conditions of the bicrystal examples (`examples/fe_case_study/2D_mode_I/Fe_27_15_10_Cu_5_40_0/`,
`3D_mode_I/Fe_27_15_10_Cu_5_40_0/`, node 7 and its node column) were regenerated; a regression test is part of
`reproduce_manuscript_results.py` (Appendix B.4). Boundary conditions exported for interface cracks with an earlier
version should be exported again.

## Citation

If you use this software, please cite the accompanying article and the archived software version
(see `CITATION.cff`):

> C. Baumann, T. Seifert, Crack-tip fields in anisotropic materials and at grain boundaries as boundary
> conditions for finite element models: the Python toolkit crack_tip_fields and its application to a grain
> boundary crack with crystal plasticity (submitted).
> Software: crack_tip_fields v1.0.0, Zenodo, https://doi.org/TODO

## License

GNU Affero General Public License v3.0, see [`LICENSE`](LICENSE).

## References

- C. Hwu, *Anisotropic Elastic Plates*, Springer, 2010.
- T. C. T. Ting, *Anisotropic Elasticity: Theory and Applications*, Oxford University Press, 1996.
- F. Brunner, P. Andric, F. Maresca, *An atomistic K-test framework for general grain boundaries
  and triclinic single crystals*, Modelling Simul. Mater. Sci. Eng. 33 (2025) 035004,
  https://doi.org/10.1088/1361-651X/adba03
