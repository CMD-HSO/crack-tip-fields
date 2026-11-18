# FE case study: K-field check in Abaqus

Abaqus input files of the finite-element case study of the manuscript (Sections 4 and 5, cases C1--C6). A circular
boundary-layer model (radius R = 1 mm, crack tip at the origin, crack along the negative x-axis as a
seam) is loaded by the displacements of the analytical crack-tip field on its outer boundary,
a = 20 mm (reference length of the interface field, 2a = 40 mm; it enters only the phase (r/2a)^(i gamma), i.e. not at all for the homogeneous cases). Mode I: K_I = 27.5 MPa√m = 869.626 MPa√mm; mixed mode: in addition K_II = 0.5 K_I and
K_III = 0.3 K_I. The J-integral is compared with the prescribed energy release rate J = k^T E k / 4
(pure mode I: K_I recovered as 2 √(J / E22)), and all stress components on a circle around the tip are
compared with the analytical field.

Units: mm, N, MPa. All 2D models share the same mesh (5184 CPE8 elements, 15 715 nodes); the 3D
slices have the same in-plane mesh with one layer of C3D20 elements (thickness 0.01 mm).

## Folder structure

Every case is in its own folder, which contains all files needed to run it (the included files must be
in the same folder as the main input file). The file names state the loading (`mode_I`, `mixed_mode`)
and, for the 3D slices, the generalised plane-strain kinematics (`gpe`).

| Folder | Case | Files |
|---|---|---|
| `2D_mode_I/isotropic_In718/` | isotropic In718 (E = 162 GPa, ν = 0.3, 650 °C) | `isotropic_In718_mode_I.inp` + `_bc.inp` |
| `2D_mode_I/Fe_27_0_0/` | Fe single crystal, Bunge (27°, 0°, 0°) | `Fe_27_0_0_mode_I.inp` + `_bc.inp` |
| `2D_mode_I/Fe_27_15_10/` | Fe single crystal, Bunge (27°, 15°, 10°) | `Fe_27_15_10_mode_I.inp` + `_bc.inp` |
| `2D_mode_I/Fe_27_15_10_Cu_5_40_0/` | Fe (27°, 15°, 10°) / Cu (5°, 40°, 0°) bicrystal, crack along the interface | `Fe_27_15_10_Cu_5_40_0_mode_I.inp` + `_bc.inp` |
| `3D_mode_I/Fe_27_0_0/` | Fe single crystal (27°, 0°, 0°), 3D generalised plane strain | `Fe_27_0_0_mode_I_gpe.inp` + `_bc.inp` + `_equations.inp` |
| `3D_mode_I/Fe_27_15_10/` | Fe single crystal (27°, 15°, 10°), 3D generalised plane strain | `Fe_27_15_10_mode_I_gpe.inp` + `_bc.inp` + `_equations.inp` |
| `3D_mode_I/Fe_27_15_10_Cu_5_40_0/` | Fe/Cu bicrystal, 3D generalised plane strain | `Fe_27_15_10_Cu_5_40_0_mode_I_gpe.inp` + `_bc.inp` + `_equations.inp` |
| `3D_mixed_mode/Fe_27_15_10_Cu_5_40_0/` | Fe/Cu bicrystal, 3D generalised plane strain, K_II = 0.5 K_I, K_III = 0.3 K_I | `Fe_27_15_10_Cu_5_40_0_mixed_mode_gpe.inp` + `_bc.inp` + `_equations.inp` |
| `3D_crystal_plasticity/C6_run*/` | C6, crystal plasticity: In718 bicrystal, Bunge (105°, 60°, 30°) / (0°, 0°, 0°), crack along the grain boundary, crystal plasticity (UMAT), R = 1, 4, 10 mm, KH and LKH, 3D generalised plane strain, K_I up to 40 MPa√m, a = 20 mm | complete model `In718_bicrystal_ssy_dup.inp` + `_equations.inp`, boundary condition `In718_bicrystal_ssy_bc.inp`, `material_card.inp`, `RUN_README.txt`, evaluated results in `results/` |
| `3D_slice/` | mesh of the 3D slice (placeholder section/material), input of `make_gpe_model.py` | `slice_3D.inp` |

Case C6 (`3D_crystal_plasticity/C6_run*/`) is a bicrystal of In718 (elastic constants C11/C12/C44 = 193.5/127.8/89.4 GPa) with crystal plasticity in both
grains, modelled like the other 3D cases (one element layer in generalised plane strain, same in-plane mesh), loaded by the interface field
with a = 20 mm up to K_I = 40 MPa√m. The crack tip is modelled as recommended by Abaqus for non-linear materials: collapsed elements with duplicate
nodes and midside nodes at the quarter points, so that the tip can blunt. The constitutive model is the
single crystal plasticity model of Schilli et al. (2022), without (KH) and with latent kinematic hardening (LKH), implemented as an Abaqus user
material (UMAT; R. Zandomeni, T. Seifert, A. Hartmaier, Mech. Mater. 222 (2026) 105821, doi:10.1016/j.mechmat.2026.105821). **The UMAT source files
are not part of this repository.** The input files contain the material cards (72 constants per grain). Six runs
(R = 1, 4 and 10 mm, KH and LKH) are stored with their evaluated results, so that the figures and tables of Section 5 can be reproduced without
Abaqus and without the UMAT (see "Small-scale yielding (C6)" below). The result files of the Abaqus jobs (`.odb`, `.dat`, ...) of all cases are
not part of the repository.

Materials: Fe C11/C12/C44 = 243.36/145.01/116.04 GPa, Cu 180.85/120.02/83.56 GPa. In the 2D models
the crystal stiffness is given already rotated into the global system (`*Elastic, type=ANISOTROPIC`,
computed with `rotate_stiffness_euler`) together with the identity orientation `GLOBAL`, because
Abaqus requires a local orientation for anisotropic elasticity; in the 3D models the unrotated crystal
stiffness is combined with an `*Orientation` from the Bunge angles (`GPE_ORI1`/`GPE_ORI2`; bicrystal:
grain 1 `GPE_GRAIN1`, y > 0, grain 2 `GPE_GRAIN2`, y < 0).

Sets: `BULK` (all elements), `GRAIN1`/`GRAIN2` (2D bicrystal), `BOUNDARY` resp. `BOUNDARY_GRAIN1`/
`BOUNDARY_GRAIN2` (outer boundary nodes, used by the app), `CRACK_FRONT`/`CRACK_TIP` (contour integral;
3D: `_BOTTOM`, `_MID`, `_TOP`).

## Reproducing the boundary conditions with the app

Load the model in `notebooks/abaqus_crack_tip_bc_app.ipynb` (2D: the main input file of the case folder,
3D: `3D_slice/slice_3D.inp`) and use

| Case | Model type / material(s) / Euler angles | Node set(s) |
|---|---|---|
| `isotropic_In718` | isotropic, `In718 (isotropic)` | `BOUNDARY` |
| `Fe_27_0_0` | anisotropic, `Fe (cubic, example)`, (27, 0, 0) | `BOUNDARY` |
| `Fe_27_15_10` | anisotropic, `Fe (cubic, example)`, (27, 15, 10) | `BOUNDARY` |
| `Fe_27_15_10_Cu_5_40_0` | interface, `Fe (cubic, example)` (27, 15, 10) / `Cu (cubic, example)` (5, 40, 0) | `BOUNDARY_GRAIN1`, `BOUNDARY_GRAIN2` |

with K_I = 869.626, K_II = K_III = 0 (mixed mode: K_II = 434.813, K_III = 260.888), a = 20, X0 = Y0 = 0,
alpha = 0 and instance prefix `Part-1-1` (detected automatically); DOFs u1, u2 for the 2D models and
u1, u2 **and u3** for the 3D slices. Save the result under the name of the `_bc.inp` file of the case
folder. The result is identical to the `*_bc.inp` files supplied here.

In `slice_3D.inp` the boundary node sets contain all nodes of the outer boundary on all three node
levels (z = 0, t/2, t; 323 nodes, 969 values), as prescribed by `make_gpe_model.py`. The `*EQUATION`
constraints of the generalised plane-strain state are written only by `make_gpe_model.py`; they couple
the interior node columns and leave the boundary columns to the boundary condition, so a boundary
condition that omits the mid-plane nodes would leave them free.

## Generating the 3D models (generalised plane strain)

The 3D models were generated from `3D_slice/slice_3D.inp` with `scripts/make_gpe_model.py`, e.g. for the
bicrystal (in the case folder `3D_mode_I/Fe_27_15_10_Cu_5_40_0/`):

```bash
python ../../../../scripts/make_gpe_model.py ../../3D_slice/slice_3D.inp --material "Fe (cubic, example)" --euler 27 15 10 \
       --material2 "Cu (cubic, example)" --euler2 5 40 0 --boundary-set BOUNDARY --out Fe_27_15_10_Cu_5_40_0_mode_I_gpe
```

| Case folder | options of `make_gpe_model.py` | `--out` |
|---|---|---|
| `3D_mode_I/Fe_27_0_0/` | `--material "Fe (cubic, example)" --euler 27 0 0` | `Fe_27_0_0_mode_I_gpe` |
| `3D_mode_I/Fe_27_15_10/` | `--material "Fe (cubic, example)" --euler 27 15 10` | `Fe_27_15_10_mode_I_gpe` |
| `3D_mode_I/Fe_27_15_10_Cu_5_40_0/` | as above plus `--material2 "Cu (cubic, example)" --euler2 5 40 0` | `Fe_27_15_10_Cu_5_40_0_mode_I_gpe` |
| `3D_mixed_mode/Fe_27_15_10_Cu_5_40_0/` | as above plus `--KI 869.626 --KII 434.813 --KIII 260.888` | `Fe_27_15_10_Cu_5_40_0_mixed_mode_gpe` |

(always with `--boundary-set BOUNDARY`). The script writes `<out>.inp`, `<out>_bc.inp` (u1, u2, u3 of the
analytical field on all nodes of the boundary node columns) and `<out>_equations.inp` (62 361
`*EQUATION` constraints u_i(z) = u_i(z_min), identical for all cases because they depend only on the
mesh) into the folder of the input file unless `--out` contains a path; run it from the case folder with
`--out` as above, or move the three files there. The boundary conditions supplied here were exported with
the app (identical values). J is evaluated at the mid-plane node (`CRACK_TIP_MID`).

## Running and evaluating

In the case folder, e.g. `3D_mixed_mode/Fe_27_15_10_Cu_5_40_0/`:

```bash
abaqus job=Fe_27_15_10_Cu_5_40_0_mixed_mode_gpe interactive
abaqus python ../../../../scripts/evaluate_kfield_abaqus.py --mode bicrystal --KII 13.75 --KIII 8.25
python ../../../../scripts/compare_stress_circle.py bc_3D_circle.csv --material "Fe (cubic, example)" --euler 27 15 10 \
       --material2 "Cu (cubic, example)" --euler2 5 40 0 --KI 869.626 --KII 434.813 --KIII 260.888
```

| Case | `evaluate_kfield_abaqus.py` (K in MPa√m) | `compare_stress_circle.py` (K in MPa√mm) |
|---|---|---|
| `isotropic_In718` | `--mode isotropic` | `--isotropic "In718 (isotropic)"` |
| `Fe_27_0_0` (2D, 3D) | `--mode single_crystal --euler1 27 0 0` | `--material "Fe (cubic, example)" --euler 27 0 0` |
| `Fe_27_15_10` (2D, 3D) | `--mode single_crystal --euler1 27 15 10` | `--material "Fe (cubic, example)" --euler 27 15 10` |
| `Fe_27_15_10_Cu_5_40_0`, mode I (2D, 3D) | `--mode bicrystal` | `--material "Fe (cubic, example)" --euler 27 15 10 --material2 "Cu (cubic, example)" --euler2 5 40 0` |
| `Fe_27_15_10_Cu_5_40_0`, mixed mode (3D) | `--mode bicrystal --KII 13.75 --KIII 8.25` | as above plus `--KI 869.626 --KII 434.813 --KIII 260.888` |

`evaluate_kfield_abaqus.py` names its output `iso`, `sc` or `bc` (plus `_3D`), e.g. `bc_3D_circle.csv`;
use `--prefix` for other names.

### Small-scale yielding (C6)

The jobs run on a cluster (UMAT, `cpus=8`; the `.odb` files are not part of the repository). The step is one static step of 1 s with 15
stored frames, frame N corresponds to K_I = 40 N / 15 MPa√m (K_I = 2.67, 5.33, 8.00, ... 40 MPa√m); the manuscript uses the frames 1 to 8
(K_I up to 21.3 MPa√m). Each run folder contains `RUN_README.txt` with the run and evaluation commands. After the job, in a folder `results/` that contains the `.odb` and the scripts of `scripts/`
(the outputs are written into this folder, as in the repository):

```bash
abaqus python evaluate_kfield_abaqus.py --mode bicrystal --odb In718_bicrystal_ssy_dup.odb --euler1 105 60 30 --euler2 0 0 0 \n    --elastic1 193500 127800 89400 --elastic2 193500 127800 89400 --step-time 1 --KI 40 --frame N --prefix ssy_fN   # N = 1..15
abaqus python c6_extract_abaqus.py --odb In718_bicrystal_ssy_dup.odb --KI 40      # element, nodal, crack-face and ring data, written to export/
python c6_analyze.py --dir results --label "R=4 mm, KH" --crit max --thr 1e-4 --analysis-dir analysis_max1e-4
```

(`evaluate_kfield_abaqus.py` and `c6_extract_abaqus.py` run with Abaqus Python, `c6_analyze.py` with Python 3.) The plastic zone is the set of
elements in which the accumulated slip (sum over the 12 slip systems, state variables SDV226 to SDV237 of the UMAT) exceeds 1e-4 in at least
one integration point; r_p is its length along the grain boundary and r_max its largest distance from the tip. K_I from J is
2 √(J / E22) with E22 = 2.3522e-5 1/MPa (beta = 0.0183, gamma = 0.0058 for this pair), J of the prescribed field J40 = 9.4090 N/mm at 40 MPa√m;
the figures use the outermost contour (15). The stresses and displacements of the K-field comparison are taken at the ring r = 1 mm.

| Folder | R [mm] | Hardening model | Material card |
|---|---|---|---|
| `C6_run1_R1mm_KH` | 1 | kinematic hardening (KH) | Schilli et al. (2022), Table 3, KH |
| `C6_run2_R1mm_LKH` | 1 | with latent kinematic hardening (LKH) | Table 3, LKH |
| `C6_run3_R4mm_KH` | 4 | KH | Table 3, KH |
| `C6_run4_R4mm_LKH` | 4 | LKH | Table 3, LKH |
| `C6_run5_R10mm_LKH` | 10 | LKH | Table 3, LKH |
| `C6_run6_R10mm_KH` | 10 | KH | Table 3, KH |

All runs have the same inner mesh (r <= 1 mm); the models of larger radius have an additional outer ring of elements, graded towards the boundary
(`scripts/make_larger_model.py`). Only the base model (R = 1 mm) is stored as a complete input file
(`C6_run1_R1mm_KH/In718_bicrystal_ssy_dup.inp` and `_equations.inp`; identical for run 2). The other run folders are written from it by
`scripts/make_c6_runs.py` (checked: the input files, equations, boundary conditions and material cards of runs 2 to 6 are reproduced
identically). The base model has the material card of the KH model (tau_0 = 168.89 MPa, c/d = 252.23 MPa, d = 152.90); for the LKH runs the card is
replaced by the one of Schilli et al. (2022), Table 3 (`--tau0 254.96 --cd 143.05 --d 32.58`). Run 4 (R = 4 mm, LKH) and run 3 (R = 4 mm, KH):

```bash
python scripts/make_c6_runs.py --base examples/fe_case_study/3D_crystal_plasticity/C6_run1_R1mm_KH --out C6_run4_R4mm_LKH --K 40 --a 20 --R 4 --tau0 254.96 --cd 143.05 --d 32.58 --no-umat
python scripts/make_c6_runs.py --base examples/fe_case_study/3D_crystal_plasticity/C6_run1_R1mm_KH --out C6_run3_R4mm_KH --K 40 --a 20 --R 4 --no-umat
```

(`--R 1` and `--R 10` for the other radii.) The hardening model (KH or LKH) is selected in the UMAT (flag `iAFHKin` of its initialisation file), which is
why the UMAT of each model variant has to be put into the run folder.

The boundary condition (`In718_bicrystal_ssy_bc.inp`) is the K-field of the interface model (K_I = 1264.91 MPa√mm, a = 20 mm, grain 1: Bunge
(105°, 60°, 30°), grain 2: (0°, 0°, 0°)); it is checked by `tests/test_fe_workflow.py`. **The UMAT source files are not part of the repository**
(the folders of the runs are given without them). Without the UMAT the jobs cannot be repeated; the evaluated results are supplied.

## Expected results

Mode I (manuscript, Table 2; K_I = 27.5 MPa√m = 869.626 MPa√mm, exactly 27.5·√1000 in the deviations; K_I,FE = 2 √(J / E22) from J on
contour 6, 0.021 mm from the tip, mid-plane node in 3D; deviation in 10⁻³ %, i.e. 1000 corresponds to 1 %):

| Case | Model | E22 [1/MPa] | J_presc [N/mm] | J [N/mm] | K_I,FE [MPa√mm] | Deviation [10⁻³ %] |
|---|---|---|---|---|---|---|
| isotropic In718 | 2D plane strain | 2.2469·10⁻⁵ | 4.2481 | 4.2481 | 869.63 | -0.06 |
| Fe (27, 0, 0) | 2D plane strain | 1.8731·10⁻⁵ | 3.5414 | 3.5414 | 869.63 | -0.03 |
| Fe (27, 0, 0) | 3D generalised plane strain | 1.8731·10⁻⁵ | 3.5414 | 3.5414 | 869.63 | +0.03 |
| Fe (27, 15, 10) | 2D plane strain | 1.8150·10⁻⁵ | 3.4314 | 3.4636 | 873.70 | +469 |
| Fe (27, 15, 10) | 3D generalised plane strain | 1.8150·10⁻⁵ | 3.4314 | 3.4314 | 869.63 | +0.03 |
| Fe/Cu | 2D plane strain | 1.9992·10⁻⁵ | 3.7798 | 3.8135 | 873.50 | +445 |
| Fe/Cu | 3D generalised plane strain | 1.9992·10⁻⁵ | 3.7798 | 3.7798 | 869.63 | +0.01 |

The 2D plane-strain models of the coupled orientations (Fe (27, 15, 10), Fe/Cu) suppress the
anti-plane displacement u3 of the analytical field and are therefore slightly too stiff (+0.47 % and +0.45 % in K_I); the 3D
slices in generalised plane strain remove this deviation.

Mixed mode (Fe/Cu, 3D GPE, K_I = 27.5, K_II = 13.75, K_III = 8.25 MPa√m): J is compared with
J = k^T E k / 4 = 5.4771 N/mm (full compliance matrix E; the mode-I part alone is 3.7798 N/mm). Result:
J = 5.4771 N/mm (contour 6, mid-plane node), i.e. +0.13·10⁻³ % in J and +0.06·10⁻³ % in the magnitude
of k, as for pure mode I.

Stresses on the circle r = 0.0974 mm (`compare_stress_circle.py`, a = 20 mm; manuscript, Table 3): largest deviation
max |sigma_FE - sigma_analytical| in % of K_I / sqrt(2 pi r)

| Case | Model | s11 | s22 | s33 | s23 | s13 | s12 |
|---|---|---|---|---|---|---|---|
| isotropic In718 | 2D, mode I | 0.21 | 0.19 | 0.08 | – | – | 0.11 |
| Fe (27, 0, 0) | 2D, mode I | 0.27 | 0.32 | 0.13 | – | – | 0.20 |
| Fe (27, 0, 0) | 3D GPE, mode I | 0.27 | 0.32 | 0.13 | – | – | 0.20 |
| Fe (27, 15, 10) | 2D, mode I | 1.82 | 1.37 | 3.35 | 6.78* | 9.73* | 1.01 |
| Fe (27, 15, 10) | 3D GPE, mode I | 0.24 | 0.29 | 0.13 | 0.04 | 0.03 | 0.20 |
| Fe/Cu | 2D, mode I | 1.22 | 1.05 | 3.20 | 6.94* | 7.91* | 0.89 |
| Fe/Cu | 3D GPE, mode I | 0.26 | 0.30 | 0.17 | 0.03 | 0.03 | 0.20 |
| Fe/Cu | 3D GPE, mixed mode | 0.28 | 0.43 | 0.18 | 0.04 | 0.05 | 0.24 |

–: component vanishes identically (decoupled orientation); *: missing in the 2D model (amplitude of the
analytical component). Under mixed-mode loading the anti-plane stresses reach 54 % of K_I / sqrt(2 pi r).

Small-scale yielding (case C6, manuscript, Section 5): the evaluated results of the six runs are in `3D_crystal_plasticity/C6_run*/results/`
(`export/`: element, nodal, crack-face and ring data per frame; `ssy_f<N>_J.csv`, `_s22.csv`, `_circle.csv`: J, opening stress and
stresses on the circle of frame N; `analysis_max1e-4/summary.csv`: plastic zone, K deviation and CTOD per frame). The tables and figures of
Section 5 are generated from them with the scripts in `scripts/c6_figures/`.