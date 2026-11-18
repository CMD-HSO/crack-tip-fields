C6, plastic zone study, run folder written by make_c6_runs.py

model radius R = 10 mm (inner mesh r <= 1 mm identical for all runs), reference length a = 20 mm
K_I at the end of the step: 40 MPa*sqrt(m) = 1264.911 MPa*sqrt(mm); one step of 1 s, frame N: K_I = 40*N/15
frames 1..15: K_I = 2.67, 5.33, 8.00, 10.67, 13.33, 16.00, 18.67, 21.33, 24.00, 26.67, 29.33, 32.00, 34.67, 37.33, 40.00 MPa*sqrt(m)
material: Schilli et al. (2022), Table 3; hardening model: see UMATHT_global_ini.for (iAFHKin)
material card: tau_0 = 168.89 MPa, c/d = 252.23 MPa, d = 152.9, c = 38565.97 MPa
NOTE: the UMAT (user material subroutine, *.for) is not part of this repository. It is described in Zandomeni et al. (2026); see the data availability statement of the manuscript.
Model variant: kinematic hardening WITHOUT latent hardening (Schilli et al. 2022, Eq. 13), Table 3 values 'without latent hardening' (final): tau_0 = 168.89, c/d = 252.23, d = 152.90.
output: field output at 15 stages; INNER = elements r <= 1 mm, RING_R1 = nodes at r = 1 mm; 15 contours

run:      SMALauncher job=In718_bicrystal_ssy_dup user=UMATHTH_SXGRADPlast_grDefo_v11sdv cpus=8
evaluate: SMALauncher python evaluate_kfield_abaqus.py --mode bicrystal --odb In718_bicrystal_ssy_dup.odb --euler1 105 60 30 --euler2 0 0 0 --elastic1 193500 127800 89400 --elastic2 193500 127800 89400 --step-time 1 --KI 40 --frame N --prefix ssy_fN   (N = 1..15)

Modification of this run (R = 10 mm, KH) compared with the base model written by make_c6_runs.py: the first run stopped at K_I = 20 MPa*sqrt(m)
(no convergence at the first element ring of the crack tip). The results of this folder were computed with the following changes in the step definition
of In718_bicrystal_ssy_dup.inp (everything else unchanged):
  *Step, name=Step-1, nlgeom=YES, inc=10000, unsymm=YES       (unsymmetric equation solver)
  0.01, 1., 1e-08, 0.05                                      (minimum increment 1e-08 instead of 1e-06)
  *Controls, parameters=time incrementation
  4, 8, 9, 16, 10, 4, 12, 20, 6, 3
