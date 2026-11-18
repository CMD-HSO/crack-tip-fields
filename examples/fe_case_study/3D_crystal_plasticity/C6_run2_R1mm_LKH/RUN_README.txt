C6, plastic zone study, run folder written by make_c6_runs.py

model radius R = 1 mm (inner mesh r <= 1 mm identical for all runs), reference length a = 20 mm
K_I at the end of the step: 40 MPa*sqrt(m) = 1264.911 MPa*sqrt(mm); one step of 1 s, frame N: K_I = 40*N/15
frames 1..15: K_I = 2.67, 5.33, 8.00, 10.67, 13.33, 16.00, 18.67, 21.33, 24.00, 26.67, 29.33, 32.00, 34.67, 37.33, 40.00 MPa*sqrt(m)
material: Schilli et al. (2022), Table 3; hardening model: see UMATHT_global_ini.for (iAFHKin)
NOTE: the UMAT (user material subroutine, *.for) is not part of this repository. It is described in Zandomeni et al. (2026); see the data availability statement of the manuscript.
output: field output at 15 stages; INNER = elements r <= 1 mm, RING_R1 = nodes at r = 1 mm; 15 contours

run:      SMALauncher job=In718_bicrystal_ssy_dup user=UMATHTH_SXGRADPlast_grDefo_v11sdv cpus=8
evaluate: SMALauncher python evaluate_kfield_abaqus.py --mode bicrystal --odb In718_bicrystal_ssy_dup.odb --euler1 105 60 30 --euler2 0 0 0 --elastic1 193500 127800 89400 --elastic2 193500 127800 89400 --step-time 1 --KI 40 --frame N --prefix ssy_fN   (N = 1..15)
