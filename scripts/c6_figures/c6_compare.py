"""c6_compare.py - compare the six C6 runs (R = 1, 4, 10 mm; KH and LKH) from the analysis_max1e-4/summary.csv files."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = sys.argv[1]            # folder with the C6_run* directories
OUT = sys.argv[2]          # output folder
KMAX = float(sys.argv[3]) if len(sys.argv) > 3 else 21.4
SUB = sys.argv[4] if len(sys.argv) > 4 else "analysis_max1e-4"
TAG = "" if SUB == "analysis" else "_" + SUB.replace("analysis_", "")   # common load range [MPa sqrt(m)]
runs = [("R = 1 mm", "C6_run1_R1mm_KH", "C6_run2_R1mm_LKH"),
        ("R = 4 mm", "C6_run3_R4mm_KH", "C6_run4_R4mm_LKH"),
        ("R = 10 mm", "C6_run6_R10mm_KH", "C6_run5_R10mm_LKH")]
col = {"R = 1 mm": "#1b6ca8", "R = 4 mm": "#d98200", "R = 10 mm": "#3a9d5d"}
ls = {"KH": "-", "LKH": "--"}
D = {}
for lab, kh, lkh in runs:
    for tag, d in (("KH", kh), ("LKH", lkh)):
        D[(lab, tag)] = np.genfromtxt(os.path.join(B, d, "results", SUB, "summary.csv"), delimiter=",", names=True)

panels = [("r_max", "largest radius with slip > 1e-3 [mm]", 1.0), ("r_max_over_R", "r_max / R [-]", 1.0),
          ("r_p_GB", "r_p along the grain boundary [mm]", 1.0), ("area", "area of the zone [mm$^2$]", 1.0),
          ("K_dev_percent", "K deviation from J [%]", 1.0), ("CTOD_fix", "CTOD at 0.01 mm [mm]", 1.0)]
fig, ax = plt.subplots(2, 3, figsize=(13, 7.5))
for a, (key, yl, s) in zip(ax.ravel(), panels):
    for (lab, tag), d in D.items():
        m = d["K_I"] <= KMAX
        a.plot(d["K_I"][m], d[key][m] * s, ls[tag], color=col[lab], marker="o", ms=3, label="%s %s" % (lab, tag))
    a.set_xlabel(r"$K_I$ [MPa$\sqrt{m}$]"); a.set_ylabel(yl); a.grid(alpha=.3)
ax[0, 0].legend(fontsize=8)
fig.suptitle("C6: all six runs, K <= %.1f MPa sqrt(m)" % KMAX)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "c6_compare%s.png" % TAG), dpi=130)

# table at the frames of the common range
rows = []
for kk in (16.0, 18.67, 21.33):
    for (lab, tag), d in D.items():
        i = int(np.argmin(abs(d["K_I"] - kk)))
        rows.append((kk, lab, tag, d["r_max"][i], d["r_max_over_R"][i], d["r_p_GB"][i], d["area"][i],
                     d["slip_max_ring2"][i], d["K_dev_percent"][i], d["CTOD_fix"][i]))
hdr = "K_I,R,variant,r_max_mm,r_max_over_R,r_p_GB_mm,area_mm2,slip_max,K_dev_percent,CTOD_mm"
with open(os.path.join(OUT, "c6_compare_table%s.csv" % TAG), "w") as f:
    f.write(hdr + "\n")
    for r in rows:
        f.write("%.2f,%s,%s,%.4f,%.4f,%.4f,%.5f,%.4f,%.2f,%.5f\n" % r)
print(hdr)
for r in rows:
    print("%.2f %-9s %-4s r_max=%.3f (%.3f R) r_p=%.4f area=%.4f slip=%.3f dK=%.2f%% CTOD=%.5f" % r[:3] + r[3:] if False else
          "%.2f %-9s %-4s r_max=%.3f (%.3f R) r_p=%.4f area=%.4f slip=%.3f dK=%.2f%% CTOD=%.5f" % r)
