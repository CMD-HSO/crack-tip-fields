"""c6_kfield.py - C6: FE fields against the prescribed interface K-field (crack_tip_fields), per run and frame.

(1) stress at the ring r = 1 mm (mid-plane nodes; group 'ring1' for R > 1 mm, the outer boundary itself for R = 1 mm)
(2) displacement at the ring r = 1 mm (lower node level, ring_u_frameNN.csv)
Both are given as the root mean square (rms) of the difference FE - K-field over all nodes, divided by the rms of the K-field itself:
    dev = sqrt(mean_nodes |f_FE - f_K|^2) / sqrt(mean_nodes |f_K|^2),   |.| = vector norm of the components
(stress: the six components s11, s22, s33, s23, s13, s12; displacement: u1, u2, u3).
For the model R = 1 mm the ring r = 1 mm is the prescribed boundary (check: displacement deviation = 0).
Usage: python c6_kfield.py <folder with the C6_run* directories> <repository root> <output folder>"""
import os, sys, csv
import numpy as np
B, REPO, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, os.path.join(REPO, "src"))
import crack_tip_fields as ctf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

model = ctf.InterfaceCrackModel("In718 (anisotropic)", (105, 60, 30), "In718 (anisotropic)", (0, 0, 0))
A = 20.0
COMP = ("s11", "s22", "s33", "s23", "s13", "s12")
runs = [("R = 1 mm", 1.0, "C6_run1_R1mm_KH", "C6_run2_R1mm_LKH", "#1b6ca8", "o"),
        ("R = 4 mm", 4.0, "C6_run3_R4mm_KH", "C6_run4_R4mm_LKH", "#d98200", "s"),
        ("R = 10 mm", 10.0, "C6_run6_R10mm_KH", "C6_run5_R10mm_LKH", "#2e8b57", "^")]


def side_name(side):
    return "bot" if side < 0 else "top"          # side 0 (shared node on the interface): grain 1, the displacement is continuous


def ana_stress(x, y, side, K):
    x2 = np.copysign(0.0, side) if abs(y) < 1e-9 and side != 0 else y
    return np.array(model.stress(x, x2, K, 0.0, 0.0, A, side_name(side)), dtype=float)


def ana_disp(x, y, side, K):
    x2 = np.copysign(0.0, side) if abs(y) < 1e-9 and side != 0 else y
    return np.array(model.displacement(x, x2, K, 0.0, 0.0, A, side=side_name(side)), dtype=float)


def rms(a):
    return np.sqrt((a ** 2).sum(axis=1).mean())


def boundary_nodes(A_, N, K, R):
    """list of (theta, FE stress, analytical stress) of the nodes at r = 1 mm"""
    grp = "boundary" if R <= 1.001 else "ring1"
    out = []
    for r_ in csv.DictReader(open(os.path.join(A_, "nodal_stress_frame%02d.csv" % N))):
        if r_["group"] != grp:
            continue
        x, y, s = float(r_["x"]), float(r_["y"]), int(r_["side"])
        out.append((np.degrees(np.arctan2(y, x)), np.array([float(r_[k]) for k in COMP]), ana_stress(x, y, s, K)))
    out.sort(key=lambda t: t[0])
    return out


if __name__ == "__main__":
    rows = []
    for lab, R, kh, lkh, c, mk in runs:
        for tag, d in (("KH", kh), ("LKH", lkh)):
            A_ = os.path.join(B, d, "results", "export")
            fr = np.genfromtxt(os.path.join(A_, "frames.csv"), delimiter=",", names=True)
            for N, Kmp in zip(fr["frame"].astype(int), fr["K_I_MPa_sqrt_m"]):
                K = Kmp * np.sqrt(1000.0)
                nd = boundary_nodes(A_, N, K, R)
                fe, an = np.array([n[1] for n in nd]), np.array([n[2] for n in nd])
                sig = rms(fe - an) / rms(an)
                sig_max = np.sqrt(((fe - an) ** 2).sum(axis=1)).max() / rms(an)       # maximum over the nodes, same reference (rms of the K-field)
                X, Y, S, U = [], [], [], []
                for r_ in csv.DictReader(open(os.path.join(A_, "ring_u_frame%02d.csv" % N))):
                    X.append(float(r_["x"])); Y.append(float(r_["y"])); S.append(int(r_["side"]))
                    U.append([float(r_["u1"]), float(r_["u2"]), float(r_["u3"])])
                Ua = np.array([ana_disp(x, y, s, K) for x, y, s in zip(X, Y, S)])
                dU = np.array(U) - Ua
                rows.append((lab, tag, N, Kmp, sig, rms(dU) / rms(Ua), sig_max, np.sqrt((dU ** 2).sum(axis=1)).max() / rms(Ua)))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "c6_kfield.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["R", "variant", "frame", "K_I", "boundary_stress_dev_rms", "ring_u_dev_rms", "boundary_stress_dev_max", "ring_u_dev_max"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], "%.3f" % r[3]] + ["%.5f" % v for v in r[4:8]])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    for lab, R, kh, lkh, c, mk in runs:
        for tag, st in (("KH", "-"), ("LKH", "--")):
            sel = [r for r in rows if r[0] == lab and r[1] == tag and r[3] <= 21.4]
            k = [r[3] for r in sel]
            kw = dict(color=c, lw=2.4, marker=mk, ms=6, alpha=.85, mfc="none" if tag == "LKH" else c)
            ax[0].plot(k, [100 * r[4] for r in sel], st, label="%s, %s" % (lab, tag), **kw)
            if R > 1.0:
                ax[1].plot(k, [100 * r[5] for r in sel], st, label="%s, %s" % (lab, tag), **kw)
    ax[0].set_ylabel("rms deviation of the stress at r = 1 mm [% of the K-field]"); ax[0].set_title("(a) stress at r = 1 mm", fontsize=10)
    ax[1].set_ylabel("rms deviation of the displacement at r = 1 mm [% of the K-field]"); ax[1].set_title("(b) displacement at r = 1 mm", fontsize=10)
    for a in ax:
        a.set_xlabel(r"prescribed $K_I$ [MPa$\sqrt{\mathrm{m}}$]"); a.grid(alpha=.3)
    ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "c6_kfield.png"), dpi=140)

    # stresses over theta as in the paper (sigma_ij sqrt(2 pi R) / K_I), FE markers and analytical lines
    Kp, tg = 18.67, "KH"
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.4), sharex=True)
    for lab, R, kh, lkh, c, mk in runs[:2]:
        A_ = os.path.join(B, kh, "results", "export")
        fr = np.genfromtxt(os.path.join(A_, "frames.csv"), delimiter=",", names=True)
        i = int(np.argmin(abs(fr["K_I_MPa_sqrt_m"] - Kp))); N = int(fr["frame"][i]); K = fr["K_I_MPa_sqrt_m"][i] * np.sqrt(1000.0)
        nd = boundary_nodes(A_, N, K, R)
        th = np.array([n[0] for n in nd]); fe = np.array([n[1] for n in nd]); an = np.array([n[2] for n in nd])
        norm = K / np.sqrt(2 * np.pi * 1.0)
        for j, a in enumerate(axs.ravel()):
            a.plot(th, an[:, j] / norm, "-", color=c, lw=1.2, alpha=.8)
            a.plot(th, fe[:, j] / norm, mk, color=c, ms=3.5, mfc="none", label="%s, FE" % lab)
    for j, a in enumerate(axs.ravel()):
        a.set_title(r"$\sigma_{%s}\sqrt{2\pi}/K_I$" % COMP[j][1:], fontsize=10); a.grid(alpha=.3)
    for a in axs[1]:
        a.set_xlabel(r"$\theta$ [deg]")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(r"stress at r = 1 mm (mm units), KH, $K_I$ = %.2f MPa$\sqrt{\mathrm{m}}$ (lines: K-field, markers: FE)" % Kp, fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "c6_kfield_sigma_theta.png"), dpi=140)
    for lab in ("R = 1 mm", "R = 4 mm", "R = 10 mm"):
        for tag in ("KH", "LKH"):
            for kk in (8.0, 16.0, 18.67):
                r = [x for x in rows if x[0] == lab and x[1] == tag and abs(x[3] - kk) < 0.05]
                if r:
                    print("%-9s %-4s K=%5.2f  stress rms dev %.2f %%  | ring displacement rms dev %.3f %%" % (lab, tag, r[0][3], 100 * r[0][4], 100 * r[0][5]))
