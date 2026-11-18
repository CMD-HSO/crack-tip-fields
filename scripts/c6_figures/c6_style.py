"""c6_style.py - common look of the paper figures (serif font as the LaTeX text, same sizes, no titles in the image)."""
import matplotlib
matplotlib.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False, "font.size": 10.4, "axes.labelsize": 10.4, "xtick.labelsize": 9.1, "ytick.labelsize": 9.1, "legend.fontsize": 9.1,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "grid.linewidth": 0.4, "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42,          # embed the fonts in PDF/EPS (journal requirement for vector figures)
})
PAPER_WIDTH = 7.0          # inch; the figures are included with width=\textwidth

# colour = hardening model (same in all figures), lightness = load K_I; marker = model radius R
COL = {"KH": "#3182bd", "LKH": "#e6550d"}
SHADES = {"KH": {8.0: "#6baed6", 16.0: "#3182bd", 21.333: "#08306b"}, "LKH": {8.0: "#fd8d3c", 16.0: "#e6550d", 21.333: "#7f2704"}}


def shade(model, K):
    k = min(SHADES[model], key=lambda v: abs(v - K))
    return SHADES[model][k]


def k_legend(ax, **kw):
    """legend of the loads: two columns, KH (blue shades) and LKH (orange shades)"""
    from matplotlib.lines import Line2D
    h = []
    for model, ls in (("KH", "-"), ("LKH", "--")):
        for K in (8.0, 16.0, 21.333):
            h.append(Line2D([], [], color=shade(model, K), lw=1.4, ls=ls, label="%s, $K_I=%.1f$" % (model, K)))
    opts = dict(ncol=2, frameon=False, columnspacing=1.0, handlelength=1.8)
    opts.update(kw)
    ax.legend(handles=h, **opts)
