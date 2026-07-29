# 22_figure1_participant_flow.py
# Produces: Figure 1 (participant flow)
# Counts come from pipeline/01_cohort_and_time_zero.py (PPMI),
# 16_build_largepd_analytic.py (LARGE-PD) and 18_build_amppd_analytic.py (AMP-PD).
# They are listed explicitly here so the figure is auditable against those scripts.

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from config import PROJECT_ROOT

matplotlib.rcParams["svg.fonttype"] = "none"
OUT = os.path.join(PROJECT_ROOT, "04_Resultados", "Figuras")
os.makedirs(OUT, exist_ok=True)

COL = {"dev": "#1F4E79", "pri": "#1B7A5A", "sec": "#B5651D", "excl": "#6B6B6B"}
FILL = {"dev": "#EAF1F8", "pri": "#E8F5F0", "sec": "#FDF1E4", "excl": "#F2F2F2"}

COLUMNS = [
    dict(key="dev", title="Development", sub="PPMI", x=0.16,
         boxes=[("Idiopathic PD who initiated levodopa,\nno prevalent dyskinesia,\nno atypical parkinsonism",
                 "n = 1,447 (303 events)"),
                ("Development analysis sample\n(complete predictor data)", "n = 813 (165 events)")],
         drops=[("Excluded: incomplete\npredictor data", "n = 634")]),
    dict(key="pri", title="External validation: primary", sub="LARGE-PD", x=0.50,
         boxes=[("LARGE-PD, Brazilian participants\nwith Parkinson's disease", "n = 257"),
                ("Complete data on the five predictors", "n = 222"),
                ("Valid levodopa duration", "n = 160"),
                ("Analysis sample (outcome assessed)", "n = 159 (37 events)")],
         drops=[("Excluded: incomplete\npredictor data", "n = 35"),
                ("Excluded: missing\nlevodopa duration", "n = 62"),
                ("Excluded: outcome\nnot assessed", "n = 1")]),
    dict(key="sec", title="External validation: secondary", sub="AMP-PD (non-PPMI)", x=0.84,
         boxes=[("AMP-PD non-PPMI cohorts (PDBP,\nSTEADY-PD3, SURE-PD3, BioFIND),\nidiopathic PD", "n = 1,481"),
                ("Observable levodopa initiation", "n = 261"),
                ("Outcome ascertainable", "n = 228"),
                ("Analysis sample (complete data)", "n = 153 (9 events)")],
         drops=[("Excluded: no observable\nlevodopa initiation", "n = 1,220"),
                ("Excluded: outcome\nnot ascertainable", "n = 33"),
                ("Excluded: incomplete\npredictor data", "n = 75")]),
]

fig, ax = plt.subplots(figsize=(15.5, 10.2))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
BW, BH, DW, DH = 0.235, 0.105, 0.155, 0.070


def box(x, y, w, h, title, value, key, fs=10.0):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.008,rounding_size=0.012",
                                lw=1.4, ec=COL[key], fc=FILL[key], zorder=3))
    ax.text(x, y + h * 0.16, title, ha="center", va="center", fontsize=fs, color="#1A1A1A", zorder=4, linespacing=1.55)
    ax.text(x, y - h * 0.30, value, ha="center", va="center", fontsize=fs + 1.4, fontweight="bold",
            color=COL[key], zorder=4)


for c in COLUMNS:
    x = c["x"]
    ax.text(x, 0.965, c["title"], ha="center", fontsize=13, fontweight="bold", color=COL[c["key"]])
    ax.text(x, 0.928, c["sub"], ha="center", fontsize=11, color="#4A4A4A")
    ax.plot([x - 0.115, x + 0.115], [0.908, 0.908], color=COL[c["key"]], lw=1.6, alpha=0.65)

    n = len(c["boxes"])
    top, bot = 0.845, 0.135
    ys = [top] if n == 1 else [top - i * (top - bot) / (n - 1) for i in range(n)]
    for (t, v), y in zip(c["boxes"], ys):
        box(x, y, BW, BH, t, v, c["key"])
    for i, (t, v) in enumerate(c["drops"]):
        y0, y1 = ys[i], ys[i + 1]
        ax.annotate("", xy=(x, y1 + BH / 2), xytext=(x, y0 - BH / 2),
                    arrowprops=dict(arrowstyle="-|>", color="#7A7A7A", lw=1.3), zorder=2)
        ym = (y0 + y1) / 2
        dx = x + (BW / 2 + DW / 2 + 0.020) * (-1 if n == 2 else 1)
        ax.plot([x, dx - DW / 2 * (1 if n != 2 else -1)], [ym, ym], color="#9A9A9A", lw=1.1, zorder=1)
        box(dx, ym, DW, DH, t, v, "excl", fs=8.6)

ax.text(0.5, 0.035,
        "PPMI participants were removed from AMP-PD to avoid overlap with development. "
        "Time-zero is levodopa initiation; the outcome is problematic dyskinesia "
        "(MDS-UPDRS item 4.1 or 4.2 of 2 or higher).",
        ha="center", fontsize=9.3, color="#5A5A5A", style="italic")

fig.tight_layout()
for ext in ("png", "svg", "pdf"):
    fig.savefig(os.path.join(OUT, "Figure1_participant_flow." + ext), dpi=200, bbox_inches="tight")
plt.close()
print("saved Figure1_participant_flow.{png,svg,pdf} to", OUT)
