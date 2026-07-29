# 26_figure4_decision_curve.py
# Produces: Figure 4 (two-panel decision-curve analysis in LARGE-PD, before and after
# recalibration). Panel B uses the intercept-plus-slope recalibration estimated within
# LARGE-PD, the same one used for Figure 3B, and is therefore an in-sample estimate.

import os, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import PROJECT_ROOT

matplotlib.rcParams["svg.fonttype"] = "none"
EV = os.path.join(PROJECT_ROOT, "08_Validacao_Externa")
OUT = os.path.join(PROJECT_ROOT, "04_Resultados", "Figuras")
os.makedirs(OUT, exist_ok=True)

A = json.load(open(os.path.join(EV, "modelo_reduzido_5var.json")))
V5, B, M = A["variaveis"], A["coeficientes"], A["means"]
GT = np.array(A["baseline_survival_grid"]["t"]); GS = np.array(A["baseline_survival_grid"]["S0"])
S0 = lambda t: np.interp(t, GT, GS, left=1.0, right=GS[-1])

L = pd.read_csv(os.path.join(EV, "dados_processados", "largepd_analitico.csv"))
lp = sum(B[k] * (L[k] - M[k]) for k in V5).values
r = np.clip(1 - S0(L["time_years"].values) ** np.exp(lp), 1e-6, 1 - 1e-6)
y = L["event"].values.astype(float)
n = len(y)

x = np.log(-np.log(1 - r))
g = sm.GLM(y, sm.add_constant(x), family=sm.families.Binomial(sm.families.links.cloglog())).fit()
r2 = 1 - np.exp(-np.exp(g.params[0] + g.params[1] * x))
print("recalibration: intercept %+.3f | slope %.2f (95%% CI %.2f to %.2f)"
      % (g.params[0], g.params[1], g.conf_int()[1][0], g.conf_int()[1][1]))
print("O:E before %.2f | after %.2f" % (y.sum() / r.sum(), y.sum() / r2.sum()))
print("predicted risk >0.90 before recalibration: %d of %d (%.0f%%)" % ((r > .9).sum(), n, 100 * (r > .9).mean()))


def nb(risk, ths):
    out = []
    for pt in ths:
        pp = risk >= pt
        TP = ((pp) & (y == 1)).sum(); FP = ((pp) & (y == 0)).sum()
        out.append(TP / n - FP / n * (pt / (1 - pt)))
    return np.array(out)


ths = np.linspace(0.01, 0.50, 100)
nb_all = y.mean() - (1 - y.mean()) * (ths / (1 - ths))

fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6), sharey=True)
for ax, risk, tag, ttl in zip(axes, [r, r2], ["A", "B"], ["Original model", "After recalibration"]):
    ax.plot(ths * 100, nb_all, color="#C0392B", lw=1.5, label="Treat all")
    ax.axhline(0, color="#9A9A9A", ls="--", lw=1.2)
    ax.plot(ths * 100, nb(risk, ths), color="#1F4E79", lw=2.6, label="Total-5 model")
    ax.set_xlim(0, 50); ax.set_ylim(-0.10, 0.25)
    ax.set_xlabel("Risk threshold (%)", fontsize=12)
    ax.text(0.02, 0.965, tag, transform=ax.transAxes, fontsize=15, fontweight="bold", va="top")
    ax.text(0.075, 0.955, ttl, transform=ax.transAxes, fontsize=12.5, va="top")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("Net benefit", fontsize=12)
axes[0].text(0.62, 0.10, "Treat none", transform=axes[0].transAxes, fontsize=10.5,
             color="#8A8A8A", style="italic")
h, lb = axes[0].get_legend_handles_labels()
fig.legend(h[::-1], lb[::-1], loc="upper center", ncol=2, frameon=False, fontsize=12,
           bbox_to_anchor=(0.5, 1.02))
fig.text(0.5, 0.005, "LARGE-PD, n = %d, %d events" % (n, int(y.sum())), ha="center",
         fontsize=10.5, color="#8A8A8A", style="italic")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
for ext in ("png", "svg", "pdf"):
    fig.savefig(os.path.join(OUT, "Figure4_decision_curve_two_panel." + ext), dpi=200, bbox_inches="tight")
plt.close()
print("saved Figure4_decision_curve_two_panel.{png,svg,pdf} to", OUT)
