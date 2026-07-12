"""
Calculadora de risco de discinesia problematica induzida por levodopa (modelo Total-6).
Coorte PPMI. Cox; formula fechada (numpy). Risco absoluto ajustado por risco competitivo (Aalen-Johansen) em 7/10a.
FERRAMENTA DE PESQUISA - nao substitui julgamento clinico.
"""
import json, os
import numpy as np
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(os.path.join(HERE, "calculator_artifacts.json"), encoding="utf-8"))
BETA, MEANS, HOR = A["coeficientes"], A["means"], A["horizontes"]
S0_HOR = {int(k): v for k, v in A["baseline_survival_hor"].items()}
GRID_T = A["baseline_survival_grid"]["t"]
GRID_S0 = A["baseline_survival_grid"]["S0"]
VARS = [i["var"] for i in A["inputs"]]
CRF = {int(k): v for k, v in A.get("competing_risk_factor", {}).items()}
CR_T = sorted(CRF) if CRF else [3, 5, 7, 10]
CR_V = [CRF.get(t, 1.0) for t in CR_T] if CRF else [1, 1, 1, 1]

def cr_factor(t):
    return float(np.interp(t, CR_T, CR_V, left=1.0, right=CR_V[-1]))

st.set_page_config(page_title="Calculadora de discinesia (PPMI)", page_icon=":brain:", layout="wide")

def linear_predictor(x):
    return sum(BETA[k] * (float(x[k]) - MEANS[k]) for k in VARS)

def risk_at(lp, s0, t):
    return float((1 - s0 ** np.exp(lp)) * cr_factor(t))

st.title("Risco de discinesia problematica induzida por levodopa")
st.caption(
    "Modelo prognostico **Total-6** (coorte PPMI) - desfecho: discinesia problematica "
    "(MDS-UPDRS 4.1>=2 OU 4.2>=2) - tempo-zero = inicio da levodopa - risco absoluto ajustado por risco competitivo (morte)."
)
st.warning(
    "Ferramenta de PESQUISA. Estima probabilidade de discinesia *problematica*, nao de qualquer discinesia. "
    "Nao substitui o julgamento clinico. Validacao externa pendente (fase 2)."
)

col_in, col_out = st.columns([1, 1.25], gap="large")

with col_in:
    st.subheader("Dados do paciente (1 consulta)")
    x = {}
    for it in A["inputs"]:
        v, lab, tp, hlp = it["var"], it["label"], it["tipo"], it.get("help", "")
        if tp == "slider":
            x[v] = st.slider(lab, float(it["min"]), float(it["max"]), float(it["default"]), step=float(it["step"]), help=hlp)
        elif tp == "number":
            x[v] = st.number_input(lab, float(it["min"]), float(it["max"]), float(it["default"]), step=float(it["step"]), help=hlp)
        elif tp == "select":
            opts = it["options"]
            choice = st.selectbox(lab, list(opts.keys()), index=list(opts.keys()).index(it["default"]), help=hlp)
            x[v] = opts[choice]

with col_out:
    st.subheader("Probabilidade estimada de discinesia problematica")
    lp = linear_predictor(x)
    probs = {h: risk_at(lp, S0_HOR[h], h) for h in HOR}
    cols = st.columns(len(HOR))
    for c, h in zip(cols, HOR):
        c.metric(str(h) + " anos", str(round(probs[h] * 100)) + "%")
    r5 = probs[5]
    if r5 < 0.10:
        cat, color = "BAIXO", "#2E7D32"
    elif r5 < 0.25:
        cat, color = "INTERMEDIARIO", "#C55A11"
    else:
        cat, color = "ALTO", "#C0392B"
    box = ("<div style='padding:8px 14px;border-radius:8px;margin:8px 0;background:" + color +
           "22;border-left:5px solid " + color + "'><b style='color:" + color + "'>Risco " + cat +
           "</b> &nbsp; probabilidade em 5 anos: <b>" + str(round(r5 * 100)) + "%</b></div>")
    st.markdown(box, unsafe_allow_html=True)
    curve = pd.DataFrame({
        "Anos desde a levodopa": GRID_T,
        "Risco acumulado (%)": [ (1 - s0 ** np.exp(lp)) * cr_factor(t) * 100 for t, s0 in zip(GRID_T, GRID_S0) ],
    }).set_index("Anos desde a levodopa")
    st.line_chart(curve, height=260)

st.divider()
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Variaveis do modelo (6):** MDS-UPDRS total (I+II+III), idade de inicio, sexo, IMC, "
                "razao TD/PIGD e congelamento da marcha (item 2.13) - todas de uma consulta, sem exame complementar.")
with c2:
    perf = A["desempenho"]
    st.markdown("**Desempenho (PPMI, n=" + str(A["n"]) + ", " + str(A["eventos"]) + " eventos):** C-index corrigido **" +
                str(perf["C_corrigido"]) + "** - leave-one-site-out **" + str(perf["C_LOSO"]) + "** - calibracao " + perf["calibracao"] + ".")
st.caption("Dados: PPMI (cut 29-Abr-2026). Relato TRIPOD. Risco de 7/10 anos ajustado por risco competitivo (morte) via Aalen-Johansen. "
           "Genetica, neuroimagem e biomarcadores testados e nao acrescentaram discriminacao.")
