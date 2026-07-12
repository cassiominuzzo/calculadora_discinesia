"""
Calculadora de risco de discinesia problematica induzida por levodopa.
Coorte PPMI. Cox; formula fechada (numpy). Risco absoluto ajustado por risco competitivo (Aalen-Johansen).
Dois modos: BASAL (Total-6, inicio da levodopa) e DINAMICO (Total-6 + responsividade, no seguimento).
FERRAMENTA DE PESQUISA - nao substitui julgamento clinico.
"""
import json, os
import numpy as np
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(os.path.join(HERE, "calculator_artifacts.json"), encoding="utf-8"))
HOR = A["horizontes"]
BETA, MEANS = A["coeficientes"], A["means"]
S0_HOR = {int(k): v for k, v in A["baseline_survival_hor"].items()}
GRID_T = A["baseline_survival_grid"]["t"]
GRID_S0 = A["baseline_survival_grid"]["S0"]
VARS = [i["var"] for i in A["inputs"]]
DYN = A.get("modelo_dinamico")
CRF = {int(k): v for k, v in A.get("competing_risk_factor", {}).items()}
CR_T = sorted(CRF) if CRF else [3, 5, 7, 10]
CR_V = [CRF.get(t, 1.0) for t in CR_T] if CRF else [1, 1, 1, 1]

def cr_factor(t):
    return float(np.interp(t, CR_T, CR_V, left=1.0, right=CR_V[-1]))

def lp_generic(x, beta, means, keys):
    return sum(beta[k] * (float(x[k]) - means[k]) for k in keys)

def risk_at(lp, s0, t):
    return float((1 - s0 ** np.exp(lp)) * cr_factor(t))

st.set_page_config(page_title="Calculadora de discinesia (PPMI)", page_icon=":brain:", layout="wide")

st.title("Risco de discinesia problematica induzida por levodopa")
st.caption(
    "Modelo prognostico (coorte PPMI) - desfecho: discinesia problematica "
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

    st.markdown("---")
    st.markdown("**Responsividade a levodopa** *(opcional - so no seguimento)*")
    tem_resp = "Nao"
    resp_val = None
    if DYN:
        tem_resp = st.radio(
            "Voce ja conhece a responsividade a levodopa deste paciente?",
            ["Nao", "Sim"],
            help="Responsividade = melhora do MDS-UPDRS III do estado OFF para ON. So existe depois que o paciente ja usou levodopa.",
        )
        if tem_resp == "Sim":
            ri = DYN["resp_input"]
            resp_val = st.number_input(
                "Responsividade motora (% de melhora do MDS-UPDRS III: OFF -> ON)",
                float(ri["min"]), float(ri["max"]), float(ri["default"]), step=float(ri["step"]),
                help="(UPDRS-III_OFF - UPDRS-III_ON) / UPDRS-III_OFF x 100. Ex.: OFF 40, ON 16 -> 60%.",
            )
            if resp_val > 50:
                st.caption("Respondedor forte (>50%) - associado a MAIOR risco de discinesia.")
            elif resp_val >= 30:
                st.caption("Respondedor moderado.")
            else:
                st.caption("Baixa responsividade - associada a MENOR risco.")

modo_dinamico = bool(DYN) and tem_resp == "Sim" and resp_val is not None

with col_out:
    if modo_dinamico:
        keys = list(DYN["coeficientes"].keys())
        beta, means = DYN["coeficientes"], DYN["means"]
        s0_hor = {int(k): v for k, v in DYN["baseline_survival_hor"].items()}
        grid_t, grid_s0 = DYN["baseline_survival_grid"]["t"], DYN["baseline_survival_grid"]["S0"]
        xf = dict(x); xf["resp"] = resp_val
        lp = lp_generic(xf, beta, means, keys)
        st.subheader("Probabilidade estimada - modo DINAMICO (com responsividade)")
    else:
        s0_hor, grid_t, grid_s0 = S0_HOR, GRID_T, GRID_S0
    