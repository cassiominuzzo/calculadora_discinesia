"""
Calculadora de risco de discinesia problematica induzida por levodopa.
Coorte PPMI. Cox; formula fechada (numpy). Risco absoluto ajustado por risco competitivo (Aalen-Johansen).
Dois modos: BASAL (Total-6, inicio da levodopa) e DINAMICO (Total-6 + responsividade, no seguimento).
Layout robusto: entradas na barra lateral, resultados na area principal (sem colunas aninhadas).
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

# ------------------ BARRA LATERAL: entradas ------------------
st.sidebar.header("Dados do paciente (1 consulta)")
x = {}
for it in A["inputs"]:
    v, lab, tp, hlp = it["var"], it["label"], it["tipo"], it.get("help", "")
    if tp == "slider":
        x[v] = st.sidebar.slider(lab, float(it["min"]), float(it["max"]), float(it["default"]), step=float(it["step"]), help=hlp)
    elif tp == "number":
        x[v] = st.sidebar.number_input(lab, float(it["min"]), float(it["max"]), float(it["default"]), step=float(it["step"]), help=hlp)
    elif tp == "select":
        opts = it["options"]
        choice = st.sidebar.selectbox(lab, list(opts.keys()), index=list(opts.keys()).index(it["default"]), help=hlp)
        x[v] = opts[choice]

st.sidebar.markdown("---")
st.sidebar.markdown("**Responsividade a levodopa** *(opcional - so no seguimento)*")
tem_resp = "Nao"
resp_val = None
if DYN:
    tem_resp = st.sidebar.radio(
        "Voce ja conhece a responsividade a levodopa deste paciente?",
        ["Nao", "Sim"],
        help="Responsividade = melhora do MDS-UPDRS III do estado OFF para ON. So existe depois que o paciente ja usou levodopa.",
    )
    if tem_resp == "Sim":
        ri = DYN["resp_input"]
        resp_val = st.sidebar.number_input(
            "Responsividade motora (% de melhora do MDS-UPDRS III: OFF -> ON)",
            float(ri["min"]), float(ri["max"]), float(ri["default"]), step=float(ri["step"]),
            help="(UPDRS-III_OFF - UPDRS-III_ON) / UPDRS-III_OFF x 100. Ex.: OFF 40, ON 16 -> 60%.",
        )
        if resp_val > 50:
            st.sidebar.caption("Respondedor forte (>50%) - associado a MAIOR risco de discinesia.")
        elif resp_val >= 30:
            st.sidebar.caption("Respondedor moderado.")
        else:
            st.sidebar.caption("Baixa responsividade - associada a MENOR risco.")

modo_dinamico = bool(DYN) and tem_resp == "Sim" and resp_val is not None

# ------------------ AREA PRINCIPAL ------------------
st.title("Risco de discinesia problematica induzida por levodopa")
st.caption(
    "Modelo prognostico (coorte PPMI) - desfecho: discinesia problematica "
    "(MDS-UPDRS 4.1>=2 OU 4.2>=2) - tempo-zero = inicio da levodopa - risco absoluto ajustado por risco competitivo (morte)."
)
st.warning(
    "Ferramenta de PESQUISA. Estima probabilidade de discinesia problematica, nao de qualquer discinesia. "
    "Nao substitui o julgamento clinico. Validacao externa pendente (fase 2)."
)
st.info("Preencha os dados do paciente na **barra lateral** (a esquerda). Se ela estiver fechada, clique na setinha `>` no canto superior esquerdo.")

try:
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
        lp = lp_generic(x, BETA, MEANS, VARS)
        st.subheader("Probabilidade estimada - modo BASAL (inicio da levodopa)")

    probs = {h: risk_at(lp, s0_hor[h], h) for h in HOR}
    mc = st.columns(len(HOR))
    for c, h in zip(mc, HOR):
        c.metric(str(h) + " anos", str(round(probs[h] * 100)) + "%")

    r5 = probs[5]
    msg = "Probabilidade em 5 anos: **" + str(round(r5 * 100)) + "%**"
    if r5 < 0.10:
        st.success("Risco BAIXO - " + msg)
    elif r5 < 0.25:
        st.warning("Risco INTERMEDIARIO - " + msg)
    else:
        st.error("Risco ALTO - " + msg)

    curve = pd.DataFrame({
        "Anos desde a levodopa": grid_t,
        "Risco acumulado (%)": [(1 - s0 ** np.exp(lp)) * cr_factor(t) * 100 for t, s0 in zip(grid_t, grid_s0)],
    }).set_index("Anos desde a levodopa")
    st.line_chart(curve, height=280)
    st.caption("Passe o mouse sobre a linha para ler o risco ano a ano.")
except Exception as e:
    st.error("Ocorreu um erro ao calcular. Detalhe abaixo (envie este texto ao desenvolvedor):")
    st.exception(e)

st.divider()
st.subheader("Qual calculadora usar? Conceitos")
cc1, cc2 = st.columns(2)
with cc1:
    st.markdown(
        "**Calculadora BASAL (6 variaveis) - modelo principal**\n\n"
        "Use **no momento de iniciar a levodopa** (ou antes de prescrever). Estima o risco de "
        "discinesia problematica com dados de **uma unica consulta**: MDS-UPDRS total (I+II+III), "
        "idade de inicio, sexo, IMC, razao TD/PIGD e congelamento da marcha (item 2.13). "
        "E o modelo validado internamente do estudo (C corrigido 0,70; leave-one-site-out 0,69; calibracao adequada)."
    )
with cc2:
    st.markdown(
        "**Calculadora DINAMICA (6 variaveis + responsividade)**\n\n"
        "Use **no seguimento**, depois que o paciente **ja respondeu a levodopa** e voce tem um par "
        "**OFF/ON** do MDS-UPDRS III. Acrescenta a **responsividade motora** [(OFF-ON)/OFF x100]: "
        "quanto **maior a melhora** com a levodopa, **maior** o risco de discinesia (o classico bom "
        "respondedor que discinesia). Serve para **refinar** o prognostico com a informacao que so o "
        "tratamento revela - nao substitui a basal."
    )

st.markdown("**Quando usar a dinamica (e quando NAO):**")
st.markdown(
    "- **Use** quando o paciente ja esta em levodopa e a resposta motora e conhecida (challenge OFF/ON "
    "ou avaliacao ON vs OFF documentada).\n"
    "- **Nao use** no primeiro dia de levodopa - ainda nao existe responsividade; nesse caso use a **basal**."
)

lim_n = str(DYN["n"]) if DYN else "-"
lim_dc = str(DYN["desempenho"]["dC_apparent"]) if DYN else "-"
st.warning(
    "**Limitacoes da calculadora dinamica (exploratoria):**\n\n"
    "1. **Ganho pequeno de discriminacao** (delta-C aparente ~ +" + lim_dc + " sobre a basal no mesmo grupo; C ~0,70). "
    "E um refinamento, nao um salto de performance.\n"
    "2. **Subgrupo menor** (n=" + lim_n + " vs 813), pois exige MDS-UPDRS III medido em OFF e em ON.\n"
    "3. A responsividade e um **marcador pos-basal** (medido durante o seguimento): interprete como "
    "**atualizacao** do risco, nao como fator causal de tempo-zero.\n"
    "4. Depende da **qualidade/definicao do challenge** OFF/ON (padronizacao do estado OFF, tempo de pico ON).\n"
    "5. **Validacao externa pendente** (fase 2) - vale para as duas calculadoras."
)

st.divider()
perf = A["desempenho"]
st.markdown("**Variaveis basais (6):** MDS-UPDRS total (I+II+III), idade de inicio, sexo, IMC, "
            "razao TD/PIGD e congelamento da marcha (item 2.13) - todas de uma consulta, sem exame complementar.")
st.markdown("**Desempenho basal (PPMI, n=" + str(A["n"]) + ", " + str(A["eventos"]) + " eventos):** C-index corrigido **" +
            str(perf["C_corrigido"]) + "** - leave-one-site-out **" + str(perf["C_LOSO"]) + "** - calibracao " + perf["calibracao"] + ".")
st.caption("Dados: PPMI (cut 29-Abr-2026). Relato TRIPOD. Risco de 7/10 anos ajustado por risco competitivo (morte) via Aalen-Johansen. "
           "Genetica, neuroimagem e biomarcadores testados e nao acrescentaram discriminacao.")
