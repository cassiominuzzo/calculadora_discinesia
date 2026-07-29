# Risk calculator for problematic levodopa-induced dyskinesia (Total-6)

RESEARCH tool. Estimates the risk of problematic dyskinesia
(MDS-UPDRS item 4.1 or 4.2 >= 2) at 3, 5, 7 and 10 years after levodopa
initiation, from six clinical variables collected at a single visit.
Does not replace clinical judgement.

Developed in PPMI (n = 813, 165 events; optimism-corrected C 0.70).
Externally validated in LARGE-PD (n = 159, 37 events): discrimination
preserved (C 0.68, 95% CI 0.60 to 0.77); absolute risk was over-predicted
and required local recalibration, so the output should be read as risk
stratification rather than exact probability outside PPMI.

Live app: https://calculadoradiscinesia-9vdp7aarb9g6euqigpkhhb.streamlit.app/
Pre-registration and model artifacts: https://doi.org/10.17605/OSF.IO/TUVBW

Contents: `app.py` (calculator), `calculator_artifacts.json` (frozen model),
`requirements.txt`.
