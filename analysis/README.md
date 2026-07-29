# Analysis code

Code for *Development and external validation of a single-visit clinical model to
predict problematic levodopa-induced dyskinesia in Parkinson's disease*.

The model itself, the frozen coefficients and the interactive calculator are in the
root of this repository (`app.py`, `calculator_artifacts.json`).

## Data availability

The three cohorts are controlled-access and are **not** redistributed here.

| Cohort | Role | Access |
|---|---|---|
| PPMI | Development | https://www.ppmi-info.org (data use agreement required) |
| AMP-PD | Secondary external validation | https://amp-pd.org (data use agreement required) |
| LARGE-PD | Primary external validation | On reasonable request to the consortium |

Set the data root before running anything:

```bash
export LID_PROJECT_ROOT=/path/to/your/data
pip install -r requirements.txt
```

## `pipeline/` — reproduces the manuscript

Run in order. Each script states at the top which manuscript element it produces.

| Script | Produces |
|---|---|
| `00_data_inventory.py` | Inventory of all raw tables |
| `01_cohort_and_time_zero.py` | Cohort and time-zero; numbers in Figure 1 |
| `02_horizon_support.py` | Follow-up supporting each horizon; at-risk row of Table 2 |
| `03_baseline_predictor_matrix.py` | Baseline predictor matrix, all candidates pre-levodopa |
| `04_table1_and_incidence.py` | Table 1; cumulative incidence |
| `05_final_model_internal_validation.py` | Table 2; optimism correction; leave-one-site-out; numbers in Supplementary Figure S1 |
| `06_dynamic_model_refit.py` | Dynamic mode of the calculator |
| `07_incremental_value_prs.py` | Polygenic risk scores |
| `08_incremental_value_imaging_csf.py` | DaTSCAN and cerebrospinal fluid |
| `09_fdr_variants.py` | False-discovery-rate variants |
| `10_supplementary_table_S1.py` | Supplementary Table S1 |
| `11_supplementary_table_S3.py` | Supplementary Table S3 |
| `12_competing_risk.py` | Aalen-Johansen adjustment, factor c(t) in Table 2 |
| `13_supplementary_table_S4.py` | Supplementary Table S4 |
| `14_head_to_head_with_CI.py` | Numbers in Supplementary Figure S2 |
| `15_freeze_reduced_model_total5.py` | Supplementary Table S2 (frozen Total-5) |
| `16_build_largepd_analytic.py` | LARGE-PD analytic table |
| `17_validate_largepd.py` | Table 3, LARGE-PD row |
| `18_build_amppd_analytic.py` | AMP-PD analytic table |
| `19_validate_amppd.py` | Table 3, AMP-PD row |
| `20_figures_km_calibration_dca.py` | Figure 3 (external calibration, before and after recalibration) |
| `21_full_audit.py` | Independent re-derivation of every key number from the raw tables |
| `22_figure1_participant_flow.py` | Figure 1 |
| `23_supplementary_figure_S1.py` | Supplementary Figure S1 |
| `24_supplementary_figure_S2.py` | Supplementary Figure S2 |
| `25_figure2_km_tertiles.py` | Figure 2, including the numbers-at-risk table |
| `26_figure4_decision_curve.py` | Figure 4, two-panel decision curve |

Every figure and table in the manuscript and in the supplement is produced by a
script in this folder. Figures are written to `04_Resultados/Figuras/` in PNG, SVG
and PDF.

## `exploratory/` — superseded

Model-selection history: the univariate screens, the hierarchical models and the
earlier candidate models that preceded the six-variable specification. Retained so
that the predictor-selection process is inspectable.

**These scripts do not reproduce the manuscript.** They were run on earlier outcome
definitions and earlier predictor sets, and the numbers they print differ from the
published ones by design. Use `pipeline/` for anything reported in the paper.

## Environment

Python 3.10.12. Exact package versions in `requirements.txt`. Random seeds are fixed
for every bootstrap and cross-validation, so results reproduce on re-running.
