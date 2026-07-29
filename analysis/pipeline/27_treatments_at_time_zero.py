# 27_treatments_at_time_zero.py
# Produces: the treatment rows of Table 1 (TRIPOD+AI item 6c)
# Levodopa daily dose, total levodopa-equivalent daily dose, dopamine agonist use and
# amantadine use, all as recorded at time-zero (the date of levodopa initiation).
# A one-month window is used because the PPMI LEDD log records start and stop dates
# with month precision; wider windows begin to capture dose escalation after initiation.

import os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from config import PROJECT_ROOT

PR = os.path.join(PROJECT_ROOT, "02_Dados_Processados")
# The raw PPMI download may sit beside the project folder rather than inside it.
_cands = [os.path.join(PROJECT_ROOT, "01_Dados_Brutos", "Historia_Medica_e_Medicacao"),
          os.path.join(os.path.dirname(PROJECT_ROOT), "01_Dados_Brutos", "Historia_Medica_e_Medicacao")]
RAW = next((p for p in _cands if os.path.isdir(p)), _cands[0])
TAB = os.path.join(PROJECT_ROOT, "04_Resultados", "Tabelas")
os.makedirs(TAB, exist_ok=True)

WINDOW_DAYS = 31

AGONIST = (r"PRAMIPEX|MIRAPEX|SIFROL|ROPINIROL|ROPINEROL|REQUIP|ADARTREL|ROTIGOTIN|NEUPRO"
           r"|APOMORPH|PIRIBEDIL|CABERGOLIN|PERGOLID|BROMOCRIPT")
AMANTADINE = r"AMANTAD|GOCOVRI|OSMOLEX|PK.?MERZ"
LEVODOPA = (r"LEVODOPA|SINEMET|MADOPAR|RYTARY|STALEVO|CREXONT|DUOPA|PROLOPA|NACOM"
            r"|ISICOM|LEVOCOMP|DHIVY|VYALEV|CARBIDOPA")

L = pd.read_csv(os.path.join(RAW, "LEDD_Concomitant_Medication_Log_29Apr2026.csv"), low_memory=False)
T = pd.read_parquet(os.path.join(PR, "tzero.parquet"))[["PATNO", "tzero_levodopa"]]

name = L["LEDTRT"].astype(str).str.upper()
L["is_agonist"] = name.str.contains(AGONIST, regex=True, na=False)
L["is_amantadine"] = name.str.contains(AMANTADINE, regex=True, na=False)
L["is_levodopa"] = name.str.contains(LEVODOPA, regex=True, na=False) & ~L["is_agonist"] & ~L["is_amantadine"]

L["start"] = pd.to_datetime(L["STARTDT"], format="%m/%Y", errors="coerce")
L["stop"] = pd.to_datetime(L["STOPDT"], format="%m/%Y", errors="coerce")
L["LEDD"] = pd.to_numeric(L["LEDD"], errors="coerce")
L["PATNO"] = pd.to_numeric(L["PATNO"], errors="coerce")
L = L.merge(T, on="PATNO", how="inner").dropna(subset=["tzero_levodopa"])

tz = L["tzero_levodopa"]
active = (L["start"].notna()
          & (L["start"] <= tz + pd.Timedelta(days=WINDOW_DAYS))
          & (L["stop"].isna() | (L["stop"] >= tz - pd.Timedelta(days=WINDOW_DAYS))))

# development sample: eligible participants complete on the candidate clinical pool
import pickle
POOL = pickle.load(open(os.path.join(PR, "sel_data.pkl"), "rb"))["POOL"]
M = pd.read_parquet(os.path.join(PR, "candidate_matrix.parquet"))
if "ageonset" not in M.columns and "ageonset_x" in M.columns:
    M = M.rename(columns={"ageonset_x": "ageonset"})
M["PATNO"] = pd.to_numeric(M["PATNO"], errors="coerce")
dev = M[M["exit"] > 0].dropna(subset=POOL)[["PATNO"]].reset_index(drop=True)

g = L[active].groupby("PATNO")
r = dev.copy()
r = r.merge(g.apply(lambda x: x.loc[x["is_levodopa"], "LEDD"].sum()).rename("levodopa_mg"), on="PATNO", how="left")
r = r.merge(g.apply(lambda x: x["LEDD"].sum()).rename("ledd_total_mg"), on="PATNO", how="left")
r = r.merge(g.apply(lambda x: x["is_agonist"].any()).rename("agonist"), on="PATNO", how="left")
r = r.merge(g.apply(lambda x: x["is_amantadine"].any()).rename("amantadine"), on="PATNO", how="left")
r[["agonist", "amantadine"]] = r[["agonist", "amantadine"]].fillna(False)

def med_iqr(s):
    s = s[s.notna() & (s > 0)]
    q = s.quantile([.25, .5, .75])
    return "%.0f [%.0f-%.0f]" % (q[.5], q[.25], q[.75]), int(len(s))

lev, n_lev = med_iqr(r["levodopa_mg"])
tot, n_tot = med_iqr(r["ledd_total_mg"])
out = pd.DataFrame([
    dict(characteristic="Levodopa daily dose at time-zero, mg", value=lev, n_available=n_lev, n_total=len(r)),
    dict(characteristic="Total levodopa-equivalent daily dose at time-zero, mg", value=tot, n_available=n_tot, n_total=len(r)),
    dict(characteristic="Dopamine agonist at time-zero, n (%)",
         value="%d (%.0f%%)" % (r.agonist.sum(), 100 * r.agonist.mean()), n_available=len(r), n_total=len(r)),
    dict(characteristic="Amantadine at time-zero, n (%)",
         value="%d (%.0f%%)" % (r.amantadine.sum(), 100 * r.amantadine.mean()), n_available=len(r), n_total=len(r)),
])
print("PPMI development sample, n = %d" % len(r))
print(out.to_string(index=False))
out.to_csv(os.path.join(TAB, "tab27_treatments_at_time_zero.csv"), index=False)
r.to_csv(os.path.join(TAB, "tab27_treatments_per_patient.csv"), index=False)
print("\nsaved tab27_treatments_at_time_zero.csv")
