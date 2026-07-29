# 16_build_largepd_analytic.py
# Produces: LARGE-PD analytic table, current-status design
# Original file in the project archive: 62_build_largepd.py

"""62 — Tabela analítica do LARGE-PD (transversal). Preditores (5, sem IMC), duração de levodopa e desfecho.
Desenho current-status: desfecho = discinesia problemática PRESENTE na avaliação; tempo = duração da levodopa.
Sem métricas de performance."""
from config import PROJECT_ROOT
import pandas as pd, numpy as np, glob, warnings
warnings.filterwarnings('ignore')
R=PROJECT_ROOT+'/Projeto_LID_v2/08_Validacao_Externa'
import os as _os
F=(R+'/dados_brutos/largepd_raw.csv') if _os.path.exists(R+'/dados_brutos/largepd_raw.csv') else (__import__('glob').glob(os.path.join(PROJECT_ROOT,'Banco*Dados*.csv'))+[''])[0]
d=pd.read_csv(F,engine='python')
def num04(x): x=pd.to_numeric(x,errors='coerce'); return x.where((x>=0)&(x<=4))
pdd=d[d['diagnostico']==0].copy()
print("PD:",len(pdd)," | sex valores:",pdd['sex'].value_counts(dropna=False).to_dict())
# CHECAGEM do mapeamento Part 3: soma(updrs1..updrs33) deve bater com updrs_escore_parte3
items3=[f'updrs{i}' for i in range(1,34)]
s33=pdd[items3].apply(pd.to_numeric,errors='coerce').sum(axis=1,min_count=30)
p3tot=pd.to_numeric(pdd['updrs_escore_parte3'],errors='coerce')
dif=(s33-p3tot).abs()
print("checagem Part3: soma(itens) vs total -> diferença mediana=%.2f, %% exatos(<=1)=%.0f%%"%(dif.median(),100*(dif<=1).mean()))
# preditores
p1=pd.to_numeric(pdd['escore_total_updrs_parte_1'],errors='coerce'); p2=pd.to_numeric(pdd['escore_total_updrs_parte_2'],errors='coerce')
pdd['updrs_totscore']=p1+p2+p3tot
pdd['ageonset']=pd.to_numeric(pdd['idadeaodiag'],errors='coerce')
pdd['SEX']=pdd['sex'].map(lambda v:1 if str(v).strip() in ['1','1.0','M','m'] else (0 if str(v).strip() in ['2','2.0','0','0.0','F','f'] else np.nan))
pdd['NP2FREZ']=num04(pdd['updrs2_13'])
# TD/PIGD (Stebbins): TD=2.10 + Part3 tremor(updrs24..updrs33); PIGD=3.10/3.11/3.12(updrs19,20,21)+2.12+2.13
TD=['updrs2_10']+[f'updrs{i}' for i in range(24,34)]
PIGD=['updrs19','updrs20','updrs21','updrs2_12','updrs2_13']
for c in TD+PIGD: pdd[c+'_n']=num04(pdd[c])
tdv=pdd[[c+'_n' for c in TD]].mean(axis=1); pgv=pdd[[c+'_n' for c in PIGD]].mean(axis=1)
pdd['td_pigd_ratio']=np.where(pgv>0,tdv/pgv,np.nan)
# duração levodopa + desfecho
aval=pd.to_datetime(pdd['data_avalclinica'],errors='coerce'); lev=pd.to_datetime(pdd['lev_emuso_desde'],errors='coerce')
pdd['dur_levodopa']=(aval-lev).dt.days/365.25
u41=pd.to_numeric(pdd['updrs4_1'],errors='coerce'); u42=pd.to_numeric(pdd['updrs4_2'],errors='coerce')
pdd['event']=((u41>=2)|(u42>=2)).astype(int); pdd['tem_desfecho']=u41.notna()|u42.notna()
V5=['updrs_totscore','ageonset','SEX','td_pigd_ratio','NP2FREZ']
pdd['dur_ok']=pdd['dur_levodopa'].notna()&(pdd['dur_levodopa']>0)
an=pdd[pdd[V5].notna().all(axis=1)&pdd['dur_ok']&pdd['tem_desfecho']].copy()
print("\n===== FUNIL LARGE-PD =====")
print(" PD:",len(pdd)," | com 5 preditores completos:",int(pdd[V5].notna().all(axis=1).sum()))
print(" + duração de levodopa válida:",int((pdd[V5].notna().all(axis=1)&pdd['dur_ok']).sum()))
print(" + desfecho medido = ANALISÁVEL:",len(an))
print(" eventos (discinesia problemática):",int(an['event'].sum()),"(%.0f%%)"%(100*an['event'].mean()))
print(" duração levodopa (anos): mediana=%.1f  p25=%.1f  p75=%.1f"%(an['dur_levodopa'].median(),an['dur_levodopa'].quantile(.25),an['dur_levodopa'].quantile(.75)))
print(" preditores (resumo):")
print(an[V5+['dur_levodopa']].describe().round(2).to_string())
out=an[['codelarge']+V5+['dur_levodopa','event']].rename(columns={'dur_levodopa':'time_years'})
out.to_csv(R+'/dados_processados/largepd_analitico.csv',index=False)
print("\n salvo largepd_analitico.csv (n=%d, eventos=%d)"%(len(out),int(out['event'].sum())))
