# 15_freeze_reduced_model_total5.py
# Produces: Frozen Total-5 model. Supplementary Table S2
# Original file in the project archive: 60_modelo_reduzido_5var.py

"""60 — Congela o modelo REDUZIDO de 5 variáveis (sem IMC), re-derivado no PPMI.
Motivo: o AMP-PD não tem peso/altura, então o IMC é impossível externamente.
Mantém a MESMA definição de desfecho, tempo-zero e variáveis do Total-6, apenas sem BMI.
Saída: modelo_reduzido_5var.json (coeficientes, médias, S0(t)).
"""
from config import PROJECT_ROOT
import pandas as pd, numpy as np, json, glob, warnings
warnings.filterwarnings('ignore')
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
B=PROJECT_ROOT+'/Projeto_LID_v2'
M=pd.read_parquet(B+'/02_Dados_Processados/candidate_matrix.parquet')
if 'ageonset' not in M.columns and 'ageonset_x' in M.columns: M=M.rename(columns={'ageonset_x':'ageonset'})
V5=['updrs_totscore','ageonset','SEX','td_pigd_ratio','NP2FREZ']
d=M[M['exit']>0].dropna(subset=V5+['exit','event']).copy().reset_index(drop=True)
print('PPMI complete-case (5 var): n=%d eventos=%d'%(len(d),int(d.event.sum())))
cph=CoxPHFitter().fit(d[['exit','event']+V5],'exit','event')
print(cph.summary[['coef','exp(coef)','p']].round(4).to_string())
means={c:float(d[c].mean()) for c in V5}
mr=pd.DataFrame([means])[V5]
grid=[round(float(x),1) for x in np.arange(0.5,13.0001,0.5)]
S0g=[round(float(x),6) for x in cph.predict_survival_function(mr,times=grid).iloc[:,0].ffill().bfill().values]
S0h={str(h):float(cph.predict_survival_function(mr,times=[h]).iloc[0,0]) for h in [3,5,7,10]}
Capp=float(concordance_index(d['exit'],-cph.predict_partial_hazard(d[V5]).values,d['event']))
art={'modelo':'Total-5 (reduzido, sem IMC) — Cox PH, congelado no PPMI',
 'motivo':'AMP-PD nao possui peso/altura; IMC impossivel externamente',
 'desfecho':'MDS-UPDRS 4.1>=2 OU 4.2>=2','horizontes':[3,5,7,10],
 'variaveis':V5,'n':len(d),'eventos':int(d.event.sum()),'C_aparente':round(Capp,4),
 'coeficientes':{c:float(cph.params_[c]) for c in V5},'means':means,
 'baseline_survival_hor':S0h,'baseline_survival_grid':{'t':grid,'S0':S0g}}
json.dump(art,open(B+'/08_Validacao_Externa/modelo_reduzido_5var.json','w'),ensure_ascii=False,indent=2)
print('\nC aparente (Total-5, PPMI):',round(Capp,4),' | salvo modelo_reduzido_5var.json')
