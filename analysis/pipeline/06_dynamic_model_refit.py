# 06_dynamic_model_refit.py
# Produces: Total-6 plus levodopa responsiveness (dynamic mode of the calculator)
# Original file in the project archive: 51_fit_dynamic_calculator.py

"""51 — Refit AUDITAVEL do modelo DINAMICO (Total-6 + responsividade continua) para a calculadora.
Grava um bloco 'modelo_dinamico' no calculator_artifacts.json, mantendo o modelo basal intacto.
responsividade = (UPDRS-III_OFF - UPDRS-III_ON)/UPDRS-III_OFF*100  (winsor [-100,100]).
"""
from config import PROJECT_ROOT
import glob, json, warnings, numpy as np, pandas as pd; warnings.filterwarnings('ignore')
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold

B=PROJECT_ROOT
PR=B+'/Projeto_LID_v2/02_Dados_Processados'
CALC=B+'/Projeto_LID_v2/07_Calculadora/calculator_artifacts.json'

def ip(d):
    d=d.copy(); d['PATNO']=pd.to_numeric(d['PATNO'],errors='coerce').astype('Int64'); return d.dropna(subset=['PATNO'])

T6=['updrs_totscore','ageonset','SEX','BMI','td_pigd_ratio','NP2FREZ']
mb=ip(pd.read_parquet(PR+'/master_baseline.parquet'))[['PATNO','resp']].dropna()
M=pd.read_parquet(PR+'/candidate_matrix.parquet')
if 'ageonset' not in M.columns and 'ageonset_x' in M.columns: M=M.rename(columns={'ageonset_x':'ageonset'})
M=ip(M)
d=M[M['exit']>0].dropna(subset=T6).merge(mb,on='PATNO',how='inner').reset_index(drop=True)
cols=T6+['resp']
d=d.dropna(subset=cols).reset_index(drop=True)
n=len(d); ev=int(d.event.sum())
print('SUBGRUPO dinamico: n=%d eventos=%d'%(n,ev))

cph=CoxPHFitter(penalizer=0.02).fit(d[['exit','event']+cols],'exit','event')
print('\n--- Cox 7 variaveis (Total-6 + resp) ---')
print(cph.summary[['coef','exp(coef)','exp(coef) lower 95%','exp(coef) upper 95%','p']].round(4).to_string())

means={c:float(d[c].mean()) for c in cols}
mean_row=pd.DataFrame([means])[cols]
grid_t=[round(float(x),1) for x in np.arange(0.5,13.0001,0.5)]
sfg=cph.predict_survival_function(mean_row, times=grid_t).iloc[:,0].ffill().bfill()
S0g=[round(float(x),5) for x in sfg.values]
sfh=cph.predict_survival_function(mean_row, times=[3,5,7,10]).iloc[:,0].ffill().bfill()
S0h={str(h):float(s) for h,s in zip([3,5,7,10],sfh.values)}

def oof(seed):
    kf=KFold(5,shuffle=True,random_state=seed); pr=np.zeros(n)
    for tr,te in kf.split(d):
        m=CoxPHFitter(penalizer=0.1).fit(d.iloc[tr][['exit','event']+cols],'exit','event')
        pr[te]=-m.predict_partial_hazard(d.iloc[te][cols]).values
    return concordance_index(d['exit'],pr,d['event'])
c_oof=float(np.mean([oof(s) for s in range(6)]))
c_app=float(concordance_index(d['exit'],-cph.predict_partial_hazard(d[cols]).values,d['event']))
# C do Total-6 SEM resp no MESMO subgrupo (delta honesto)
cph6=CoxPHFitter(penalizer=0.02).fit(d[['exit','event']+T6],'exit','event')
c_app6=float(concordance_index(d['exit'],-cph6.predict_partial_hazard(d[T6]).values,d['event']))
print('\nC apparent (7var)=%.4f | C OOF (7var)=%.4f | C apparent Total-6 no mesmo n=%.4f | dC=%.4f'%(c_app,c_oof,c_app6,c_app-c_app6))

hr=cph.summary.loc['resp']
dyn={
 'nota': 'Modelo dinamico: Total-6 + responsividade a levodopa (continua, % de melhora UPDRS-III OFF->ON). Uso em SEGUIMENTO, apos challenge OFF/ON. Exploratorio.',
 'coeficientes':{c:float(cph.params_[c]) for c in cols},
 'means':means,
 'baseline_survival_hor':S0h,
 'baseline_survival_grid':{'t':grid_t,'S0':S0g},
 'resp_hr_por_10pct':round(float(np.exp(cph.params_['resp']*10)),3),
 'resp_hr_bruto':round(float(hr['exp(coef)']),4),
 'resp_p':float(hr['p']),
 'resp_input':{'min':-20.0,'max':100.0,'step':5.0,'default':float(round(means['resp'],0))},
 'n':n,'eventos':ev,
 'desempenho':{'C_OOF':round(c_oof,3),'C_apparent':round(c_app,3),'C_Total6_mesmo_n':round(c_app6,3),'dC_apparent':round(c_app-c_app6,3)}
}
A=json.load(open(CALC,encoding='utf-8'))
A['modelo_dinamico']=dyn
json.dump(A,open(CALC,'w',encoding='utf-8'),ensure_ascii=False,indent=2)
print('\nOK: bloco modelo_dinamico gravado em calculator_artifacts.json')
print('HR resp por +10%% de melhora = %.3f | media resp = %.1f%% | default input = %.0f'%(dyn['resp_hr_por_10pct'],means['resp'],dyn['resp_input']['default']))
