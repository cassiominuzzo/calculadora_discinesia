# 19_validate_amppd.py
# Produces: AMP-PD secondary validation. Table 3
# Original file in the project archive: 65_figuras_en_e_amppd.py

"""65 — Figuras em ingles (titulos calculados dos dados) + resultados do AMP-PD salvos (auditavel)."""
from config import PROJECT_ROOT
import json, numpy as np, pandas as pd, glob, warnings
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
R=PROJECT_ROOT+'/Projeto_LID_v2/08_Validacao_Externa'
A=json.load(open(R+'/modelo_reduzido_5var.json')); V5=A['variaveis']; B=A['coeficientes']; M=A['means']
GT=np.array(A['baseline_survival_grid']['t']); GS=np.array(A['baseline_survival_grid']['S0'])
def S0(t): return np.interp(t,GT,GS,left=1.0,right=GS[-1])
L=pd.read_csv(R+'/dados_processados/largepd_analitico.csv')
L['lp']=sum(B[k]*(L[k]-M[k]) for k in V5); L['risk']=1-S0(L['time_years'].values)**np.exp(L['lp'].values)
y=L['event'].values.astype(float); r=L['risk'].values; n=len(y); ev=int(y.sum()); oe=y.sum()/r.sum()
# Fig 1 calibration (titulo calculado)
L['grp']=pd.qcut(r,4,labels=False,duplicates='drop')
g=L.groupby('grp').agg(pred=('risk','mean'),obs=('event','mean'),nn=('event','size')); se=np.sqrt(g['obs']*(1-g['obs'])/g['nn'])
fig,ax=plt.subplots(figsize=(5,5)); ax.plot([0,1],[0,1],'--',color='gray',label='perfect calibration')
ax.errorbar(g['pred'],g['obs'],yerr=1.96*se,fmt='o-',color='#1F4E79',capsize=4,label='LARGE-PD (Total-5)')
ax.set_xlabel('Predicted risk (mean per quartile)'); ax.set_ylabel('Observed proportion with dyskinesia')
ax.set_title('Calibration - LARGE-PD (n=%d, %d events)\nO:E = %.2f (model over-predicts risk)'%(n,ev,oe))
ax.set_xlim(0,1.02); ax.set_ylim(0,1.02); ax.legend(); fig.tight_layout(); fig.savefig(R+'/resultados/fig_calibration_largepd.png',dpi=140); plt.close()
# Fig 2 DCA
ths=np.linspace(0.01,0.5,50); nb=[];nba=[]
for pt in ths:
    pp=r>=pt; TP=((pp)&(y==1)).sum(); FP=((pp)&(y==0)).sum(); nb.append(TP/n-FP/n*(pt/(1-pt))); nba.append(y.mean()-(1-y.mean())*(pt/(1-pt)))
fig,ax=plt.subplots(figsize=(6,4.2)); ax.plot(ths*100,nb,color='#1F4E79',lw=2,label='Total-5 model')
ax.plot(ths*100,nba,color='#C0392B',lw=1.2,label='Treat all'); ax.axhline(0,color='gray',lw=1,label='Treat none')
ax.set_xlabel('Risk threshold (%)'); ax.set_ylabel('Net benefit'); ax.set_ylim(-0.1,0.25)
ax.set_title('Decision-curve analysis - LARGE-PD'); ax.legend(); fig.tight_layout(); fig.savefig(R+'/resultados/fig_dca_largepd_en.png',dpi=140); plt.close()
# AMP-PD resultados salvos
Ad=pd.read_csv(R+'/dados_processados/amppd_nonppmi_completo.csv'); Ad['lp']=sum(B[k]*(Ad[k]-M[k]) for k in V5)
c_amp=float(concordance_index(Ad['time_years'],-Ad['lp'],Ad['event'])); Ad['risk3']=1-S0(3.0)**np.exp(Ad['lp'])
obs3=1-float(KaplanMeierFitter().fit(Ad['time_years'],Ad['event']).predict(3.0)); oe3=obs3/Ad['risk3'].mean()
res={'coorte':'AMP-PD nao-PPMI (secundaria, subpoderada)','n':int(len(Ad)),'eventos':int(Ad.event.sum()),
     'C_index':round(c_amp,3),'OE_3anos':round(oe3,2),'obs_3a':round(obs3,3),'esp_3a':round(float(Ad['risk3'].mean()),3),
     'nota':'apenas exploratorio (9 eventos)'}
json.dump(res,open(R+'/resultados/resultados_amppd.json','w'),ensure_ascii=False,indent=2)
print('figuras EN regeneradas (titulos calculados: O:E=%.2f) e resultados_amppd.json salvo (C=%.3f, O:E3a=%.2f)'%(oe,c_amp,oe3))
