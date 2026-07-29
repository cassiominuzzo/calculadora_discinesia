# 20_figures_km_calibration_dca.py
# Produces: Source figures for Figures 2, 3 and 4
# Original file in the project archive: 66_figuras_competitivas.py

"""66 — Figuras competitivas. Fig1: incidencia acumulada por tercil de risco (PPMI, discriminacao).
Fig2: calibracao antes/depois no LARGE-PD (painel duplo)."""
from config import PROJECT_ROOT
import json, numpy as np, pandas as pd, glob, warnings
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
import statsmodels.api as sm
R=PROJECT_ROOT+'/Projeto_LID_v2'
RES=R+'/08_Validacao_Externa/resultados'

# ---------- FIG 1: KM por tercil de risco predito (PPMI, modelo Total-6) ----------
C=json.load(open(R+'/07_Calculadora/calculator_artifacts.json'))
T6=[i['var'] for i in C['inputs']]; B=C['coeficientes']; M=C['means']
cm=pd.read_parquet(R+'/02_Dados_Processados/candidate_matrix.parquet')
if 'ageonset' not in cm.columns and 'ageonset_x' in cm.columns: cm=cm.rename(columns={'ageonset_x':'ageonset'})
d=cm[cm['exit']>0].dropna(subset=T6).copy()
d['lp']=sum(B[k]*(d[k]-M[k]) for k in T6)
d['grp']=pd.qcut(d['lp'],3,labels=['Low risk','Intermediate risk','High risk'])
lr=multivariate_logrank_test(d['exit'],d['grp'],d['event'])
fig,ax=plt.subplots(figsize=(6.2,4.6)); col={'Low risk':'#2E7D32','Intermediate risk':'#C55A11','High risk':'#C0392B'}
kmf=KaplanMeierFitter()
for g in ['Low risk','Intermediate risk','High risk']:
    s=d[d['grp']==g]; kmf.fit(s['exit'],s['event'],label=g)
    ci=(1-kmf.survival_function_).iloc[:,0]
    ax.plot(ci.index,ci.values*100,color=col[g],lw=2.2,label='%s (n=%d)'%(g,len(s)))
ax.set_xlabel('Years since levodopa initiation'); ax.set_ylabel('Cumulative incidence of problematic dyskinesia (%)')
ax.set_xlim(0,10); ax.set_ylim(0,None)
p=lr.p_value; ax.set_title('Risk stratification by the model (PPMI development cohort)')
ax.text(0.60,0.06,'log-rank p < 0.001' if p<0.001 else 'log-rank p = %.3f'%p,transform=ax.transAxes,fontsize=10)
ax.legend(loc='upper left',fontsize=9); fig.tight_layout(); fig.savefig(RES+'/fig1_km_risk_tertiles_ppmi.png',dpi=140); plt.close()
print("Fig1 KM: log-rank p=%.2e"%p)

# ---------- FIG 2: calibracao antes/depois (LARGE-PD) ----------
A=json.load(open(R+'/08_Validacao_Externa/modelo_reduzido_5var.json')); V5=A['variaveis']; B5=A['coeficientes']; M5=A['means']
GT=np.array(A['baseline_survival_grid']['t']); GS=np.array(A['baseline_survival_grid']['S0'])
def S0(t): return np.interp(t,GT,GS,left=1.0,right=GS[-1])
L=pd.read_csv(R+'/08_Validacao_Externa/dados_processados/largepd_analitico.csv')
L['lp']=sum(B5[k]*(L[k]-M5[k]) for k in V5)
r=(1-S0(L['time_years'].values)**np.exp(L['lp'].values)); r=np.clip(r,1e-6,1-1e-6)
y=L['event'].values.astype(float); n=len(y)
# recalibracao (nivel + slope) via GLM cloglog do status sobre cloglog(r)
x=np.log(-np.log(1-r)); Xd=sm.add_constant(x)
g=sm.GLM(y,Xd,family=sm.families.Binomial(sm.families.links.cloglog())).fit()
r2=1-np.exp(-np.exp(g.params[0]+g.params[1]*x))
oe1=y.sum()/r.sum(); oe2=y.sum()/r2.sum()
def panel(ax,rr,title):
    q=pd.qcut(rr,4,labels=False,duplicates='drop'); df=pd.DataFrame({'r':rr,'y':y,'q':q})
    gg=df.groupby('q').agg(pred=('r','mean'),obs=('y','mean'),nn=('y','size')); se=np.sqrt(gg['obs']*(1-gg['obs'])/gg['nn'])
    ax.plot([0,1],[0,1],'--',color='gray',label='perfect calibration')
    ax.errorbar(gg['pred'],gg['obs'],yerr=1.96*se,fmt='o-',color='#1F4E79',capsize=4)
    ax.set_xlim(0,1.02);ax.set_ylim(0,1.02);ax.set_xlabel('Predicted risk (mean per quartile)');ax.set_title(title)
fig,(a1,a2)=plt.subplots(1,2,figsize=(10,5))
panel(a1,r,'A. Original model (O:E = %.2f)'%oe1); a1.set_ylabel('Observed proportion with dyskinesia'); a1.legend(loc='upper left',fontsize=9)
panel(a2,r2,'B. After recalibration (O:E = %.2f)'%oe2)
fig.suptitle('External calibration in LARGE-PD (n=159, 37 events)',y=1.02,fontsize=12)
fig.tight_layout(); fig.savefig(RES+'/fig2_calibration_before_after_largepd.png',dpi=140,bbox_inches='tight'); plt.close()
print("Fig2 calibracao: O:E antes=%.2f depois=%.2f"%(oe1,oe2))
print("salvas em resultados/")

# ---------- FIG 3: decision-curve analysis (padrao uniforme) ----------
ths=np.linspace(0.01,0.5,50); nb=[];nba=[]
for pt in ths:
    pp=r>=pt; TP=((pp)&(y==1)).sum(); FP=((pp)&(y==0)).sum()
    nb.append(TP/n-FP/n*(pt/(1-pt))); nba.append(y.mean()-(1-y.mean())*(pt/(1-pt)))
fig,ax=plt.subplots(figsize=(6.2,4.6))
ax.plot(ths*100,nb,color='#1F4E79',lw=2.2,label='Total-5 model')
ax.plot(ths*100,nba,color='#C0392B',lw=1.4,label='Treat all')
ax.axhline(0,color='gray',lw=1.2,label='Treat none')
ax.set_xlabel('Risk threshold (%)'); ax.set_ylabel('Net benefit'); ax.set_ylim(-0.1,0.25); ax.set_xlim(0,50)
ax.set_title('Decision-curve analysis (LARGE-PD)'); ax.legend(loc='upper right',fontsize=9)
fig.tight_layout(); fig.savefig(RES+'/fig3_dca_largepd.png',dpi=140); plt.close()
print("Fig3 DCA salva (padrao uniforme)")
