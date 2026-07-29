# 17_validate_largepd.py
# Produces: LARGE-PD external validation. Table 3
# Original file in the project archive: 63_validacao_largepd.py

"""63 — VALIDAÇÃO EXTERNA (pré-registrada). Modelo Total-5 congelado.
LARGE-PD (primário, current-status): discriminação, calibração, DCA, sensibilidade, recalibração.
"""
from config import PROJECT_ROOT
import json, numpy as np, pandas as pd, warnings, glob
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
from scipy.optimize import brentq
R=PROJECT_ROOT+'/Projeto_LID_v2/08_Validacao_Externa'
A=json.load(open(R+'/modelo_reduzido_5var.json')); V5=A['variaveis']; B=A['coeficientes']; M=A['means']
GT=np.array(A['baseline_survival_grid']['t']); GS=np.array(A['baseline_survival_grid']['S0'])
def S0(t): return np.interp(t,GT,GS,left=1.0,right=GS[-1])
L=pd.read_csv(R+'/dados_processados/largepd_analitico.csv')
L['lp']=sum(B[k]*(L[k]-M[k]) for k in V5)
L['risk']=1-S0(L['time_years'].values)**np.exp(L['lp'].values)
y=L['event'].values.astype(float); r=L['risk'].values; lp=L['lp'].values
n=len(y); ev=int(y.sum())
print("LARGE-PD: n=%d eventos=%d (%.0f%%) | duração>13a: %d pac"%(n,ev,100*y.mean(),int((L.time_years>13).sum())))

def bootci(fn,B_=1000,seed=1):
    rng=np.random.default_rng(seed); idx=np.arange(n); out=[]
    for _ in range(B_):
        b=rng.choice(idx,n,replace=True)
        if y[b].sum() in (0,len(b)): continue
        try: out.append(fn(b))
        except: pass
    return np.nanpercentile(out,[2.5,97.5])

# --- Discriminação ---
auc_r=roc_auc_score(y,r); ci_r=bootci(lambda b: roc_auc_score(y[b],r[b]))
auc_lp=roc_auc_score(y,lp); ci_lp=bootci(lambda b: roc_auc_score(y[b],lp[b]))
print("\n[DISCRIMINAÇÃO]")
print(" C/AUC do risco predito : %.3f (IC95 %.3f–%.3f)"%(auc_r,ci_r[0],ci_r[1]))
print(" C/AUC do preditor linear: %.3f (IC95 %.3f–%.3f)"%(auc_lp,ci_lp[0],ci_lp[1]))

# --- Calibração ---
OE=y.sum()/r.sum(); ci_oe=bootci(lambda b: y[b].sum()/r[b].sum())
Xs=sm.add_constant(lp); ms=sm.Logit(y,Xs).fit(disp=0); slope=ms.params[1]; sl_ci=ms.conf_int()[1]
print("\n[CALIBRAÇÃO]")
print(" Eventos observados=%d | esperados(soma riscos)=%.1f"%(ev,r.sum()))
print(" Razão O:E = %.2f (IC95 %.2f–%.2f)  [<1 = superestima risco]"%(OE,ci_oe[0],ci_oe[1]))
print(" Slope de calibração = %.2f (IC95 %.2f–%.2f)  [ideal=1]"%(slope,sl_ci[0],sl_ci[1]))
L['grp']=pd.qcut(r,4,labels=False,duplicates='drop')
cal=L.groupby('grp').apply(lambda g: pd.Series({'pred_medio':g.risk.mean(),'obs':g.event.mean(),'n':len(g),'ev':int(g.event.sum())}))
print(" Calibração por quartil de risco:"); print(cal.round(3).to_string())

# --- Recalibração pré-planejada (escala do hazard acumulado p/ O:E=1) ---
H=-np.log(S0(L['time_years'].values)); 
def OEc(c): rr=1-np.exp(-c*H*np.exp(lp)); return y.sum()/rr.sum()-1
try:
    c_star=brentq(OEc,0.01,20)
    r_rc=1-np.exp(-c_star*H*np.exp(lp)); auc_rc=roc_auc_score(y,r_rc)
    print("\n[RECALIBRAÇÃO no geral] fator do hazard = %.3f -> O:E=1 por construção; C mantém %.3f (discriminação não muda)"%(c_star,auc_rc))
except Exception as e: print("recalib erro:",e); c_star=np.nan

# --- DCA ---
print("\n[CURVA DE DECISÃO] net benefit (modelo vs tratar-todos vs tratar-ninguém):")
for pt in [0.10,0.20,0.30,0.40]:
    pp=r>=pt; TP=((pp)&(y==1)).sum(); FP=((pp)&(y==0)).sum()
    nb=TP/n-FP/n*(pt/(1-pt)); nb_all=y.mean()-(1-y.mean())*(pt/(1-pt))
    print("  limiar %2.0f%%: modelo=%.3f | tratar-todos=%.3f | tratar-ninguém=0.000"%(100*pt,nb,nb_all))

# --- Sensibilidade: duração <=10 anos (dentro do horizonte de desenvolvimento) ---
s=L[L.time_years<=10]; ys=s.event.values.astype(float); rs=s.risk.values
print("\n[SENSIBILIDADE t<=10a] n=%d eventos=%d | C/AUC=%.3f | O:E=%.2f"%(len(s),int(ys.sum()),roc_auc_score(ys,rs),ys.sum()/rs.sum()))

res={'coorte':'LARGE-PD','n':n,'eventos':ev,'taxa':round(float(y.mean()),3),
 'C_AUC_risco':round(auc_r,3),'C_AUC_risco_ic':[round(ci_r[0],3),round(ci_r[1],3)],
 'C_AUC_lp':round(auc_lp,3),'OE':round(OE,2),'OE_ic':[round(ci_oe[0],2),round(ci_oe[1],2)],
 'slope':round(slope,2),'slope_ic':[round(sl_ci[0],2),round(sl_ci[1],2)],'recalib_fator':round(float(c_star),3)}
json.dump(res,open(R+'/resultados/resultados_largepd.json','w'),indent=2,ensure_ascii=False)
cal.to_csv(R+'/resultados/calibracao_largepd.csv')
L[['codelarge','time_years','lp','risk','event']].to_csv(R+'/resultados/largepd_predicoes.csv',index=False)
print("\nsalvo em resultados/.")
