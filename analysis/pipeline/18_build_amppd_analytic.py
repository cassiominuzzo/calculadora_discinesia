# 18_build_amppd_analytic.py
# Produces: AMP-PD non-PPMI analytic table
# Original file in the project archive: 61_build_amppd_nonppmi.py

"""61 (v3) — Tabela analítica NÃO-PPMI do AMP-PD usando colunas code_* (numéricas 0-4).
Tempo-zero=início levodopa (No->Yes); preditor por visita não-nula mais próxima do basal;
desfecho code_4.1>=2 OU code_4.2>=2; elegível=sem discinesia no t0. Sem métricas de performance."""
from config import PROJECT_ROOT
import pandas as pd, numpy as np, glob, warnings
warnings.filterwarnings('ignore')
R=PROJECT_ROOT+'/Projeto_LID_v2/08_Validacao_Externa'
AMP=R+'/AMP PD/Descompactado'; OUT=R+'/dados_processados'
COH=['PDBP','Steady','Sure','BioFIND']
def num04(x): x=pd.to_numeric(x,errors='coerce'); return x.where((x>=0)&(x<=4))
TDp2=['code_upd2210_tremor']
TDp3=['code_upd2315a_postural_tremor_of_right_hand','code_upd2315b_postural_tremor_of_left_hand',
 'code_upd2316a_kinetic_tremor_of_right_hand','code_upd2316b_kinetic_tremor_of_left_hand',
 'code_upd2317a_rest_tremor_amplitude_right_upper_extremity','code_upd2317b_rest_tremor_amplitude_left_upper_extremity',
 'code_upd2317c_rest_tremor_amplitude_right_lower_extremity','code_upd2317d_rest_tremor_amplitude_left_lower_extremity',
 'code_upd2317e_rest_tremor_amplitude_lip_or_jaw','code_upd2318_consistency_of_rest_tremor']
PIp2=['code_upd2212_walking_and_balance','code_upd2213_freezing']
PIp3=['code_upd2310_gait','code_upd2311_freezing_of_gait','code_upd2312_postural_stability']
FREZ='code_upd2213_freezing'
part=pd.read_csv(AMP+'/amp_pd_participants_.csv'); st=part.set_index('participant_id')['study']
cc=pd.read_csv(AMP+'/amp_pd_case_control_.csv')
pdmask=cc['diagnosis_latest'].astype(str).str.contains("Parkinson",case=False)& ~cc['diagnosis_latest'].astype(str).str.contains("Atypical|Vascular|Drug|Secondary|Supranuclear|Multiple System|Corticobasal",case=False)
keep=set(part[part.study.isin(COH)].merge(cc[pdmask][['participant_id']],on='participant_id')['participant_id'])
print("FUNIL: PD idiopática coortes-alvo:",len(keep))
dem=pd.read_csv(AMP+'/Demographics_.csv')
SEX=dem.dropna(subset=['sex']).groupby('participant_id')['sex'].first().map(lambda v:1 if str(v).lower().startswith('m') else (0 if str(v).lower().startswith('f') else np.nan))
ageb=dem.dropna(subset=['age_at_baseline']).groupby('participant_id')['age_at_baseline'].first()
mh=pd.read_csv(AMP+'/PD_Medical_History_.csv',usecols=['participant_id','visit_month','on_levodopa','age_at_diagnosis'])
mh=mh[mh.participant_id.isin(keep)]
aad=mh.dropna(subset=['age_at_diagnosis']).groupby('participant_id')['age_at_diagnosis'].first()
g=mh.dropna(subset=['on_levodopa'])
yes=g[g.on_levodopa=='Yes'].groupby('participant_id')['visit_month'].min().rename('t0_month')
m=g.merge(yes,on='participant_id'); nb=m[(m.on_levodopa=='No')&(m.visit_month<m.t0_month)].groupby('participant_id')['visit_month'].max().rename('base_month')
t0=pd.concat([yes,nb],axis=1).dropna()
print(" início levodopa observável:",len(t0),{c:int(sum(st.get(i)==c for i in t0.index)) for c in COH})
p1=pd.read_csv(AMP+'/MDS_UPDRS_Part_I_.csv',usecols=['participant_id','visit_month','mds_updrs_part_i_summary_score'])
p2=pd.read_csv(AMP+'/MDS_UPDRS_Part_II_.csv',usecols=['participant_id','visit_month','mds_updrs_part_ii_summary_score']+TDp2+PIp2)
p3=pd.read_csv(AMP+'/MDS_UPDRS_Part_III_.csv',usecols=['participant_id','visit_month','mds_updrs_part_iii_summary_score']+TDp3+PIp3)
def nearest(dfp,col):
    d=dfp[['participant_id','visit_month',col]].dropna(subset=[col]).merge(t0,left_on='participant_id',right_index=True)
    d=d[d.visit_month<=d.t0_month]
    if d.empty: return pd.Series(dtype=float,name=col)
    d['dist']=(d.visit_month-d.base_month).abs()
    return d.sort_values(['participant_id','dist']).groupby('participant_id')[col].first()
B=pd.DataFrame(index=list(t0.index))
B['i']=nearest(p1,'mds_updrs_part_i_summary_score'); B['ii']=nearest(p2,'mds_updrs_part_ii_summary_score'); B['iii']=nearest(p3,'mds_updrs_part_iii_summary_score')
B['updrs_totscore']=B[['i','ii','iii']].sum(axis=1,min_count=3)
B['NP2FREZ']=num04(nearest(p2,FREZ))
for c in TDp2+PIp2: B[c]=num04(nearest(p2,c))
for c in TDp3+PIp3: B[c]=num04(nearest(p3,c))
B['tdv']=B[TDp2+TDp3].mean(axis=1); B['pgv']=B[PIp2+PIp3].mean(axis=1)
B['td_pigd_ratio']=np.where(B['pgv']>0,B['tdv']/B['pgv'],np.nan)
p4=pd.read_csv(AMP+'/MDS_UPDRS_Part_IV_.csv',usecols=['participant_id','visit_month','code_upd2401_time_spent_with_dyskinesias','code_upd2402_functional_impact_of_dyskinesias'])
p4=p4[p4.participant_id.isin(set(t0.index))].copy()
p4['dysk']=(pd.to_numeric(p4.code_upd2401_time_spent_with_dyskinesias,errors='coerce')>=2)|(pd.to_numeric(p4.code_upd2402_functional_impact_of_dyskinesias,errors='coerce')>=2)
out=[]
for pid,gg in p4.groupby('participant_id'):
    tt=t0.loc[pid,'t0_month']; gg=gg.sort_values('visit_month')
    if gg[gg.visit_month<=tt]['dysk'].any(): continue
    post=gg[gg.visit_month>tt]
    if post.empty: continue
    ev=post[post.dysk]
    out.append({'participant_id':pid,'event':int(len(ev)>0),'time_years':((ev.visit_month.min() if len(ev) else post.visit_month.max())-tt)/12.0})
outc=pd.DataFrame(out).set_index('participant_id')
print(" elegíveis c/ desfecho válido:",len(outc))
df=B.join(outc,how='inner')
df['SEX']=df.index.map(SEX)
df['ageonset']=df.index.map(aad); df['ageonset_src']='age_at_diagnosis'
miss=df['ageonset'].isna(); df.loc[miss,'ageonset']=df.index[miss].map(ageb); df.loc[miss,'ageonset_src']='age_at_baseline(proxy)'
df['study']=df.index.map(st); df=df.reset_index().rename(columns={'index':'participant_id'})
V5=['updrs_totscore','ageonset','SEX','td_pigd_ratio','NP2FREZ']
comp=df.dropna(subset=V5+['event','time_years'])
print("\n===== DATASET PRONTO =====  montado:",len(df)," | completo(5+desfecho):",len(comp))
print(" não-nulos:",{v:int(df[v].notna().sum()) for v in V5})
tot_ev=0;tot_n=0
for c in COH:
    s=comp[comp.study==c]
    if len(s): tot_ev+=int(s.event.sum());tot_n+=len(s); print(f"   {c:8s}: n={len(s):4d} | eventos={int(s.event.sum()):3d} ({100*s.event.mean():.0f}%) | fu_med={s.time_years.median():.1f}a | proxy_idade={int((s.ageonset_src!='age_at_diagnosis').sum())}")
print(f"   {'TOTAL':8s}: n={tot_n:4d} | eventos={tot_ev:3d}")
df.to_parquet(OUT+'/amppd_nonppmi_analitico.parquet'); comp.to_csv(OUT+'/amppd_nonppmi_completo.csv',index=False)
print(" salvo em dados_processados/.")
