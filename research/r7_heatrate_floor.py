"""R7: does the cushion->price slope SCALE with gas (multiplicative), or ADD to it (additive), or is gas a FLOOR
that binds when high?  1000 MW at $4 gas vs 1000 MW at $10 gas."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from model import data as _D, panel
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
df = panel.build(); df = df.dropna(subset=['cush','px','aeco']).copy(); df=df[df.index>='2024-09-01']
df['lpx']=np.log1p(df.px); df['g']=df.aeco.clip(lower=0.25); df['lg']=np.log(df.g)
def ols(y,X):
    A=np.column_stack([np.ones(len(X)),X]); b,*_=np.linalg.lstsq(A,y,rcond=None); e=y-A@b
    s2=(e@e)/(len(y)-A.shape[1]); se=np.sqrt(np.diag(s2*np.linalg.pinv(A.T@A))); return b,b/se,1-(e@e)/((y-y.mean())**2).sum()
print(f'{len(df)} days, AECO {df.aeco.min():.2f}..{df.aeco.max():.2f}')
print('\n=== A. interaction: log px ~ cush + gas + cush*gas   (does the slope on cushion depend on gas?) ===')
for lab,y in [('flat',df.lpx),('off-peak',np.log1p(df.px_op)),('peak',np.log1p(df.px_pk))]:
    X=np.column_stack([df.cush/1000, df.g, (df.cush/1000)*df.g]); b,t,r2=ols(y.values,X)
    print(f'{lab:<9} R2={r2:.3f}  cush {b[1]:+.3f} (t{t[1]:+.1f})  gas {b[2]:+.3f} (t{t[2]:+.1f})  cush*gas {b[3]:+.3f} (t{t[3]:+.1f})')
print('\n=== B. elasticity: log px ~ cush + log(gas)  ->  price ∝ gas^gamma.  gamma=1 is the pure heat-rate model ===')
for lab,y in [('flat',df.lpx),('off-peak',np.log1p(df.px_op)),('peak',np.log1p(df.px_pk))]:
    X=np.column_stack([df.cush/1000,(df.cush/1000)**2,df.lg]); b,t,r2=ols(y.values,X)
    print(f'{lab:<9} R2={r2:.3f}  gamma={b[3]:+.3f} (t{t[3]:+.1f})')
    for lo,hi in [(0,1500),(1500,2400),(2400,9000)]:
        s=df[(df.cush>=lo)&(df.cush<hi)]; X=np.column_stack([s.cush/1000,s.lg]); b,t,_=ols(y.loc[s.index].values,X)
        print(f'          cushion {lo:>4}-{hi:<4}: gamma={b[2]:+.2f} (t{t[2]:+.1f}) n={len(s)}')
print('\n=== C. off-peak price vs gas, by cushion band: intercept and slope (the implied heat rate in GJ/MWh) ===')
for lo,hi in [(0,1200),(1200,1800),(1800,2400),(2400,3000),(3000,9000)]:
    s=df[(df.cush>=lo)&(df.cush<hi)]; b,t,r2=ols(s.px_op.values,s.g.values[:,None]); bk,tk,_=ols(s.px_pk.values,s.g.values[:,None])
    print(f'cush {lo:>4}-{hi:<4} n={len(s):3d}  off-peak = {b[0]:5.1f} + {b[1]:4.1f} x gas (t{t[1]:+.1f})   ratio px_op/gas median {np.median(s.px_op/s.g):5.1f}    peak = {bk[0]:5.1f} + {bk[1]:4.1f} x gas (t{tk[1]:+.1f})')
print('\n=== D. walk-forward monthly read (cushion known), four gas treatments ===')
from model.curve import Curve, band_of, BETA, BETA_OP, BETA_PK
gasf=_D.aeco_forward(); rng=np.random.default_rng(1)
def kfit(tr,y,xs,bw=250):
    return np.array([(np.exp(-0.5*((tr.cush-c)/bw)**2)*y).sum()/np.exp(-0.5*((tr.cush-c)/bw)**2).sum() for c in xs])
rows=[]
for m in pd.period_range('2025-04','2026-09',freq='M'):
    t0,t1=m.start_time,m.end_time; te=df[(df.index>=t0)&(df.index<=t1)]; tr=df[(df.index<t0)&(df.index>=t0-pd.Timedelta(days=365))].copy()
    if len(te)<15 or len(tr)<250: continue
    gs=gasf[(gasf.m==t0)&(gasf.ed<=t0-pd.Timedelta(days=25))&(gasf.ed>=t0-pd.Timedelta(days=45))].gas; gf=float(gs.mean()) if len(gs) else tr.aeco.tail(30).mean()
    gfe=max(gf,0.25); c=te.cush.values
    r={'m':str(m),'real':te.px.mean(),'gas_fwd':gf,'gas_real':te.aeco.mean()}
    r['none']=np.expm1(kfit(tr,tr.lpx,c)).mean()                                              # ignore gas
    r['additive']=(np.expm1(kfit(tr,np.log1p((tr.px-BETA*tr.aeco).clip(lower=0)),c))+BETA*gf).mean()   # current model
    r['ratio']=(np.expm1(kfit(tr,np.log1p(tr.px/tr.g),c))*gfe).mean()                          # pure heat rate x gas
    tr['lg']=np.log(tr.g); gam=0.35
    r['elastic']=(np.expm1(kfit(tr,tr.lpx-gam*tr.lg,c)+gam*np.log(gfe))).mean()               # price ∝ gas^0.35
    # floor: scarcity curve (gas-free) but never below the marginal gas unit: 8.5 GJ/MWh x gas + $3 off-peak, 9.5 x gas + $8 peak, blended 16/8
    sc_op=np.expm1(kfit(tr,np.log1p(tr.px_op),c)); sc_pk=np.expm1(kfit(tr,np.log1p(tr.px_pk),c))
    r['floor']=((16*np.maximum(sc_pk,9.5*gfe+8)+8*np.maximum(sc_op,8.5*gfe+3))/24).mean()
    r['add+floor']=((16*np.maximum(np.expm1(kfit(tr,np.log1p((tr.px_pk-BETA_PK*tr.aeco).clip(lower=0)),c))+BETA_PK*gf,9.5*gfe+8)+8*np.maximum(np.expm1(kfit(tr,np.log1p((tr.px_op-BETA_OP*tr.aeco).clip(lower=0)),c))+BETA_OP*gf,8.5*gfe+3))/24).mean()
    rows.append(r)
R=pd.DataFrame(rows).set_index('m'); print(R.round(1).to_string())
for c in ['none','additive','ratio','elastic','floor','add+floor']:
    e=R[c]-R.real; print(f'{c:<10} MAE {e.abs().mean():5.2f}  bias {e.mean():+5.2f}  corr {R[c].corr(R.real):.2f}')
print('\n=== E. what each form says for 1,000 MW and 2,500 MW at $2, $4, $10 gas (trailing year to Aug-26) ===')
tr=df[(df.index>='2025-09-01')&(df.index<'2026-09-01')].copy(); tr['lg']=np.log(tr.g)
for cc in [1000,2500]:
    for g in [2,4,10]:
        add=(16*(np.expm1(kfit(tr,np.log1p((tr.px_pk-BETA_PK*tr.aeco).clip(lower=0)),[cc]))+BETA_PK*g)+8*(np.expm1(kfit(tr,np.log1p((tr.px_op-BETA_OP*tr.aeco).clip(lower=0)),[cc]))+BETA_OP*g))/24
        rat=np.expm1(kfit(tr,np.log1p(tr.px/tr.g),[cc]))*g
        sc_op=np.expm1(kfit(tr,np.log1p(tr.px_op),[cc])); sc_pk=np.expm1(kfit(tr,np.log1p(tr.px_pk),[cc]))
        fl=(16*np.maximum(sc_pk,9.5*g+8)+8*np.maximum(sc_op,8.5*g+3))/24
        af=(16*np.maximum(np.expm1(kfit(tr,np.log1p((tr.px_pk-BETA_PK*tr.aeco).clip(lower=0)),[cc]))+BETA_PK*g,9.5*g+8)+8*np.maximum(np.expm1(kfit(tr,np.log1p((tr.px_op-BETA_OP*tr.aeco).clip(lower=0)),[cc]))+BETA_OP*g,8.5*g+3))/24
        print(f'cushion {cc} MW, gas ${g:>2}:  additive ${add[0]:6.1f}   ratio ${rat[0]:6.1f}   floor ${fl[0]:6.1f}   additive+floor ${af[0]:6.1f}')
