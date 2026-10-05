import json, numpy as np
from scipy import stats
d=json.load(open('../results/results.json')); META=d['meta']; RAW=d['results']
seen={}
for r in RAW:
    k=(r['exp'],r['method'],r['alpha'],r['sigma'],r['mal'],r['seed'])
    if k not in seen: seen[k]=r
U=list(seen.values())
CL=META['classes']; NCLS=len(CL)
def get(exp,method,alpha,sigma=0.0,mal=0):
    rs=sorted([r for r in U if r['exp']==exp and r['method']==method and r['alpha']==alpha and r['sigma']==sigma and r['mal']==mal],key=lambda r:r['seed'])
    assert len(rs)==3,(exp,method,alpha,sigma,mal,len(rs)); return rs
def ms(rs,f='test_f1'):
    v=np.array([r[f] for r in rs],float); return float(v.mean()),float(v.std(ddof=1))
def vals(rs,f='test_f1'): return [float(r[f]) for r in rs]
def paired(a,b,f='test_f1'):
    x=np.array([r[f] for r in a],float)-np.array([r[f] for r in b],float)
    m=float(x.mean()); s=float(x.std(ddof=1)); se=s/np.sqrt(len(x))
    t=m/se if se>0 else float('inf'); p=float(2*(1-stats.t.cdf(abs(t),df=2))); ci=float(stats.t.ppf(0.975,2)*se)
    return dict(diff=m,ci=ci,p=p,npos=int((x>0).sum()),n=len(x))
def holm(ps):
    idx=np.argsort(ps); m=len(ps); adj=[0]*m; prev=0
    for rank,i in enumerate(idx):
        a=min(1.0,(m-rank)*ps[i]); prev=max(prev,a); adj[i]=prev
    return adj
def eps(sig,K,T,delta=1e-5):
    if sig<=0: return float('inf')
    s2=sig**2*K; a=np.arange(1.01,3000,0.05); return float(np.min(T*a/(2*s2)+np.log(1/delta)/(a-1)))
T_=META['rounds']; K_=META['K']
CEN=get('central','Centralized','none')
FA={a:get('hetero','FedAvg',a) for a in ['iid','0.5','0.1']}
FP={a:get('hetero','FedProx mu=0.001',a) for a in ['iid','0.5','0.1']}
FP01=get('hetero','FedProx mu=0.01','0.1')
CLIP=get('dp','Clip only (sigma=0)','0.5')
DP={s:get('dp',f'Clip+DP sigma={s}','0.5',s) for s in [0.03,0.05,0.1,0.3]}
ABL={n:get('ablation',n,'0.1',0.0 if n=='- DP noise' else 0.05) for n in ['Full SPP-FLIDS','- FedProx','- Clipping','- DP noise','- SecAgg']}
MAL={(meth,m):get('malicious',meth,'0.5',0.0 if meth=='FedAvg' else 0.05,m) for meth in ['FedAvg','SPP-FLIDS'] for m in [0,1,2]}
SPP05=MAL[('SPP-FLIDS',0)]   # FedProx + clip + DP(sigma*=0.05) at alpha=0.5
if __name__=='__main__':
    print(ms(CEN),ms(FA['iid']),ms(SPP05),ms(ABL['Full SPP-FLIDS']))
    print(paired(MAL[('SPP-FLIDS',2)],MAL[('FedAvg',2)],'asr'))
    print(paired(MAL[('FedAvg',2)],MAL[('FedAvg',0)],'asr'))
    print(eps(0.05,10,20),eps(0.05,1,20))
