import json, math, numpy as np, itertools, sys
WS=['structured','prose','coding','ja']
C0={'structured':52.62,'prose':53.69,'coding':57.09,'ja':50.09}; SLOPE=4.46
def ms(w,K): return C0[w]+SLOPE*K
def load_L(path):
    d=json.load(open(path)); out={w:[] for w in WS}
    for r in d['runs']:
        if 'chunks' not in r: continue
        out[r['workload']].append([x[1]-1 for x in r['chunks'][1:]])
    return out
class EMA:
    def __init__(s,kset=(4,5,7),alpha=0.25,margin=1.0,min_steps=4,kmax=7,sat='max'):
        s.kset=sorted(kset);s.alpha=alpha;s.margin=margin;s.min_steps=min_steps;s.kmax=kmax;s.sat=sat;s.ema=None;s.n=0
    def observe(s,k,a):
        o=float(max(s.kmax,k)) if (a>=k and s.sat!='n') else float(a)
        s.ema=o*s.alpha+(s.kmax*(1-s.alpha) if s.ema is None else s.ema*(1-s.alpha)); s.n+=1
    def choose(s):
        if s.ema is None or s.n<s.min_steps: return s.kmax
        tgt=int(math.ceil(s.ema+s.margin)); c=[v for v in s.kset if v<=min(tgt,s.kmax)]
        return max(c) if c else min(s.kset)
class Fixed:
    def __init__(s,K): s.K=K
    def observe(s,k,a): pass
    def choose(s): return s.K
class Hazard:
    """Per-request per-position conditional acceptance h_i (EMA over censored observations);
    choose K in kset maximising (1+sum_{i<=K} prod h)/ms(K)."""
    def __init__(s,w,kset=(4,5,7),beta=0.15,prior=None,min_steps=4,kmax=7,cost=None):
        s.w=w;s.kset=sorted(kset);s.beta=beta;s.kmax=kmax;s.min_steps=min_steps;s.n=0
        s.h=np.array(prior if prior is not None else [0.8,0.7,0.7,0.7,0.6,0.6,0.6],float)
        s.cost=cost or (lambda K: ms(w,K))
    def observe(s,k,a):
        s.n+=1
        # positions 1..min(a+1,k) observed: pos<=a accepted, pos a+1 rejected (if a<k)
        for i in range(1,k+1):
            if i<=a: s.h[i-1]+=s.beta*(1-s.h[i-1])
            elif i==a+1: s.h[i-1]+=s.beta*(0-s.h[i-1]); break
    def choose(s):
        if s.n<s.min_steps: return s.kmax
        surv=np.cumprod(s.h)
        best=None
        for K in s.kset:
            v=(1+surv[:K].sum())/s.cost(K)
            if best is None or v>best[0]: best=(v,K)
        return best[1]
def run(policy_factory,Ls,w,lag=2):
    toks=0;t=0;acc=0;steps=0;Ksum=0
    for L in Ls:
        p=policy_factory(w); Ks=[]; As=[]
        for i,l in enumerate(L):
            while len(As)>0 and False: pass
            # feed observations up to i-1-lag
            K=p.choose() if True else None
            Ks.append(K); a=min(l,K); As.append(a)
            j=i-lag
            if j>=0: p.observe(Ks[j],As[j])
            toks+=a+1; t+=ms(w,K); acc+=a; steps+=1; Ksum+=K
    return toks/t*1000, acc/steps, Ksum/steps, t/steps
