import json, numpy as np, math, sys
sys.path.insert(0, __import__('os').path.dirname(__file__))
from policy_eval import EMA, Fixed, Hazard, ms, WS
def load_probe(path):
    d=json.load(open(path)); out={}
    for w,rec in d.items():
        n=len(rec['ref_ids']); L=np.full(n,-1)
        bad=0
        for p,v in rec['probes'].items():
            p=int(p)
            if v['L'] is None: continue
            m=v['ref_match']
            if m<1: bad+=1; continue   # anchor differs from ref: unknown
            L[p]=min(v['L'], m-1)
            if m-1 < v['L']+1 and m < len(v['first'])+len(v['verify']): bad+=1
        out[w]={'L':L,'n':n,'bad':bad,'text':rec['ref_text']}
    return out
def replay(policy, L, w, n_tokens=200, lag=2, cost=ms, fill=None):
    p=0; t=0.0; steps=0; acc=0; Ks=[]; As=[]
    while p < n_tokens-1:
        K=policy.choose()
        l=L[p] if p < len(L) else 0
        if l<0: l = fill if fill is not None else 0
        a=min(l,K); Ks.append(K); As.append(a)
        j=len(As)-1-lag
        if j>=0: policy.observe(Ks[j],As[j])
        t+=cost(w,K); steps+=1; acc+=a; p+=a+1
    toks=min(p+1,n_tokens)   # tokens emitted by verify steps ~ p (index of last emitted)
    return {'tok_s':p/t*1000,'acc':acc/steps,'K':np.mean(Ks),'ms':t/steps,'steps':steps,'Ks':Ks,'As':As}
