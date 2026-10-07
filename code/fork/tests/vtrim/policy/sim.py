import json, math, numpy as np
KSET=[4,5,7]; KMAX=7
def ema_policy(a_hist_obs, lag, alpha=0.25, margin=1.0, min_steps=4, kset=KSET):
    """Return K for each step given the sequence of observations available; generic simulator of production EMA.
    a_obs: list of (num_draft, num_acc). K_t computed from observations of steps < t-lag."""
    pass
def simulate_trace(a, lag, alpha=0.25, margin=1.0, min_steps=4, kset=KSET, sat='max'):
    """a: observed accepted counts of the trace (under production). Return inferred K per step and consistency."""
    ema=None; n=0; Ks=[]; obs=[]
    for t in range(len(a)):
        # observations available: steps 0..t-1-lag
        while len(obs) < max(0, t-lag):
            i=len(obs); k=Ks[i]; acc=a[i]
            o = float(max(KMAX,k)) if (acc>=k and sat!='n') else float(acc)
            ema = o*alpha + (KMAX*(1-alpha) if ema is None else ema*(1-alpha)); n+=1
            obs.append(o)
        if ema is None or n<min_steps:
            K=KMAX
        else:
            tgt=int(math.ceil(ema+margin)); c=[v for v in kset if v<=min(tgt,KMAX)]
            K=max(c) if c else min(kset)
        Ks.append(K)
    return Ks
