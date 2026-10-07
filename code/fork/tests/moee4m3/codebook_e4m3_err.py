# CPU/numpy only. Exact e4m3 rounding error of the EXL3 "mcg" codebook (all 65536 states), plus
# activation e4m3 error per-token vs per-128-block on synthetic rotated rows. Does NOT use real routed weights.
import numpy as np
MCG,MASK,FLIP=0xCBAC1FED,0x8FFF8FFF,0x3B603B60
s=np.arange(65536,dtype=np.uint64); x=((s*MCG)&0xFFFFFFFF); x=(x&MASK)^FLIP
v=((x&0xFFFF).astype(np.uint16).view(np.float16).astype(np.float64)+(x>>16).astype(np.uint16).view(np.float16).astype(np.float64)).astype(np.float16).astype(np.float64)
def e4m3(x):
    x=np.clip(x,-448,448); a=np.abs(x); e=np.maximum(np.floor(np.log2(np.maximum(a,2**-9))),-6); q=2.0**(e-3); return np.sign(x)*np.round(a/q)*q
print(f"codebook rms {np.sqrt((v**2).mean()):.4f} max|v| {np.abs(v).max():.3f} frac|v|<2^-6 {(np.abs(v)<2**-6).mean():.4%}")
for sc in [1,4,16,64]:
    r=np.sqrt(((e4m3(v*sc)/sc-v)**2).mean()/(v**2).mean()); print(f"weight(codebook) e4m3 prescale x{sc}: rel_rms {r*100:.3f}%")
n=128; i=np.arange(n); par=np.array([bin(t).count('1')&1 for t in range(n)]); H=np.where(par[i[:,None]&i[None,:]]==1,-1.,1.)/np.sqrt(n)
rng=np.random.default_rng(0); T,K=2048,4096
X=rng.standard_normal((T,K))*np.exp(rng.standard_normal((T,1))); X[:,rng.choice(K,16,replace=False)]*=30
X*=rng.choice([-1.,1.],K)  # suh sign
Xr=(X.reshape(T,-1,n)@H.T).reshape(T,K)
for name,A in (("unrotated+outliers",X),("rotated (routed domain)",Xr)):
    sa=np.abs(A).max(1,keepdims=True)/448; ra=np.sqrt(((e4m3(A/sa)*sa-A)**2).sum()/(A**2).sum())
    Ab=A.reshape(T,-1,n); sb=np.abs(Ab).max(2,keepdims=True)/448; rb=np.sqrt(((e4m3(Ab/sb)*sb-Ab)**2).sum()/(Ab**2).sum())
    print(f"act {name}: per-token {ra*100:.3f}%  per-128-block {rb*100:.3f}%")
# output error of one GEMM vs production-like fp16 path, codebook weights drawn uniformly over states
W=v[rng.integers(0,65536,(K,1024))]
Y=Xr@W; sa=np.abs(Xr).max(1,keepdims=True)/448
for lab,A8 in (("per-token",e4m3(Xr/sa)*sa),):
    Y8=A8@e4m3(W); print(f"GEMM out rel_l2 e4m3-path vs exact ({lab} act, direct weight cvt): {np.linalg.norm(Y8-Y)/np.linalg.norm(Y)*100:.3f}%")
