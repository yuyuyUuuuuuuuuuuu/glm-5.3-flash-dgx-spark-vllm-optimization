# CPU only: e4m3 round-to-nearest error of the EXL3 mcg codebook values (uniform state histogram) as a function of a
# non-power-of-two prescale c (W8 = e4m3(c * W_q), 1/c folded into the epilogue scale). A power of two cannot change
# anything; this checks whether the codebook's discrete structure leaves room at other c.
import numpy as np
MCG,MASK,FLIP=0xCBAC1FED,0x8FFF8FFF,0x3B603B60
s=np.arange(65536,dtype=np.uint64); x=(s*MCG)&0xFFFFFFFF; x=(x&MASK)^FLIP
CB=((x&0xFFFF).astype(np.uint16).view(np.float16).astype(np.float64)+(x>>16).astype(np.uint16).view(np.float16).astype(np.float64)).astype(np.float16).astype(np.float64)
def e4m3(a):
    a=np.clip(a,-448,448); m=np.abs(a); e=np.maximum(np.floor(np.log2(np.maximum(m,2**-9))),-6); q=2.0**(e-3); return np.sign(a)*np.round(m/q)*q
def err(c):
    v=(CB*c).astype(np.float16).astype(np.float64)   # the kernel would form c*v in fp16
    return np.sqrt(((e4m3(v)/c-CB)**2).mean()/(CB**2).mean())
best=[]
for c in np.linspace(1.0,2.0,401):
    best.append((err(c),c))
best.sort()
print("c=1: %.4f%%"%(100*err(1.0)))
print("best 5:", ", ".join("c=%.4f %.4f%%"%(c,100*e) for e,c in best[:5]))
print("worst:", "c=%.4f %.4f%%"%(best[-1][1],100*best[-1][0]))
# distinct codebook values and how many e4m3 bins they hit
print("distinct fp16 codebook values:", len(np.unique(CB)), " distinct e4m3 values:", len(np.unique(e4m3(CB))))
# continuous Gaussian reference
g=np.random.default_rng(0).standard_normal(1<<20); print("gaussian e4m3 rel_rms %.4f%%"%(100*np.sqrt(((e4m3(g)-g)**2).mean()/(g**2).mean())))
