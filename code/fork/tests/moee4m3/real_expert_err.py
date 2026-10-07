# CPU/numpy only. Real layer-10 routed experts (EXL3 TR3 4bpw) from nodeC read-only shards:
# per-GEMM and per-expert-FFN output error of the e4m3 path vs the production path (fp16 operands, fp32 acc),
# both vs an fp64 reference of the SAME quantized weights. ACTIVATIONS ARE SYNTHETIC (see ACT below).
import json, os, struct, sys, numpy as np
D=os.path.join(os.environ.get("TF_EXL3_ASSETS") or os.path.expanduser("~/tf-exl3-assets"), "moee4m3-shards/")
IDX=json.load(open(os.path.join(os.environ.get("TF_EXL3_MODELS") or os.path.expanduser("~/models"), "GLM-5.3-Flash-EXL3-TR3-4bpw-partial/model.safetensors.index.json")))["weight_map"]
hdr={}
def tensor(name):
    f=IDX[name]
    if f not in hdr:
        b=open(D+f,'rb'); n=struct.unpack('<Q',b.read(8))[0]; hdr[f]=(json.loads(b.read(n)),8+n)
    h,off=hdr[f]; m=h[name]; a,z=m['data_offsets']
    dt={'I16':np.int16,'F16':np.float16,'BF16':np.uint16,'F32':np.float32,'I32':np.int32}[m['dtype']]
    x=np.fromfile(D+f,dtype=dt,count=(z-a)//np.dtype(dt).itemsize,offset=off+a).reshape(m['shape'])
    if m['dtype']=='BF16': x=(x.astype(np.uint32)<<16).view(np.float32)
    return x
MCG,MASK,FLIP=0xCBAC1FED,0x8FFF8FFF,0x3B603B60
s=np.arange(65536,dtype=np.uint64); x=(s*MCG)&0xFFFFFFFF; x=(x&MASK)^FLIP
CB=((x&0xFFFF).astype(np.uint16).view(np.float16).astype(np.float64)+(x>>16).astype(np.uint16).view(np.float16).astype(np.float64)).astype(np.float16)
p=np.arange(256); lane,j=p//8,p%8; ROWS=2*(lane%4)+(j&1)+8*((j>>1)&1); COLS=lane//4+8*(j>>2)
def unpack(t,bits=4):
    w=t.view(np.uint16).astype(np.uint64); w=w[...,0::2]|(w[...,1::2]<<16); nw=8*bits
    first=p*bits+bits-16+256*bits; last=first+16
    i0,i1=(first//32)%nw,((last-1)//32)%nw; sh=((last-1)//32+1)*32-last
    st=(((w[...,i0]<<32)|w[...,i1])>>sh.astype(np.uint64))&0xFFFF
    kt,nt=st.shape[:2]; out=np.zeros((kt,16,nt,16),np.float16)
    out[:,ROWS,:,COLS]=CB[st.astype(np.int64)].transpose(2,0,1); return out.reshape(kt*16,nt*16), st
n=128; ii=np.arange(n); par=np.array([bin(t).count('1')&1 for t in range(n)]); H=np.where(par[ii[:,None]&ii[None,:]]==1,-1.,1.)/np.sqrt(n)
def rot(a): sh=a.shape; return (a.reshape(*sh[:-1],-1,n)@H).reshape(sh)
def e4m3(a):
    a=np.clip(a,-448,448); m=np.abs(a); e=np.maximum(np.floor(np.log2(np.maximum(m,2**-9))),-6); q=2.0**(e-3); return np.sign(a)*np.round(m/q)*q
def q8(a, mode):
    if mode=='tok': sc=np.abs(a).max(-1,keepdims=True)/448
    else: b=a.reshape(a.shape[0],-1,n); sc=np.repeat(np.abs(b).max(-1,keepdims=True)/448,n,-1).reshape(a.shape)
    sc=np.where(sc==0,1,sc); return e4m3(a/sc)*sc
def lin(xin, W, suh, svh, path, amode='tok'):
    xh=rot(xin*suh)                               # routed domain
    if path=='ref':  y=xh@W.astype(np.float64)
    elif path=='prod': y=(xh.astype(np.float16).astype(np.float32)@W.astype(np.float32)).astype(np.float64)
    else: y=(q8(xh,amode).astype(np.float32)@e4m3(W.astype(np.float64)).astype(np.float32)).astype(np.float64)
    return rot(y)*svh
def rel(a,b): return np.linalg.norm(a-b)/np.linalg.norm(b)
rng=np.random.default_rng(0); L=int(sys.argv[1]) if len(sys.argv)>1 else 10; T=256
def acts(kind):
    X=rng.standard_normal((T,4096))
    if kind=='heavy':   # per-token norm spread + 16 outlier channels x30 (SYNTHETIC)
        X*=np.exp(rng.standard_normal((T,1))); X[:,rng.choice(4096,16,replace=False)]*=30
    return X
experts=[int(a) for a in (sys.argv[2].split(',') if len(sys.argv)>2 else "0,37,101,150,222,287".split(','))]
hist=np.zeros(65536)
print("expert act | gate prod/e4m3tok/e4m3blk | FFN out: prod-vs-ref  e4m3tok-vs-ref  e4m3tok-vs-prod  e4m3blk-vs-prod")
for e in experts:
    P=lambda k,t: tensor(f"model.language_model.layers.{L}.mlp.experts.{e}.{k}.{t}")
    Ws={}
    for k in ('gate_proj','up_proj','down_proj'):
        W,st=unpack(P(k,'trellis')); hist+=np.bincount(st.ravel().astype(np.int64),minlength=65536)
        Ws[k]=(W,P(k,'suh').astype(np.float64),P(k,'svh').astype(np.float64))
    for kind in ('gauss','heavy'):
        X=acts(kind); r={}
        for path,am in (('ref','tok'),('prod','tok'),('e8','tok'),('e8b','blk')):
            pth='e8' if path.startswith('e8') else path
            g=lin(X,*Ws['gate_proj'],pth,am); u=lin(X,*Ws['up_proj'],pth,am)
            h=g/(1+np.exp(-g))*u
            if pth=='prod': h=h.astype(np.float16).astype(np.float64)
            r[path]=(g,lin(h,*Ws['down_proj'],pth,am))
        print(f"{e:4d} {kind:5s} | {rel(r['prod'][0],r['ref'][0]):.2e} {rel(r['e8'][0],r['ref'][0]):.2e} {rel(r['e8b'][0],r['ref'][0]):.2e} |"
              f" {rel(r['prod'][1],r['ref'][1]):.2e} {rel(r['e8'][1],r['ref'][1]):.2e} {rel(r['e8'][1],r['prod'][1]):.2e} {rel(r['e8b'][1],r['prod'][1]):.2e}",flush=True)
pr=hist/hist.sum(); v=CB.astype(np.float64)
print(f"real state histogram: weight e4m3 rel_rms {np.sqrt((pr*(e4m3(v)-v)**2).sum()/(pr*v**2).sum())*100:.3f}% (uniform-state 2.669%); frac |w|<2^-6 {(pr*(np.abs(v)<2**-6)).sum():.4%}")
