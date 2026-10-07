#!/usr/bin/env python3
"""Part A(2) of the moe2 review: is the e4m3 path's ~6.5 % per-expert error small or large RELATIVE TO THE EXL3 4bpw
quantization noise production already accepts in the same experts?

The routed experts' BF16 originals are not on nodeC, but the SHARED expert of the same layers is stored in BF16
(layer 10: shard 00004; layers 3/23/43: models/GLM-5.3-Flash-EXL3-TR3-4bpw-partial/bf16_samples) and has exactly a
routed expert's shapes (gate/up 2048x4096, down 4096x2048, SwiGLU). So:

  1. quantize it with the image's own exllamav3 0.0.43 quantizer (the version the checkpoint was made with:
     quant_method exl3 0.0.43, K=4, mcg codebook), LDLQ with a Hessian from calibration activations
     (gate and up share one H and one su, as in the checkpoint: _exl3_shared_w13_suh), down's H from the SwiGLU
     activations of the BF16 gate/up;
  2. on held-out activations from the same distribution, compute the FFN output of
       ref   = the BF16 weights in fp32 (the unquantized model),
       prod  = the decoded trellis with production's arithmetic (fp16 operands, fp32 accumulate),
       e4m3  = the e4m3 spec (emu_spec_copy: weights -> e4m3, per-row e4m3 activations), down input scale per
               TP=2 rank (1024 local intermediate rows), as the kernel does,
     plus error-reduction variants (down in fp16; gate/up activations as two e4m3 terms; per-32 activation groups);
  3. report rel-L2 vs ref and the ADDED noise variance ratio r = (e(e4m3,ref)^2 - e(prod,ref)^2) / e(prod,ref)^2.
     For trellis quantization D ~ 2^(-2 R), so the e4m3 path costs as much as lowering the routed experts'
     bitrate by dR = 0.5*log2(1+r) bpw.

Activations are SYNTHETIC (no captured hidden states exist on nodeC): 'ln' = Gaussian times layer 10's real
post_attention_layernorm weight (the MoE input's real per-channel scale profile), 'heavy' = 'ln' with a log-normal
per-token norm spread and 16 outlier channels x30. The quantizer is run on a calibration draw, the errors are
measured on a different draw of the same distribution.
Run: tests/gpu_run.sh python3 tests/moe2/exl3_vs_e4m3_noise.py   (GPU_RUN_RO must mount the shard + model dirs)
"""
import importlib.util, json, math, os, struct, sys, time
import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
SH = os.path.join(os.environ.get("TF_EXL3_ASSETS") or os.path.expanduser("~/tf-exl3-assets"), "moee4m3-shards/")
PART = os.path.join(os.environ.get("TF_EXL3_MODELS") or os.path.expanduser("~/models"), "GLM-5.3-Flash-EXL3-TR3-4bpw-partial/")
IDX = json.load(open(PART + "model.safetensors.index.json"))["weight_map"]
dev = torch.device("cuda")

spec = importlib.util.spec_from_file_location("emu", "/w/tests/moee4m3/emu_spec_copy.py")
emu = importlib.util.module_from_spec(spec); spec.loader.exec_module(emu)
import types
# exllamav3's package __init__ pulls in the generator / attention stack (flash_attn, kbnf); the quantizer needs none of
# it: register the package (and its sub-packages up to the quantizer) without running their __init__ files.
_root = importlib.util.find_spec("exllamav3").submodule_search_locations[0]
for _name, _sub in (("exllamav3", ""), ("exllamav3.modules", "modules"), ("exllamav3.modules.quant", "modules/quant"),
                    ("exllamav3.modules.quant.exl3_lib", "modules/quant/exl3_lib")):
    _m = types.ModuleType(_name); _m.__path__ = [os.path.join(_root, _sub)] if _sub else [_root]
    sys.modules[_name] = _m
from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3

_hdr = {}
def st_tensor(name):
    f = IDX[name]
    if f not in _hdr:
        b = open(SH + f, "rb"); n = struct.unpack("<Q", b.read(8))[0]; _hdr[f] = (json.loads(b.read(n)), 8 + n)
    h, off = _hdr[f]; m = h[name]; a, z = m["data_offsets"]
    assert m["dtype"] == "BF16"
    x = np.fromfile(SH + f, dtype=np.uint16, count=(z - a) // 2, offset=off + a).reshape(m["shape"])
    return torch.from_numpy((x.astype(np.uint32) << 16).view(np.float32))

MAN = {m["name"]: m for m in json.load(open(PART + "bf16_samples/manifest.json"))}
def sample_tensor(name):
    m = MAN[name]
    x = np.fromfile(PART + "bf16_samples/" + m["file"], dtype=np.uint16).reshape(m["shape"])
    return torch.from_numpy((x.astype(np.uint32) << 16).view(np.float32))

def shared(L):
    get = st_tensor if L == 10 else sample_tensor
    return {p: get(f"model.language_model.layers.{L}.mlp.shared_experts.{p}_proj.weight").to(dev)
            for p in ("gate", "up", "down")}

LN = st_tensor("model.language_model.layers.10.post_attention_layernorm.weight").to(dev)

_Q = {}
def _basis():
    if "q" not in _Q:
        g = torch.Generator(device=dev).manual_seed(77)
        _Q["q"] = torch.linalg.qr(torch.randn(4096, 4096, device=dev, generator=g))[0]
    return _Q["q"]

def acts(kind, n, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    X = torch.randn(n, 4096, device=dev, generator=g) * LN
    if kind == "spec":   # power-law covariance spectrum (eigenvalue k^-1) in a random basis, then the LN profile
        Q = _basis()
        X = ((torch.randn(n, 4096, device=dev, generator=g) * torch.arange(1, 4097, device=dev).pow(-0.5)) @ Q) * LN
        X *= 4096 ** 0.5 / X.pow(2).mean().sqrt() * LN.pow(2).mean().sqrt() / 4096 ** 0.5
    if kind == "heavy":
        X *= torch.exp(0.5 * torch.randn(n, 1, device=dev, generator=g))
        idx = torch.randperm(4096, device=dev, generator=g)[:16]
        X[:, idx] *= 30
    return X

def H_data(X):
    X = X.float()
    return {"H": X.T @ X, "first_key": "k", "count": X.shape[0], "finalized": False, "num_total": X.numel(),
            "inf_nan": torch.zeros(2, dtype=torch.long, device=dev), "device": dev}

def quant(W_out_in, Hd, seed):
    qa = {"seed": seed, "K": 4, "devices": [0], "device_ratios": None, "apply_out_scales": None, "mcg": True}
    wq, perr, t = quantize_exl3(W_out_in.T.contiguous().float(), Hd, qa, True)
    return wq, perr, t, qa

LIMIT = float("inf")   # shared expert: production's SwiGLU limit for the shared path is not applied here

def swiglu(g, u): return torch.nn.functional.silu(g) * u

def e4m3_tok(x, group=None):
    if group is None: return emu._quant_tok(x)
    s = x.shape; return emu._quant_tok(x.reshape(*s[:-1], s[-1] // group, group)).reshape(s)

def two_term(x):
    """x ~= q1 + q2: q1 = per-row e4m3 of x, q2 = per-row e4m3 of the residual (two e4m3 mma per k step)."""
    q1 = emu._quant_tok(x); return q1 + emu._quant_tok(x - q1)

def proj(x, pk, mode, aq):
    """pk = (Wq fp16 [K,N] rotated domain, suh, svh). mode: 'prod' | 'e4m3' | 'f16w' (fp16 weights + aq)."""
    wq, suh, svh = pk
    xs = emu._rotate(x.half().float() * suh.float())
    if mode == "prod":
        y = xs.half().float() @ wq.float()
    else:
        w = wq.to(torch.float8_e4m3fn).float() if mode == "e4m3" else wq.float()
        y = aq(xs) @ w
    return emu._rotate(y) * svh.float()

def ffn_q(x, P, gu_mode, gu_aq, dn_mode, dn_aq, tp=2):
    g = proj(x, P["gate"], gu_mode, gu_aq); u = proj(x, P["up"], gu_mode, gu_aq)
    a = swiglu(g, u)
    if gu_mode == "prod":
        a = a.half().float()
    out = 0
    I = a.shape[1] // tp
    for r in range(tp):    # TP=2: each rank holds 1024 intermediate rows; the down input scale is per rank
        sl = slice(r * I, (r + 1) * I)
        wq, suh, svh = P["down"]
        out = out + proj(a[:, sl].half().float(), (wq[sl], suh[sl], svh), dn_mode, dn_aq)
    return out

def rel(a, b): return (torch.linalg.vector_norm((a - b).double()) / torch.linalg.vector_norm(b.double())).item()

print(f"torch {torch.__version__}  device {torch.cuda.get_device_name()}  tf32 {torch.backends.cuda.matmul.allow_tf32}",
      flush=True)
NCAL, NTE = 16384, 4096
rows = []
for L in [int(a) for a in (sys.argv[1] if len(sys.argv) > 1 else "10,3,23,43").split(",")]:
    W = shared(L)
    for kind in ("ln", "heavy", "spec"):
        t0 = time.time()
        Xc = acts(kind, NCAL, 1000 + L); Xt = acts(kind, NTE, 2000 + L)
        hd = H_data(Xc)
        Pq = {}; perrs = {}
        for p in ("gate", "up"):
            wq, perr, t, qa = quant(W[p], hd, seed=7)
            Pq[p] = (emu.decode_wq(t["trellis"]), t["suh"], t["svh"]); perrs[p] = perr
            # the decode of the packed trellis reproduces the quantizer's own reconstruction
            chk = rel(proj(Xt[:256], Pq[p], "prod", None), Xt[:256].half().float() @ wq.float())
            assert chk < 2e-2, (p, chk)
        Ac = swiglu(Xc @ W["gate"].T, Xc @ W["up"].T)
        wq, perr, t, qa = quant(W["down"], H_data(Ac), seed=7)
        Pq["down"] = (emu.decode_wq(t["trellis"]), t["suh"], t["svh"]); perrs["down"] = perr
        del Ac, Xc
        ref = swiglu(Xt.double() @ W["gate"].T.double(), Xt.double() @ W["up"].T.double()) @ W["down"].T.double()
        ref = ref.float()
        prod = ffn_q(Xt, Pq, "prod", None, "prod", None)
        V = {
            "prod (production EXL3 4bpw)": prod,
            "e4m3 (GLM53_MOE_E4M3 spec)": ffn_q(Xt, Pq, "e4m3", e4m3_tok, "e4m3", e4m3_tok),
            "e4m3 act groups of 32": ffn_q(Xt, Pq, "e4m3", lambda v: e4m3_tok(v, 32), "e4m3", lambda v: e4m3_tok(v, 32)),
            "e4m3 gate/up, fp16 down": ffn_q(Xt, Pq, "e4m3", e4m3_tok, "prod", None),
            "e4m3 w, 2-term act gate/up, e4m3 down": ffn_q(Xt, Pq, "e4m3", two_term, "e4m3", e4m3_tok),
            "e4m3 w, 2-term act gate/up + down": ffn_q(Xt, Pq, "e4m3", two_term, "e4m3", two_term),
            "e4m3 w, 2-term act gate/up, fp16 down": ffn_q(Xt, Pq, "e4m3", two_term, "prod", None),
            "e4m3 weights only (fp32 act)": ffn_q(Xt, Pq, "e4m3", lambda v: v, "e4m3", lambda v: v),
            "e4m3 activations only (fp16 w)": ffn_q(Xt, Pq, "f16w", e4m3_tok, "f16w", e4m3_tok),
        }
        ep = rel(prod, ref)
        print(f"\n== layer {L} shared expert, activations '{kind}' (cal {NCAL} / test {NTE} rows, {time.time()-t0:.0f} s)"
              f"  exllamav3 proxy_err gate {perrs['gate']:.5f} up {perrs['up']:.5f} down {perrs['down']:.5f}"
              f"  out-scales {qa['apply_out_scales']}", flush=True)
        print(f"   {'path':42s} {'vs BF16 ref':>11s} {'vs prod':>9s} {'added var r':>11s} {'= dbpw':>7s}")
        for name, y in V.items():
            e = rel(y, ref); ev = rel(y, prod); r = (e * e - ep * ep) / (ep * ep)
            dr = 0.5 * math.log2(1 + r) if r > -1 else float("nan")
            print(f"   {name:42s} {e:11.4%} {ev:9.4%} {r:11.3f} {dr:7.3f}", flush=True)
            rows.append(dict(layer=L, act=kind, path=name, vs_ref=e, vs_prod=ev, r=r, dbpw=dr))
        torch.cuda.empty_cache()
json.dump(rows, open("/w/docs/logs/moe2/exl3_vs_e4m3_noise.json", "w"), indent=1)
