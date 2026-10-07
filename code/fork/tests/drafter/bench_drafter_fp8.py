"""B1 + C1: drafter linears and the candidate lm_head, BF16 vs production-style FP8 Marlin, on nodeC (one GB10).

Weights: incoai/GLM-5.3-Flash-DFlash2 @ dc77ff1c (public HF, CC BY-NC-ND 4.0; local measurement only) and the target
lm_head.weight of Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw @ 9eaebb7c (public HF, Range-fetched). Both mounted read-only.
Shapes: TP=2 rank-0 shards as vLLM builds them (QKVParallelLinear / RowParallelLinear / MergedColumnParallelLinear /
ReplicatedLinear fc / ParallelLMHead vocab half).
BF16 path: vllm UnquantizedLinearMethod.apply (-> default_unquantized_gemm = F.linear), i.e. what the drafter runs today.
FP8 path: vllm.model_executor.layers.quantization.exl3.Glm53DenseFp8Method (process_weights_after_loading + apply),
the exact class GLM53_DENSE_FP8 uses for the target (per-output-channel e4m3, Marlin).
Timing: one CUDA graph per sequence (the 5 layers' 20 projections in execution order, or the 5 matrices of one kind),
replayed; the weights of one replay (>= 126 MB) evict L2 before the next, so every read is cold (production's verify
forward also evicts L2 between drafter steps). Each timing is the median of 15 reps (each rep = mean of `iters`
replays) and is printed with the [min-max] of those 15 reps. The candidate lm_head is timed in 3 independent trials
(new graph, new input) for every B = 1..8; its summary uses the median over trials.
The fc runs eagerly in production (combine_hidden_states, outside the drafter graph), so it is timed three ways:
graph replay, eager back-to-back calls (GPU time, host work overlapped) and eager one call at a time with a host sync
around each call (host dispatch of the Marlin wrapper included).
Asserts: Marlin output == x @ dequant(W)^T (rel <= 1e-2) and finite for every matrix; timings finite.
"""
import json
import os
import statistics
import sys
import time

import torch

avail = int(next(l for l in open("/proc/meminfo") if l.startswith("MemAvailable")).split()[1]) * 1024
assert avail > 40 * 2**30, f"host MemAvailable {avail / 2**30:.1f} GiB < 40"
torch.cuda.set_per_process_memory_fraction(min(1.0, 8 * 2**30 / torch.cuda.get_device_properties(0).total_memory))

from safetensors import safe_open  # noqa: E402

from vllm.model_executor.layers.linear import UnquantizedLinearMethod  # noqa: E402
from vllm.model_executor.layers.quantization.exl3 import Glm53DenseFp8Method  # noqa: E402

# The image's exl3.py has Glm53DenseFp8Method(group); the launcher overlay's has (group, prefix), and its
# process_weights_after_loading asks vLLM for the TP world size (a TP=3 gate), so a TP group must exist: a
# single-rank one here (the gate only differs at TP=3; the shards below are the TP=2 rank-0 shapes either way).
import inspect  # noqa: E402

_TAKES_PREFIX = "prefix" in inspect.signature(Glm53DenseFp8Method.__init__).parameters
print(f"production module Glm53DenseFp8Method{inspect.signature(Glm53DenseFp8Method.__init__)}", flush=True)


def fp8_method(group, prefix):
    return Glm53DenseFp8Method(group, prefix) if _TAKES_PREFIX else Glm53DenseFp8Method(group)


def _single_rank_tp():
    import socket

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
    from vllm.distributed import parallel_state as ps

    if ps.model_parallel_is_initialized():
        return
    with socket.socket() as s_:
        s_.bind(("127.0.0.1", 0))
        port = s_.getsockname()[1]
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                     distributed_init_method=f"tcp://127.0.0.1:{port}", backend="nccl")
        ensure_model_parallel_initialized(1, 1)


_single_rank_tp()

DRAFT = os.environ.get("DRAFT_DIR", os.path.join(os.environ.get("TF_EXL3_MODELS") or os.path.expanduser("~/models"), "GLM-5.3-Flash-DFlash2-dc77ff1c"))
LMH = os.environ.get("LMHEAD_DIR", os.path.join(os.environ.get("TF_EXL3_MODELS") or os.path.expanduser("~/models"), "GLM-5.3-Flash-EXL3-TR3-4bpw-partial/lm_head"))
dev = "cuda"
torch.manual_seed(0)
FAIL = []
cfg = json.load(open(os.path.join(DRAFT, "config.json")))
H, NH, NKV, HD, I, NL = (cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"],
                         cfg["intermediate_size"], cfg["num_hidden_layers"])
TP = 2
qh, kvh, ih = NH // TP * HD, NKV // TP * HD, I // TP
print(f"drafter: hidden {H}, heads {NH}/{NKV} x {HD}, MLP {I}, {NL} layers; TP=2 rank-0 shards: q {qh}, kv {kvh}, MLP {ih}")


class L(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.weight = torch.nn.Parameter(w, requires_grad=False)
        self.output_size_per_partition, self.input_size_per_partition = w.shape
        self.bias = None


def fp8_of(w, want_deq=True):
    """(layer quantized by the production class, dequantized reference or None)."""
    lay = L(w.clone())
    m = fp8_method("draft", "model.layers.45.mlp.down_proj")
    m.process_weights_after_loading(lay)
    assert lay.weight.dtype == torch.float8_e4m3fn or lay.weight.dtype == torch.int32, lay.weight.dtype
    torch.cuda.empty_cache()
    if not want_deq:
        return lay, m, None
    wf = w.float()
    s = wf.abs().amax(dim=1).clamp(min=1e-12) / 448.0
    deq = (wf / s[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn).float() * s.to(w.dtype).float()[:, None]
    return lay, m, deq


BF = UnquantizedLinearMethod()

# ------------------------------------------------------------------ rank-0 drafter weights
mats = []  # (layer, kind, W[N,K] bf16 on GPU)
with safe_open(os.path.join(DRAFT, "model.safetensors"), framework="pt", device="cpu") as f:
    g = lambda n: f.get_tensor(n)
    for l in range(NL):
        p = f"layers.{l}."
        q, k, v = g(p + "self_attn.q_proj.weight"), g(p + "self_attn.k_proj.weight"), g(p + "self_attn.v_proj.weight")
        mats.append((l, "qkv_proj", torch.cat([q[:qh], k[:kvh], v[:kvh]]).to(dev)))
        mats.append((l, "o_proj", g(p + "self_attn.o_proj.weight")[:, :qh].contiguous().to(dev)))
        ga, up = g(p + "mlp.gate_proj.weight"), g(p + "mlp.up_proj.weight")
        mats.append((l, "gate_up_proj", torch.cat([ga[:ih], up[:ih]]).to(dev)))
        mats.append((l, "down_proj", g(p + "mlp.down_proj.weight")[:, :ih].contiguous().to(dev)))
    fc = g("fc.weight").to(dev)
    norm_w = g("norm.weight").to(dev)
KINDS = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


def acts(M, k, outl):
    x = torch.randn(M, k, device=dev)
    if outl:
        idx = torch.randperm(k, device=dev)[: max(1, k // 256)]
        x[:, idx] *= 30.0
    return x.to(torch.bfloat16)


# ------------------------------------------------------------------ accuracy on the real weights
print("\n== accuracy, real drafter weights, M=8 rows (rel. Frobenius error of the output vs BF16 weights)")
q8 = []
acc_rows = []
for (l, kind, w) in mats + [(-1, "fc", fc)]:
    lay, m, deq = fp8_of(w)
    r = {"layer": l, "kind": kind, "shape": tuple(w.shape)}
    for outl in (False, True):
        x = acts(8, w.shape[1], outl)
        y = m.apply(lay, x)
        ok = torch.isfinite(y).all().item()
        r["out" + ("_o" if outl else "")] = rel(y, x.float() @ w.float().t())
        r["kvd" + ("_o" if outl else "")] = rel(y, x.float() @ deq.t())
        if not ok or r["kvd" + ("_o" if outl else "")] > 1e-2:
            FAIL.append(f"Marlin mismatch {kind} layer {l} outliers={outl}: {r['kvd' + ('_o' if outl else '')]:.3e} finite={ok}")
    acc_rows.append(r)
    q8.append((l, kind, w, lay, m))
for kind in KINDS + ("fc",):
    v = [r for r in acc_rows if r["kind"] == kind]
    print(f"  {kind:12s} {str(v[0]['shape']):>14s} x{len(v)}: out_rel median/max {statistics.median(r['out'] for r in v):.3e}/{max(r['out'] for r in v):.3e}"
          f"  with 1/256 outlier channels x30 {statistics.median(r['out_o'] for r in v):.3e}/{max(r['out_o'] for r in v):.3e}"
          f"  | Marlin vs dequant max {max(max(r['kvd'], r['kvd_o']) for r in v):.2e}")
allmax = max(r["out"] for r in acc_rows if r["kind"] != "fc")
print(f"  drafter layers max out_rel {allmax:.3e}; production-accepted target groups (docs/logs/fp8_groups_check.log): "
      "kda 2.79e-2/2.84e-2, dense 2.28e-2/2.30e-2 (median/max)")


# ------------------------------------------------------------------ timing helpers
def graph_ms(fns, reps=15, iters=40):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            for fn in fns:
                fn()
    torch.cuda.current_stream().wait_stream(s)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for fn in fns:
            fn()
    gr.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(iters):
            gr.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) / iters)
    del gr
    torch.cuda.empty_cache()
    return T(ts)


class T(float):
    """A median timing (ms) that also remembers the spread of its reps."""

    def __new__(cls, ts):
        self = super().__new__(cls, statistics.median(ts))
        self.lo, self.hi = min(ts), max(ts)
        return self

    def s(self, scale=1.0, fmt="{:.3f}"):
        return (fmt + " [" + fmt + "-" + fmt + "]").format(self * scale, self.lo * scale, self.hi * scale)


def eager_ms(fns, reps=15, iters=40):
    for fn in fns:
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(iters):
            for fn in fns:
                fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) / iters)
    return T(ts)


def eager_sync_ms(fns, reps=15, iters=40):
    """Wall clock of one pass over fns with a device sync before and after: host dispatch + GPU, serialized."""
    for fn in fns:
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        acc = 0.0
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for fn in fns:
                fn()
            torch.cuda.synchronize()
            acc += time.perf_counter() - t0
        ts.append(acc / iters * 1000)
    return T(ts)


def seq(entries, M, mode):
    """entries: [(w_bf16, fp8_layer, fp8_method)] -> list of callables, one input per matrix."""
    fns = []
    for (w, lay, m) in entries:
        x = torch.randn(M, w.shape[1], device=dev).to(torch.bfloat16)
        if mode == "bf16":
            lb = L(w)
            fns.append(lambda lb=lb, x=x: BF.apply(lb, x))
        else:
            fns.append(lambda lay=lay, m=m, x=x: m.apply(lay, x))
    return fns


results = {}
print("\n== drafter decoder layers (5 x qkv/o/gate_up/down, rank-0 TP=2), one forward on M = 8*B rows, CUDA-graph replay, cold weights")
bytes_bf16 = sum(w.numel() * 2 for (l, _, w, _, _) in q8 if l >= 0)
bytes_fp8 = bytes_bf16 / 2
print(f"  bytes per forward per rank: BF16 {bytes_bf16 / 1e9:.3f} GB, FP8 {bytes_fp8 / 1e9:.3f} GB (+ per-channel scales)")
layer_entries = [(w, lay, m) for (l, _, w, lay, m) in q8 if l >= 0]
for B in (1, 2, 4, 8):
    M = 8 * B
    tb = graph_ms(seq(layer_entries, M, "bf16"))
    tf = graph_ms(seq(layer_entries, M, "fp8"))
    per_kind = []
    for kind in KINDS:
        ents = [(w, lay, m) for (l, k, w, lay, m) in q8 if k == kind]
        kb, kf = graph_ms(seq(ents, M, "bf16")), graph_ms(seq(ents, M, "fp8"))
        per_kind.append(f"{kind} {kb:.3f}->{kf:.3f}")
    results[("layers", B)] = (tb, tf)
    print(f"  B={B} M={M:3d}: BF16 {tb.s()} ms ({bytes_bf16 / tb / 1e6:.0f} GB/s)  FP8 {tf.s()} ms ({bytes_fp8 / tf / 1e6:.0f} GB/s)"
          f"  saved {tb - tf:+.3f} ms/step\n      per kind, each the sum of its 5 matrices (one per layer), BF16->FP8 ms: [" + ", ".join(per_kind) + "]")
    if not all(map(lambda t: t == t and t > 0, (tb, tf))):
        FAIL.append(f"non-finite timing B={B}")
tb, tf = eager_ms(seq(layer_entries, 8, "bf16")), eager_ms(seq(layer_entries, 8, "fp8"))
print(f"  (eager, no graph, B=1: BF16 {tb.s()} ms, FP8 {tf.s()} ms, saved {tb - tf:+.3f} ms)")

print("\n== fc 20480 -> 4096 (replicated; runs EAGERLY in production, on the target's 1+n tokens per request: M = Tk*B)")
fcl, fcm = q8[-1][3], q8[-1][4]
for Tk in (5, 8):
    for B in (1, 8):
        M = Tk * B
        line = []
        for how, fnt in (("graph", lambda f: graph_ms(f, iters=200)), ("eager", lambda f: eager_ms(f, iters=200)),
                         ("eager+sync", lambda f: eager_sync_ms(f, iters=100))):
            tb = fnt(seq([(fc, fcl, fcm)] * 1, M, "bf16"))
            tf = fnt(seq([(fc, fcl, fcm)] * 1, M, "fp8"))
            results[("fc", how, Tk, B)] = (tb, tf)
            line.append(f"{how}: BF16 {tb.s(1000, '{:.1f}')} FP8 {tf.s(1000, '{:.1f}')} saved {(tb - tf) * 1000:+.1f}")
            if not all(map(lambda t: t == t and t > 0, (tb, tf))):
                FAIL.append(f"non-finite fc timing {how} T={Tk} B={B}")
        print(f"  T={Tk} B={B} M={M:3d} (us): " + " | ".join(line))
del mats, q8, acc_rows, layer_entries, fcl, fcm, fc, w, lay, m, deq, x, y, ents
import gc  # noqa: E402

gc.collect()
torch.cuda.empty_cache()
print(f"  [memory] allocated after the layer benchmarks: {torch.cuda.memory_allocated() / 2**30:.2f} GiB")

# ================================================================== C1: candidate lm_head copy
print("\n== C: candidate lm_head (target's BF16 lm_head, vocab half per rank = 77440 x 4096), drafter M = 7*B rows")
man = json.load(open(os.path.join(LMH, "manifest.json")))
Vv, Hh = man["shape"]
raw = torch.from_file(os.path.join(LMH, "lm_head.weight.bin"), shared=False, size=Vv * Hh, dtype=torch.bfloat16)
Wl = raw.view(Vv, Hh)
halves = [Wl[: Vv // 2].to(dev), Wl[Vv // 2:].to(dev)]
del raw, Wl
l0, m0, _ = fp8_of(halves[0], want_deq=False)
torch.cuda.empty_cache()
lm_bytes = halves[0].numel() * 2
TRIALS = 3
for B in range(1, 9):
    M = 7 * B
    tr_b, tr_f = [], []
    for _ in range(TRIALS):   # independent trials: new graph, new input
        tr_b.append(graph_ms(seq([(halves[0], l0, m0)], M, "bf16"), iters=40))
        tr_f.append(graph_ms(seq([(halves[0], l0, m0)], M, "fp8"), iters=40))
    tb, tf = statistics.median(tr_b), statistics.median(tr_f)
    results[("lm_head", B)] = (tb, tf)
    print(f"  B={B} M={M:3d}: BF16 {tb:.3f} ms ({lm_bytes / tb / 1e6:.0f} GB/s)  FP8 {tf:.3f} ms ({lm_bytes / 2 / tf / 1e6:.0f} GB/s)  saved {tb - tf:+.3f} ms/step"
          f"   trials BF16 " + ", ".join(t.s() for t in tr_b) + " | FP8 " + ", ".join(t.s() for t in tr_f))
    if not all(t == t and t > 0 for t in tr_b + tr_f):
        FAIL.append(f"non-finite lm_head timing B={B}")
fp8_bytes = l0.weight.numel() * l0.weight.element_size() + l0.weight_scale.numel() * l0.weight_scale.element_size() + l0.workspace.numel() * l0.workspace.element_size()
print(f"  memory of the FP8 copy per rank: {fp8_bytes / 2**20:.1f} MiB (packed weight {l0.weight.numel() * l0.weight.element_size() / 2**20:.1f} MiB,"
      f" scales {l0.weight_scale.numel() * l0.weight_scale.element_size() / 2**10:.1f} KiB, workspace {l0.workspace.numel() * l0.workspace.element_size() / 2**10:.1f} KiB)")

print("\n== C: candidate quality (indicative; synthetic hidden states, both vocab halves = the TP-gathered logits)")
gc.collect()
torch.cuda.empty_cache()
print(f"  [memory] allocated before quantizing the second half: {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
l1, m1, _ = fp8_of(halves[1], want_deq=False)
torch.cuda.empty_cache()


def logits(h, mode):
    if mode == "bf16":
        return torch.cat([BF.apply(L(halves[0]), h), BF.apply(L(halves[1]), h)], dim=-1).float()
    return torch.cat([m0.apply(l0, h), m1.apply(l1, h)], dim=-1).float()


def rmsnorm(x):
    return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5) * norm_w.float()).to(torch.bfloat16)


Wfull = torch.cat(halves)
for label, make in (
    ("random hidden (norm(z) * drafter norm.weight)", lambda n: rmsnorm(torch.randn(n, Hh, device=dev))),
    ("peaked hidden (aligned with a random token's lm_head row)",
     lambda n: rmsnorm(Wfull[torch.randint(0, Vv, (n,), device=dev)].float() * 40 + torch.randn(n, Hh, device=dev))),
):
    top1 = ov = same = tvs = 0.0
    n_tot = 0
    for _ in range(8):
        h = make(512)
        a, b = logits(h, "bf16"), logits(h, "fp8")
        ta, tb_ = a.topk(16, dim=-1).indices, b.topk(16, dim=-1).indices
        top1 += (ta[:, 0] == tb_[:, 0]).float().sum().item()
        inter = (ta[:, :, None] == tb_[:, None, :]).any(-1).float().sum(-1)
        ov += inter.sum().item() / 16
        same += (inter == 16).float().sum().item()
        tvs += 0.5 * (a.softmax(-1) - b.softmax(-1)).abs().sum(-1).sum().item()
        n_tot += h.shape[0]
    pmax = a.softmax(-1).max(-1).values.median().item()
    print(f"  {label}: median max-prob {pmax:.3f} | top-1 agreement {top1 / n_tot:.4f} | top-16 set overlap {ov / n_tot:.4f}"
          f" | identical top-16 sets {same / n_tot:.4f} | softmax TV {tvs / n_tot:.4f}")

print("\n== summary: ms saved per speculative step per rank (BF16 - FP8, graph replay, cold; lm_head = median of 3 trials)")
for B in (1, 2, 4, 8):
    lb, lf = results[("layers", B)]
    hb, hf = results[("lm_head", B)]
    print(f"  B={B}: decoder layers {lb - lf:+.3f} ms | candidate lm_head copy {hb - hf:+.3f} ms | both {lb - lf + hb - hf:+.3f} ms")
sv = [results[("lm_head", B)][0] - results[("lm_head", B)][1] for B in range(1, 9)]
print(f"  candidate lm_head copy over B=1..8: median {statistics.median(sv):.3f} ms, range {min(sv):.3f}-{max(sv):.3f} ms")
for Tk, B in ((5, 1), (8, 8)):
    print(f"  fc (T={Tk}, B={B}), saved us: " + ", ".join(
        f"{how} {(results[('fc', how, Tk, B)][0] - results[('fc', how, Tk, B)][1]) * 1000:+.1f}" for how in ("graph", "eager", "eager+sync")))
if torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()
print("ALL PASSED" if not FAIL else "FAILED:\n  " + "\n  ".join(FAIL))
sys.exit(1 if FAIL else 0)
