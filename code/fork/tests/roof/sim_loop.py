"""Node1 simulator of production's per-step EAGER loop around the target's CUDA graph, for the eager-anchor triggers
t0 / t5 of GLM53_DEC_FP8ROOF (fp8_roof.py, docs/DEC_FP8ROOF.md), through the real integration (the launcher overlay's
Glm53DenseFp8Method + fp8_gemv wrappers + fp8_roof hooks). One step, in production's order (R15 trace, rank 0):
  drafter fc        4096 x 20480 FP8 ("draft", "model.fc"; fp8_gemv with the roof TABLE entry), eager, hooked
  drafter body      5 x (qkv 3072x4096, o 4096x2048, gate_up 12288x4096, down 4096x6144 FP8 GEMVs + attention /
                    all-reduce / conv spins); production runs it torch.compiled: no roof role, no fork
  drafter lm_head   77440 x 4096 FP8 built like glm53_runtime.convert_lm_head_fp8 (no process_weights_after_loading:
                    classified at its first call), eager, hooked                                        -> t0
  drafter tail      logits all-gather 86 us + top-k / prepare 40 us, then the host step loop H us (a GPU spin: DRAM
                    idle, like the host gap it stands for)
  target            ONE CUDA graph (captured once, no prefetch inside): embedding all-reduce 130 us, hc_prenorm 14 us,
                    layer-0 KDA in_proj 12576 x 4096 (the t0 target), then NB x 51 MB FP8 GEMVs + spins (the rest of
                    the forward, enough to cycle L2), final all-reduce 40 us
  target lm_head    the same lm_head object, eager, hooked                                              -> t5
  sampler tail      logits all-gather 60 us + ~100 us of small kernels (spins) -> next step's fc
Configs (ROOF_LOOP_CONFIGS, ';'-separated 'name:triggers:mib', the first must be off): paired rounds of S steps,
alternating order; ms per step (CUDA events), then CUPTI per-call fc / layer-0 in_proj / lm_head durations for
the first two configs.
Usage: sim_loop.py [M=5] [H=1000] [NB=8] [S=4] [rounds=15] [prof_rounds=4]
"""
import json
import os
import statistics
import sys
import tempfile
import types
from collections import defaultdict
from pathlib import Path

R_ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(R_)); sys.path.insert(0, str(R_ / "tests"))
import torch  # noqa: E402

a = [int(v) for v in sys.argv[1:]]
M, H, NB, S, ROUNDS, PROF = (a + [5, 1000, 8, 4, 15, 4][len(a):])[:6]
dev = "cuda"
for v in ("GLM53_FP8_GEMV", "GLM53_DEC_FP8ROOF"):
    os.environ[v] = "1"
os.environ.setdefault("GLM53_FP8_GEMV_MAX_M", "16")


def single_rank_tp():
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


single_rank_tp()
import vllm.model_executor.layers.quantization.exl3 as prod  # noqa: E402
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_layer_for_marlin  # noqa: E402
import fp8_gemv as F  # noqa: E402
import fp8_roof as RF  # noqa: E402
import tf_exl3_moe as T  # noqa: E402
from torch.utils.cpp_extension import load_inline  # noqa: E402

inc = T._cuda_include_shim()
SPN = load_inline("roofspin3", cpp_sources="void spin(int64_t,int64_t);", cuda_sources=r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
__global__ void spin_k(long long c) { long long t0 = clock64(); while (clock64() - t0 < c) {} }
void spin(int64_t c, int64_t b) { spin_k<<<(unsigned)b, 128, 0, at::cuda::getCurrentCUDAStream()>>>(c); }
''', functions=["spin"], extra_cuda_cflags=["-O3", *inc], extra_cflags=[*inc])
e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
SPN.spin(1000, 48); torch.cuda.synchronize()
e0.record(); SPN.spin(2_000_000, 48); e1.record(); torch.cuda.synchronize()
CYC = 2_000_000 / (e0.elapsed_time(e1) * 1000)


def spin(us, blocks=48):
    SPN.spin(int(us * CYC), blocks)


class FakeMoE:
    def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        return x


assert F.install(prod)["installed"]
rep = RF.install(types.SimpleNamespace(Exl3MoEMethod=FakeMoE, __file__="<sim>"))
print(f"fp8_roof: {rep}", flush=True)
cls = prod.Glm53DenseFp8Method
g = torch.Generator(device=dev).manual_seed(0)


class L(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.weight = torch.nn.Parameter(w, requires_grad=False)
        self.output_size_per_partition, self.input_size_per_partition = w.shape
        self.bias = None


def build(n, k, group, prefix):
    w = (torch.randn(n, k, device=dev, generator=g) * 0.02).to(torch.bfloat16)
    lay, m = L(w), cls(group, prefix)
    m.process_weights_after_loading(lay)
    del w
    return lay, m


def lm_head_like(n, k):
    """glm53_runtime.convert_lm_head_fp8: FP8 Marlin holder + Glm53DenseFp8Method("lm_head", "lm_head"), ready."""
    h = torch.nn.Module()
    h.output_size_per_partition, h.input_size_per_partition = n, k
    h.orig_dtype = torch.bfloat16
    fp8 = torch.empty(n, k, dtype=torch.float8_e4m3fn, device=dev)
    sc = torch.empty(n, dtype=torch.float32, device=dev)
    for a_ in range(0, n, 8192):
        wf = torch.randn(min(8192, n - a_), k, device=dev, generator=g) * 0.02
        s_ = wf.abs().amax(dim=1).clamp(min=1e-12) / 448.0
        fp8[a_:a_ + 8192] = (wf / s_[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        sc[a_:a_ + 8192] = s_
    h.weight = torch.nn.Parameter(fp8, requires_grad=False)
    h.weight_scale = torch.nn.Parameter(sc.to(torch.bfloat16), requires_grad=False)
    h.weight_block_size = None
    prepare_fp8_layer_for_marlin(h, size_k_first=False)
    h.glm53_fp8_n, h.glm53_fp8_k = n, k
    mm = cls("lm_head", "lm_head")
    mm.ready = True
    return h, mm


fc = build(4096, 20480, "draft", "model.fc")
drafter = [(build(3072, 4096, "sim", f"sim.d{i}.qkv"), build(4096, 2048, "sim", f"sim.d{i}.o"),
            build(12288, 4096, "sim", f"sim.d{i}.gu"), build(4096, 6144, "sim", f"sim.d{i}.dn")) for i in range(5)]
lm = lm_head_like(77440, 4096)
in0 = build(12576, 4096, "kda", "model.layers.0.self_attn.in_proj_qkvbfg_a")
body = [build(12288, 4096, "sim", f"sim.t{i}") for i in range(NB)]
torch.cuda.empty_cache()
assert RF.STATE.reg.get(("fc", -1)) is fc[0].weight and RF.STATE.reg.get(("kda_in", 0)) is in0[0].weight


def lin(lm_, h):
    return lm_[1].apply(lm_[0], h)


xf = (torch.randn(M, 20480, device=dev, generator=g) * 0.5).to(torch.bfloat16)
xh = (torch.randn(M, 4096, device=dev, generator=g) * 0.5).to(torch.bfloat16)
xa = (torch.randn(M, 2048, device=dev, generator=g) * 0.5).to(torch.bfloat16)
xd = (torch.randn(M, 6144, device=dev, generator=g) * 0.5).to(torch.bfloat16)

# the target: one graph, captured with no roof trigger active (nothing is forked inside it in any config)
RF.STATE.triggers = frozenset()
tx = xh.clone()


def target(h):
    spin(130); spin(14)
    y = lin(in0, h)[:, :4096]
    for b in body:
        spin(40)
        y = lin(b, y)[:, :4096] * 0.5 + h
    spin(40)
    return y


s0 = torch.cuda.Stream(); s0.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s0):
    target(tx)
torch.cuda.current_stream().wait_stream(s0); torch.cuda.synchronize()
RF.STATE.pending.clear()
tg = torch.cuda.CUDAGraph()
with torch.cuda.graph(tg):
    ty = target(tx)
torch.cuda.synchronize()


def step():
    lin(fc, xf)                                           # drafter fc (joins t5)
    for qkv, o, gu, dn in drafter:
        spin(40); lin(qkv, xh); spin(20); lin(o, xa); spin(30); spin(40); lin(gu, xh); lin(dn, xd); spin(30)
    lin(lm, xh)                                           # drafter lm_head  -> t0
    spin(86, 8); spin(40); spin(H)                        # all-gather, top-k / prepare, host step loop
    tg.replay()                                           # target (graph: no Python)
    lin(lm, xh)                                           # target lm_head   -> t5
    spin(60, 8); spin(100)                                # all-gather, sampler


def parse(spec):
    parts = spec.split(":")
    name = parts[0]
    trig = parts[1] if len(parts) > 1 else name
    trig = frozenset() if trig == "off" else frozenset(trig.split(","))
    mib, why = RF._parse_mib(parts[2] if len(parts) > 2 else "")
    assert why is None, why
    return name, trig, mib


cfgs = [parse(c) for c in os.environ.get("ROOF_LOOP_CONFIGS", "off;on:t0,t5").split(";")]
assert not cfgs[0][1], "the first config must be off"


def use(cfg):
    _, trig, mib = cfg
    RF.STATE.triggers, RF.STATE.mib = trig, mib
    RF.STATE.pf = True


for cfg in cfgs:                                          # warm-up: every config once, and the lazy lm_head key
    use(cfg)
    RF.COUNTERS.clear()
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    print(f"warm-up {cfg[0]:>8s} (triggers {sorted(cfg[1])}, MiB t0 {cfg[2]['t0']} t5 {cfg[2]['t5']}): counters "
          f"{dict(sorted(RF.COUNTERS.items()))}", flush=True)
res = {c[0]: [] for c in cfgs}
for r in range(ROUNDS):
    for cfg in (cfgs if r % 2 == 0 else cfgs[::-1]):
        use(cfg)
        step()                                            # settle the order-dependent state (pending, lm count)
        e0.record()
        for _ in range(S):
            step()
        e1.record(); torch.cuda.synchronize()
        res[cfg[0]].append(e0.elapsed_time(e1) / S)
off = statistics.median(res[cfgs[0][0]])
print(f"M={M}, host gap H={H} us, target body {NB} x 50.3 MB, L2 "
      f"{torch.cuda.get_device_properties(0).L2_cache_size / 2**20:.0f} MiB, {ROUNDS} rounds x {S} steps; ms per step "
      f"(median), saving vs off, paired ratio median [min, max]")
for cfg in cfgs:
    v = res[cfg[0]]
    m = statistics.median(v)
    rr = [b / a_ for a_, b in zip(res[cfgs[0][0]], v)]
    print(f"  {cfg[0]:>8s}: {m:8.3f} ms  saving {(off - m) * 1000:7.1f} us  ratio {statistics.median(rr):.4f} "
          f"[{min(rr):.4f}, {max(rr):.4f}]  spread {(max(v) - min(v)) / m * 100:.1f}%", flush=True)

if PROF and len(cfgs) > 1:
    from torch.profiler import ProfilerActivity, profile
    A, B = cfgs[0], cfgs[1]
    sched = []
    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        for r in range(PROF):
            for cfg in ((A, B) if r % 2 == 0 else (B, A)):
                use(cfg)
                for _ in range(3):
                    step()
                    sched.append(cfg[0])
        torch.cuda.synchronize()
    tmp = tempfile.mkdtemp()
    pr.export_chrome_trace(f"{tmp}/loop.json")
    ev = json.load(open(f"{tmp}/loop.json"))["traceEvents"]
    ks = sorted([e for e in ev if e.get("cat") == "kernel" and e.get("ph") == "X"], key=lambda e: e["ts"])
    # per step: the fc is the first 4096-row fp8 kernel with grid of the fc; classify by name + grid: the first fp8
    # kernel of a step is the fc, the lm_head kernels have the largest grid, layer-0 in_proj is the fp8 kernel right after
    # the 130 + 14 us spins inside the graph (the first fp8 kernel after the first lm_head of the step)
    fp8 = [e for e in ks if "fp8_gemv" in e["name"] or "Marlin" in e["name"]]
    grid = lambda e: tuple(e["args"].get("grid") or ())   # noqa: E731
    lm_grid = max((grid(e) for e in fp8), key=lambda gr_: gr_[0])
    steps = []
    cur = None
    for e in fp8:
        if cur is None or (cur.get("lmT") is not None and grid(e) != lm_grid):
            cur = {"fc": e["dur"], "lmD": None, "in0": None, "lmT": None}
            steps.append(cur)
            continue
        if grid(e) == lm_grid:
            if cur["lmD"] is None:
                cur["lmD"] = e["dur"]
            else:
                cur["lmT"] = e["dur"]
        elif cur["lmD"] is not None and cur["in0"] is None:
            cur["in0"] = e["dur"]
    assert len(steps) == len(sched), f"{len(steps)} steps in the trace vs {len(sched)} run"
    per = {A[0]: defaultdict(list), B[0]: defaultdict(list)}
    for st_, name in zip(steps, sched):
        for k_, v_ in st_.items():
            if v_ is not None:
                per[name][k_].append(v_)
    NBY = {"fc": 4096 * 20480, "lmD": 77440 * 4096, "in0": 12608 * 4096, "lmT": 77440 * 4096}
    print(f"per-call kernel time (CUPTI), {PROF} rounds x 3 steps per config; median us (GB/s of the weight bytes)")
    for k_ in ("fc", "lmD", "in0", "lmT"):
        va, vb = per[A[0]][k_], per[B[0]][k_]
        ma, mb = statistics.median(va), statistics.median(vb)
        print(f"  {k_:>4s}: {A[0]} {ma:8.1f} ({NBY[k_] / ma / 1e3:5.1f})  {B[0]} {mb:8.1f} ({NBY[k_] / mb / 1e3:5.1f})  "
              f"saving {ma - mb:6.1f} us")
    pfk = [e["dur"] for e in ks if "l2_prefetch" in e["name"]]
    print(f"  l2_prefetch kernels: {len(pfk)} in {PROF * 3} {B[0]} steps, median {statistics.median(pfk) if pfk else 0:.1f} us")
