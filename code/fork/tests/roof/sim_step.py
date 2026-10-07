"""Node1 decode-step simulator for GLM53_DEC_FP8ROOF's L2 prefetch, through the REAL integration (production's
Glm53DenseFp8Method from the launcher overlay + fp8_gemv wrappers + fp8_roof hooks; a stand-in Exl3MoEMethod class
wrapped the same way). Every FP8 linear runs at its production shape, config and prefix; the latency-bound work
between them is replaced by stand-ins with the production trace's median durations (R15 p6h, rank 0, prose):
  attention : AR 25 us | mHC 12 + 6 us  (ROOF_SIM_ATTN_PRE=1, the default; 0 = the first session's simulator)
  KDA layer : in_proj | f_b, g_b | conv 7 us | recurrent: read 2 MiB state + write D MiB per-token states, 24 us |
              norm 3 us | o_proj
  MLA layer : fused_qkv_a | norm 2 us | q_b | attention: read A MiB of BF16 weights/KV + 150 us | o_proj
  MoE block : AR 25 us | mHC 13 + 6 us | router (2.4 MB read) | topk 3 us | rot_in 24 us | routed experts: evict_first
              streaming of R MiB (FP8 GEMVs, pol on) | shared gate_up + down on an aux stream (vLLM's runner order) |
              epilogue 8 us
  dense MLP : AR 25 us | mHC 19 us | gate_up | act 4 us | down
Layers 0-2 KDA + dense, then (MLA, KDA, KDA, KDA) + MoE repeated: NL layers. One forward (M rows) captured twice
(prefetch on / off), replayed in alternating rounds; ms per forward, medians, paired ratio.
Usage: sim_step.py [NL=15] [M=5] [D=10] [A=16] [R=160] [rounds=15]   (env GLM53_DEC_FP8ROOF_PF etc. as production;
       ROOF_SIM_STANDIN=spin|chase|mix: what the latency-bound stand-ins are made of, see spin())
"""
import inspect
import os
import statistics
import sys
import types
from pathlib import Path

R_ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(R_)); sys.path.insert(0, str(R_ / "tests"))
import torch  # noqa: E402

a = [int(v) for v in sys.argv[1:]]
NL, M, D, A, RM, ROUNDS = (a + [15, 5, 10, 16, 160, 15][len(a):])[:6]
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
import fp8_gemv as F  # noqa: E402
import fp8_roof as RF  # noqa: E402
import tf_exl3_moe as T  # noqa: E402
from torch.utils.cpp_extension import load_inline  # noqa: E402

inc = T._cuda_include_shim()
S = load_inline("roofspin3", cpp_sources="void spin(int64_t,int64_t); void chase(torch::Tensor,int64_t,int64_t,int64_t);", cuda_sources=r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
__global__ void spin_k(long long c) { long long t0 = clock64(); while (clock64() - t0 < c) {} }
void spin(int64_t c, int64_t b) { spin_k<<<(unsigned)b, 128, 0, at::cuda::getCurrentCUDAStream()>>>(c); }
// ROOF_SIM_STANDIN=chase stand-in: a DRAM-latency-bound kernel (dependent loads over a buffer >> L2), lane 0 of each block chases
// its own random cycle; small L2 footprint (48 chains). Sensitive to loaded DRAM latency, unlike spin.
__global__ void chase_k(const int* __restrict__ buf, long long nlines, long long steps, int salt, int* out) {
  if (threadIdx.x != 0) return;
  unsigned long long s = ((unsigned long long)(blockIdx.x + 1) * 2654435761ull + (unsigned long long)salt * 40503ull) % (unsigned long long)nlines;
  int idx = (int)(s * 32);
  for (long long i = 0; i < steps; ++i) idx = __ldcg(buf + idx);
  if (idx == -7) out[0] = idx;
}
void chase(torch::Tensor buf, int64_t steps, int64_t blocks, int64_t salt) {
  chase_k<<<(unsigned)blocks, 32, 0, at::cuda::getCurrentCUDAStream()>>>(buf.data_ptr<int>(), buf.numel() / 32, steps, (int)salt, nullptr);
}
''', functions=["spin", "chase"], extra_cuda_cflags=["-O3", *inc], extra_cflags=[*inc])
e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
S.spin(1000, 48); torch.cuda.synchronize()
e0.record(); S.spin(2_000_000, 48); e1.record(); torch.cuda.synchronize()
CYC = 2_000_000 / (e0.elapsed_time(e1) * 1000)


# ROOF_SIM_STANDIN (adversarial review): what the latency-bound windows are made of. spin (default: clock64 spins, blind
# to DRAM contention, the optimistic end) | chase (dependent loads over a 256 MiB buffer: DRAM-latency-bound, slowed by
# a concurrent prefetch, the pessimistic end) | mix (half spin, half chase).
STANDIN = os.environ.get("ROOF_SIM_STANDIN", "spin")
assert STANDIN in ("spin", "chase", "mix"), STANDIN
if STANDIN != "spin":
    NLINES = (256 << 20) // 128
    _perm = torch.randperm(NLINES, device=dev, dtype=torch.int64)
    CBUF = torch.zeros(NLINES * 32, dtype=torch.int32, device=dev)
    CBUF[_perm * 32] = (torch.roll(_perm, -1) * 32).to(torch.int32)
    del _perm
    S.chase(CBUF, 100, 48, 1); torch.cuda.synchronize()
    e0.record(); S.chase(CBUF, 4000, 48, 2); e1.record(); torch.cuda.synchronize()
    LAT_US = e0.elapsed_time(e1) * 1000 / 4000
    print(f"stand-in {STANDIN}: unloaded dependent-load latency {LAT_US * 1000:.0f} ns/step", flush=True)
_salt = [0]


def spin(us, blocks=48):
    if STANDIN == "spin":
        S.spin(int(us * CYC), blocks)
        return
    f = 1.0 if STANDIN == "chase" else 0.5
    if f < 1.0:
        S.spin(int(us * (1 - f) * CYC), blocks)
    _salt[0] += 1
    S.chase(CBUF, max(1, int(us * f / LAT_US)), 48, _salt[0])


class FakeMoE:
    def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        return layer.moe_fn(x)


assert F.install(prod)["installed"]
rep = RF.install(types.SimpleNamespace(Exl3MoEMethod=FakeMoE, __file__="<sim>"))
print(f"fp8_roof: {rep}", flush=True)
cls = prod.Glm53DenseFp8Method
assert "prefix" in inspect.signature(cls.__init__).parameters, "bind the launcher overlay exl3.py (GPU_RUN_BIND)"
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


kinds = ["kda"] * 3 + [("mla", "kda", "kda", "kda")[(i - 3) % 4] for i in range(3, NL)]
W = {}
for Lx, kind in enumerate(kinds):
    p = f"model.layers.{Lx}."
    if kind == "kda":
        W[(Lx, "in")] = build(12576, 4096, "kda", p + "self_attn.in_proj_qkvbfg_a")
        W[(Lx, "fb")] = build(4096, 128, "kda", p + "self_attn.f_b_proj")
        W[(Lx, "gb")] = build(4096, 128, "kda", p + "self_attn.g_b_proj")
        W[(Lx, "o")] = build(4096, 4096, "kda", p + "self_attn.o_proj")
    else:
        W[(Lx, "qkv_a")] = build(2048, 4096, "mla", p + "self_attn.fused_qkv_a_proj")
        W[(Lx, "q_b")] = build(8192, 1536, "mla", p + "self_attn.q_b_proj")
        W[(Lx, "o")] = build(4096, 8192, "mla", p + "self_attn.o_proj")
    if Lx < 3:
        W[(Lx, "gu")] = build(12288, 4096, "dense", p + "mlp.gate_up_proj")
        W[(Lx, "dn")] = build(4096, 6144, "dense", p + "mlp.down_proj")
    else:
        W[(Lx, "sgu")] = build(2048, 4096, "shared", p + "mlp.shared_experts.gate_up_proj")
        W[(Lx, "sdn")] = build(4096, 1024, "shared", p + "mlp.shared_experts.down_proj")
torch.cuda.empty_cache()
# stand-ins' data
nst = max(1, D // 2)
h0 = [torch.randn(1, 2 << 18, device=dev) for _ in range(NL)]
st = [torch.empty(nst, 2 << 18, device=dev) for _ in range(NL)]
attw = [torch.randn(max(A, 1) << 19, device=dev).to(torch.bfloat16) for _ in range(NL)]     # A MiB each
router = [torch.randn(288, 4096, device=dev).to(torch.bfloat16) for _ in range(NL)]
moe_w = [build(12288, 4096, "draft", f"sim.moe{i}")[0] for i in range(max(1, -(-RM * 2**20 // (12288 * 4096))) + 2)]
nmoe = max(1, round(RM * 2**20 / (12288 * 4096)))
E = F.ext()
ym = torch.empty(64, 12288, dtype=torch.bfloat16, device=dev)
red = torch.empty((), dtype=torch.float32, device=dev)
aux = torch.cuda.Stream()
moe_cursor = [0]


TRACE = None      # list of role names while an eager forward is traced (ROOF_SIM_PROFILE)
ROLE = {"in": "kda_in", "fb": "kda_fb", "gb": "kda_gb", "qkv_a": "mla_qkv_a", "q_b": "mla_q_b", "gu": "dense_gu",
        "dn": "dense_dn", "sgu": "sh_gu", "sdn": "sh_dn"}


def moe_fn(h):
    rows = h.shape[0]
    for _ in range(nmoe):
        if TRACE is not None:
            TRACE.append("routed")
        lw = moe_w[moe_cursor[0] % len(moe_w)]
        moe_cursor[0] += 1
        r16 = min(rows, 16)   # the stand-in config is instantiated for MB <= 2 only; the weight bytes are the same
        E.fp8_gemv_out(ym[:r16], h[:r16], lw.weight, lw.weight_scale.view(-1), None, 12288, 4096, 16, 4, 2, True, True)
    return h * 0.5


moe_layers = {}
for Lx in range(3, NL):
    ml = torch.nn.Module()
    ml.layer_name = f"model.layers.{Lx}.mlp.experts"
    ml.moe_fn = moe_fn
    moe_layers[Lx] = ml
moe = FakeMoE()


def lin(key, h):
    lay, m = W[key]
    if TRACE is not None:
        TRACE.append(ROLE.get(key[1]) or f"{kinds[key[0]]}_o")
    return m.apply(lay, h)


ATTN_PRE = os.environ.get("ROOF_SIM_ATTN_PRE", "1") != "0"


def attn_pre():
    """production: every attention block starts with the all-reduce of the previous MLP, mhc_fused_tilelang (~12 us) and
    mhc_pre_big_fuse_with_norm (~6 us) -- t3's window (R15 trace: MoE down -> next in_proj / qkv_a median 56-58 us).
    ROOF_SIM_ATTN_PRE=0 drops it (the first session's simulator, docs/logs/dec_fp8roof/explore/step*.log)."""
    if ATTN_PRE:
        spin(25); spin(12); spin(6)


def kda(Lx, h):
    attn_pre()
    y = lin((Lx, "in"), h)
    fb = lin((Lx, "fb"), y[:, 12320:12448])
    gb = lin((Lx, "gb"), y[:, 12448:12576])
    spin(7)                                            # conv
    if D:
        st[Lx].copy_(h0[Lx].expand(nst, -1))           # recurrent: per-token state writes
    spin(24)
    core = (y[:, :4096] * torch.sigmoid(fb) + gb * 0.01)
    spin(3)                                            # norm
    return lin((Lx, "o"), core)


def mla(Lx, h):
    attn_pre()
    a_ = lin((Lx, "qkv_a"), h)
    spin(2)
    q = lin((Lx, "q_b"), a_[:, :1536])
    if A:
        torch.sum(attw[Lx], dim=0, out=red)            # attention's own memory traffic (normal loads)
    spin(150)
    return lin((Lx, "o"), q * 0.05)


def moe_block(Lx, h):
    spin(25); spin(19)                                 # all-reduce, mHC
    ev = torch.cuda.Event(); ev.record(); aux.wait_event(ev)
    torch.mv(router[Lx], h[0], out=None)               # router GEMV stand-in (2.4 MB)
    spin(3); spin(24)                                  # topk, rot_in
    routed = moe.apply(moe_layers[Lx], h, None, None, None, None)
    with torch.cuda.stream(aux):
        su = lin((Lx, "sgu"), h)
        sh = lin((Lx, "sdn"), torch.nn.functional.silu(su[:, :1024]) * su[:, 1024:])
    torch.cuda.current_stream().wait_stream(aux)
    spin(8)                                            # epilogue
    return routed + sh


def dense(Lx, h):
    spin(25); spin(19)
    u = lin((Lx, "gu"), h)
    spin(4)
    return lin((Lx, "dn"), torch.nn.functional.silu(u[:, :6144]) * u[:, 6144:])


def rms(h):
    return (h.float() * torch.rsqrt(h.float().pow(2).mean(-1, keepdim=True) + 1e-6)).to(torch.bfloat16)


def forward(h):
    for Lx, kind in enumerate(kinds):
        h = rms(h + (kda(Lx, h) if kind == "kda" else mla(Lx, h)))
        h = rms(h + (dense(Lx, h) if Lx < 3 else moe_block(Lx, h)))
    spin(25)
    return h


def parse_cfg(spec):
    """'name:triggers:mib:ctas:pol', e.g. 'all', 'off', 't2,t3', 'big:all:t1=16,t3=16:16:0'."""
    parts = spec.split(":")
    name = parts[0]
    trig = parts[1] if len(parts) > 1 else name
    trig = frozenset() if trig == "off" else frozenset(RF.TRIGGERS) if trig == "all" else frozenset(trig.split(","))
    mib, _ = RF._parse_mib(parts[2] if len(parts) > 2 else "")
    ctas = int(parts[3]) if len(parts) > 3 and parts[3] else RF.DEFAULT_CTAS
    pol = int(parts[4]) if len(parts) > 4 and parts[4] else 0
    return name, trig, mib, ctas, pol


_plan_new = RF.plan


def _plan_v1(src_role, L):
    """The first committed plan (16d5cf9), for paired comparisons (config names starting with 'v1'): t3 only after a
    routed MoE and only to the first linear of the next layer (no dense down_proj trigger, no MLA q_b bytes)."""
    if src_role == "dense_dn":
        return "t3", []
    trig, out = _plan_new(src_role, L)
    return trig, (out[:1] if src_role == "moe" else out)


def capture(cfg):
    name, trig, mib, ctas, pol = cfg
    RF.plan = _plan_v1 if name.startswith("v1") else _plan_new
    RF.STATE.pf = bool(trig)
    RF.STATE.triggers, RF.STATE.mib, RF.STATE.ctas, RF.STATE.pol = trig, mib, ctas, pol
    x = (torch.randn(M, 4096, device=dev, generator=g) * 0.5).to(torch.bfloat16)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        forward(x)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    RF.COUNTERS.clear()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        y = forward(x)
    c = dict(RF.COUNTERS)
    gr.replay(); torch.cuda.synchronize()
    RF.STATE.pf = False
    RF.plan = _plan_new
    return gr, x, y, c


cfgs = [parse_cfg(c) for c in os.environ.get("ROOF_SIM_CONFIGS", "off;all").split(";")]
assert cfgs[0][1] == frozenset(), "the first config must be off (the reference)"
graphs = {}
for cfg in cfgs:
    graphs[cfg[0]] = capture(cfg)
    c = graphs[cfg[0]][3]
    print(f"captured {cfg[0]:>10s} (triggers {sorted(cfg[1])}, MiB {cfg[2]}, CTAs {cfg[3]}, pol {cfg[4]}): forks "
          + " ".join(f"{t}={c.get('fork_' + t, 0)}" for t in RF.TRIGGERS)
          + f", joined {c.get('joined', 0)}, safety {c.get('joined_late_safety', 0)}", flush=True)
xs = (torch.randn(M, 4096, device=dev, generator=g) * 0.5).to(torch.bfloat16)
ref = None
same = True
for name, (gr, x, y, _) in graphs.items():
    x.copy_(xs); gr.replay(); torch.cuda.synchronize()
    if ref is None:
        ref = y.clone()
    same &= torch.equal(ref, y)
print(f"every configuration's forward output bitwise equal to off: {same}; finite {bool(torch.isfinite(ref).all())}",
      flush=True)
res = {k: [] for k in graphs}
names = list(graphs)
for r in range(ROUNDS):
    for name in (names if r % 2 == 0 else names[::-1]):
        gr = graphs[name][0]
        e0.record()
        for _ in range(3):
            gr.replay()
        e1.record(); torch.cuda.synchronize()
        res[name].append(e0.elapsed_time(e1) / 3)
off = statistics.median(res[names[0]])
nk = sum(1 for k in kinds if k == "kda")
print(f"NL={NL} ({nk} KDA, {NL - nk} MLA, {NL - 3} MoE), M={M}, D={D} MiB, A={A} MiB, routed {nmoe} x 51.4 MB, "
      f"attention pre-block (AR + mHC) {'on' if ATTN_PRE else 'off'}, L2 {torch.cuda.get_device_properties(0).L2_cache_size / 2**20:.0f} MiB, "
      f"{ROUNDS} rounds; ms per forward (median), saving vs off, paired ratio median [min, max]")
for name in names:
    m = statistics.median(res[name])
    rr = [b / a_ for a_, b in zip(res[names[0]], res[name])]
    print(f"  {name:>10s}: {m:8.3f} ms  saving {off - m:6.3f} ms ({(off - m) / off * 100:5.2f}%, "
          f"{(off - m) / NL * 1000:5.1f} us/layer)  ratio {statistics.median(rr):.4f} [{min(rr):.4f}, {max(rr):.4f}]  "
          f"spread {(max(res[name]) - min(res[name])) / m * 100:.1f}%", flush=True)


# ---------------------------------------------------------------------------------------------------------------
# ROOF_SIM_PROFILE=<rounds>: per-call kernel durations of every FP8 linear inside the captured graphs (torch.profiler /
# CUPTI), off vs the second config, paired rounds. Kernel -> role: an eager forward is traced once (Python call order
# = launch order = CUPTI correlation order) to learn each role's (kernel name, grid) signature; in a replay the k-th
# kernel (by start time) of a signature is the k-th call of that signature (stream order + the aux join make it so).
PROF = int(os.environ.get("ROOF_SIM_PROFILE", "0") or 0)
if PROF:
    import gzip
    import json
    import tempfile
    from collections import defaultdict
    from torch.profiler import ProfilerActivity, profile

    def kernels(tr):
        ev = json.load(open(tr))["traceEvents"]
        ks = [e for e in ev if e.get("cat") == "kernel" and e.get("ph") == "X"]
        launches = sorted(e["args"]["correlation"] for e in ev if e.get("cat") == "cuda_runtime"
                          and e.get("name") == "cudaGraphLaunch")
        return ks, launches

    def sig(e):
        return (e["name"], tuple(e["args"].get("grid") or ()))

    isfp8 = lambda e: "fp8_gemv" in e["name"]     # noqa: E731
    ispf = lambda e: "l2_prefetch" in e["name"]   # noqa: E731
    # (1) eager trace: role -> signature
    RF.STATE.pf = False
    x0 = (torch.randn(M, 4096, device=dev, generator=g) * 0.5).to(torch.bfloat16)
    TRACE = []
    tmp = tempfile.mkdtemp()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as pr:
        forward(x0)
        torch.cuda.synchronize()
    order = list(TRACE)
    TRACE = None
    pr.export_chrome_trace(f"{tmp}/eager.json")
    ks, _ = kernels(f"{tmp}/eager.json")
    fk = sorted([e for e in ks if isfp8(e)], key=lambda e: e["args"]["correlation"])
    assert len(fk) == len(order), f"eager: {len(fk)} fp8 kernels vs {len(order)} calls"
    sig_roles = defaultdict(list)                  # signature -> [role of its k-th call]
    for e, role in zip(fk, order):
        sig_roles[sig(e)].append(role)
    # (2) replays under the profiler: rounds x (off x2, cfg x2), alternating which goes first
    a_name, b_name = names[0], names[1]
    sched = []
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as pr:
        for r in range(PROF):
            for name in ((a_name, b_name) if r % 2 == 0 else (b_name, a_name)):
                for _ in range(2):
                    graphs[name][0].replay()
                    sched.append(name)
        torch.cuda.synchronize()
    pr.export_chrome_trace(f"{tmp}/graph.json")
    ks, launches = kernels(f"{tmp}/graph.json")
    assert len(launches) == len(sched), f"{len(launches)} graph launches vs {len(sched)} scheduled"
    cfg_of = dict(zip(launches, sched))
    per = {a_name: defaultdict(list), b_name: defaultdict(list)}
    pfk = {a_name: [], b_name: []}
    byrep = defaultdict(list)
    for e in ks:
        byrep[e["args"]["correlation"]].append(e)
    skip = 1                                      # the first replay of each config: warm-up
    seen = defaultdict(int)
    for c in launches:
        name = cfg_of[c]
        seen[name] += 1
        if seen[name] <= skip:
            continue
        rep_k = sorted(byrep[c], key=lambda e: e["ts"])
        bysig = defaultdict(list)
        for e in rep_k:
            if isfp8(e):
                bysig[sig(e)].append(e)
            elif ispf(e):
                pfk[name].append(e["dur"])
        for sg, roles in sig_roles.items():
            got = bysig.get(sg, [])
            assert len(got) == len(roles), f"replay: signature {sg[1]} {len(got)} kernels vs {len(roles)} calls"
            for e, role in zip(got, roles):
                per[name][role].append(e["dur"])
    NB = {"kda_in": 12608 * 4096, "kda_fb": 4096 * 128, "kda_gb": 4096 * 128, "kda_o": 4096 * 4096,
          "mla_qkv_a": 2048 * 4096, "mla_q_b": 8192 * 1536, "mla_o": 4096 * 8192, "dense_gu": 12288 * 4096,
          "dense_dn": 4096 * 6144, "sh_gu": 2048 * 4096, "sh_dn": 4096 * 1024, "routed": 12288 * 4096}
    nfwd = {n: max(1, seen[n] - skip) for n in (a_name, b_name)}
    print(f"per-call FP8 kernel time inside the graphs (CUPTI), M={M}, {PROF} rounds x 2 replays, first replay of each "
          f"config dropped; median us (GB/s of the weight bytes) [p10-p90 of {b_name}]")
    print(f"  {'role':>10s} {'calls/fwd':>9s} {'MB':>6s} {a_name + ' us':>10s} {'GB/s':>6s} {b_name + ' us':>10s} "
          f"{'GB/s':>6s} {'saving us':>9s}   {b_name} p10..p90")
    tot_a = tot_b = 0.0
    for role in NB:
        va, vb = per[a_name].get(role, []), per[b_name].get(role, [])
        if not va:
            continue
        n_per = len(va) // nfwd[a_name]
        ma, mb = statistics.median(va), statistics.median(vb)
        tot_a += sum(va) / nfwd[a_name]
        tot_b += sum(vb) / nfwd[b_name]
        q = sorted(vb)
        print(f"  {role:>10s} {n_per:9d} {NB[role] / 1e6:6.2f} {ma:10.1f} {NB[role] / ma / 1e3:6.1f} {mb:10.1f} "
              f"{NB[role] / mb / 1e3:6.1f} {ma - mb:9.1f}   {q[len(q) // 10]:.1f}..{q[len(q) * 9 // 10]:.1f}")
    pa = sum(pfk[a_name]) / nfwd[a_name]
    pb = sum(pfk[b_name]) / nfwd[b_name]
    print(f"  sum of FP8 kernel time per forward: {a_name} {tot_a / 1e3:.3f} ms, {b_name} {tot_b / 1e3:.3f} ms "
          f"(saving {(tot_a - tot_b) / 1e3:.3f} ms); l2_prefetch kernels per forward: {b_name} "
          f"{len(pfk[b_name]) // nfwd[b_name]} launches, {pb / 1e3:.3f} ms of side-stream time (off: {pa:.1f} us)")
