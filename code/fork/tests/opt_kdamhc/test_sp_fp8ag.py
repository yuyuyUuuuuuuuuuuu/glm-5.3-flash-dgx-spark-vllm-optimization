"""opt-kdamhc: GLM53_SP_FP8AG checks (nodeC, production image). Exit 1 on any failure.

U1  per-token e4m3 quant is row-local: quant(shard0) ++ quant(shard1) == quant(full), bytes and scales, at the KDA
    in_proj input shape (T = 13,824 and the odd tail 4,463 with the SP zero pad row).
U2  w8a8_forward(preq=gathered shard quants) == w8a8_forward(full bf16) BITWISE, KDA in_proj rank shape
    [12576 x 4096] at M = 13,824 / 4,463, custom CUTLASS GEMM and the cutlass_mm pieces.
U3  the apply hook: a Glm53DenseFp8Method stand-in wrapped by fp8_w8a8._make_apply; prequant_serves() True for the
    served layer, the marker input gives the bitwise output of the bf16 input, a stale marker is never applied, and a
    declining layer rebuilds bf16 rows (counted) instead of reading the stride-0 marker.
U4  timing (median of 9): quant_per_token at T vs T/2 rows; w8a8_forward full vs preq.
N   two processes on the one GB10 (NCCL socket loopback, like tests/mhc2/nccl_sp2_pipeline.py): the helpers exec'd
    verbatim from the SP- and SP2-patched + sp-fp8ag-patched model.py, real PyNcclCommunicator:
      N1 plain SP: _spfp8_gather(shard) -> marker -> in_proj output == in_proj(sp_all_gather(shard)) bitwise, both
         ranks; T = 13,824 and 4,464 (even) ...
      N2 SP2 pipelined helper (_sp2_fused_post_pre_gather, attention phase) fp8 vs bf16 gather: in_proj outputs bitwise
         equal, both ranks;
      N3 loopback timing of the bf16 vs fp8 all-gather (socket loopback is ~9x slower than RoCE; ratio only).
Run: GPU_RUN_SHM=2g GPU_RUN_RO=$TF_EXL3_KITS/tf-exl3-deploy16.r16z2/overlay:$TF_EXL3_KITS/tf-exl3-deploy16.r16z2/site \
     flock /tmp/tf-gpu-bench.lock tests/gpu_run.sh python3 tests/opt_kdamhc/test_sp_fp8ag.py
"""
from __future__ import annotations

import os
import statistics
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (ROOT, os.path.join(ROOT, "tests"), os.path.join(ROOT, "overlay")):
    sys.path.insert(0, p)
EXT_DIR = os.environ.get("W8A8_EXT_DIR", os.path.join(os.environ.get("TF_EXL3_KITS") or os.path.expanduser("~"), "tf-exl3-deploy16.r16z2/overlay"))
sys.path.append(EXT_DIR)
sys.path.append(os.environ.get("KIT_SITE_DIR", os.path.join(os.environ.get("TF_EXL3_KITS") or os.path.expanduser("~"), "tf-exl3-deploy16.r16z2/site")))

import torch  # noqa: E402

N_IN, K_IN = 12576, 4096          # KDA in_proj per rank (TP=2)
FAIL: list[str] = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg, flush=True)
    if not cond:
        FAIL.append(msg)


def act(T, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rs = torch.rand(T, 1, generator=g, device="cuda") * 3 + 0.05          # per-token magnitude spread
    return (torch.randn(T, K_IN, generator=g, device="cuda") * rs).bfloat16()


def tmed(fn, n=9):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


class FakeMethod:
    """The attributes fp8_w8a8.serve()/prequant_serves() read on production's Glm53DenseFp8Method."""
    ready = True
    group = "kda"
    prefix = "model.layers.0.self_attn.in_proj_qkvbfg_a"

    def apply(self, layer, x, bias=None):        # production's path stand-in: Marlin W8A16
        from fp8_bench_common import marlin
        return marlin(layer, x.reshape(-1, K_IN), N_IN, K_IN)


def setup_w8a8(seed=3):
    import fp8_w8a8 as W
    from fp8_bench_common import quantize_like_prod
    W.ext()
    W.STATE.custom = hasattr(W.ext(), "fp8_w8a8_gemm")
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(N_IN, K_IN, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    layer, _, _ = quantize_like_prod(w)
    del w
    if not hasattr(FakeMethod.apply, "_tf_w8a8_hook"):
        FakeMethod.apply = W._make_apply(FakeMethod.apply)
    layer.quant_method = FakeMethod()
    layer.bias = None
    W.STATE.enabled, W.STATE.groups = True, frozenset({"kda"})
    W.CFG.only = frozenset({"kda.in_proj_qkvbfg_a"})
    v = W.selftest(layer, N_IN, K_IN, None, "test kda.in_proj")
    return W, layer, v


def unit():
    import fp8_w8a8 as W
    W, layer, v = setup_w8a8()
    print(f"selftest: {v}; custom GEMM for [{N_IN}x{K_IN}]: {W.use_custom(N_IN, K_IN)}", flush=True)
    check(v[0], "U0 self-test passed on the test layer")
    for T in (13824, 4463):
        x = act(T, 11 + T)
        S = -(-T // 2)
        xp = torch.cat([x, x.new_zeros(2 * S - T, K_IN)], 0)            # sp_shard's zero pad
        qf, sf = W.quant_per_token(x)
        q0, s0 = W.quant_per_token(xp[:S].contiguous())
        q1, s1 = W.quant_per_token(xp[S:].contiguous())
        qc, sc = torch.cat([q0, q1], 0)[:T], torch.cat([s0, s1], 0)[:T]
        check(torch.equal(qc.view(torch.uint8), qf.view(torch.uint8)) and torch.equal(sc, sf),
              f"U1 T={T}: quant(shards) == quant(full) bytes + scales")
        for mode in ("custom", "cutlass_mm"):
            saved = W.STATE.custom
            W.STATE.custom = saved and mode == "custom"
            yf = W.w8a8_forward(x, layer.weight, layer.weight_scale, N_IN, K_IN)
            marker = torch.zeros((1, 1), dtype=torch.bfloat16, device="cuda").expand(T, K_IN)
            yp = W.w8a8_forward(marker, layer.weight, layer.weight_scale, N_IN, K_IN, preq=(qc, sc))
            W.STATE.custom = saved
            check(torch.equal(yf, yp), f"U2 T={T} {mode}: w8a8_forward(preq) == w8a8_forward(bf16) bitwise")
    # U3 the hook
    T = 13824
    x = act(T, 99)
    check(W.prequant_serves(layer, T), "U3 prequant_serves(served layer, 13824) is True")
    check(not W.prequant_serves(layer, 256), "U3 prequant_serves(M below min_m) is False")
    y_ref = layer.quant_method.apply(layer, x)
    q, s = W.quant_per_token(x)
    c0 = dict(W.COUNTERS)
    marker = W.prequant_marker(layer, q, s)
    y_m = layer.quant_method.apply(layer, marker)
    check(torch.equal(y_ref, y_m), "U3 marker input -> bitwise the bf16-input output through the apply hook")
    check(W.COUNTERS.get("w8a8_preq", 0) == c0.get("w8a8_preq", 0) + 1 and not W.PREQ,
          "U3 served from the prequant (counter +1, PREQ empty after)")
    # stale: a marker registered, then a DIFFERENT real tensor comes in -> must ignore the entry
    W.prequant_marker(layer, q, s)
    y_s = layer.quant_method.apply(layer, x)
    check(torch.equal(y_s, y_ref) and W.COUNTERS.get("preq_stale", 0) >= 1, "U3 stale entry never applied")
    # declining layer: selection filter excludes it -> production path on rows rebuilt from fp8 (never the marker)
    W.CFG.only = frozenset({"mla.o_proj"})
    marker = W.prequant_marker(layer, q, s)
    y_d = layer.quant_method.apply(layer, marker)
    deq = (q.float() * s).bfloat16()
    from fp8_bench_common import marlin
    check(torch.equal(y_d, marlin(layer, deq, N_IN, K_IN)) and W.COUNTERS.get("preq_materialized", 0) >= 1,
          "U3 declining layer: served from the dequantized fp8 rows (counted), not the stride-0 marker")
    W.CFG.only = frozenset({"kda.in_proj_qkvbfg_a"})
    # U4 timing
    xh = x[: T // 2].contiguous()
    tq_full = tmed(lambda: W.quant_per_token(x))
    tq_half = tmed(lambda: W.quant_per_token(xh))
    tf = tmed(lambda: W.w8a8_forward(x, layer.weight, layer.weight_scale, N_IN, K_IN))
    mk = torch.zeros((1, 1), dtype=torch.bfloat16, device="cuda").expand(T, K_IN)
    tp = tmed(lambda: W.w8a8_forward(mk, layer.weight, layer.weight_scale, N_IN, K_IN, preq=(q, s)))
    print(f"U4 quant_per_token: T={T} {tq_full:.3f} ms, T/2 {tq_half:.3f} ms (saved per KDA layer "
          f"{tq_full - tq_half:.3f} ms); w8a8_forward full {tf:.3f} ms, preq {tp:.3f} ms", flush=True)


# ------------------------------------------------------------------------------------------------ NCCL two processes
def _trees():
    import shutil
    import tempfile
    import patch_mhc_sp as P
    import patch_mhc_sp2 as P2
    import patch_sp_fp8ag as F
    site = os.environ.get("GLM53_SITEPKG", "/usr/local/lib/python3.12/dist-packages")
    src = open(os.path.join(site, "vllm/models/glm5next/nvidia/model.py")).read()
    sp = P.prepare(src, P.MODEL_HUNKS)
    return F.prepare(sp), F.prepare(P2.prepare(sp))


def _exec_helpers(src, extra):
    import logging
    from vllm.platforms import current_platform
    ns = {"torch": torch, "MHC_SP_ACTIVE": True, "current_platform": current_platform,
          "logger": logging.getLogger("t"), "sp_shard": None}
    ns.update(extra)
    a = src.index("# [glm53-sp-fp8ag] sequence-parallel fp8 all-gather")
    b = src.index("class Glm5NextDecoderLayer(nn.Module):")
    if "MHC_SP2_K = 2" in src:
        a2 = src.index("MHC_SP2_K = 2")
        b2 = src.index("from vllm.multimodal import MULTIMODAL_REGISTRY")
        exec(compile(src[a2:b2], "<sp2 helpers>", "exec"), ns)
    exec(compile(src[a:b], "<fp8ag helpers>", "exec"), ns)
    ns.update(extra)
    return ns


def _worker(rank, port, q):
    os.environ.update({
        "NCCL_HOSTID": f"f8ag-host{rank}", "NCCL_IB_DISABLE": "1", "NCCL_SOCKET_IFNAME": "lo",
        "NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "1", "NCCL_NVLS_ENABLE": "0",
        "NCCL_MIN_NCHANNELS": "8", "NCCL_MAX_NCHANNELS": "8", "NCCL_DEBUG": "WARN"})
    import torch.distributed as dist
    torch.cuda.set_device(0)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    comm = PyNcclCommunicator(dist.group.WORLD, torch.device("cuda:0"))
    import vllm.distributed as vd
    vd.get_tensor_model_parallel_world_size = lambda: 2
    W, lin, v = setup_w8a8(seed=3)                     # same seed: identical weights on both ranks
    res = {"rank": rank, "fail": [], "t": {}}

    def chk(c, m):
        if not c:
            res["fail"].append(m)

    def ag_bf16(x):
        out = torch.empty((2 * x.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
        comm.all_gather(out, x.contiguous())
        return out

    sp_src, sp2_src = _trees()
    layer = types.SimpleNamespace(layer_kind="kda", is_sequence_parallel=False, layer_idx=3,
                                  self_attn=types.SimpleNamespace(in_proj_qkvbfg_a=lin))
    ns = _exec_helpers(sp_src, {"sp_all_gather": ag_bf16})
    ns["_spfp8_comm"] = lambda: comm
    for T in (13824, 4464):
        x = act(T, 500 + T)                                       # the full layer input, identical on both ranks
        S = T // 2
        xs = x[rank * S:(rank + 1) * S].contiguous()
        ref = lin.quant_method.apply(lin, ag_bf16(xs)[:T])
        chk(ns["_spfp8_lin"](layer, T) is lin, f"N1 T={T}: _spfp8_lin selects the served in_proj")
        mk = ns["_spfp8_gather"](layer, xs, T)
        chk(mk.stride() == (0, 0) and tuple(mk.shape) == (T, K_IN), f"N1 T={T}: a stride-0 marker comes back")
        out = lin.quant_method.apply(lin, mk)
        chk(torch.equal(out, ref), f"N1 T={T}: in_proj(fp8 gather) == in_proj(bf16 gather) bitwise")
        torch.cuda.synchronize()
        if T == 13824:
            def t_bf16():
                ag_bf16(xs)

            def t_fp8():
                qq, ss = W.quant_per_token(xs)
                qa = torch.empty((T, K_IN), dtype=qq.dtype, device="cuda")
                sa = torch.empty((T, 1), dtype=torch.float32, device="cuda")
                ns["_spfp8_ag_pair"](comm, qa, sa, qq, ss, torch.cuda.current_stream())
            res["t"]["ag_bf16_ms"] = tmed(t_bf16, 5)
            res["t"]["ag_fp8_quant_ms"] = tmed(t_fp8, 5)
    # N1b: a layer whose W8A8 verdict differs between the ranks -> both ranks must take the bf16 gather
    layer_b = types.SimpleNamespace(layer_kind="kda", is_sequence_parallel=False, layer_idx=7,
                                    self_attn=types.SimpleNamespace(in_proj_qkvbfg_a=lin))
    real = W.prequant_serves
    W.prequant_serves = (lambda l, m: False) if rank == 1 else real
    T = 13824
    x = act(T, 4242)
    xs = x[rank * (T // 2):(rank + 1) * (T // 2)].contiguous()
    got = ns["_spfp8_lin"](layer_b, T)
    W.prequant_serves = real
    chk(got is None and layer_b._spfp8_ok is False, "N1b asymmetric verdict -> agreed bf16 on both ranks")
    xg = ns["_spfp8_gather"](layer_b, xs, T)
    chk(xg.stride() != (0, 0) and torch.equal(xg, ag_bf16(xs)[:T]), "N1b asymmetric verdict: bf16 rows gathered")
    # N2: the SP2 pipelined helper, attention phase
    sys.path.insert(0, os.path.join(ROOT, "tests", "mhc_sp"))
    from bench_mhc_halving import make
    T = 13824
    B, S = T // 4, T // 2
    w = make(T, seed=5)
    ns2 = _exec_helpers(sp2_src, {"sp_all_gather": ag_bf16})
    ns2["_SP2"].update(k=2, B=B, N=2, r=rank, T=T, comm=comm, pend=None)
    ns2["_spfp8_comm"] = lambda: comm
    norm = types.SimpleNamespace(weight=types.SimpleNamespace(data=w["norm"]), variance_epsilon=1e-5)
    other = types.SimpleNamespace(weight=types.SimpleNamespace(data=w["norm"]), variance_epsilon=1e-5)
    layer2 = types.SimpleNamespace(layer_kind="kda", is_sequence_parallel=False, layer_idx=5, rms_norm_eps=1e-5, hc_eps=1e-6,
                                   mhc_post_mult_value=2.0, mhc_sinkhorn_iterations=20, input_layernorm=norm,
                                   post_attention_layernorm=other,
                                   self_attn=types.SimpleNamespace(in_proj_qkvbfg_a=lin))

    def shard(t):
        return torch.cat([t[rank * B:(rank + 1) * B], t[(rank + 2) * B:(rank + 3) * B]], 0)

    outs = {}
    for nm, nrm in (("fp8", norm), ("bf16", other)):
        r = ns2["_sp2_fused_post_pre_gather"](layer2, shard(w["x"]), shard(w["res"]), shard(w["post"]),
                                              shard(w["comb"]), w["fn"], w["scale"], w["base"], nrm, T)
        xg = r[3]
        if nm == "fp8":
            chk(xg.stride() == (0, 0), "N2 SP2 attention phase of a served KDA layer returns the marker")
        else:
            chk(xg.stride() != (0, 0), "N2 SP2 non-attention norm returns bf16 rows")
        outs[nm] = (lin.quant_method.apply(lin, xg), r[0])
    chk(torch.equal(outs["fp8"][0], outs["bf16"][0]), "N2 SP2: in_proj(fp8 pipelined gather) == in_proj(bf16) bitwise")
    chk(torch.equal(outs["fp8"][1], outs["bf16"][1]), "N2 SP2: residual unchanged")
    # N2b: under SP2 the first-prefill verdict all-reduce (compute stream, same communicator) must be ordered after
    # every op already on the side stream (in-flight reduce-scatter slices): a slow side-stream write must be visible
    # to the compute stream right after _spfp8_lin returns
    torch.cuda.synchronize()
    cs = ns2["_sp2_stream"]()
    probe = torch.zeros(1, device="cuda")
    with torch.cuda.stream(cs):
        torch.cuda._sleep(400_000_000)                          # ~0.2-0.4 s of GPU spin on the side stream
        probe.fill_(1.0)
    layer_c = types.SimpleNamespace(layer_kind="kda", is_sequence_parallel=False, layer_idx=9,
                                    self_attn=types.SimpleNamespace(in_proj_qkvbfg_a=lin))
    got = ns2["_spfp8_lin"](layer_c, T)
    seen = probe.clone()                                        # compute stream, after the verdict all-reduce
    chk(got is lin and float(seen.item()) == 1.0, "N2b SP2 verdict all-reduce waits for the side stream")
    torch.cuda.synchronize()
    q.put(res)
    dist.destroy_process_group()


def nccl():
    import socket
    import torch.multiprocessing as mp
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=_worker, args=(r, port, q)) for r in range(2)]
    for p in ps:
        p.start()
    got = [q.get(timeout=1200) for _ in ps]
    for p in ps:
        p.join(60)
    for r in sorted(got, key=lambda d: d["rank"]):
        for m in r["fail"]:
            check(False, f"rank {r['rank']}: {m}")
        print(f"  rank {r['rank']}: {len(r['fail'])} failures; loopback timing {r['t']}", flush=True)
    check(all(not r["fail"] for r in got) and len(got) == 2, "N both ranks: every NCCL check passed")


if __name__ == "__main__":
    print(f"gpu {torch.cuda.get_device_name(0)}", flush=True)
    if os.environ.get("SKIP_UNIT") != "1":
        unit()
    if os.environ.get("SKIP_NCCL") != "1":
        nccl()
    print(f"RESULT: {'ALL OK' if not FAIL else f'{len(FAIL)} FAILED'}", flush=True)
    sys.exit(1 if FAIL else 0)
