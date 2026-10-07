"""opt-dense: GLM53_DENSE_W8A8_FP8AG (fp8 sequence-parallel all-gather into a served KDA in_proj), nodeC.

Single process (default):
 A  _attn_gather_line finds exactly the attention gather of the SP-patched production Glm5NextDecoderLayer.forward
    (patch_mhc_sp.prepare over the image's model.py) and of the stock forward, never the MLP / aux / final gathers;
 B  _attn_forward_ok: the image's Glm5NextLinearAttention passes; MLA attention and a forward that reads
    hidden_states elsewhere fail;
 C  numerics: 2 ranks emulated (the gather stub concatenates the per-shard quantizations), real Marlin in_proj
    (12576 x 4096): in_proj(placeholder) == W8A8 in_proj(bf16 full gather) BITWISE at T 13824 / 4289 (odd, padded) /
    6000 / 1024;
 D  the wrapper end to end through a decoder-shaped fixture (fp8ag_fake.py): the attention gather goes fp8, the
    second gather stays bf16; agreement False -> bf16 path; BadAttn -> bf16 path;
 E  a pending gather met by another projection is materialized as bf16(q*s) (warned, counted);
 F  timing on one GPU: per-token quant of the full T vs the shard.
Two ranks (argv 'nccl'): vLLM TP=2 over loopback NCCL (two processes on the one GB10, as tests/mhc2): the real
tensor_model_parallel_all_gather of fp8-as-uint8 + fp32 scales and the real CPU MIN agreement; in_proj output on
each rank == W8A8 on the bf16 gathered rows (bitwise); wall of bf16 AG vs fp8 AG (+ shard quant) on the loopback.
Usage: source tests/w8a82/env.sh; tests/gpu_run.sh python3 tests/optdense/test_fp8ag.py [nccl]"""
import os
import statistics
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import torch  # noqa: E402

dev = "cuda"
FAILS = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAILS.append(msg)


def tmed(fn, n=9):
    for _ in range(2): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def setup_layer(seed=0):
    for v in ("GLM53_FP8_GEMV", "GLM53_FP8_GEMV_MAX_M", "TF_EXL3_MOE", "GLM53_KDA_BF16_LARGE_M", "GLM53_DEC_FP8ROOF"):
        os.environ.pop(v, None)
    os.environ["GLM53_FP8_LARGE_M"] = "1"
    os.environ["GLM53_DENSE_FP8"] = "dense,kda,mla,shared"
    os.environ["GLM53_DENSE_W8A8"] = "1"
    os.environ["GLM53_DENSE_W8A8_ONLY"] = "kda.in_proj_qkvbfg_a,mla.o_proj"
    os.environ["GLM53_DENSE_W8A8_FP8AG"] = "1"
    from harness import load_prod
    from test_fp8_integrate import L
    prod = load_prod()
    import fp8_gemv as F
    import fp8_w8a8 as W
    F.install(prod)
    rep = W.install(prod)
    print("install:", rep, "fp8ag", W.CFG.fp8ag, flush=True)
    cls = prod.Glm53DenseFp8Method
    g = torch.Generator(device=dev).manual_seed(seed)
    mk = []
    for i in range(2):
        w = (torch.randn(12576, 4096, device=dev, generator=g) * 0.02).to(torch.bfloat16)
        lay = L(w)
        m = cls("kda", f"model.layers.{i}.self_attn.in_proj_qkvbfg_a")
        m.process_weights_after_loading(lay)
        lay.quant_method = m
        lay.forward = (lambda mm, ll: (lambda x: (cls.apply(mm, ll, x), None)))(m, lay)
        mk.append((m, lay))
    return W, cls, mk


def quant_shards(W, x_full, ranks=2):
    pad = (-x_full.shape[0]) % ranks
    xp = torch.nn.functional.pad(x_full, (0, 0, 0, pad)) if pad else x_full
    R = xp.shape[0] // ranks
    shards = [xp[r * R:(r + 1) * R] for r in range(ranks)]
    return shards, [W.quant_per_token(s.contiguous()) for s in shards]


def single():
    from harness import gpu_guard
    gpu_guard(8.0)
    from test_fp8_integrate import single_rank_tp
    single_rank_tp()
    W, cls, mk = setup_layer()
    (m0, lay0), (m1, lay1) = mk
    check(W.CFG.fp8ag and W.STATE.enabled, "install parsed GLM53_DENSE_W8A8_FP8AG=1")

    # A. the attention gather line of the real SP-patched decoder forward
    sys.path.insert(0, str(ROOT / "overlay"))
    import patch_mhc_sp as P
    import vllm.models.glm5next.nvidia.model as MM
    stock_src = Path(MM.__file__).read_text()
    out_dir = Path(os.environ.get("FP8AG_TMP", "/w/.fp8ag_tmp")) if os.access("/w", os.W_OK) else None
    import tempfile
    d = Path(tempfile.mkdtemp(prefix="fp8ag-"))
    for label, src in (("stock", stock_src), ("sp-patched", P.prepare(stock_src, P.MODEL_HUNKS))):
        a = src.index("class Glm5NextDecoderLayer")
        b = src.index("\nclass ", a + 10)
        cls_src = src[a:b]
        f0 = cls_src.index("    def forward(")
        f1 = cls_src.index("\n    def ", f0 + 10)
        meth = cls_src[f0:f1]
        mod = d / f"dec_{label.replace('-', '_')}.py"
        mod.write_text("from __future__ import annotations\nclass Dec:\n" + meth + "\n")
        ns = {}
        exec(compile(mod.read_text(), str(mod), "exec"), ns)
        code = ns["Dec"].forward.__code__
        ln = W._attn_gather_line(code)
        lines = mod.read_text().splitlines()
        ok = ln is not None and lines[ln - 1].strip().startswith("x = sp_all_gather(x)") and \
            any(lines[j].strip().startswith("x = self.self_attn(") for j in range(ln, min(ln + 3, len(lines))))
        n_gathers = sum(1 for t in lines if "sp_all_gather(" in t)
        check(ok, f"A {label}: attention gather at line {ln} ({lines[ln - 1].strip() if ln else None!r}); "
                  f"{n_gathers} sp_all_gather lines in the forward, exactly one chosen")
        if label == "sp-patched":                  # the quickwins transplant: source only in linecache
            import linecache
            tsrc = textwrap.dedent(meth)
            fname = "<glm53-quickwins vllm.models.glm5next.nvidia.model.Glm5NextDecoderLayer.forward>"
            linecache.cache[fname] = (len(tsrc), None, tsrc.splitlines(True), fname)
            ns2 = {"__name__": "x"}
            exec(compile("from __future__ import annotations\n" + tsrc, fname, "exec"), ns2)
            linecache.cache[fname] = (len(tsrc), None, ("from __future__ import annotations\n" + tsrc)
                                      .splitlines(True), fname)
            ln2 = W._attn_gather_line(ns2["forward"].__code__)
            tl = linecache.getlines(fname)
            check(ln2 is not None and tl[ln2 - 1].strip().startswith("x = sp_all_gather(x)"),
                  f"A transplanted (linecache-only source, as glm53_prefill_quickwins.transplant): line {ln2}")
    del out_dir

    # B. AST check of the attention classes
    from vllm.models.glm5next.nvidia.kda import Glm5NextLinearAttention
    check(W._attn_forward_ok(Glm5NextLinearAttention), "B Glm5NextLinearAttention.forward passes the AST check")
    try:
        from vllm.models.glm5next.nvidia.attention import Glm5NextMLAAttention as MLA
        check(not W._attn_forward_ok(MLA), "B Glm5NextMLAAttention.forward fails the AST check")
    except ImportError as e:
        print("B MLA import skipped:", e)
    import fp8ag_fake as FK
    check(W._attn_forward_ok(FK.GoodAttn) and not W._attn_forward_ok(FK.BadAttn), "B fixture Good passes, Bad fails")

    # C. numerics, emulated 2 ranks
    g = torch.Generator(device=dev).manual_seed(9)
    for T in (13824, 4289, 6000, 1024):
        x_full = (torch.randn(T, 4096, device=dev, generator=g) * 0.7).to(torch.bfloat16)
        x_full[:, 7] *= 30.0                                             # an outlier channel
        ref = cls.apply(m0, lay0, x_full)
        shards, qs = quant_shards(W, x_full)
        W.AG.gather = lambda t: torch.cat([t, qs[1][0].view(torch.uint8) if t.dtype == torch.uint8 else qs[1][1]], 0)
        c0 = dict(W.COUNTERS)
        ph = W.fp8ag_gather(shards[0], lay0)
        y = cls.apply(m0, lay0, ph[:T])
        served = W.COUNTERS.get("fp8ag_served", 0) - c0.get("fp8ag_served", 0)
        check(torch.equal(y, ref) and served == 1 and W.AG.pending is None,
              f"C T={T}: in_proj(fp8-gathered placeholder) == W8A8 in_proj(bf16 full) bitwise "
              f"(max |d| {(y.float() - ref.float()).abs().max().item():.3g}), served {served}, pending cleared")
        # full-T quant == concatenated shard quants (the row-locality the whole item rests on)
        qf, sf = W.quant_per_token(x_full)
        qc = torch.cat([qs[0][0], qs[1][0]])[:T]
        sc = torch.cat([qs[0][1], qs[1][1]])[:T]
        check(torch.equal(qf.view(torch.uint8), qc.view(torch.uint8)) and torch.equal(sf, sc),
              f"C T={T}: per-token quant of the full rows == the shard quants concatenated (bytes and scales)")
        W.AG.gather = None

    # D. the wrapper through the decoder-shaped fixture
    T = 6000
    x_full = (torch.randn(T, 4096, device=dev, generator=g) * 0.7).to(torch.bfloat16)
    shards, qs = quant_shards(W, x_full)
    ref = cls.apply(m0, lay0, x_full)

    def bf16_gather(t):                                                  # the 'other rank' holds shard 1
        return torch.cat([t, shards[1] if t.shape[1] == 4096 else t], 0)

    W.AG.orig = bf16_gather
    W.AG.gather = lambda t: torch.cat([t, qs[1][0].view(torch.uint8) if t.dtype == torch.uint8 else qs[1][1]], 0)
    votes = []
    W.AG.agree = lambda b: (votes.append(b), b)[1]
    FK.sp_all_gather = W._sp_all_gather_fp8
    pos = torch.arange(T, device=dev)
    dec = FK.FakeDecoder(FK.GoodAttn(lay0))
    c0 = dict(W.COUNTERS)
    y, x2 = dec(pos, shards[0])
    dg = W.COUNTERS.get("fp8ag_gathers", 0) - c0.get("fp8ag_gathers", 0)
    ds = W.COUNTERS.get("fp8ag_served", 0) - c0.get("fp8ag_served", 0)
    check(torch.equal(y, ref) and dg == 1 and ds == 1 and votes == [True] and x2.shape == (T, 64),
          f"D fixture: attention gather fp8 ({dg} gather, {ds} served), output == W8A8 bf16 path bitwise, second "
          f"gather bf16 (shape {tuple(x2.shape)}), one agreement vote {votes}")
    y2, _ = dec(pos, shards[0])
    check(torch.equal(y2, ref) and votes == [True], "D second forward: cached decision (no new vote), same output")
    dec_no = FK.FakeDecoder(FK.GoodAttn(lay0))
    W.AG.agree = lambda b: (votes.append(b), False)[1]
    c0 = dict(W.COUNTERS)
    y3, _ = dec_no(pos, shards[0])
    check(torch.equal(y3, ref) and W.COUNTERS.get("fp8ag_gathers", 0) == c0.get("fp8ag_gathers", 0),
          "D agreement False (the other rank votes no): bf16 gather, same output")
    W.AG.agree = lambda b: (votes.append(b), b)[1]
    dec_bad = FK.FakeDecoder(FK.BadAttn(lay0))
    c0 = dict(W.COUNTERS)
    nv = len(votes)
    dec_bad(pos, shards[0])
    check(W.COUNTERS.get("fp8ag_gathers", 0) == c0.get("fp8ag_gathers", 0) and votes[nv:] == [False],
          "D BadAttn (reads hidden_states elsewhere): votes no, bf16 gather")

    # E. a pending gather met by another projection
    ph = W.fp8ag_gather(shards[0], lay0)
    c0 = dict(W.COUNTERS)
    x_other = (torch.randn(512, 4096, device=dev, generator=g)).to(torch.bfloat16)
    cls.apply(m1, lay1, x_other)
    mat = W.COUNTERS.get("fp8ag_materialized", 0) - c0.get("fp8ag_materialized", 0)
    qc = torch.cat([qs[0][0], qs[1][0]])
    sc = torch.cat([qs[0][1], qs[1][1]])
    check(mat == 1 and W.AG.pending is None and torch.equal(ph, (qc.float() * sc).to(torch.bfloat16)),
          "E pending + another projection first: placeholder materialized as bf16(q*s), counted")
    W.AG.orig = W.AG.gather = W.AG.agree = None

    # F. timing: quant full vs shard
    for T in (13824, 4289):
        x_full = torch.randn(T, 4096, device=dev, generator=g).to(torch.bfloat16)
        shards, _ = quant_shards(W, x_full)
        s0 = shards[0].contiguous()
        tf = tmed(lambda: W.quant_per_token(x_full))
        ts = tmed(lambda: W.quant_per_token(s0))
        print(f"F T={T}: per-token quant full {tf:.3f} ms -> shard {ts:.3f} ms ({ts - tf:+.3f} per KDA layer); "
              f"gather payload {T * 4096 * 2 / 2**20:.1f} MiB bf16 -> {T * 4096 / 2**20 + T * 4 / 2**20:.1f} MiB fp8+scales",
              flush=True)
    print("counters:", W.summary())


def _nccl_worker(rank, port, q):
    os.environ.update({"NCCL_HOSTID": f"fp8ag-host{rank}", "NCCL_IB_DISABLE": "1", "NCCL_SOCKET_IFNAME": "lo",
                       "NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "1", "NCCL_NVLS_ENABLE": "0",
                       "NCCL_DEBUG": "WARN"})
    try:
        torch.cuda.set_device(0)
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
        from vllm.distributed.parallel_state import set_custom_all_reduce
        set_custom_all_reduce(False)              # one GPU, two "hosts": no CUDA-IPC custom all-reduce
        with set_current_vllm_config(VllmConfig()):
            init_distributed_environment(world_size=2, rank=rank, local_rank=0,
                                         distributed_init_method=f"tcp://127.0.0.1:{port}", backend="nccl")
            ensure_model_parallel_initialized(2, 1)
        from vllm.distributed import tensor_model_parallel_all_gather, get_tp_group
        W, cls, mk = setup_layer(seed=0)                        # same weights on both ranks (seeded)
        m0, lay0 = mk[0]
        g = torch.Generator(device=dev).manual_seed(21)
        res = []
        for T in (13824, 4289):
            x_full = (torch.randn(T, 4096, device=dev, generator=g) * 0.7).to(torch.bfloat16)
            pad = (-T) % 2
            xp = torch.nn.functional.pad(x_full, (0, 0, 0, pad)) if pad else x_full
            R = xp.shape[0] // 2
            shard = xp[rank * R:(rank + 1) * R].contiguous()
            ref = cls.apply(m0, lay0, x_full)
            gat = tensor_model_parallel_all_gather(shard, 0)[:T]
            ok_g = torch.equal(gat, x_full)
            agreed = W._agree_min(True)
            agreed_mixed = W._agree_min(rank == 0)
            ph = W.fp8ag_gather(shard, lay0)
            y = cls.apply(m0, lay0, ph[:T])
            ok = torch.equal(y, ref) and ok_g and agreed and not agreed_mixed
            torch.cuda.synchronize()

            def bf16():
                tensor_model_parallel_all_gather(shard, 0)

            def fp8():
                W.fp8ag_gather(shard, lay0)
                W.AG.pending = None

            get_tp_group().barrier()
            tb = tmed(bf16, 7)
            get_tp_group().barrier()
            t8 = tmed(fp8, 7)
            res.append((T, ok, ok_g, agreed, agreed_mixed, tb, t8))
        q.put((rank, res, None))
    except Exception as e:  # noqa: BLE001
        import traceback
        q.put((rank, None, traceback.format_exc()))


def nccl():
    import socket
    import torch.multiprocessing as mp
    with socket.socket() as s_:
        s_.bind(("127.0.0.1", 0))
        port = s_.getsockname()[1]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=_nccl_worker, args=(r, port, q)) for r in range(2)]
    for p in ps:
        p.start()
    out = [q.get(timeout=900) for _ in ps]
    for p in ps:
        p.join(60)
    for rank, res, err in sorted(out, key=lambda t: t[0]):
        if err:
            check(False, f"rank {rank} raised:\n{err}")
            continue
        for T, ok, ok_g, agreed, agreed_mixed, tb, t8 in res:
            check(ok, f"NCCL rank {rank} T={T}: bf16 gather == x_full {ok_g}; agree(True,True)={agreed}, "
                      f"agree(rank0 only)={agreed_mixed}; in_proj(fp8 gather) == W8A8(bf16 full) bitwise")
            print(f"NCCL rank {rank} T={T}: loopback wall bf16 AG {tb:.2f} ms vs shard quant + fp8 AG + scale AG "
                  f"{t8:.2f} ms (loopback socket, NOT production's RoCE)", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "nccl":
        nccl()
    else:
        single()
    print("RESULT:", "FAIL" if FAILS else "PASS", f"({len(FAILS)} failures)")
    sys.exit(1 if FAILS else 0)
