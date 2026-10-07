"""mhc2 tests on nodeC (one GPU, single process): GLM53_MHC_SP2 (overlay/patch_mhc_sp2.py) on top of r16x's GLM53_MHC_SP.

1. patch mechanics: GLM53_MHC_SP2=0 touches nothing; =1 without GLM53_MHC_SP=1 refuses; =1 on a model.py that
   patch_mhc_sp.py did not patch refuses (fail-closed, nothing written); =1 on the r16x-patched copy patches model.py +
   extends the quickwins / moeglue fingerprint lines; a second run is a no-op (byte-identical).
2. fingerprint chain: the live SP2-patched forwards' fingerprints are the patch's; quickwins mhc_aux + mhc_mean
   transplants succeed on the SP2 source; moeglue WARM_VERIFIED accepts the transplanted layer forward.
3. kernel mirror: _sp2_post_pre_into on sub-chunk row slices == the production mhc_fused_post_pre_tilelang on the
   same rows, bitwise, for sub-chunks of 3,456 / 1,728 / 1,538 rows (and == the rows of a whole-shard call).
4. 2-rank emulation (tests/mhc_sp/test_mhc_sp_all.py's harness: real patched forwards, production mHC ops, rank-
   sharded projections, a CROSS-TOKEN attention core): for T in (6,400 = pipelined k=2, 6,403 = pipelined with a
   padded last block, 3,075 = odd T on the single-shard path, 4,096 = r16x's case) the SP2 run is BITWISE equal to
   the TP run (every layer's wired inputs, final hidden states, aux states, both ranks); collective counts match the
   plan; decode T=8 stays TP; quickwins mhc_aux/mhc_mean live on top.

Run: flock /tmp/tf-gpu-bench.lock tests/gpu_run.sh python3 tests/mhc2/test_mhc_sp2.py
"""
from __future__ import annotations

import importlib
import io
import os
import shutil
import sys
import tempfile
import threading

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "overlay"))
sys.path.insert(0, os.path.join(ROOT, "tests", "mhc_sp"))
import test_mhc_sp_all as H  # noqa: E402  (the r16x 2-rank emulation harness)

FAILURES: list[str] = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg, flush=True)
    if not cond:
        FAILURES.append(msg)


class PerRank(dict):
    """gm._SP2 replacement: one state dict per emulated rank (the two ranks are threads of one process)."""

    def __init__(self, proto):
        super().__init__()
        self.proto, self.tl = dict(proto), threading.local()

    def _d(self):
        d = getattr(self.tl, "d", None)
        if d is None:
            d = self.tl.d = dict(self.proto)
        return d

    def __getitem__(self, k):
        return self._d()[k]

    def __setitem__(self, k, v):
        self._d()[k] = v

    def update(self, *a, **k):
        self._d().update(*a, **k)


def mirror_test():
    print("\n== 3. kernel mirror: _sp2_post_pre_into on row slices vs the production op")
    from vllm.model_executor.kernels.mhc.tilelang import mhc_fused_post_pre_tilelang
    sys.path.insert(0, os.path.join(ROOT, "tests", "mhc_sp"))
    from bench_mhc_halving import make
    import types
    src = open(os.environ["GLM53_GLM5NEXT_MODEL_PY"]).read()
    a = src.index("def _sp2_post_pre_into(")
    b = src.index("def _sp2_fused_post_pre_gather(")
    ns = {"torch": torch}
    exec(compile(src[a:b], "<sp2 mirror>", "exec"), ns)
    into = ns["_sp2_post_pre_into"]
    args = (1e-5, 1e-6, 1e-6, 2.0, 20)
    for S, k in ((6912, 2), (6912, 4), (3076, 2)):
        w = make(S, seed=3)
        full = mhc_fused_post_pre_tilelang(w["x"], w["res"], w["post"], w["comb"], w["fn"], w["scale"], w["base"],
                                           *args, 1, 1, w["norm"], 1e-5)
        B = S // k
        rc = torch.empty_like(w["res"])
        pc = torch.empty(S, 4, dtype=torch.float32, device="cuda")
        cc = torch.empty(S, 16, dtype=torch.float32, device="cuda")
        li = torch.empty(S, 4096, dtype=torch.bfloat16, device="cuda")
        ok_sub = True
        for j in range(k):
            lo, hi = j * B, (j + 1) * B
            into(w["x"][lo:hi], w["res"][lo:hi], w["post"].reshape(S, 4)[lo:hi], w["comb"][lo:hi], w["fn"],
                 w["scale"], w["base"], *args, w["norm"], 1e-5, rc[lo:hi], pc[lo:hi], cc[lo:hi], li[lo:hi])
            ref = mhc_fused_post_pre_tilelang(w["x"][lo:hi], w["res"][lo:hi], w["post"][lo:hi], w["comb"][lo:hi],
                                              w["fn"], w["scale"], w["base"], *args, 1, 1, w["norm"], 1e-5)
            ok_sub &= (torch.equal(rc[lo:hi], ref[0]) and torch.equal(pc[lo:hi], ref[1].reshape(B, 4))
                       and torch.equal(cc[lo:hi], ref[2].reshape(B, 16)) and torch.equal(li[lo:hi], ref[3]))
        torch.cuda.synchronize()
        ok_full = (torch.equal(rc, full[0]) and torch.equal(pc, full[1].reshape(S, 4))
                   and torch.equal(cc, full[2].reshape(S, 16)) and torch.equal(li, full[3]))
        check(ok_sub, f"S={S} k={k} (sub-chunks of {B} rows): mirror == production op on the same rows, bitwise")
        check(ok_full, f"S={S} k={k}: sub-chunk rows == the whole-shard production call's rows, bitwise")
        del w, full
        torch.cuda.empty_cache()


def emulation(gm, P2, ex, T, expect_k, bitwise=True):
    """r16x SP disabled vs SP2: both through the same patched module (MHC_SP_MIN_TOKENS chooses SP vs TP)."""
    H.T_PREFILL = T
    w = H.make_weights()
    sp_out, sp_counts, _ = H.run_both_ranks(gm, w, ex, T, True)
    ks = sorted({d.get("k") for d in getattr(gm._SP2, "seen", [])}) if hasattr(gm._SP2, "seen") else None
    tp_out, tp_counts, _ = H.run_both_ranks(gm, w, ex, T, False)
    L = H.LAYERS
    # every collective of an SP2 forward: per phase k AG + k RS (the standalone-pre phases do k AG too), aux k AG,
    # final k AG
    exp_ag = (2 * L + len(H.AUX_AT) + 1) * expect_k
    exp_rs = 2 * L * expect_k
    check(sp_counts["ag"] == exp_ag and sp_counts["rs"] == exp_rs and sp_counts["ar"] == 0,
          f"T={T}: SP2 collectives {sp_counts} == plan (k={expect_k}: {exp_ag} AG / {exp_rs} RS / 0 AR)")
    (h_sp, aux_sp, st_sp, sti_sp), (h_tp, aux_tp, st_tp, sti_tp) = sp_out[0], tp_out[0]
    if not bitwise:
        # shard < 1,537 rows: compute_num_split picks split_k > 1 on the shard and 1 on the full T (r16x's own
        # tolerance class, docs/MHC_SP.md section 3): report the size of the fp32-order difference instead
        d = (h_sp.float() - h_tp.float())
        rel = (d.norm() / h_tp.float().norm()).item()
        print(f"       T={T}: final hidden maxabs {d.abs().max().item():.3e}, rel-L2 {rel:.3e}, "
              f"bitwise-equal elements {(h_sp == h_tp).float().mean().item() * 100:.2f} %")
        check(rel < 2e-2 and torch.isfinite(h_sp).all().item(),
              f"T={T}: final hidden states within the split-K fp32-order tolerance (rel-L2 {rel:.2e})")
        return rel
    check(torch.equal(h_sp, h_tp) and torch.equal(sp_out[1][0], tp_out[1][0]),
          f"T={T}: final hidden states, both ranks: SP2 == TP bitwise")
    check(len(aux_sp) == len(H.AUX_AT) and all(torch.equal(a, b) for a, b in zip(aux_sp, aux_tp))
          and all(torch.equal(a, b) for a, b in zip(sp_out[1][1], tp_out[1][1])),
          f"T={T}: aux hidden states, both ranks: SP2 == TP bitwise")
    # the rank shards: rank r's rows in the global order (k interleaved blocks of B, padded rows dropped)
    k = expect_k
    N = H.TP
    B = -(-T // (N * k))
    def rows_of(r, S):
        if k == 1:
            half = -(-T // N)
            idx = list(range(r * half, min((r + 1) * half, T)))
            return idx, len(idx)
        idx = []
        for j in range(k):
            lo = (r + N * j) * B
            idx += [i for i in range(lo, lo + B) if i < T]
        return idx, len(idx)
    bad = []
    for i in range(L):
        for f in range(4):
            b = st_tp[i][f]
            for r in range(N):
                a = sp_out[r][2][i][f]
                if not torch.is_tensor(a) or not torch.is_tensor(b):
                    if not (a is None and b is None):
                        bad.append((i, f, r, "none"))
                    continue
                idx, n = rows_of(r, a.shape[0])
                if k == 1:
                    sel = a[:n]
                else:
                    keep = []
                    for j in range(k):
                        lo = (r + N * j) * B
                        keep += [j * B + t for t in range(B) if lo + t < T]
                    sel = a[keep]
                if not H.eq_maybe_none(sel, b[idx]):
                    bad.append((i, f, r))
    check(not bad, f"T={T}: every layer's (x, residual, post, comb), both ranks' shards == TP rows bitwise "
                   f"(bad: {bad[:6]})")


def main():
    tdir = tempfile.mkdtemp(prefix="mhcsp2-")
    os.environ["GLM53_GLM5NEXT_MODEL_PY"] = os.path.join(tdir, "model.py")
    os.environ["GLM53_QUICKWINS_PY"] = os.path.join(tdir, "qw.py")
    os.environ["GLM53_MOEGLUE_PY"] = os.path.join(tdir, "mg.py")
    shutil.copy(H.MODEL_IMG, os.environ["GLM53_GLM5NEXT_MODEL_PY"])
    shutil.copy(H.QW_SRC, os.environ["GLM53_QUICKWINS_PY"])
    shutil.copy(H.MG_SRC, os.environ["GLM53_MOEGLUE_PY"])
    MP = os.environ["GLM53_GLM5NEXT_MODEL_PY"]
    try:
        import patch_mhc_sp as P
        import patch_mhc_sp2 as P2
        print("== 1. patch mechanics")
        stock = open(MP).read()
        os.environ["GLM53_MHC_SP2"] = "1"
        os.environ["GLM53_MHC_SP"] = "1"
        try:
            P2.main()
            check(False, "SP2 on a stock (non-SP) model.py refuses")
        except SystemExit as e:
            check("does not carry the r16x" in str(e) and open(MP).read() == stock,
                  f"SP2 on a stock model.py refuses, nothing written ({str(e)[:80]})")
        buf, saved = io.StringIO(), sys.stdout
        sys.stdout = buf
        try:
            P.main()
        finally:
            sys.stdout = saved
        r16x = {p: open(p).read() for p in (MP, os.environ["GLM53_QUICKWINS_PY"], os.environ["GLM53_MOEGLUE_PY"])}
        os.environ["GLM53_MHC_SP2"] = "0"
        check(P2.main() == 0 and all(open(p).read() == s for p, s in r16x.items()),
              "GLM53_MHC_SP2=0: files untouched (== r16x SP bytes)")
        os.environ["GLM53_MHC_SP2"] = "1"
        os.environ["GLM53_MHC_SP"] = ""
        try:
            P2.main()
            check(False, "SP2 without GLM53_MHC_SP=1 refuses")
        except SystemExit as e:
            check("needs GLM53_MHC_SP=1" in str(e) and all(open(p).read() == s for p, s in r16x.items()),
                  "GLM53_MHC_SP2=1 without GLM53_MHC_SP=1 refuses, nothing written")
        os.environ["GLM53_MHC_SP"] = "1"
        check(P2.main() == 0, "SP2 installs on the r16x SP-patched copy")
        after = {p: open(p).read() for p in r16x}
        check(all(after[p] != r16x[p] for p in r16x), "model.py + both fingerprint tables changed")
        sys.stdout = buf = io.StringIO()
        try:
            rc = P2.main()
        finally:
            sys.stdout = saved
        check(rc == 0 and "already present" in buf.getvalue() and all(open(p).read() == after[p] for p in after),
              "second run: no-op, byte-identical")
        # the r16x SP patch must also stay a no-op on the SP2 tree (bundle second pass: SP runs before SP2)
        sys.stdout = buf = io.StringIO()
        try:
            rc = P.main()
        finally:
            sys.stdout = saved
        check(rc == 0 and all(open(p).read() == after[p] for p in after),
              f"second bundle pass: patch_mhc_sp.py on the SP2 tree is a no-op ({buf.getvalue().strip()[:70]})")

        print("\n== 2. fingerprint chain")
        import integrate
        importlib.invalidate_caches()
        gm = importlib.import_module("vllm.models.glm5next.nvidia.model")
        import linecache
        fname = "<glm53-mhc-sp2 patched model.py>"
        linecache.cache[fname] = (len(after[MP]), None, after[MP].splitlines(True), fname)
        exec(compile(after[MP], fname, "exec"), gm.__dict__)
        fps = P2.sp2_fingerprints(after[MP])
        check(integrate.source_fingerprint(gm.Glm5NextDecoderLayer.forward) == fps["layer_forward_sp2"],
              "live layer forward fingerprint == layer_forward_sp2")
        check(integrate.source_fingerprint(gm.Glm5NextModel.forward) == fps["model_forward_sp2"],
              "live model forward fingerprint == model_forward_sp2")
        os.environ["GLM53_PREFILL_QUICKWINS"] = "mhc_aux,mhc_mean"
        QW = H._load("qw_patched2", os.environ["GLM53_QUICKWINS_PY"])
        r = QW.install(["mhc_aux", "mhc_mean"])
        check("mhc_aux" in r["items"] and "mhc_mean" in r["items"]
              and not ({"mhc_aux", "mhc_mean"} & set(QW._STATE["refused"])),
              f"quickwins mhc_aux+mhc_mean transplant on the SP2 source (refused: {QW._STATE.get('refused')})")
        MG = H._load("mg_patched2", os.environ["GLM53_MOEGLUE_PY"])
        bad = [k for k, v in MG._vllm_fingerprints().items() if v not in MG.WARM_VERIFIED[k]]
        check(not bad, f"moeglue WARM_VERIFIED accepts the SP2+quickwins layer forward (bad: {bad})")

        mirror_test()

        print("\n== 4. 2-rank emulation (production mHC ops, cross-token attention core)")
        import vllm.distributed as vd
        real = (vd.get_tensor_model_parallel_world_size, vd.get_pp_group,
                getattr(vd, "get_tensor_model_parallel_rank", None))
        vd.get_tensor_model_parallel_world_size = lambda: H.TP
        vd.get_tensor_model_parallel_rank = lambda: H.TL.rank
        pp = type("PP", (), {"is_first_rank": True, "is_last_rank": True, "world_size": 1})()
        vd.get_pp_group = lambda: pp
        gm.get_pp_group = lambda: pp
        gm.get_tensor_model_parallel_world_size = lambda: H.TP
        ex = H.Exchange()
        gm.sp_all_gather = lambda x: ex.all_gather(H.TL.rank, x)

        def _rs(x):     # the stock sp_reduce_scatter pads to a multiple of tp (odd T)
            pad = (-x.shape[0]) % H.TP
            if pad:
                x = torch.nn.functional.pad(x, (0, 0, 0, pad))
            return ex.reduce_scatter(H.TL.rank, x)
        gm.sp_reduce_scatter = _rs

        def _shard(x):  # the stock sp_shard (pads odd T)
            pad = (-x.shape[0]) % H.TP
            if pad:
                x = torch.nn.functional.pad(x, (0, 0) * (x.ndim - 1) + (0, pad))
            c = x.shape[0] // H.TP
            return x[H.TL.rank * c:(H.TL.rank + 1) * c]
        gm.sp_shard = _shard
        gm._SP2 = PerRank(gm._SP2)
        gm._sp2_comm = lambda: "emulated"
        gm._sp2_stream = lambda: torch.cuda.current_stream()   # stream mechanics: tests/mhc2/nccl_sp2_pipeline.py

        def _ag_into(out, inp, stream):
            out.copy_(ex.all_gather(H.TL.rank, inp))

        def _rs_into(out, inp, stream):
            out.copy_(ex.reduce_scatter(H.TL.rank, inp))
        gm._sp2_ag_into, gm._sp2_rs_into = _ag_into, _rs_into
        for T, k in ((6400, 2), (6403, 2), (3075, 1), (4096, 1)):
            emulation(gm, P2, ex, T, k)
        # below 3,074 tokens the shard runs split-K (r16x's even T included): same tolerance class, odd or even
        r_even = emulation(gm, P2, ex, 3000, 1, bitwise=False)
        r_odd = emulation(gm, P2, ex, 3001, 1, bitwise=False)
        check(r_odd < 4 * max(r_even, 1e-6), f"odd T=3001 difference ({r_odd:.2e}) is of r16x's even-T=3000 "
                                              f"class ({r_even:.2e})")
        H.T_PREFILL = 6400
        w = H.make_weights()
        dec_sp, dec_counts, _ = H.run_both_ranks(gm, w, ex, H.T_DECODE, True)
        dec_tp, _, _ = H.run_both_ranks(gm, w, ex, H.T_DECODE, False)
        check(torch.equal(dec_sp[0][0], dec_tp[0][0]) and dec_counts["ag"] == 0 and dec_counts["rs"] == 0,
              f"decode T={H.T_DECODE}: TP path, no SP collectives, bitwise ({dec_counts})")
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            cap = bool(gm._mhc_sp_now(6400))
        check(cap is False, "the SP gate is False inside a CUDA-graph capture")
        os.environ["GLM53_PREFILL_QUICKWINS"] = ""
        QW.uninstall_item("mhc_aux")
        QW.uninstall_item("mhc_mean")
        vd.get_tensor_model_parallel_world_size, vd.get_pp_group = real[0], real[1]
    finally:
        for k in ("GLM53_MHC_SP", "GLM53_MHC_SP2", "GLM53_GLM5NEXT_MODEL_PY", "GLM53_QUICKWINS_PY",
                  "GLM53_MOEGLUE_PY"):
            os.environ.pop(k, None)
        shutil.rmtree(tdir, ignore_errors=True)
    print(f"\n{'ALL OK' if not FAILURES else f'{len(FAILURES)} FAILURES: ' + '; '.join(FAILURES)}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
