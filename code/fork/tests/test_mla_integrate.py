"""GLM53_MLA_PREFILL wiring against production's own backend file (flashinfer_mla_sparse_sm90.py.patched, mounted by
tests/mla_env.sh): the wrapped FlashInferMLASparseSM90Impl.forward_mqa must (1) be inert without the env var, (2) serve
eligible eager calls with the exact kernel and match production's forward_mqa (planned with the exact valid counts)
within FA2's own error, (3) hand small / captured / non-eligible calls to production unchanged (bitwise)."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mla_prefill_common as C  # noqa: E402

FAIL = []


def check(name, ok, msg=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {msg}", flush=True)
    if not ok:
        FAIL.append(name)


def main() -> int:
    import importlib
    os.environ.pop("GLM53_MLA_PREFILL", None)
    import glm53_mla_prefill as M
    mod = importlib.import_module(M.TARGET_MODULE)
    orig = mod.FlashInferMLASparseSM90Impl.forward_mqa
    M.plugin_install()
    check("inert without GLM53_MLA_PREFILL", mod.FlashInferMLASparseSM90Impl.forward_mqa is orig)
    os.environ["GLM53_MLA_PREFILL"] = "1"
    os.environ["TF_EXL3_JIT"] = "1"
    M.plugin_install()
    wrapped = mod.FlashInferMLASparseSM90Impl.forward_mqa
    check("installed", wrapped is not orig and getattr(wrapped, "__glm53_mla_prefill__", False))
    M.plugin_install()
    check("idempotent", mod.FlashInferMLASparseSM90Impl.forward_mqa is wrapped)

    check(f"default kv_rows = max(min_tokens, 1024) + 1", M.STATE.kv_rows == max(M.STATE.min_tokens, 1024) + 1,
          str(M.STATE.kv_rows))
    for T, start, regime in ((2048, 12000, "sticky"), (300, 0, "indep")):
        case = C.Case(T, start, regime, seed=21)
        state = mod._SM90State(torch.device("cuda"), 32, torch.float8_e4m3fn, T, C.TOPK, kv_lora_rank=512,
                               qk_rope_head_dim=0, sm_scale=C.SM_SCALE)
        mod._SM90_STATE = state
        state.plan(T, case.valid.cpu())           # exact lengths (production plans 2048 + ctx % 4: see docs)
        topk_buf = torch.full((T, C.TOPK), -1, dtype=torch.int32, device="cuda")
        topk_buf.copy_(case.topk)
        impl = SimpleNamespace(num_heads=32, kv_lora_rank=512, qk_rope_head_dim=0, head_size=512,
                               use_fp8_kv_cache=True, topk_indices_buffer=topk_buf, scale=C.SM_SCALE)
        meta = SimpleNamespace(req_id_per_token=case.req_id, block_table=case.block_table, block_size=C.PBS,
                               num_decode_tokens=0)
        layer = SimpleNamespace(_k_scale_float=1.0)
        # production layout: ql_nope = bmm output [N, B, L] transposed to [B, N, L]
        q_nope = case.q.transpose(0, 1).contiguous().transpose(0, 1)
        q_pe = q_nope.new_empty(T, 32, 0)
        cache = case.cache.view(torch.float8_e4m3fn)
        state.kv_indices.fill_(-7)                 # sentinel: what an older call left in the process-wide buffer
        o_ref, _ = orig(impl, (q_nope, q_pe), cache, meta, layer)
        ki_prod = state.kv_indices.clone()
        state.kv_indices.fill_(-7)
        calls0 = M.STATE.calls
        o_new, lse = wrapped(impl, (q_nope, q_pe), cache, meta, layer)
        check(f"T={T} served by the exact kernel", M.STATE.calls == calls0 + 1 and lse is None)
        # decode's FA2 plan reads past its rows into this buffer (docs/HANDOFF.md): the wrapper must leave the exact
        # bytes production leaves (whole buffer: rows [0, T) written, the rest untouched)
        # opt-kdamhc: the write-back covers the first STATE.kv_rows rows (every row an FA2 call - < min_tokens rows -
        # can read beyond its own); rows past it keep the older call's entries
        R = M.STATE.kv_rows
        W = C.TOPK
        rows = T if R is None else min(T, R)
        exp = ki_prod.clone()
        exp[rows * W:] = -7
        check(f"T={T} kv_indices rows [0, {rows}) == production's (bitwise), rows past it untouched "
              f"(kv_rows {R}, min_tokens {M.STATE.min_tokens})", torch.equal(state.kv_indices, exp)
              and (R is None or R > M.STATE.min_tokens))
        reach = max(n * W + 3 for n in range(1, M.STATE.min_tokens))   # last entry an FA2 call of n rows reads
        check(f"T={T} every entry an FA2 call (< {M.STATE.min_tokens} rows) reads past its rows == production's",
              torch.equal(state.kv_indices[:min(reach + 1, T * W)], ki_prod[:min(reach + 1, T * W)]))
        M.STATE.kv_rows = None                       # GLM53_MLA_PREFILL_KV_ROWS=all: the r16 whole-buffer behaviour
        state.kv_indices.fill_(-7)
        wrapped(impl, (q_nope, q_pe), cache, meta, layer)
        check(f"T={T} KV_ROWS=all: kv_indices left for decode == production's (bitwise, whole buffer)",
              torch.equal(state.kv_indices, ki_prod))
        M.STATE.kv_rows = R
        M.STATE.write_kv_indices = False             # the deploy-r16 defect (test-only switch)
        state.kv_indices.fill_(-7)
        wrapped(impl, (q_nope, q_pe), cache, meta, layer)
        check(f"T={T} regression guard: without the write the buffer keeps the older call's entries",
              not torch.equal(state.kv_indices, ki_prod) and bool((state.kv_indices == -7).all()))
        M.STATE.write_kv_indices = True
        check(f"T={T} output layout == production's", o_new.stride() == o_ref.stride() and o_new.shape == o_ref.shape,
              f"{o_new.stride()} vs {o_ref.stride()}")
        ref = C.reference(case, torch.arange(T))
        st_new, st_fa = C.err_stats(o_new, ref), C.err_stats(o_ref, ref)
        check(f"T={T} parity vs fp32 within FA2's", st_new["rel_l2_max"] <= 1.5 * st_fa["rel_l2_max"] + 1e-4,
              f"new {st_new} | production {st_fa}")
        # fallbacks are bitwise production
        M.STATE.min_tokens = T + 1
        o_small, _ = wrapped(impl, (q_nope, q_pe), cache, meta, layer)
        check(f"T={T} below MIN_TOKENS -> production (bitwise)", torch.equal(o_small, o_ref))
        M.STATE.min_tokens = 256
        impl_rope = SimpleNamespace(**{**impl.__dict__, "qk_rope_head_dim": 64})
        why = M._ineligible(impl_rope, (q_nope, q_pe), cache, T)
        check(f"T={T} rope geometry refused", why == "rope", str(why))
        impl_bf16 = SimpleNamespace(**{**impl.__dict__, "use_fp8_kv_cache": False})
        check(f"T={T} bf16 cache refused", M._ineligible(impl_bf16, (q_nope, q_pe), cache, T) == "kv dtype")
        M.STATE.mixed = False
        meta_mixed = SimpleNamespace(**{**meta.__dict__, "num_decode_tokens": 4})
        o_mixed, _ = wrapped(impl, (q_nope, q_pe), cache, meta_mixed, layer)
        check(f"T={T} GLM53_MLA_PREFILL_MIXED=0 keeps mixed steps on production", torch.equal(o_mixed, o_ref))
        M.STATE.mixed = True
    # capture: a graph captured through the wrapper must contain production's kernel
    T = 512
    case = C.Case(T, 4000, "indep", seed=23)
    state = mod._SM90State(torch.device("cuda"), 32, torch.float8_e4m3fn, T, C.TOPK, kv_lora_rank=512,
                           qk_rope_head_dim=0, sm_scale=C.SM_SCALE)
    mod._SM90_STATE = state
    state.plan(T, case.valid.cpu())
    topk_buf = case.topk.clone()
    impl = SimpleNamespace(num_heads=32, kv_lora_rank=512, qk_rope_head_dim=0, head_size=512, use_fp8_kv_cache=True,
                           topk_indices_buffer=topk_buf, scale=C.SM_SCALE)
    meta = SimpleNamespace(req_id_per_token=case.req_id, block_table=case.block_table, block_size=C.PBS,
                           num_decode_tokens=0)
    layer = SimpleNamespace(_k_scale_float=1.0)
    q_nope = case.q.transpose(0, 1).contiguous().transpose(0, 1)
    q_pe = q_nope.new_empty(T, 32, 0)
    cache = case.cache.view(torch.float8_e4m3fn)
    o_eager_prod, _ = orig(impl, (q_nope, q_pe), cache, meta, layer)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        wrapped(impl, (q_nope, q_pe), cache, meta, layer)       # warm-up outside capture
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    before = dict(M.STATE.fallbacks)
    with torch.cuda.graph(g):
        o_cap, _ = wrapped(impl, (q_nope, q_pe), cache, meta, layer)
    g.replay()
    torch.cuda.synchronize()
    check("captured call -> production kernel (bitwise)",
          M.STATE.fallbacks.get("capturing", 0) == before.get("capturing", 0) + 1 and torch.equal(o_cap, o_eager_prod))
    print("fallback counts", M.STATE.fallbacks, "calls", M.STATE.calls)
    # Production's host-planned FA2 lengths vs the valid counts the real indexer layout produces (image ops:
    # expand_pools_and_append_tail(pool_ids[:, :511]) as in sparse_attn_indexer_kpool.py:580/874, then the backend's
    # triton_convert_req_index_to_global_index). The new kernel reads the device-side valid counts instead.
    from vllm.models.glm5next.nvidia.ops.kpool_compress import expand_pools_and_append_tail
    pos = torch.cat([torch.arange(2040, 2056), torch.arange(5000, 5004), torch.arange(99996, 100000)]).cuda()
    seq = (pos + 1).to(torch.int32)
    pool_len = seq // 4
    g = torch.Generator(device="cpu").manual_seed(0)
    pool_ids = torch.full((pos.numel(), 512), -1, dtype=torch.int64)
    for r, n in enumerate(pool_len.tolist()):
        sel = torch.randperm(n, generator=g)[:512]
        pool_ids[r, : sel.numel()] = sel
    expanded = expand_pools_and_append_tail(pool_ids.cuda()[:, :511], seq, 4)
    buf = torch.full((pos.numel(), C.TOPK), -1, dtype=torch.int32, device="cuda")
    buf[:, : expanded.shape[1]] = expanded
    nb = (100000 + 63) // 64
    bt = torch.arange(nb, dtype=torch.int32, device="cuda")[None]
    _, valid = C.convert(torch.zeros(pos.numel(), dtype=torch.int32, device="cuda"), bt, buf)
    ctx = pos + 1
    plan = torch.where(ctx <= C.TOPK, ctx, C.TOPK + ctx % 4)
    extra = (plan - valid.long()).cpu()
    print("ctx      ", ctx.tolist())
    print("valid    ", valid.tolist())
    print("prod plan", plan.tolist())
    check("real indexer layout: valid = ctx (ctx < 2048) else 2044 + ctx % 4",
          bool((valid.long().cpu() == torch.where(ctx < 2048, ctx, 2044 + ctx % 4).cpu()).all()))
    print(f"INFO production FA2 plans {int(extra.max())} extra keys on {int((extra > 0).sum())}/{extra.numel()} rows "
          "(every row with ctx >= 2048): the -1 tail is clamped to slot 0 and rows with ctx % 4 != 0 also read the "
          "first ctx % 4 entries of the NEXT row's kv_indices")
    print("ALL PASSED" if not FAIL else f"FAILED: {FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
