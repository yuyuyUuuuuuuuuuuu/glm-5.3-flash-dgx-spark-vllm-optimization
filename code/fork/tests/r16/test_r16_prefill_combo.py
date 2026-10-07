"""deploy-r16 prefill interaction: GLM53_PREFILL_QUICKWINS (mla_index + mla_bmm) and GLM53_MLA_PREFILL on the same
production functions (FlashInferMLASparseSM90Impl.forward_mqa of the production-mounted flashinfer_mla_sparse_sm90.py,
MLAAttention._v_up_proj), installed in the plugin order of integrate.plugin_register (quickwins first, MLA wraps the
recompiled forward_mqa). Run with tests/mla_env.sh (production's SM90_KV mounts) through tests/r16/gpu.sh.

  X.1 layering: forward_mqa = MLA wrapper, its production callee = quickwins' recompiled forward_mqa
  X.2 per case (T 2048 @ctx 12000 sticky, 300 @0 indep, 4608 @86000 indep), production's own planned FA2 lengths:
      a. quickwins' forward_mqa alone == production's (bitwise; its fast index path taken, T >= 256)
      b. the stack serves the call with the exact kernel (MLA counter), parity vs fp32 within FA2's error
      c. MIXED=0 + decode rows, and T below MLA's MIN_TOKENS -> the call falls through to quickwins' forward_mqa:
         bitwise == production's, quickwins' fast index path taken
      d. _v_up_proj (quickwins mla_bmm) on the MLA kernel's output (its layout) == production's bmm (bitwise)
  X.3 CUDA-graph capture through the stack -> production's kernel (both refuse while capturing), replay == eager
"""
from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import mla_prefill_common as C  # noqa: E402

FAIL = []


def check(name, ok, msg=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {msg}", flush=True)
    if not ok:
        FAIL.append(name)


def main() -> int:
    for v in [k for k in os.environ if k.startswith("GLM53_MLA_PREFILL") or k.startswith("GLM53_PREFILL_QUICKWINS")]:
        os.environ.pop(v)
    import glm53_mla_prefill as M
    import glm53_prefill_quickwins as Q
    sm = importlib.import_module(M.TARGET_MODULE)
    mla_mod = importlib.import_module(Q.M_MLA)
    Impl = sm.FlashInferMLASparseSM90Impl
    prod_fmqa = Impl.forward_mqa
    prod_vup = mla_mod.MLAAttention._v_up_proj
    Q.install(["mla_bmm", "mla_index"])
    qw_fmqa = Impl.forward_mqa
    check("X.1 quickwins recompiled forward_mqa", getattr(qw_fmqa, "_glm53_qw", False) and
          qw_fmqa._glm53_qw_orig is prod_fmqa)
    os.environ["GLM53_MLA_PREFILL"] = "1"
    M.plugin_install()
    stack = Impl.forward_mqa
    inner = [c.cell_contents for c in (stack.__closure__ or ()) if callable(c.cell_contents)
             and getattr(c.cell_contents, "__name__", "") == "forward_mqa"]
    check("X.1 MLA wrapper over quickwins' forward_mqa", getattr(stack, "__glm53_mla_prefill__", False) and
          inner == [qw_fmqa], f"inner {inner}")
    vup = mla_mod.MLAAttention._v_up_proj
    check("X.1 quickwins _v_up_proj", getattr(vup, "_glm53_qw", False))

    N, P, L, V = 32, 256, 512, 256
    g = torch.Generator(device="cuda").manual_seed(5)
    w = (torch.randn(N * (P + V), L, device="cuda", generator=g) * 0.05).to(torch.bfloat16)     # kv_b_proj.weight
    W_UV_t = w.T.view(L, N, P + V).split([P, V], dim=-1)[1].transpose(0, 1)                    # production's view
    me = SimpleNamespace(num_heads=N, kv_lora_rank=L, v_head_dim=V, W_UV=W_UV_t,
                         is_aiter_triton_fp4_bmm_enabled=False, is_aiter_triton_fp8_bmm_enabled=False)

    for T, start, regime in ((2048, 12000, "sticky"), (300, 0, "indep"), (4608, 86000, "indep")):
        case = C.Case(T, start, regime, seed=31)
        state = sm._SM90State(torch.device("cuda"), 32, torch.float8_e4m3fn, T, C.TOPK, kv_lora_rank=512,
                              qk_rope_head_dim=0, sm_scale=C.SM_SCALE)
        sm._SM90_STATE = state
        ctx = torch.arange(start + 1, start + T + 1)
        state.plan(T, torch.where(ctx <= C.TOPK, ctx, C.TOPK + ctx % 4).to(torch.int32))   # production's plan
        topk_buf = case.topk.clone()
        impl = SimpleNamespace(num_heads=32, kv_lora_rank=512, qk_rope_head_dim=0, head_size=512,
                               use_fp8_kv_cache=True, topk_indices_buffer=topk_buf, scale=C.SM_SCALE)
        meta = SimpleNamespace(req_id_per_token=case.req_id, block_table=case.block_table, block_size=C.PBS,
                               num_decode_tokens=0)
        layer = SimpleNamespace(_k_scale_float=1.0)
        q_nope = case.q.transpose(0, 1).contiguous().transpose(0, 1)
        q_pe = q_nope.new_empty(T, 32, 0)
        cache = case.cache.view(torch.float8_e4m3fn)
        o_prod, _ = prod_fmqa(impl, (q_nope, q_pe), cache, meta, layer)
        f0 = Q.STATS["mla_index_fast"]
        o_qw, _ = qw_fmqa(impl, (q_nope, q_pe), cache, meta, layer)
        check(f"X.2a T={T} quickwins forward_mqa == production (bitwise), fast index path",
              torch.equal(o_qw, o_prod) and Q.STATS["mla_index_fast"] == f0 + 1)
        c0 = M.STATE.calls
        o_new, lse = stack(impl, (q_nope, q_pe), cache, meta, layer)
        ref = C.reference(case, torch.arange(T))
        st_new, st_fa = C.err_stats(o_new, ref), C.err_stats(o_prod, ref)
        check(f"X.2b T={T} stack -> exact kernel, parity vs fp32 within FA2's",
              M.STATE.calls == c0 + 1 and lse is None and o_new.stride() == o_prod.stride() and
              st_new["rel_l2_max"] <= 1.5 * st_fa["rel_l2_max"] + 1e-4,
              f"new {st_new} | production (its plan) {st_fa}")
        M.STATE.mixed = False
        f0 = Q.STATS["mla_index_fast"]
        o_mix, _ = stack(impl, (q_nope, q_pe), cache, SimpleNamespace(**{**meta.__dict__, "num_decode_tokens": 4}),
                         layer)
        M.STATE.mixed = True
        M.STATE.min_tokens = T + 1
        o_small, _ = stack(impl, (q_nope, q_pe), cache, meta, layer)
        M.STATE.min_tokens = 256
        check(f"X.2c T={T} MIXED=0 / below MIN_TOKENS -> quickwins' forward_mqa == production (bitwise)",
              torch.equal(o_mix, o_prod) and torch.equal(o_small, o_prod) and Q.STATS["mla_index_fast"] == f0 + 2)
        for lbl, x in (("MLA kernel output", o_new), ("FA2 output", o_prod)):
            ref_o = torch.empty(T, N * V, device="cuda", dtype=torch.bfloat16)
            prod_vup(me, x, ref_o)
            got = torch.full_like(ref_o, 3.0)
            b0 = Q.STATS["mla_bmm_fast"]
            vup(me, x, got)
            check(f"X.2d T={T} _v_up_proj on the {lbl} == production bmm (bitwise)",
                  torch.equal(got, ref_o) and Q.STATS["mla_bmm_fast"] == b0 + 1)
        del case, state, o_prod, o_qw, o_new, o_mix, o_small, cache, q_nope
        torch.cuda.empty_cache()

    # X.3 capture
    T = 512
    case = C.Case(T, 4000, "indep", seed=23)
    state = sm._SM90State(torch.device("cuda"), 32, torch.float8_e4m3fn, T, C.TOPK, kv_lora_rank=512,
                          qk_rope_head_dim=0, sm_scale=C.SM_SCALE)
    sm._SM90_STATE = state
    ctx = torch.arange(4001, 4001 + T)
    state.plan(T, torch.where(ctx <= C.TOPK, ctx, C.TOPK + ctx % 4).to(torch.int32))
    impl = SimpleNamespace(num_heads=32, kv_lora_rank=512, qk_rope_head_dim=0, head_size=512, use_fp8_kv_cache=True,
                           topk_indices_buffer=case.topk.clone(), scale=C.SM_SCALE)
    meta = SimpleNamespace(req_id_per_token=case.req_id, block_table=case.block_table, block_size=C.PBS,
                           num_decode_tokens=0)
    layer = SimpleNamespace(_k_scale_float=1.0)
    q_nope = case.q.transpose(0, 1).contiguous().transpose(0, 1)
    q_pe = q_nope.new_empty(T, 32, 0)
    cache = case.cache.view(torch.float8_e4m3fn)
    o_ref, _ = prod_fmqa(impl, (q_nope, q_pe), cache, meta, layer)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        stack(impl, (q_nope, q_pe), cache, meta, layer)
    torch.cuda.current_stream().wait_stream(s)
    gr = torch.cuda.CUDAGraph()
    cap0, f0 = M.STATE.fallbacks.get("capturing", 0), Q.STATS["mla_index_fast"]
    with torch.cuda.graph(gr):
        o_cap, _ = stack(impl, (q_nope, q_pe), cache, meta, layer)
    gr.replay()
    torch.cuda.synchronize()
    check("X.3 captured call -> production kernel through both hooks (bitwise), no fast path inside the capture",
          M.STATE.fallbacks.get("capturing", 0) == cap0 + 1 and Q.STATS["mla_index_fast"] == f0 and
          torch.equal(o_cap, o_ref))
    print("quickwins STATS", {k: v for k, v in Q.STATS.items() if k.startswith("mla")}, "| MLA calls", M.STATE.calls,
          "fallbacks", M.STATE.fallbacks)
    print("ALL PASSED" if not FAIL else f"FAILED: {FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
