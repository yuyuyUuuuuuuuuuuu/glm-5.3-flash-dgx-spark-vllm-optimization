"""Second reviewer's nodeC checks of GLM53_KDA_STRIDED_QKV: paths the first rigs did not walk.

Run (patched files bound over the image's own paths, so the REAL package import is exercised; the stock side is
read from PRISTINE copies via GLM53_FLA_* because SITE_VLLM then holds the patched text - without them
kda_strided_common aborts "the overlay did not change the image's text"):
  S=<dir with the image's pristine fused_recurrent.py.orig / kda.py.orig and the overlay-patched *.patched>
  O=/usr/local/lib/python3.12/dist-packages/vllm/third_party/flash_linear_attention/ops
  flock /tmp/tf-gpu-bench.lock env GPU_RUN_RO="$S" \
      GPU_RUN_ENV="GLM53_FLA_FUSED_RECURRENT_PY=$S/fused_recurrent.py.orig;GLM53_FLA_KDA_PY=$S/kda.py.orig" \
      GPU_RUN_BIND="$S/fused_recurrent.py.patched=$O/fused_recurrent.py;$S/kda.py.patched=$O/kda.py" \
      tests/gpu_run.sh python3 tests/review2_kdaqkv_paths.py

1. Real import path: `vllm.third_party.flash_linear_attention.ops.kda` and the production model module
   `vllm.models.glm5next.nvidia.kda` import the overlay-patched files (marker present) - the earlier rigs loaded the
   patched text only as a shadow package.
2. Breakable-CUDA-graph eager replay (PIECEWISE / mixed steps): eager_break_during_capture replays _forward with
   vllm.utils.torch_utils.weak_ref_tensor(arg) views. Those must keep the strides (else the patched kernel would
   read the wrong tokens where stock's .contiguous() had copied); patched on weak refs == stock on the originals.
3. The other launcher the overlay edits (fused_recurrent_gated_delta_rule, non-KDA; not used by GLM-5.3 but shipped
   in the same file): patched == stock bitwise on its (contiguous) inputs, varlen + spec shapes.
4. Stock vs patched through the real glm5next split: projected -> split -> causal_conv1d_update in place ->
   qkv.split -> reshape, exactly the calls of models/glm5next/nvidia/kda.py (spec path), bitwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kda_strided_common as KC  # noqa: E402

import torch  # noqa: E402

from harness import Checks, run_main  # noqa: E402

DEV = "cuda"
H, D = 32, 128
P = H * D
W = 3 * P + H + 2 * D
MARK = "# [glm53-kda-strided-qkv] vLLM #55736"


def real_import(checks: Checks):
    import inspect

    import vllm.third_party.flash_linear_attention.ops.fused_recurrent as fr
    import vllm.third_party.flash_linear_attention.ops.kda as kda
    src_k, src_f = inspect.getsource(kda), inspect.getsource(fr)
    ok = checks(MARK in src_k and MARK in src_f, "real package: the bound files are not the patched ones")
    ok &= checks(hasattr(fr, "token_stride"), "real package: token_stride missing")
    import vllm.models.glm5next.nvidia.kda as gk  # noqa: F401  (the production model module imports ops.kda)
    ok &= checks(gk.fused_recurrent_kda is kda.fused_recurrent_kda,
                 "glm5next kda does not use the patched fused_recurrent_kda")
    if ok:  # (printed unconditionally before, i.e. also right after the CHECK FAILED lines of an unbound run)
        print("  ok   real import: vllm...ops.{fused_recurrent,kda} patched; glm5next.nvidia.kda binds the patched "
              "fused_recurrent_kda", flush=True)
    return kda


def make_spec_layer(ns, ql, gen):
    n = ns * ql
    projected = torch.randn(n, W, generator=gen).to(torch.bfloat16).to(DEV)
    g = torch.randn(1, n, H, D, generator=gen).to(torch.bfloat16).to(DEV)
    state = (0.1 * torch.randn(n + 2, H, D, D, generator=gen)).to(DEV)
    slots = (torch.randperm(n + 1, generator=gen)[:n] + 1).to(torch.int32).to(DEV)
    return dict(projected=projected, g=g, state=state,
                idx=slots.view(ns, ql) if ql > 1 else slots,
                cu=torch.arange(0, n + 1, ql, dtype=torch.int32, device=DEV),
                nacc=torch.randint(1, ql + 1, (ns,), generator=gen, dtype=torch.int32).to(DEV) if ql > 1 else None,
                a_log=(0.5 * torch.randn(H, generator=gen)).to(DEV),
                g_bias=(0.1 * torch.randn(P, generator=gen)).to(DEV))


def views(projected):
    qkv, beta_raw, _f, _g = projected.split([3 * P, H, D, D], dim=-1)
    q, k, v = (x.reshape(1, -1, H, D) for x in qkv.split(P, dim=-1))
    return q, k, v, beta_raw.unsqueeze(0)


def call(fn, q, k, v, g, beta, L, state, out=None):
    return fn(q=q, k=k, v=v, g=g, beta=beta, initial_state=state, use_qk_l2norm_in_kernel=True,
              cu_seqlens=L["cu"], ssm_state_indices=L["idx"], num_accepted_tokens=L["nacc"], out=out,
              sigmoid_beta=True, a_log=L["a_log"], g_bias=L["g_bias"], compute_gate=True, lower_bound=-5.0)


def weak_ref_replay(checks: Checks, stock, real_kda):
    from vllm.utils.torch_utils import weak_ref_tensor
    gen = torch.Generator().manual_seed(4242)
    for ns, ql in ((1, 8), (3, 5), (4, 1)):
        L = make_spec_layer(ns, ql, gen)
        q, k, v, beta = views(L["projected"])
        wq, wk, wv, wb = (weak_ref_tensor(x) for x in (q, k, v, beta))
        same = all(a.stride() == b.stride() and a.data_ptr() == b.data_ptr() and a.shape == b.shape
                   for a, b in ((q, wq), (k, wk), (v, wv), (beta, wb)))
        checks(same, f"[{ns}x{ql}] weak_ref_tensor changed shape/stride/data_ptr of a strided view")
        st_s, st_p = L["state"].clone(), L["state"].clone()
        o_s, _ = call(stock.fused_recurrent_kda, q, k, v, L["g"], beta, L, st_s)
        o_p, _ = call(real_kda.fused_recurrent_kda, wq, wk, wv, weak_ref_tensor(L["g"]), wb, L, st_p)
        checks(torch.equal(o_s, o_p) and torch.equal(st_s, st_p),
               f"[{ns}x{ql}] patched(weak refs) != stock(originals)")
        checks(not torch.equal(st_s, L["state"]), f"[{ns}x{ql}] vacuous: no state written")
        print(f"  ok   weak_ref replay {ns}x{ql}: strides kept {q.stride()} / beta {beta.stride()}; "
              f"patched on weak refs == stock bitwise", flush=True)


def glm5next_split_with_conv(checks: Checks, stock, real_kda):
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
    gen = torch.Generator().manual_seed(99)
    for ns, ql in ((1, 8), (2, 6), (1, 5)):
        L = make_spec_layer(ns, ql, gen)
        n = ns * ql
        conv_w = (0.2 * torch.randn(3 * P, 4, generator=gen)).to(torch.bfloat16).to(DEV)
        cstate0 = torch.randn(n + 2, 3 * P, 3 + ql - 1, generator=gen).to(torch.bfloat16).to(DEV)
        outs, states = [], []
        for mod in (stock, real_kda):
            proj = L["projected"].clone()
            cst = cstate0.clone()
            qkv = proj.split([3 * P, H, D, D], dim=-1)[0]
            qkv_spec = causal_conv1d_update(qkv, cst, conv_w, None, activation="silu",
                                            conv_state_indices=L["idx"][:, 0] if ql > 1 else L["idx"],
                                            num_accepted_tokens=L["nacc"], query_start_loc=L["cu"],
                                            max_query_len=ql)
            checks(qkv_spec.data_ptr() == proj.data_ptr(), "conv did not write in place (layout assumption)")
            q, k, v = (x.reshape(1, -1, H, D) for x in qkv_spec.split(P, dim=-1))
            beta = proj.split([3 * P, H, D, D], dim=-1)[1].unsqueeze(0)
            core = torch.empty(1, n, H, D, dtype=torch.bfloat16, device=DEV)
            st = L["state"].clone()
            call(mod.fused_recurrent_kda, q, k, v, L["g"], beta, L, st, out=core[0, :n].unsqueeze(0))
            outs.append(core)
            states.append(st)
        checks(torch.equal(outs[0], outs[1]) and torch.equal(states[0], states[1]),
               f"[{ns}x{ql}] glm5next split + in-place conv: patched != stock")
        print(f"  ok   glm5next spec split + in-place causal_conv1d_update {ns}x{ql}: bitwise", flush=True)


def non_kda_launcher(checks: Checks, stock_fr):
    """Its production-style uses: varlen decode (1 token per sequence, 1-D state indices), dense batch without
    indices (state per sequence), and spec verify (2-D indices + num_accepted_tokens)."""
    import vllm.third_party.flash_linear_attention.ops.fused_recurrent as fr
    gen = torch.Generator().manual_seed(7)
    cases = (("varlen decode", 1, 5, 4, 8), ("spec verify", 1, 8, 16, 32))
    for name, B, T, Hq, HV in cases:
        K = 128
        q = torch.randn(B, T, Hq, K, generator=gen).to(torch.bfloat16).to(DEV)
        k = torch.nn.functional.normalize(torch.randn(B, T, Hq, K, generator=gen), dim=-1).to(torch.bfloat16).to(DEV)
        v = torch.randn(B, T, HV, K, generator=gen).to(torch.bfloat16).to(DEV)
        g = torch.nn.functional.logsigmoid(torch.rand(B, T, HV, generator=gen)).to(DEV)
        beta = torch.rand(B, T, HV, generator=gen).sigmoid().to(torch.bfloat16).to(DEV)
        kw = {}
        if name == "varlen decode":
            kw["cu_seqlens"] = torch.arange(0, T + 1, dtype=torch.int32, device=DEV)
            # slots >= 1: slot 0 is the null block - that sequence is skipped and its o rows stay uninitialized
            # (stock vs stock then differs too once the allocator hands out dirty memory)
            kw["ssm_state_indices"] = (torch.randperm(T + 3, generator=gen)[:T] + 1).to(torch.int32).to(DEV)
            h0 = (0.1 * torch.randn(T + 4, HV, K, K, generator=gen)).to(DEV)
        else:
            kw["cu_seqlens"] = torch.tensor([0, 4, 8], dtype=torch.int32, device=DEV)
            kw["ssm_state_indices"] = torch.arange(1, 9, dtype=torch.int32, device=DEV).view(2, 4)
            kw["num_accepted_tokens"] = torch.tensor([2, 4], dtype=torch.int32, device=DEV)
            h0 = (0.1 * torch.randn(10, HV, K, K, generator=gen)).to(DEV)
        s_s, s_p = h0.clone(), h0.clone()
        o_s, _ = stock_fr.fused_recurrent_gated_delta_rule(q, k, v, g, beta, initial_state=s_s,
                                                           use_qk_l2norm_in_kernel=True, **kw)
        o_p, _ = fr.fused_recurrent_gated_delta_rule(q, k, v, g, beta, initial_state=s_p,
                                                     use_qk_l2norm_in_kernel=True, **kw)
        torch.cuda.synchronize()
        if not (torch.equal(o_s, o_p) and torch.equal(s_s, s_p)):
            print(f"    DIAG {name}: o max|d| {(o_s.float() - o_p.float()).abs().max().item()}, state max|d| "
                  f"{(s_s - s_p).abs().max().item()}, o shapes {tuple(o_s.shape)} {tuple(o_p.shape)} "
                  f"same ptr {o_s.data_ptr() == o_p.data_ptr()}", flush=True)
        checks(torch.equal(o_s, o_p) and torch.equal(s_s, s_p), f"non-KDA launcher {name}: patched != stock")
        checks(not torch.equal(s_s, h0), f"non-KDA {name}: vacuous (no state written)")
        if torch.equal(o_s, o_p) and torch.equal(s_s, s_p):
            print(f"  ok   non-KDA fused_recurrent_gated_delta_rule {name} B={B} T={T} H={Hq} HV={HV}: bitwise", flush=True)


def main() -> None:
    checks = Checks()
    # stock = the image's text (read by KC from SITE_VLLM, which is now bound to the patched files -> use the
    # original text shipped next to this test run: GLM53_FLA_* env overrides point at the pristine copies)
    real_kda = real_import(checks)   # first: the real package registers its CustomOp; KC's copies tolerate that
    stock, _patched_shadow, _pk, _ = KC.load_stock_and_patched()
    import importlib
    stock_fr = importlib.import_module(stock.__name__.rsplit(".", 1)[0] + ".fused_recurrent")
    weak_ref_replay(checks, stock, real_kda)
    glm5next_split_with_conv(checks, stock, real_kda)
    non_kda_launcher(checks, stock_fr)
    print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    checks.summary()


if __name__ == "__main__":
    run_main(main)
