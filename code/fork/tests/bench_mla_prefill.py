"""Sparse-MLA prefill microbenchmark at production shapes (docs/MLA_PREFILL.md).

Variants:
  v0   production FA2 (FlashInfer 0.6.18 BatchMLAPagedAttentionWrapper, fa2) with production's host-planned lengths
  v0x  the same kernel planned with the exact valid counts
  l1   FlashInfer's SM120 multi-group sparse prefill kernel (GLM_NSA: FP8 QK, 2-pass FP8 P.V) on a temporary
       656-byte packed KV + Q padded to 576 (ceiling probe; not exact)
  l2   GLM53 exact kernel (kernels/mla_prefill/mla_prefill.cu): bf16 QK and PV, fp8 KV dequantized in registers
  hook the whole forward_mqa call: production's (flashinfer_mla_sparse_sm90.py.patched: index conversion, clamp,
       copy into the wrapper's kv_indices, FA2 run; planned with production's lengths) vs the GLM53_MLA_PREFILL
       wrapper (index conversion + the exact kernel), q in production's head-major layout
Usage (inside the production image with fi618 mounted, see tests/mla_env.sh):
  python3 tests/bench_mla_prefill.py --T 13824 --start 0 --regime sticky --variants v0,v0x,l1,l2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mla_prefill_common as C  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=13824)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--regime", default="sticky")
    ap.add_argument("--variants", default="v0,v0x,l1")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--ref-rows", type=int, default=384)
    ap.add_argument("--k-scale", type=float, default=1.0)
    ap.add_argument("--json", default="")
    ap.add_argument("--l2-variants", default="3")
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = "cuda"
    t0 = time.time()
    case = C.Case(a.T, a.start, a.regime, seed=1, k_scale=a.k_scale)
    pairs = case.pairs()
    flop = 4.0 * pairs * C.HEADS * C.D
    print(f"case T={a.T} start={a.start} regime={a.regime} pairs={pairs} ({pairs / a.T:.1f}/row) "
          f"useful={flop / 1e12:.3f} TFLOP  lens_prod-valid: max {int((case.lens_prod - case.valid).max())} "
          f"rows>valid {int((case.lens_prod > case.valid).sum())}  setup {time.time() - t0:.1f}s", flush=True)
    variants = a.variants.split(",")
    g = torch.Generator(device="cpu").manual_seed(7)
    rows = torch.cat([torch.arange(min(64, a.T)), torch.randint(0, a.T, (max(a.ref_rows - 64, 0),), generator=g),
                      torch.arange(max(a.T - 16, 0), a.T)]).unique()
    ref = C.reference(case, rows) if a.ref_rows > 0 else None
    res = {"T": a.T, "start": a.start, "regime": a.regime, "pairs": pairs, "tflop": flop / 1e12, "variants": {}}

    def report(name, fn, out_fn, extra=""):
        med, best = C.cuda_time(fn, 3, a.iters)
        out = out_fn()
        st = C.err_stats(out[rows], ref) if ref is not None else {}
        res["variants"][name] = {"ms_med": med, "ms_best": best, "tflops": flop / med / 1e9, **st}
        print(f"{name:5s} {med:8.3f} ms (best {best:8.3f})  {flop / med / 1e9:6.1f} TFLOPS  "
              + "  ".join(f"{k} {v:.3e}" for k, v in st.items()) + f"  {extra}", flush=True)

    if "v0" in variants or "v0x" in variants:
        fa = C.ProdFA2(a.T + 64)   # review: +64 rows so the last row's 2048+ctx%4 plan stays in bounds
        fa.fill(case.slots)
        holder = {}
        if "v0" in variants:
            fa.plan(a.T, case.lens_prod)
            report("v0", lambda: holder.__setitem__("o", fa.run(case.q, case.cache, a.k_scale)), lambda: holder["o"],
                   "(production lengths)")
        if "v0x" in variants:
            fa.plan(a.T, case.valid)
            report("v0x", lambda: holder.__setitem__("o", fa.run(case.q, case.cache, a.k_scale)), lambda: holder["o"],
                   "(exact lengths)")
        del fa
    if "l1" in variants:
        ext = C.build_ext("mla_l1_ext", "kernels/mla_prefill/l1_mg.cu",
                          ["-DFLASHINFER_ENABLE_FP8_E8M0", "-DFLASHINFER_ENABLE_FP4_E2M1"])
        packed = torch.empty(case.num_blocks * C.PBS, 656, dtype=torch.uint8, device=dev)
        q576 = torch.zeros(a.T, C.HEADS, 576, dtype=torch.bfloat16, device=dev)
        out = torch.empty(a.T, C.HEADS, C.D, dtype=torch.bfloat16, device=dev)
        lse = torch.empty(a.T, C.HEADS, dtype=torch.float32, device=dev)
        idx = case.slots.contiguous()

        def l1_kernel():
            ext.l1_mg(q576, packed, idx, case.valid, out, lse, C.SM_SCALE)

        def l1_full():
            ext.pack656(case.cache, packed, a.k_scale)
            q576[..., : C.D].copy_(case.q)
            l1_kernel()
        l1_full()
        pk, _ = C.cuda_time(lambda: ext.pack656(case.cache, packed, a.k_scale), 2, 10)
        qc, _ = C.cuda_time(lambda: q576[..., : C.D].copy_(case.q), 2, 10)
        report("l1k", l1_kernel, lambda: out, f"(kernel only; smem {ext.SMEM} B)")
        report("l1", l1_full, lambda: out, f"(pack {pk:.3f} ms + q576 copy {qc:.3f} ms + kernel)")
        del packed, q576
    if "l2" in variants:
        sys.path.insert(0, str(C.REPO))
        import glm53_mla_prefill as M
        ext = M.load_ext(jit=True)
        out = torch.empty(a.T, C.HEADS, C.D, dtype=torch.bfloat16, device=dev)
        for var in [int(v) for v in a.l2_variants.split(",")]:
            def l2(var=var):
                M.run(ext, case.q, case.cache, case.slots, case.valid, out, C.SM_SCALE, a.k_scale, variant=var)
            l2()
            report(f"l2v{var}", l2, lambda: out)
    if "hook" in variants:
        import importlib
        import os
        from types import SimpleNamespace
        sys.path.insert(0, str(C.REPO))
        os.environ["GLM53_MLA_PREFILL"] = "1"
        import glm53_mla_prefill as M
        M.load_ext(jit=True)
        mod = importlib.import_module(M.TARGET_MODULE)
        orig = mod.FlashInferMLASparseSM90Impl.forward_mqa
        M.install(mod)
        wrapped = mod.FlashInferMLASparseSM90Impl.forward_mqa
        state = mod._SM90State(torch.device(dev), 32, torch.float8_e4m3fn, a.T + 64, C.TOPK, kv_lora_rank=512,
                               qk_rope_head_dim=0, sm_scale=C.SM_SCALE)
        mod._SM90_STATE = state
        state.plan(a.T, case.lens_prod.cpu())
        impl = SimpleNamespace(num_heads=32, kv_lora_rank=512, qk_rope_head_dim=0, head_size=512,
                               use_fp8_kv_cache=True, topk_indices_buffer=case.topk.contiguous(), scale=C.SM_SCALE)
        meta = SimpleNamespace(req_id_per_token=case.req_id, block_table=case.block_table, block_size=C.PBS,
                               num_decode_tokens=0)
        layer = SimpleNamespace(_k_scale_float=a.k_scale)
        q_nope = case.q.transpose(0, 1).contiguous().transpose(0, 1)
        q_pe = q_nope.new_empty(a.T, 32, 0)
        cache = case.cache.view(torch.float8_e4m3fn)
        holder = {}
        report("hook0", lambda: holder.__setitem__("o", orig(impl, (q_nope, q_pe), cache, meta, layer)[0]),
               lambda: holder["o"], "(production forward_mqa, production lengths)")
        report("hook1", lambda: holder.__setitem__("o", wrapped(impl, (q_nope, q_pe), cache, meta, layer)[0]),
               lambda: holder["o"], f"(GLM53_MLA_PREFILL forward_mqa, variant {M.STATE.variant})")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
