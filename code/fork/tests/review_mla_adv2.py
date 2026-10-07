"""Adversarial review of GLM53_MLA_PREFILL, part 2 (continuation). Not part of the author's suite.
G  persistent kernel next to a busy second stream: bitwise equal to the isolated result, timing under contention
H  small-T calls (MIN_TOKENS region, mixed steps): new kernel vs production FA2 at contexts 1.5k / 15k / 100k
I  peak device memory of production forward_mqa vs the wrapped one at T=13824
J  variants 4 / 3 / 2 at T=13824: bitwise relation, run-to-run determinism
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mla_prefill_common as C  # noqa: E402
import glm53_mla_prefill as M  # noqa: E402

FAIL = []


def check(name, ok, msg=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {msg}", flush=True)
    if not ok:
        FAIL.append(name)


def part_g(ext):
    T = 13824
    case = C.Case(T, 0, "sticky", seed=1)
    out0 = torch.empty(T, 32, 512, dtype=torch.bfloat16, device="cuda")
    M.run(ext, case.q, case.cache, case.slots, case.valid, out0, C.SM_SCALE, 1.0, variant=4)
    torch.cuda.synchronize()
    a = torch.randn(8192, 8192, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(8192, 8192, dtype=torch.bfloat16, device="cuda")
    c = torch.empty_like(a)
    sb = torch.cuda.Stream()
    sa = torch.cuda.Stream()
    for var in (4, 3):
        outs = [torch.empty_like(out0) for _ in range(6)]
        iso, _ = C.cuda_time(lambda: M.run(ext, case.q, case.cache, case.slots, case.valid, outs[0], C.SM_SCALE,
                                           1.0, variant=var), 2, 5)
        torch.cuda.synchronize()
        ev = []
        with torch.cuda.stream(sb):
            for _ in range(60):
                torch.mm(a, b, out=c)
        with torch.cuda.stream(sa):
            for o in outs:
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                M.run(ext, case.q, case.cache, case.slots, case.valid, o, C.SM_SCALE, 1.0, variant=var)
                e1.record()
                ev.append((e0, e1))
        torch.cuda.synchronize()
        ts = [x.elapsed_time(y) for x, y in ev]
        same = all(torch.equal(o, out0) for o in outs)
        check(f"G v{var} concurrent with a busy stream: bitwise equal to isolated v4", same,
              f"isolated {iso:.2f} ms, under contention {min(ts):.2f}-{max(ts):.2f} ms")


def part_h(ext):
    for start in (1500, 15000, 100000):
        for T in (256, 512, 1024):
            case = C.Case(T, start, "indep", seed=T + start)
            out = torch.empty(T, 32, 512, dtype=torch.bfloat16, device="cuda")
            fa = C.ProdFA2(T + 64)
            fa.fill(case.slots)
            fa.plan(T, case.lens_prod)
            t_fa, _ = C.cuda_time(lambda: fa.run(case.q, case.cache, 1.0), 3, 15)
            r = {}
            for var in (4, 3):
                r[var], _ = C.cuda_time(lambda: M.run(ext, case.q, case.cache, case.slots, case.valid, out,
                                                      C.SM_SCALE, 1.0, variant=var), 3, 15)
            print(f"H start={start:6d} T={T:5d}: FA2 {t_fa:7.3f} ms | v4 {r[4]:7.3f} ms | v3 {r[3]:7.3f} ms "
                  f"| v4 speedup {t_fa / r[4]:.2f}x", flush=True)
            check(f"H start={start} T={T} v4 not slower than FA2", r[4] <= t_fa * 1.02)
            del fa


def part_i():
    os.environ["GLM53_MLA_PREFILL"] = "1"
    mod = importlib.import_module(M.TARGET_MODULE)
    orig = mod.FlashInferMLASparseSM90Impl.forward_mqa
    M.install(mod)
    wrapped = mod.FlashInferMLASparseSM90Impl.forward_mqa
    T = 13824
    case = C.Case(T, 0, "sticky", seed=1)
    state = mod._SM90State(torch.device("cuda"), 32, torch.float8_e4m3fn, T + 64, C.TOPK, kv_lora_rank=512,
                           qk_rope_head_dim=0, sm_scale=C.SM_SCALE)
    mod._SM90_STATE = state
    state.plan(T, case.lens_prod.cpu())
    impl = SimpleNamespace(num_heads=32, kv_lora_rank=512, qk_rope_head_dim=0, head_size=512, use_fp8_kv_cache=True,
                           topk_indices_buffer=case.topk.contiguous(), scale=C.SM_SCALE)
    meta = SimpleNamespace(req_id_per_token=case.req_id, block_table=case.block_table, block_size=C.PBS,
                           num_decode_tokens=0)
    layer = SimpleNamespace(_k_scale_float=1.0)
    q_nope = case.q.transpose(0, 1).contiguous().transpose(0, 1)
    q_pe = q_nope.new_empty(T, 32, 0)
    cache = case.cache.view(torch.float8_e4m3fn)
    res = {}
    for name, fn in (("production", orig), ("wrapped", wrapped), ("production2", orig), ("wrapped2", wrapped)):
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        o, _ = fn(impl, (q_nope, q_pe), cache, meta, layer)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base
        res[name] = peak
        print(f"I {name:11s} peak over baseline {peak / 2**20:8.1f} MiB (out {o.numel() * 2 / 2**20:.1f} MiB, "
              f"strides {tuple(o.stride())})", flush=True)
        del o
    check("I wrapped peak <= production peak", res["wrapped2"] <= res["production2"],
          f"{res['wrapped2'] / 2**20:.1f} vs {res['production2'] / 2**20:.1f} MiB")


def part_j(ext):
    T = 13824
    case = C.Case(T, 30000, "indep", seed=9)
    outs = {}
    for var in (4, 3, 2):
        o = torch.empty(T, 32, 512, dtype=torch.bfloat16, device="cuda")
        M.run(ext, case.q, case.cache, case.slots, case.valid, o, C.SM_SCALE, 0.0371, variant=var)
        outs[var] = o
    torch.cuda.synchronize()
    check("J v4 == v3 bitwise (same algorithm, persistent vs one CTA per token)", torch.equal(outs[4], outs[3]))
    d = (outs[4].float() - outs[2].float()).abs().max().item()
    print(f"J v4 vs v2 max abs diff {d:.3e} (different running-max schedule, expected non-zero)", flush=True)
    rep = torch.empty_like(outs[4])
    ok = True
    for _ in range(5):
        M.run(ext, case.q, case.cache, case.slots, case.valid, rep, C.SM_SCALE, 0.0371, variant=4)
        ok &= torch.equal(rep, outs[4])
    check("J v4 run-to-run bitwise deterministic (5 runs, T=13824, 30k ctx, extra-scale path)", ok)


def main() -> int:
    ext = M.load_ext(jit=True)
    parts = sys.argv[1:] or ["g", "h", "i", "j"]
    for p in parts:
        {"g": lambda: part_g(ext), "h": lambda: part_h(ext), "i": part_i, "j": lambda: part_j(ext)}[p]()
        torch.cuda.empty_cache()
    print("ALL PASSED" if not FAIL else f"FAILED: {FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
