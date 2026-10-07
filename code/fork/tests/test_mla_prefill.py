"""GLM53_MLA_PREFILL kernel tests (docs/MLA_PREFILL.md): conversion exactness, parity vs fp32 reference and vs
production FA2 (exact lengths), edge rows (valid = 0/1/63/64/65, -1 tails), kv_scale != 1."""
from __future__ import annotations

import sys
from pathlib import Path

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


def main() -> int:
    ext = M.load_ext(jit=True)
    for var in (4, 3, 2):
        print(f"=== kernel variant {var}")
        run_variant(ext, var)
    print("ALL PASSED" if not FAIL else f"FAILED: {FAIL}")
    return 1 if FAIL else 0


def run_variant(ext, var):
    def mrun(*a, **k):
        return M.run(*a, **k, variant=var)

    def chk(name, ok, msg=""):
        check(f"v{var} {name}", ok, msg)
    print("ext VERSION", ext.VERSION, "smem", ext.SMEM)
    # 1. e4m3 -> bf16 conversion for all 256 codes (NaN codes 0x7f/0xff excluded), scale 1, pow2, non-pow2
    codes = torch.arange(256, dtype=torch.uint8, device="cuda")
    f = codes.view(torch.float8_e4m3fn)
    ok_codes = ~torch.isnan(f.float())
    for s in (1.0, 0.5, 4.0, 0.0371):
        got = ext.conv_probe(s).view(torch.bfloat16)
        want = (f.to(torch.bfloat16) * torch.tensor(s, dtype=torch.bfloat16, device="cuda")) if s != 1.0 else f.to(
            torch.bfloat16)
        same = (got.view(torch.int16) == want.view(torch.int16)) | (got.float() == want.float())
        chk(f"conv e4m3->bf16 scale={s}", bool(same[ok_codes].all()),
              f"mismatch {int((~same & ok_codes).sum())} codes")
    # 2. parity at a small production-like case (all regimes of row length incl. short rows and the tail)
    for T, start, regime in ((1024, 0, "sticky"), (1024, 3000, "indep"), (512, 20000, "local")):
        case = C.Case(T, start, regime, seed=3)
        out = torch.empty(T, C.HEADS, C.D, dtype=torch.bfloat16, device="cuda")
        mrun(ext, case.q, case.cache, case.slots, case.valid, out, C.SM_SCALE, case.k_scale)
        rows = torch.arange(T)
        ref = C.reference(case, rows)
        st = C.err_stats(out, ref)
        fa = C.ProdFA2(T)
        fa.fill(case.slots)
        fa.plan(T, case.valid)
        o_fa = fa.run(case.q, case.cache, case.k_scale)
        st_fa = C.err_stats(o_fa, ref)
        st_x = C.err_stats(out, o_fa.float())
        chk(f"parity T={T} start={start} {regime}", st["rel_l2_max"] <= 1.5 * st_fa["rel_l2_max"] + 1e-4,
              f"l2 vs fp32 {st}  | FA2 vs fp32 {st_fa} | l2 vs FA2 {st_x}")
    # 3. edge rows: valid in {0, 1, 63, 64, 65, width}; -1 tails; duplicated slots
    T = 8
    case = C.Case(T, 5000, "indep", seed=5)
    valid = torch.tensor([0, 1, 63, 64, 65, 2047, 128, 2], dtype=torch.int32, device="cuda")
    slots = case.slots.clone()
    for r in range(T):
        slots[r, int(valid[r]):] = -1
    case.valid, case.slots = valid, slots
    out = torch.full((T, C.HEADS, C.D), 7.0, dtype=torch.bfloat16, device="cuda")
    mrun(ext, case.q, case.cache, slots, valid, out, C.SM_SCALE, 1.0)
    chk("valid=0 row -> zeros", bool((out[0] == 0).all()))
    ref = C.reference(case, torch.arange(1, T))
    st = C.err_stats(out[1:], ref)
    fa = C.ProdFA2(T)
    fa.fill(slots)
    fa.plan(T, valid.clamp(min=1))
    o_fa = fa.run(case.q, case.cache, 1.0)
    st_fa = C.err_stats(o_fa[1:], ref)
    per_row = [round(C.err_stats(out[r:r + 1], ref[r - 1:r])["rel_l2_max"], 5) for r in range(1, T)]
    per_row_fa = [round(C.err_stats(o_fa[r:r + 1], ref[r - 1:r])["rel_l2_max"], 5) for r in range(1, T)]
    chk("edge rows parity", st["rel_l2_max"] <= 1.5 * st_fa["rel_l2_max"] + 1e-4,
          f"{st} | FA2 {st_fa} | per row l2 {per_row} FA2 {per_row_fa}")
    # 4. kv_scale != 1 (non power of two: FA2's double rounding) and power of two
    for ks in (0.0371, 2.0):
        case = C.Case(256, 4000, "indep", seed=9, k_scale=ks)
        out = torch.empty(256, C.HEADS, C.D, dtype=torch.bfloat16, device="cuda")
        mrun(ext, case.q, case.cache, case.slots, case.valid, out, C.SM_SCALE, ks)
        ref = C.reference(case, torch.arange(256))
        fa = C.ProdFA2(256)
        fa.fill(case.slots)
        fa.plan(256, case.valid)
        o_fa = fa.run(case.q, case.cache, ks)
        st, st_fa = C.err_stats(out, ref), C.err_stats(o_fa, ref)
        chk(f"kv_scale={ks}", st["rel_l2_max"] <= 1.5 * st_fa["rel_l2_max"] + 1e-4, f"{st} | FA2 {st_fa}")
    # 5. determinism: two runs bitwise equal
    case = C.Case(512, 3000, "sticky", seed=11)
    o1 = torch.empty(512, C.HEADS, C.D, dtype=torch.bfloat16, device="cuda")
    o2 = torch.empty_like(o1)
    mrun(ext, case.q, case.cache, case.slots, case.valid, o1, C.SM_SCALE, 1.0)
    mrun(ext, case.q, case.cache, case.slots, case.valid, o2, C.SM_SCALE, 1.0)
    chk("deterministic", bool(torch.equal(o1, o2)))


if __name__ == "__main__":
    sys.exit(main())
