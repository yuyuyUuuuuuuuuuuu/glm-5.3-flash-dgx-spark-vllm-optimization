"""Adversarial review of GLM53_MLA_PREFILL (branch mlaprefill). Not part of the author's suite.
Parts: A random-valid persistent stress, B peaked attention numerics, C production-length error attribution
(which rows carry the doc's 0.9-1.4 worst-row error), D multi-request mixed step through the hook,
E odd T values for the persistent grid."""
from __future__ import annotations

import importlib
import math
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


def per_row_err(out, ref):
    o, r = out.float(), ref.float()
    num = (o - r).pow(2).sum(-1).sqrt()
    den = r.pow(2).sum(-1).sqrt().clamp_min(1e-30)
    return (num / den).amax(-1)            # [rows] max over heads


def fa2_exact(case, T, slots, valid, k_scale=1.0):
    fa = C.ProdFA2(T + 1)
    fa.fill(slots)
    fa.plan(T, valid.clamp(min=1))
    return fa.run(case.q, case.cache, k_scale)


def part_a(ext):
    """Random per-row valid counts (incl. 0, 1..31, 33, 2047) through the persistent kernel: tokens with 0 sub-tiles
    and partial sub-tiles interleaved, so the stage ring / mbarrier phases must stay in step across tokens."""
    torch.manual_seed(1)
    for T, start in ((3000, 6000), (97, 30000)):
        case = C.Case(T, start, "indep", seed=31)
        g = torch.Generator(device="cpu").manual_seed(5)
        valid = torch.randint(0, 2048, (T,), generator=g, dtype=torch.int32)
        special = torch.tensor([0, 1, 2, 31, 32, 33, 63, 64, 65, 2047, 0, 0, 5], dtype=torch.int32)
        valid[: special.numel()] = special
        valid[torch.randperm(T, generator=g)[: T // 10]] = 0
        valid = torch.minimum(valid, case.valid.cpu())
        valid = valid.cuda()
        slots = case.slots.clone()
        col = torch.arange(C.TOPK, device="cuda")[None]
        slots[col >= valid[:, None]] = -1
        case.valid, case.slots = valid, slots
        out = torch.full((T, 32, 512), 3.0, dtype=torch.bfloat16, device="cuda")
        for var in (4, 3):
            M.run(ext, case.q, case.cache, slots, valid, out, C.SM_SCALE, 1.0, variant=var)
            torch.cuda.synchronize()
            z = valid == 0
            check(f"A T={T} v{var} valid=0 rows are zero", bool((out[z] == 0).all()), f"{int(z.sum())} rows")
            rows = torch.nonzero(~z).flatten().cpu()
            ref = C.reference(case, rows)
            e_new = per_row_err(out[rows.cuda()], ref)
            o_fa = fa2_exact(case, T, slots, valid)
            e_fa = per_row_err(o_fa[rows.cuda()], ref)
            check(f"A T={T} v{var} random valid parity", float(e_new.max()) <= 1.5 * float(e_fa.max()) + 1e-4,
                  f"new max {float(e_new.max()):.3e} mean {float(e_new.mean()):.3e} | FA2 max {float(e_fa.max()):.3e}"
                  f" mean {float(e_fa.mean()):.3e}")


def part_b(ext):
    """Peaked attention: larger q scale -> logits std ~8-20. Checks the running-max schedule (32 vs 16 keys) and
    bf16 P rounding stay in FA2's error class when a few keys dominate."""
    for sigma in (8.0, 24.0):
        for regime in ("sticky", "indep"):
            case = C.Case(768, 12000, regime, seed=41, q_sigma=sigma)
            out = torch.empty(768, 32, 512, dtype=torch.bfloat16, device="cuda")
            M.run(ext, case.q, case.cache, case.slots, case.valid, out, C.SM_SCALE, 1.0)
            rows = torch.arange(768)
            ref = C.reference(case, rows)
            st = C.err_stats(out, ref)
            o_fa = fa2_exact(case, 768, case.slots, case.valid)
            st_fa = C.err_stats(o_fa, ref)
            check(f"B sigma={sigma} {regime} parity", st["rel_l2_max"] <= 1.5 * st_fa["rel_l2_max"] + 1e-4,
                  f"new {st} | FA2 {st_fa}")


def part_c():
    """The doc's 'production plan error 0.9-1.4' on synthetic data: which rows? The test harness sizes the FA2
    kv_indices to exactly T rows, so the LAST row's 2048 + ctx%4 plan reads past the end of that allocation."""
    for T, start in ((1791, 13824), (512, 15000)):
        case = C.Case(T, start, "sticky", seed=1)
        rows = torch.arange(T)
        ref = C.reference(case, rows)
        pads = ((0, "harness (kv_indices = T rows, last row reads past the end)"),) if os.environ.get("PAD0") else ()
        for pad, label in pads + ((64, "kv_indices with 64 spare zero rows (last row reads slot 0 like the -1 tail)"),):
            fa = C.ProdFA2(T + pad)
            fa.fill(case.slots)
            fa.plan(T, case.lens_prod)
            o = fa.run(case.q, case.cache, 1.0)
            e = per_row_err(o, ref)
            top = torch.topk(e, 3)
            e_wo_last = e[:-1]
            print(f"C T={T} {label}: max {float(e.max()):.3e} at rows {top.indices.tolist()} "
                  f"({[round(float(v), 4) for v in top.values]}); mean {float(e.mean()):.3e}; "
                  f"excluding last row: max {float(e_wo_last.max()):.3e} mean {float(e_wo_last.mean()):.3e}",
                  flush=True)
        o_x = fa2_exact(case, T, case.slots, case.valid)
        ex = per_row_err(o_x, ref)
        print(f"C T={T} FA2 exact lengths: max {float(ex.max()):.3e} mean {float(ex.mean()):.3e}", flush=True)


def part_d():
    """A mixed step through the wrapped forward_mqa: 3 requests (decode rows of a 60k-context request with 8 spec
    rows, a 700-row prefill chunk at 3000, decode rows of a 1500-context request), separate block tables, one cache.
    Output vs fp32 reference with the exact valid counts, and vs production forward_mqa planned as the builder does."""
    os.environ["GLM53_MLA_PREFILL"] = "1"
    mod = importlib.import_module(M.TARGET_MODULE)
    M.install(mod)
    wrapped = mod.FlashInferMLASparseSM90Impl.forward_mqa
    g = torch.Generator(device="cpu").manual_seed(77)
    reqs = [(60000 - 8, 8, "sticky"), (3000, 700, "indep"), (1500 - 4, 4, "sticky")]   # (first pos, rows, regime)
    nb_total = sum((s + n + C.PBS - 1) // C.PBS for s, n, _ in reqs) + 16
    perm = torch.randperm(nb_total, generator=g).to(torch.int32)
    maxb = max((s + n + C.PBS - 1) // C.PBS for s, n, _ in reqs)
    bt = torch.zeros(len(reqs), maxb, dtype=torch.int32)
    used = 0
    for i, (s, n, _) in enumerate(reqs):
        nb = (s + n + C.PBS - 1) // C.PBS
        bt[i, :nb] = perm[used: used + nb]
        used += nb
    bt = bt.cuda()
    chan = torch.ones(C.D)
    kv = (torch.randn(nb_total, C.PBS, C.D, generator=g) * chan).to(torch.float8_e4m3fn).view(torch.uint8).cuda()
    topk = torch.cat([torch.from_numpy(C.topk_logical(s, n, rg, seed=i)) for i, (s, n, rg) in enumerate(reqs)]).cuda()
    req_id = torch.cat([torch.full((n,), i, dtype=torch.int32) for i, (s, n, _) in enumerate(reqs)]).cuda()
    T = int(req_id.numel())
    q = (torch.randn(T, 32, C.D, generator=g) * 1.6).to(torch.bfloat16).cuda()
    slots, valid = C.convert(req_id, bt, topk)
    ctx = torch.cat([torch.arange(s, s + n) + 1 for s, n, _ in reqs])
    lens_prod = torch.where(ctx <= C.TOPK, ctx, C.TOPK + ctx % C.KPOOL).to(torch.int32)
    state = mod._SM90State(torch.device("cuda"), 32, torch.float8_e4m3fn, 4096, C.TOPK, kv_lora_rank=512,
                           qk_rope_head_dim=0, sm_scale=C.SM_SCALE)
    mod._SM90_STATE = state
    state.plan(T, lens_prod)
    impl = SimpleNamespace(num_heads=32, kv_lora_rank=512, qk_rope_head_dim=0, head_size=512, use_fp8_kv_cache=True,
                           topk_indices_buffer=topk.contiguous(), scale=C.SM_SCALE)
    meta = SimpleNamespace(req_id_per_token=req_id, block_table=bt, block_size=C.PBS, num_decode_tokens=8)
    layer = SimpleNamespace(_k_scale_float=1.0)
    q_nope = q.transpose(0, 1).contiguous().transpose(0, 1)
    q_pe = q_nope.new_empty(T, 32, 0)
    cache = kv.view(torch.float8_e4m3fn)
    calls0 = M.STATE.calls
    o_new, _ = wrapped(impl, (q_nope, q_pe), cache, meta, layer)
    check("D mixed 3-request step served by the kernel", M.STATE.calls == calls0 + 1)
    case = SimpleNamespace(cache=kv, k_scale=1.0, valid=valid, slots=slots, q=q)
    ref = C.reference(case, torch.arange(T))
    e_new = per_row_err(o_new, ref)
    # production forward_mqa (need the unwrapped function): rebuild from the module file's class dict via closure
    cells = dict(zip(wrapped.__code__.co_freevars, wrapped.__closure__))
    prod = cells["orig"].cell_contents
    o_prod, _ = prod(impl, (q_nope, q_pe), cache, meta, layer)
    e_prod = per_row_err(o_prod, ref)
    state.plan(T, valid.cpu())
    o_px, _ = prod(impl, (q_nope, q_pe), cache, meta, layer)
    e_px = per_row_err(o_px, ref)
    b = [0, 8, 708, 712]
    for i in range(3):
        sl = slice(b[i], b[i + 1])
        print(f"D req {i}: new max {float(e_new[sl].max()):.3e} | production plan {float(e_prod[sl].max()):.3e} "
              f"(last row {float(e_prod[b[i + 1] - 1]):.3e}) | FA2 exact {float(e_px[sl].max()):.3e}", flush=True)
    check("D mixed step parity vs FA2 exact", float(e_new.max()) <= 1.5 * float(e_px.max()) + 1e-4,
          f"new {float(e_new.max()):.3e} FA2x {float(e_px.max()):.3e}")


def part_e(ext):
    """Odd T for the persistent grid (T < #SM, T = #SM +- 1, multi-wave remainders)."""
    for T in (1, 47, 48, 49, 95, 257):
        case = C.Case(T, 9000, "indep", seed=T)
        out = torch.empty(T, 32, 512, dtype=torch.bfloat16, device="cuda")
        M.run(ext, case.q, case.cache, case.slots, case.valid, out, C.SM_SCALE, 1.0, variant=4)
        ref = C.reference(case, torch.arange(T))
        st = C.err_stats(out, ref)
        o_fa = fa2_exact(case, T, case.slots, case.valid)
        st_fa = C.err_stats(o_fa, ref)
        check(f"E T={T} v4", st["rel_l2_max"] <= 1.5 * st_fa["rel_l2_max"] + 1e-4,
              f"{st['rel_l2_max']:.3e} vs FA2 {st_fa['rel_l2_max']:.3e}")


def part_f():
    """Production's +4 plan keys with a vLLM-like null block: global block 0 zeroed and never in a block table.
    Separates the slot-0 copies (logit 0 on a zero key) from the next-row spill."""
    for regime in ("sticky", "indep"):
        T, start = 1791, 13824
        case = C.Case(T, start, regime, seed=1)
        nb = case.num_blocks
        g = torch.Generator(device="cpu").manual_seed(3)
        nbr = case.block_table.shape[1]
        bt = (torch.randperm(nb - 1, generator=g)[:nbr] + 1).to(torch.int32).cuda()[None]
        case.block_table = bt
        case.cache[0].zero_()
        case.slots, case.valid = C.convert(case.req_id, bt, case.topk)
        rows = torch.arange(T)
        ref = C.reference(case, rows)
        fa = C.ProdFA2(T + 64)
        fa.fill(case.slots)
        fa.plan(T, case.lens_prod)
        e = per_row_err(fa.run(case.q, case.cache, 1.0), ref)
        # the same plan, but with the next-row spill replaced by slot 0 (zero key): isolates the slot-0 copies
        fa2 = C.ProdFA2(T + 64)
        padded = torch.full((T, C.TOPK + 64), -1, dtype=torch.int32, device="cuda")
        padded[:, : C.TOPK] = case.slots
        fa2.kv_indices.zero_()
        # rows laid out with stride TOPK in the wrapper: emulate by planning lengths min(prod, TOPK) (no spill)
        fa2.fill(case.slots)
        fa2.plan(T, torch.minimum(case.lens_prod, torch.full_like(case.lens_prod, C.TOPK)))
        e0 = per_row_err(fa2.run(case.q, case.cache, 1.0), ref)
        print(f"F {regime} null block zero: production plan max {float(e.max()):.3e} mean {float(e.mean()):.3e} | "
              f"slot-0 copies only (no spill) max {float(e0.max()):.3e} mean {float(e0.mean()):.3e}", flush=True)


def main() -> int:
    ext = M.load_ext(jit=True)
    parts = sys.argv[1:] or ["a", "b", "c", "d", "e"]
    if "a" in parts:
        part_a(ext)
    if "b" in parts:
        part_b(ext)
    if "c" in parts:
        part_c()
    if "d" in parts:
        part_d()
    if "e" in parts:
        part_e(ext)
    if "f" in parts:
        part_f()
    print("ALL PASSED" if not FAIL else f"FAILED: {FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
