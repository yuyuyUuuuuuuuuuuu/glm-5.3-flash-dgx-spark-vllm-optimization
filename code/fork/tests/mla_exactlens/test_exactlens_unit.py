#!/usr/bin/env python3
"""[glm53-mla-exactlens] unit + real-op test (nodeC, production image + the production SM90 mounts).
Run: tests/hostloop_gpu.sh python3 tests/mla_exactlens/test_exactlens_unit.py
U1 fingerprints of the production builder functions (printed; must be in glm53_mla_exactlens.VERIFIED)
U2 arithmetic: exact_from_planned(production's planned lens) == the indexer's valid count for ctx 1..40000,
   kpool path and short-prefill rows; never larger than production's plan
U3 self_test() on the image's real ops (persistent_topk / top_k_per_row_prefill / expand / convert) passes
U4 production's own _kv_lens_host on host metadata, then exact_lens(): decode-only, mixed decode+long prefill,
   short-prefill step with a ctx == 2048 prefill row (stays 2048) next to a decode row with ctx == 2048 (2044),
   a spec-verify request (rows ctx..ctx+7)
U5 install_now() on the real module: build replaced, idempotent; a module whose build differs is refused
"""
import sys, types
sys.path.insert(0, "/w")
import torch

FAILS = []
def ck(ok, msg):
    print(("ok   " if ok else "FAIL ") + msg)
    if not ok:
        FAILS.append(msg)

def main():
    import glm53_mla_exactlens as X
    import vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90 as S
    B = S.FlashInferMLASparseSM90Builder
    fb, fk = X.source_fingerprint(B.build), X.source_fingerprint(B._kv_lens_host)
    print(f"info fingerprints: build={fb} _kv_lens_host={fk}")
    ck(fb in X.VERIFIED["FlashInferMLASparseSM90Builder.build"], f"U1 build fingerprint {fb} verified")
    ck(fk in X.VERIFIED["FlashInferMLASparseSM90Builder._kv_lens_host"], f"U1 _kv_lens_host fingerprint {fk} verified")

    topk, kpool = 2048, 4
    ctx = torch.arange(1, 40001, dtype=torch.int64)
    planned = torch.where(ctx <= topk, ctx, topk + ctx % kpool).to(torch.int32)
    ex = X.exact_from_planned(planned, None, topk, kpool).long()
    ck(torch.equal(ex, X.valid_reference(ctx, None, topk, kpool)), "U2 kpool path: exact == valid for ctx 1..40000")
    ck(bool((ex <= planned.long()).all()), "U2 exact <= production's plan everywhere")
    over = (planned.long() - ex)
    ck(int(over[ctx >= topk].min()) == 4 and int(over[ctx < topk].max()) == 0,
       "U2 production over-plans by exactly 4 keys on every ctx >= 2048 row, by 0 below")
    cs = torch.arange(1, topk + 1, dtype=torch.int64)
    ps = torch.where(cs <= topk, cs, topk + cs % kpool).to(torch.int32)
    es = X.exact_from_planned(ps, torch.ones_like(cs, dtype=torch.bool), topk, kpool).long()
    ck(torch.equal(es, cs), "U2 short-prefill rows (ctx <= 2048, identity top-k): exact == ctx (incl. ctx == 2048)")

    ok, detail = X.self_test(topk, kpool)
    print("info self_test:", detail)
    ck(ok, "U3 self_test on the image's real ops passes")

    fake = types.SimpleNamespace(_async_scheduling=False, _index_topk=topk, _index_kpool=kpool,
                                 vllm_config=types.SimpleNamespace(
                                     speculative_config=types.SimpleNamespace(num_speculative_tokens=7)))

    def cam(reqs):
        # reqs: list of (seq_len, q_len) in reordered order (decodes first)
        ql = torch.tensor([q for _, q in reqs], dtype=torch.int32)
        qsl = torch.zeros(len(reqs) + 1, dtype=torch.int32)
        qsl[1:] = torch.cumsum(ql, 0)
        return types.SimpleNamespace(num_reqs=len(reqs), num_actual_tokens=int(qsl[-1]), max_query_len=int(ql.max()),
                                     query_start_loc_cpu=qsl,
                                     seq_lens_cpu_upper_bound=torch.tensor([s for s, _ in reqs], dtype=torch.int32),
                                     positions=None)

    def rows_ctx(reqs):
        return torch.cat([torch.arange(s - q + 1, s + 1, dtype=torch.int64) for s, q in reqs])

    cases = {
        "decode-only (spec verify 8 rows, two requests across 2048)": ([(2051, 8), (100003, 8)], False),
        "mixed decode + long prefill chunk": ([(2049, 5), (6000, 1392)], False),
        "short-prefill step: decode ctx 2048 + prefill ending at 2048": ([(2048, 1), (2048, 300)], True),
        "short-prefill step: decode ctx 9000 + prefill of 1000": ([(9000, 6), (1000, 1000)], True),
        "plain decode rows q_len 1": ([(2048, 1), (2052, 1), (4097, 1)], False),
    }
    for name, (reqs, short_step) in cases.items():
        c = cam(reqs)
        n, lens = B._kv_lens_host(fake, c)
        new = X.exact_lens(fake, c, n, lens).long()
        rc = rows_ctx(reqs)
        # which rows are short-prefill rows: prefill requests (q_len > 8) in a step whose prefill seq_lens <= 2048
        short_row = torch.cat([torch.full((q,), short_step and q > 8, dtype=torch.bool) for _, q in reqs])
        want = X.valid_reference(rc, short_row, topk, kpool)
        ck(torch.equal(new, want), f"U4 {name}: exact_lens == indexer valid count ({int((lens.long() - new).sum())} "
                                   f"keys fewer than production's plan)")

    # U5 install on the real module (on a copy of the class so the imported module stays pristine for others)
    mod = types.SimpleNamespace(FlashInferMLASparseSM90Builder=type("B2", (B,), {"build": B.build}), _SM90_STATE=None)
    ok1 = X.install_now(mod)
    ok2 = X.install_now(mod)
    ck(ok1 and ok2 and getattr(mod.FlashInferMLASparseSM90Builder.build, "_glm53_exactlens", False),
       "U5 install_now replaces build (fingerprint ok) and is idempotent")
    def other_build(self, a, b, fast_build=False):
        return None
    mod2 = types.SimpleNamespace(FlashInferMLASparseSM90Builder=type("B3", (B,), {"build": other_build}), _SM90_STATE=None)
    ck(not X.install_now(mod2) and not getattr(mod2.FlashInferMLASparseSM90Builder.build, "_glm53_exactlens", False),
       "U5 a builder whose build source differs is refused (production path)")
    print(f"RESULT: {'ALL OK' if not FAILS else str(len(FAILS)) + ' FAIL'}")
    return 1 if FAILS else 0

sys.exit(main())
