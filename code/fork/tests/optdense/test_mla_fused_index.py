"""opt-dense: GLM53_MLA_PREFILL fused index pass (glm53_mla_prefill.fused_index) vs the old chain
(triton_convert_req_index_to_global_index + clamp_ + copy_ into the SM90 wrapper's kv_indices).

For each case: the wrapped forward_mqa with fused_index=False and =True from the same sentinel-filled kv_indices:
  - attention output bitwise equal (the kernel reads the same valid prefix, the same valid counts);
  - the WHOLE kv_indices buffer bitwise equal (rows [0, T) = production's bytes, the rest untouched);
  - valid counts == triton_convert's;
then the per-call wall time of both (median of 15, the kernel included) and of the index part alone.
Run: source tests/mla_env.sh; tests/gpu_run.sh python3 tests/optdense/test_mla_fused_index.py"""
from __future__ import annotations

import os
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import mla_prefill_common as C  # noqa: E402

FAIL = []


def check(name, ok, msg=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {msg}", flush=True)
    if not ok:
        FAIL.append(name)


def tmed(fn, n=15):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def main() -> int:
    import importlib
    os.environ["GLM53_MLA_PREFILL"] = "1"
    os.environ["TF_EXL3_JIT"] = "1"
    import glm53_mla_prefill as M
    mod = importlib.import_module(M.TARGET_MODULE)
    M.plugin_install()
    wrapped = mod.FlashInferMLASparseSM90Impl.forward_mqa
    check("installed, fused index on by default", getattr(wrapped, "__glm53_mla_prefill__", False) and M.STATE.fused_index)
    cases = [(13824, 0, "sticky"), (13824, 13824, "sticky"), (4289, 27648, "indep"), (2048, 12000, "sticky"),
             (300, 0, "indep"), (1791, 100000, "indep")]
    for T, start, regime in cases:
        case = C.Case(T, start, regime, seed=31 + T % 97)
        state = mod._SM90State(torch.device("cuda"), 32, torch.float8_e4m3fn, T + 64, C.TOPK, kv_lora_rank=512,
                               qk_rope_head_dim=0, sm_scale=C.SM_SCALE)
        mod._SM90_STATE = state
        topk_buf = torch.full((T + 64, C.TOPK), -1, dtype=torch.int32, device="cuda")
        topk_buf[:T].copy_(case.topk)
        impl = SimpleNamespace(num_heads=32, kv_lora_rank=512, qk_rope_head_dim=0, head_size=512,
                               use_fp8_kv_cache=True, topk_indices_buffer=topk_buf, scale=C.SM_SCALE)
        meta = SimpleNamespace(req_id_per_token=case.req_id, block_table=case.block_table, block_size=C.PBS,
                               num_decode_tokens=0)
        layer = SimpleNamespace(_k_scale_float=1.0)
        q_nope = case.q.transpose(0, 1).contiguous().transpose(0, 1)
        q_pe = q_nope.new_empty(T, 32, 0)
        cache = case.cache.view(torch.float8_e4m3fn)
        M.STATE.fused_index = False
        state.kv_indices.fill_(-7)
        o_old, _ = wrapped(impl, (q_nope, q_pe), cache, meta, layer)
        ki_old = state.kv_indices.clone()
        M.STATE.fused_index = True
        state.kv_indices.fill_(-7)
        f0 = M.STATE.fused_index_calls
        o_new, _ = wrapped(impl, (q_nope, q_pe), cache, meta, layer)
        ki_new = state.kv_indices.clone()
        check(f"T={T} start={start} {regime}: fused pass used", M.STATE.fused_index_calls == f0 + 1)
        check(f"T={T}: attention output bitwise == old chain", torch.equal(o_old, o_new))
        check(f"T={T}: whole kv_indices buffer bitwise == old chain (production's bytes)", torch.equal(ki_old, ki_new),
              f"{int((ki_old != ki_new).sum())} differ")
        _s, v = M.fused_index(case.req_id, case.block_table, topk_buf[:T], C.PBS, torch.empty_like(state.kv_indices))
        check(f"T={T}: valid counts == triton_convert's", torch.equal(v, case.valid))
        # timing: whole forward_mqa, and the index part alone
        def run(f):
            M.STATE.fused_index = f
            wrapped(impl, (q_nope, q_pe), cache, meta, layer)
        t_old = tmed(lambda: run(False))
        t_new = tmed(lambda: run(True))
        from vllm.v1.attention.backends.mla.sparse_utils import triton_convert_req_index_to_global_index
        ti = topk_buf[:T]

        def idx_old():
            s, vv = triton_convert_req_index_to_global_index(case.req_id, case.block_table, ti, BLOCK_SIZE=C.PBS,
                                                            NUM_TOPK_TOKENS=C.TOPK, return_valid_counts=True)
            state.kv_indices[: T * C.TOPK].copy_(s.reshape(-1).clamp_(min=0).to(torch.int32))
        i_old = tmed(idx_old)
        i_new = tmed(lambda: M.fused_index(case.req_id, case.block_table, ti, C.PBS, state.kv_indices))
        print(f"TIME T={T} start={start}: forward_mqa old {t_old:.3f} ms -> fused {t_new:.3f} ms ({t_new - t_old:+.3f}); "
              f"index part old {i_old:.3f} -> fused {i_new:.3f} ms", flush=True)
        M.STATE.fused_index = True
    # inputs outside the fused pass -> None (the old chain serves)
    bad = M.fused_index(torch.zeros(4, dtype=torch.int64, device="cuda"), torch.zeros(1, 4, dtype=torch.int32,
                        device="cuda"), torch.zeros(4, 2048, dtype=torch.int32, device="cuda"), 64,
                        torch.zeros(4 * 2048, dtype=torch.int32, device="cuda"))
    check("int64 req_id -> declined (old chain)", bad is None)
    bad = M.fused_index(torch.zeros(4, dtype=torch.int32, device="cuda"), torch.zeros(1, 4, dtype=torch.int32,
                        device="cuda"), torch.zeros(4, 2048, dtype=torch.int32, device="cuda"), 64,
                        torch.zeros(3 * 2048, dtype=torch.int32, device="cuda"))
    check("kv_indices too small -> declined", bad is None)
    print("ALL PASSED" if not FAIL else f"FAILED {FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
