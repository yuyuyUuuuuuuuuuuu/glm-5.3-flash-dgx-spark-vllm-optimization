"""r16z5rev: arm (a) numerics of the MLA prefill wrapper - the r16z5 DEFAULT (fused index pass + the limited
kv_indices write-back through the private [T, W] buffer) against production r16z2's path (triton_convert + clamp +
copy, every row written back), on production's backend file (tests/mla_env.sh mounts).

For each T (300 < kv_rows; 2048 / 4608 / 13824 > kv_rows) and two kv_indices pre-states (a -7 sentinel and a random
"older call" fill):
  (1) the attention output of every arm is BITWISE production's path,
  (2) every kv_indices entry an FA2 call of n < min_tokens rows can read (rows 0..n incl. the ctx % 4 overhang of row n)
      is bitwise production's, and rows past kv_rows keep the older call's bytes (nothing else written),
  (3) the peak allocation of the default arm vs production's path (the private [T, W] buffer replaces production's
      topk_slots, so the peak must not exceed production's).
Arms: prod = (FUSED_INDEX=0, KV_ROWS=all) [r16z2], z4 = (fused, all) [r16z4 zero copy], z5 = (fused, limited) [r16z5
default], z5nf = (FUSED_INDEX=0, limited).
Run: source tests/mla_env.sh && GPU_RUN_ENV_EXTRA="TF_EXL3_JIT=1" tests/gpu_run.sh python3 tests/r16z5rev/mla_compose_equiv.py
"""
from __future__ import annotations

import os
import sys
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
    import importlib
    os.environ["GLM53_MLA_PREFILL"] = "1"
    os.environ["TF_EXL3_JIT"] = "1"
    os.environ.pop("GLM53_MLA_PREFILL_KV_ROWS", None)
    os.environ.pop("GLM53_MLA_PREFILL_FUSED_INDEX", None)
    import glm53_mla_prefill as M
    mod = importlib.import_module(M.TARGET_MODULE)
    M.plugin_install()
    wrapped = mod.FlashInferMLASparseSM90Impl.forward_mqa
    check("installed with the kit defaults: fused index on, kv_rows limited",
          M.STATE.fused_index is True and M.STATE.kv_rows == max(M.STATE.min_tokens, 1024) + 1,
          f"fused={M.STATE.fused_index} kv_rows={M.STATE.kv_rows} min_tokens={M.STATE.min_tokens}")
    R = M.STATE.kv_rows
    arms = {"prod": (False, None), "z4": (True, None), "z5": (True, R), "z5nf": (False, R)}
    W = C.TOPK
    for T, start, regime in ((300, 0, "indep"), (2048, 12000, "sticky"), (4608, 30000, "sticky"),
                             (13824, 60000, "sticky")):
        case = C.Case(T, start, regime, seed=T)
        state = mod._SM90State(torch.device("cuda"), 32, torch.float8_e4m3fn, T, W, kv_lora_rank=512,
                               qk_rope_head_dim=0, sm_scale=C.SM_SCALE)
        mod._SM90_STATE = state
        topk_buf = torch.full((T, W), -1, dtype=torch.int32, device="cuda")
        topk_buf.copy_(case.topk)
        impl = SimpleNamespace(num_heads=32, kv_lora_rank=512, qk_rope_head_dim=0, head_size=512,
                               use_fp8_kv_cache=True, topk_indices_buffer=topk_buf, scale=C.SM_SCALE)
        meta = SimpleNamespace(req_id_per_token=case.req_id, block_table=case.block_table, block_size=C.PBS,
                               num_decode_tokens=0)
        layer = SimpleNamespace(_k_scale_float=1.0)
        q_nope = case.q.transpose(0, 1).contiguous().transpose(0, 1)
        q_pe = q_nope.new_empty(T, 32, 0)
        cache = case.cache.view(torch.float8_e4m3fn)
        g = torch.Generator(device="cuda").manual_seed(7 + T)
        older = torch.randint(0, 1 << 20, state.kv_indices.shape, generator=g, device="cuda",
                              dtype=state.kv_indices.dtype)
        for pre_name, pre in (("sentinel", None), ("older-call", older)):
            res = {}
            for arm, (fi, kvr) in arms.items():
                M.STATE.fused_index, M.STATE.kv_rows = fi, kvr
                if pre is None:
                    state.kv_indices.fill_(-7)
                else:
                    state.kv_indices.copy_(pre)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
                calls0, fic0 = M.STATE.calls, M.STATE.fused_index_calls
                o, lse = wrapped(impl, (q_nope, q_pe), cache, meta, layer)
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() - base
                served_fused = M.STATE.fused_index_calls == fic0 + 1
                check(f"T={T} {pre_name} arm {arm}: exact kernel served (fused pass {'ran' if fi else 'off'})",
                      M.STATE.calls == calls0 + 1 and lse is None and served_fused == fi)
                res[arm] = (o.clone(), state.kv_indices.clone(), peak)
            o_p, ki_p, pk_p = res["prod"]
            reach = max(n * W + 3 for n in range(1, M.STATE.min_tokens))   # last entry an FA2 call of n rows reads
            lim = min(reach + 1, ki_p.numel())
            for arm in ("z4", "z5", "z5nf"):
                o, ki, pk = res[arm]
                check(f"T={T} {pre_name} {arm}: output bitwise == production's path", torch.equal(o, o_p))
                check(f"T={T} {pre_name} {arm}: every entry an FA2 call (< {M.STATE.min_tokens} rows) reads == "
                      f"production's", torch.equal(ki[:lim], ki_p[:lim]))
                kvr = arms[arm][1]
                rows = T if kvr is None else min(T, kvr)
                exp = ki_p.clone()
                if pre is None:
                    exp[rows * W:] = -7
                else:
                    exp[rows * W:] = pre[rows * W:]
                check(f"T={T} {pre_name} {arm}: rows [0,{rows}) production's, rest untouched",
                      torch.equal(ki, exp))
                print(f"     T={T} {pre_name} {arm}: peak alloc {pk / 2**20:.1f} MiB vs production path "
                      f"{pk_p / 2**20:.1f} MiB", flush=True)
                check(f"T={T} {pre_name} {arm}: peak allocation <= production path's + 1 MiB", pk <= pk_p + (1 << 20),
                      f"{pk / 2**20:.1f} vs {pk_p / 2**20:.1f} MiB")
        del state, case, topk_buf, older
        mod._SM90_STATE = None
        torch.cuda.empty_cache()
    print("ALL OK" if not FAIL else f"FAILED {len(FAIL)}: {FAIL}", flush=True)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
