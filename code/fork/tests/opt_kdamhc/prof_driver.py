"""opt-kdamhc in-container driver (tests/handoff/run.sh with HANDOFF_DRIVER=/w/tests/opt_kdamhc/prof_driver.py): kernel
profile of ONE prefill step of --T tokens on the REAL engine of the production image (handoff mini, 10 layers: 5 KDA +
5 DSA, per-rank TP=2 head counts), production flags from the run.sh environment, no speculative decoding.

Warm-up: one prefill of the same length (other cache salt) so every JIT/TileLang/autotune is done. Then torch.profiler
(CUDA activity) around a second prefill. Output (stdout + <out>/prof.json): per-kernel-name totals (ms, calls), the
step's GPU busy time, and the ordered kernel sequence (name, ms) so the per-layer glue between the big kernels can be
read off.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--label", default="")
ap.add_argument("--model", default=os.environ.get("KL_MODEL", os.path.join(os.environ.get("TF_EXL3_MODELS") or os.path.expanduser("~/models"), "GLM-5.3-Flash-handoff-mini-w8a8")))
ap.add_argument("--T", type=int, default=13824)
ap.add_argument("--mnbt", type=int, default=16384)
ap.add_argument("--mhc", default="")
ap.add_argument("--mhcmod", type=int, default=0, help="1 = install glm53_mhc_fused (the production hook, JIT kernel)")
ap.add_argument("--kv-bytes", type=int, default=2 << 30)
ap.add_argument("--gpu-util", type=float, default=0.2)
ap.add_argument("--det-topk", type=int, default=0, help="1 = kl_driver's deterministic sort top-k (the profiles "
                "before 2026-10-03 00:50 had it on: +~20 ms per MLA layer at 13,824 rows vs production's kernel)")
ap.add_argument("--stack-of", default="", help="print the Python stacks of CPU ops whose name contains this")
ap.add_argument("--reps", type=int, default=0, help="untraced timed prefills before the profiled one (+ peak memory)")
ARGS = ap.parse_args()
os.makedirs(ARGS.out, exist_ok=True)
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
sys.path.insert(0, "/w/tests/w8a82")
sys.path.insert(0, "/w/tests/opt_kdamhc")
sys.argv = [sys.argv[0], "--out", ARGS.out]
import torch  # noqa: E402
import kl_driver as K  # noqa: E402


def main():
    from vllm import LLM, SamplingParams
    if ARGS.det_topk:
        K.install_topk()   # kl_driver's deterministic stable-sort top-k: NOT production's topKPerRowPrefill (O(T^2) sort)
    if ARGS.mhc:
        import mhc_hook
        mhc_hook.install(ARGS.mhc)
    if ARGS.mhcmod:
        os.environ["GLM53_MHC_FUSED_JIT"] = "1"
        # loaded by file path: putting /w first on sys.path would shadow site-packages' fp8_w8a8 (the armed
        # integrate.py then silently skips the W8A8 plugin)
        import importlib.util
        spec = importlib.util.spec_from_file_location("glm53_mhc_fused", "/w/glm53_mhc_fused.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["glm53_mhc_fused"] = mod
        spec.loader.exec_module(mod)
        mod.install()
    llm = LLM(model=ARGS.model, skip_tokenizer_init=True, tensor_parallel_size=1, dtype="bfloat16",
              max_model_len=32768, max_num_seqs=2, max_num_batched_tokens=ARGS.mnbt, enable_prefix_caching=False,
              kv_cache_dtype="fp8", kv_cache_memory_bytes=ARGS.kv_bytes, gpu_memory_utilization=ARGS.gpu_util,
              enable_flashinfer_autotune=False, seed=0, compilation_config={"cudagraph_capture_sizes": [1, 2]})
    sp = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True, detokenize=False, seed=0)
    ids = K.prompt_ids(ARGS.T - 1, 7)
    for w in range(2):
        t0 = time.time()
        llm.generate([{"prompt_token_ids": ids, "cache_salt": f"prof-w{w}"}], sp, use_tqdm=False)
        torch.cuda.synchronize()
        print(f"warm-up {w}: {time.time() - t0:.2f}s", flush=True)
    # untraced step time + the step's transient memory above the post-warm-up baseline (in-process engine)
    base_alloc = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    walls = []
    for r in range(ARGS.reps):
        t0 = time.time()
        llm.generate([{"prompt_token_ids": ids, "cache_salt": f"prof-t{r}"}], sp, use_tqdm=False)
        torch.cuda.synchronize()
        walls.append(time.time() - t0)
    peak = torch.cuda.max_memory_allocated()
    if walls:
        walls.sort()
        print(f"== timing {ARGS.label}: T={ARGS.T} mnbt={ARGS.mnbt} untraced wall median {walls[len(walls) // 2]:.4f}s "
              f"min {walls[0]:.4f}s ({ARGS.reps} reps; {ARGS.T / walls[len(walls) // 2]:.0f} tok/s); transient peak "
              f"{(peak - base_alloc) / 2**20:.0f} MiB above the {base_alloc / 2**30:.2f} GiB baseline; torch reserved "
              f"{torch.cuda.memory_reserved() / 2**30:.2f} GiB", flush=True)
    from torch.profiler import ProfilerActivity, profile
    t0 = time.time()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], with_stack=bool(ARGS.stack_of)) as prof:
        llm.generate([{"prompt_token_ids": ids, "cache_salt": "prof-run"}], sp, use_tqdm=False)
        torch.cuda.synchronize()
    wall = time.time() - t0
    if ARGS.stack_of:
        seen = set()
        for e in prof.events():
            if ARGS.stack_of in e.name and e.name.startswith("aten::"):
                chain, p = [], e.cpu_parent
                while p is not None and len(chain) < 40:
                    chain.append(p.name)
                    p = p.cpu_parent
                st = tuple(e.stack or ()) + tuple(chain)
                if st in seen:
                    continue
                seen.add(st)
                print(f"== stack of {e.name}:", flush=True)
                for f in st[:40]:
                    print("     ", f[:200], flush=True)
    evs = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    evs.sort(key=lambda e: e.time_range.start)
    tot = collections.defaultdict(lambda: [0.0, 0])
    seq = []
    busy = 0.0
    for e in evs:
        d = (e.time_range.end - e.time_range.start) / 1000.0
        tot[e.name][0] += d
        tot[e.name][1] += 1
        busy += d
        seq.append((e.name[:120], round(d, 4), e.time_range.start))
    rows = sorted(tot.items(), key=lambda kv: -kv[1][0])
    if ARGS.mhcmod:
        import glm53_mhc_fused
        print(f"glm53_mhc_fused: calls {glm53_mhc_fused.STATE['calls']} prod {glm53_mhc_fused.STATE['prod']} "
              f"installed {glm53_mhc_fused.STATE['installed']}", flush=True)
    print(f"== profile {ARGS.label}: T={ARGS.T}, wall {wall:.3f}s (with profiler), GPU kernel sum {busy:.1f} ms, "
          f"{len(evs)} GPU events", flush=True)
    for name, (ms, n) in rows[:70]:
        print(f"{ms:9.2f} ms {n:5d}x  {name[:150]}")
    t_first = seq[0][2] if seq else 0
    json.dump({"label": ARGS.label, "T": ARGS.T, "busy_ms": busy, "wall_s": wall,
               "totals": [[n, ms, c] for n, (ms, c) in rows],
               "seq": [[n, d, (t - t_first) / 1000.0] for n, d, t in seq]},
              open(os.path.join(ARGS.out, "prof.json"), "w"))


if __name__ == "__main__":
    main()
