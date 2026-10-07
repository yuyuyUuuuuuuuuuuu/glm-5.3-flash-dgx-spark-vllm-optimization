"""opt-kdamhc in-container driver (tests/handoff/run.sh with HANDOFF_DRIVER=/w/tests/opt_kdamhc/dvp_driver.py): the REAL
vLLM engine of the production image on the 10-layer handoff mini, no speculative decoding.

Per prompt (real-text windows, deterministic): A = greedy decode of --gen tokens (decode path: the mHC decode branch,
the recurrent KDA, CUDA graphs) with the top-20 logprobs; B = ONE fresh prefill of prompt + A's tokens (own cache salt)
whose full log-softmax rows at the generated positions are captured. Decode-vs-prefill consistency
KL(A || B) = sum over A's top-20 of pA (log pA - log pB) per generated position (the same quantity
tools/prodcheck/kpool_decode_consistency.py measures on production, top-20 truncated). B's rows at every 16th prompt
position are also saved, so arms can be compared on prefill (KL vs --ref arm).

--mhc fused0|fused1 installs tests/opt_kdamhc/mhc_hook.py (prefill mHC post+prenorm GEMM in one kernel; fused0 =
decode-consistent fp32 dot products) before the engine is built.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--label", default="")
ap.add_argument("--model", default=os.environ.get("KL_MODEL", os.path.join(os.environ.get("TF_EXL3_MODELS") or os.path.expanduser("~/models"), "GLM-5.3-Flash-handoff-mini")))
ap.add_argument("--prompts", default="3000,3000,3000,3000")
ap.add_argument("--gen", type=int, default=96)
ap.add_argument("--mhc", default="")
ap.add_argument("--ref", default="")
ap.add_argument("--mnbt", type=int, default=16384)
ap.add_argument("--kv-bytes", type=int, default=1 << 30)
ap.add_argument("--gpu-util", type=float, default=0.125)
ARGS = ap.parse_args()
os.makedirs(ARGS.out, exist_ok=True)
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
for _v in ("VLLM_PREFIX_CACHE_RETENTION_INTERVAL", "VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA", "GLM53_APC_PRIOR_CHECKPOINT"):
    os.environ.pop(_v, None)
sys.path.insert(0, "/w/tests/w8a82")
sys.path.insert(0, "/w/tests/opt_kdamhc")
sys.argv = [sys.argv[0], "--out", ARGS.out]          # kl_driver parses its own args at import
import torch  # noqa: E402
import kl_driver as K  # noqa: E402  (prompt windows, deterministic top-k)

LOG = open(os.path.join(ARGS.out, "dvp.log"), "a")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


CAP = {"cur": None, "lo": 0, "rows": {}}


def install_capture():
    import vllm.v1.worker.gpu.sample.prompt_logprob as PL
    orig = PL.PromptLogprobsWorker.compute_prompt_logprobs

    def compute_prompt_logprobs(self, logits_fn, hidden_states, input_batch, all_token_ids, num_computed_tokens,
                                prompt_lens):
        key = CAP["cur"]
        if key is not None:
            qsl = input_batch.query_start_loc_np if hasattr(input_batch, "query_start_loc_np") else \
                input_batch.query_start_loc.cpu().numpy()
            start = int(input_batch.num_computed_prefill_tokens_np[0])
            n = int(qsl[1] - qsl[0])
            rows = [r for r in range(n) if start + r >= CAP["lo"] - 1 or ((start + r) % 16 == 15 and start + r >= 31)]
            if rows:
                with torch.no_grad():
                    lg = logits_fn(hidden_states[int(qsl[0]) + torch.tensor(rows, device=hidden_states.device)])
                    lp = torch.log_softmax(lg.float(), dim=-1)
                for r, row in zip(rows, lp):
                    CAP["rows"].setdefault(key, {})[start + r] = row.cpu()
        return orig(self, logits_fn, hidden_states, input_batch, all_token_ids, num_computed_tokens, prompt_lens)

    PL.PromptLogprobsWorker.compute_prompt_logprobs = compute_prompt_logprobs


def main():
    from vllm import LLM, SamplingParams
    K.install_topk()
    install_capture()
    if ARGS.mhc:
        import mhc_hook
        mhc_hook.install(ARGS.mhc)
    log(f"== dvp run {ARGS.label!r}: {json.dumps(vars(ARGS))}")
    t0 = time.time()
    llm = LLM(model=ARGS.model, skip_tokenizer_init=True, tensor_parallel_size=1, dtype="bfloat16",
              max_model_len=32768, max_num_seqs=4, max_num_batched_tokens=ARGS.mnbt, enable_prefix_caching=False,
              kv_cache_dtype="fp8", kv_cache_memory_bytes=ARGS.kv_bytes, gpu_memory_utilization=ARGS.gpu_util,
              enable_flashinfer_autotune=False, seed=0, max_logprobs=20,
              compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8]})
    log(f"engine up in {time.time() - t0:.0f}s")
    spA = SamplingParams(temperature=0.0, max_tokens=ARGS.gen, ignore_eos=True, logprobs=20, detokenize=False, seed=0)
    spB = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True, prompt_logprobs=1, detokenize=False, seed=0)
    allk, prompts, gens, saved = [], [], [], {}
    for i, n in enumerate(int(x) for x in ARGS.prompts.split(",") if x.strip()):
        ids = K.prompt_ids(n, i)
        oa = llm.generate([{"prompt_token_ids": ids, "cache_salt": f"dvp-A{i}"}], spA, use_tqdm=False)[0].outputs[0]
        gen = [int(t) for t in oa.token_ids]
        CAP["cur"], CAP["lo"] = i, len(ids)
        llm.generate([{"prompt_token_ids": ids + gen, "cache_salt": f"dvp-B{i}"}], spB, use_tqdm=False)
        CAP["cur"] = None
        rows = CAP["rows"].get(i, {})
        ks = []
        for j, d in enumerate(oa.logprobs or []):
            pos = len(ids) - 1 + j                     # B's row at pos predicts gen[j]
            if pos not in rows:
                continue
            lb = rows[pos]
            kl = 0.0
            for t, v in d.items():
                la = float(v.logprob)
                kl += float(torch.tensor(la).exp()) * (la - float(lb[int(t)]))
            ks.append(kl)
        allk.extend(ks)
        prompts.append(ids)
        gens.append(gen)
        saved[i] = {p: r for p, r in rows.items() if p < len(ids) - 1}
        log(f"prompt {i}: {n} tokens, gen {len(gen)}; dvp KL mean {sum(ks) / max(len(ks), 1):.5f} over {len(ks)} "
            f"positions; first gen {gen[:6]}")
    kt = torch.tensor(allk)
    res = {"label": ARGS.label, "mhc": ARGS.mhc, "positions": len(allk), "dvp_kl_mean": kt.mean().item(),
           "dvp_kl_p50": kt.quantile(0.5).item(), "dvp_kl_p95": kt.quantile(0.95).item(), "dvp_kl_max": kt.max().item()}
    if ARGS.mhc:
        import mhc_hook
        res["hook_calls"] = mhc_hook.STATE["calls"]
    torch.save({"rows": saved, "prompts": prompts, "gens": gens}, os.path.join(ARGS.out, "prefill_rows.pt"))
    for refdir in [r for r in ARGS.ref.split(",") if r]:
        ref = torch.load(os.path.join(refdir, "prefill_rows.pt"))
        if ref["prompts"] != prompts:
            log("reference prompts differ; prefill KL skipped")
            continue
        kls, agree = [], 0
        for i, rows in saved.items():
            for pos, lp in rows.items():
                lr = ref["rows"][i].get(pos)
                if lr is None:
                    continue
                kls.append(float((lr.exp() * (lr - lp)).sum()))
                agree += int(lr.argmax() == lp.argmax())
        t = torch.tensor(kls)
        res.update({"prefill_kl_vs_ref": t.mean().item(), "prefill_kl_p99": t.quantile(0.99).item(),
                    "prefill_top1_agree": agree / max(len(kls), 1), "prefill_positions": len(kls),
                    "same_greedy_as_ref": ref["gens"] == gens})
    log("DVP " + json.dumps(res))
    with open(os.path.join(ARGS.out, "dvp.jsonl"), "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
