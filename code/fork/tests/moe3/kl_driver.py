"""moe3 in-container driver (run by tests/handoff/run.sh with HANDOFF_DRIVER=/w/tests/moe3/kl_driver.py and
HANDOFF_MODEL=$TF_EXL3_MODELS/GLM-5.3-Flash-handoff-moe-mini-tp2r0): the REAL vLLM engine of the production image,
composed like production (the kit's overlay chain + this worktree's bundle), on the 10-layer handoff MoE mini cut to
TP=2 rank-0 shapes (tests/moe3/build_mini_moe_tp2.py: real KDA / DSA / mHC, layers 3..9 = 32 real layer-10 EXL3
experts at local intermediate 1024 = production's per-rank shape), prefill-only, no speculative decoding.
KL between configurations at prompt positions (prompt_logprobs -> fp32 log-softmax at every --stride-th position):
--save stores the reference arm, --ref <dir>[,<dir>] compares (KL mean / p50 / p95 / p99 / max, top-1 agreement,
mean NLL), plus the GLM53_MOE_FUSED16 / GLM53_MOE_E4M3 serving counters (which schedule served how many calls).
Deterministic in-process: the sparse indexer's prefill top-k is replaced by a stable full sort; same prompts and cache
salts in every arm. Adapted from tests/w8a82/kl_driver.py (branch w8a82).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--label", default="")
ap.add_argument("--model", default=os.environ.get("KL_MODEL", os.path.join(os.environ.get("TF_EXL3_MODELS") or os.path.expanduser("~/models"), "GLM-5.3-Flash-handoff-moe-mini-tp2r0")))
ap.add_argument("--prompts", default="14000,6000,3001")
ap.add_argument("--stride", type=int, default=16)
ap.add_argument("--save", type=int, default=0)
ap.add_argument("--ref", default="")
ap.add_argument("--mnbt", type=int, default=16384)
ap.add_argument("--kv-bytes", type=int, default=1 << 30)
ap.add_argument("--gpu-util", type=float, default=0.125)
ARGS = ap.parse_args()
os.makedirs(ARGS.out, exist_ok=True)
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
for _v in ("VLLM_PREFIX_CACHE_RETENTION_INTERVAL", "VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA", "GLM53_APC_PRIOR_CHECKPOINT"):
    os.environ.pop(_v, None)   # no drafter / no APC here: the launcher retention knobs do not apply
import torch  # noqa: E402

LOG = open(os.path.join(ARGS.out, "kl.log"), "a")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


_TEXT = {}


def prompt_ids(n, i):
    from tokenizers import Tokenizer
    if "ids" not in _TEXT:
        tok = Tokenizer.from_file(os.path.join(os.environ.get("TF_EXL3_MODELS") or os.path.expanduser("~/models"), "GLM-OCR/tokenizer.json"))
        files = sorted(glob.glob("/usr/share/common-licenses/*")) + sorted(glob.glob("/usr/lib/python3.12/*.py"))
        t = "".join("\n\n### %s\n%s" % (os.path.basename(f), open(f, errors="ignore").read()) for f in files)
        _TEXT["ids"] = [int(x) for x in tok.encode(t).ids]
    ids = _TEXT["ids"]
    s = random.Random("moe3-kl-%d" % i).randrange(0, len(ids) - n - 1)
    return ids[s:s + n]


# ---------------------------------------------------------------- deterministic sparse top-k (stable full sort)
def install_topk():
    ns = torch.ops._C
    orig = ns.top_k_per_row_prefill

    def top_k_per_row_prefill(logits, ks, ke, out, num_rows, s0, s1, k):
        n = int(num_rows)
        if torch.cuda.is_current_stream_capturing():
            return orig(logits, ks, ke, out, num_rows, s0, s1, k)
        cols = logits.shape[1]
        ar = torch.arange(cols, device=logits.device)
        valid = (ar[None, :] >= ks[:n, None]) & (ar[None, :] < ke[:n, None])
        v = torch.where(valid, logits[:n].float(), torch.full((1, 1), float("-inf"), device=logits.device))
        kk = min(int(k), cols)
        o = torch.sort(-v, dim=1, stable=True).indices[:, :kk]
        sv = valid.gather(1, o)
        rel = o - ks[:n, None].to(torch.int64)
        out[:n, :kk] = torch.where(sv, rel, torch.full_like(rel, -1)).to(out.dtype)
        if kk < int(k):
            out[:n, kk:int(k)] = -1

    ns.top_k_per_row_prefill = top_k_per_row_prefill


# ---------------------------------------------------------------- logits capture at prompt positions
CAP = {"cur": None, "rows": {}}


def install_logits_capture():
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
            rows = [r for r in range(n) if (start + r) % ARGS.stride == ARGS.stride - 1 and start + r >= 31]
            if rows:
                with torch.no_grad():
                    lg = logits_fn(hidden_states[int(qsl[0]) + torch.tensor(rows, device=hidden_states.device)])
                    lp = torch.log_softmax(lg.float(), dim=-1)
                for r, row in zip(rows, lp):
                    CAP["rows"].setdefault(key, {})[start + r] = row.cpu()
        return orig(self, logits_fn, hidden_states, input_batch, all_token_ids, num_computed_tokens, prompt_lens)

    PL.PromptLogprobsWorker.compute_prompt_logprobs = compute_prompt_logprobs


def moe_counters():
    import sys as _s
    out = {}
    for mod in ("glm53_moe_fused16", "glm53_moe_e4m3"):
        m = _s.modules.get(mod)
        if m is not None:
            out[mod] = {k: v for k, v in m.STATS.items() if isinstance(v, (int, bool))}
    return out


def main():
    from vllm import LLM, SamplingParams
    install_topk()
    install_logits_capture()
    flags = {k: v for k, v in sorted(os.environ.items()) if k.startswith(("GLM53_", "TF_EXL3"))}
    log(f"== kl run {ARGS.label!r}: {json.dumps(vars(ARGS))}")
    log(f"   flags {json.dumps(flags)}")
    t0 = time.time()
    llm = LLM(model=ARGS.model, skip_tokenizer_init=True, tensor_parallel_size=1, dtype="bfloat16",
              max_model_len=32768, max_num_seqs=4, max_num_batched_tokens=ARGS.mnbt, enable_prefix_caching=False,
              kv_cache_dtype="fp8", kv_cache_memory_bytes=ARGS.kv_bytes, gpu_memory_utilization=ARGS.gpu_util,
              enable_flashinfer_autotune=False, seed=0, max_logprobs=20,
              compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8]})
    log(f"engine up in {time.time() - t0:.0f}s")
    log(f"moe counters after load: {json.dumps(moe_counters())}")
    sp = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True, prompt_logprobs=1, detokenize=False, seed=0)
    prompts = []
    for i, n in enumerate(int(x) for x in ARGS.prompts.split(",") if x.strip()):
        ids = prompt_ids(n, i)
        prompts.append(ids)
        CAP["cur"] = i
        llm.generate([{"prompt_token_ids": ids, "cache_salt": f"moe3-kl-{i}"}], sp, use_tqdm=False)
        CAP["cur"] = None
        log(f"prompt {i}: {n} tokens, captured {len(CAP['rows'].get(i, {}))} positions")
    log(f"moe counters after the prompts: {json.dumps(moe_counters())}")
    if ARGS.save:
        torch.save({"rows": CAP["rows"], "prompts": prompts}, os.path.join(ARGS.out, "logprobs.pt"))
        log(f"saved {sum(len(v) for v in CAP['rows'].values())} positions to {ARGS.out}/logprobs.pt")
    for refdir in [r for r in ARGS.ref.split(",") if r]:
        ref = torch.load(os.path.join(refdir, "logprobs.pt"))
        assert ref["prompts"] == prompts, "reference prompts differ"
        kls, agree, nll_r, nll_t = [], 0, [], []
        for i, rows in CAP["rows"].items():
            for pos, lp in rows.items():
                lr = ref["rows"][i][pos]
                p = lr.exp()
                kls.append(float((p * (lr - lp)).sum()))
                agree += int(lr.argmax() == lp.argmax())
                if pos + 1 < len(prompts[i]):
                    t = prompts[i][pos + 1]
                    nll_r.append(-float(lr[t]))
                    nll_t.append(-float(lp[t]))
        kt = torch.tensor(kls)
        res = {"label": ARGS.label, "ref": refdir, "positions": len(kls), "kl_mean": kt.mean().item(),
               "kl_p50": kt.quantile(0.5).item(), "kl_p95": kt.quantile(0.95).item(),
               "kl_p99": kt.quantile(0.99).item(), "kl_max": kt.max().item(), "top1_agree": agree / len(kls),
               "nll_ref": sum(nll_r) / len(nll_r), "nll_this": sum(nll_t) / len(nll_t)}
        log("KL " + json.dumps(res))
        with open(os.path.join(ARGS.out, "kl.jsonl"), "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
