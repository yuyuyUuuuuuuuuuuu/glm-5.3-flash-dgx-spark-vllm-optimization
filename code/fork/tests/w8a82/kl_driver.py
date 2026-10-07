"""w8a82 in-container driver (run by tests/handoff/run.sh with HANDOFF_DRIVER=/w/tests/w8a82/kl_driver.py and
HANDOFF_MODEL=<mini-w8a8>): the REAL vLLM engine of the production image on the 10-layer handoff mini (real KDA /
DSA / mHC / dense-MLP weights, production quant path), prefill-only, no speculative decoding.

  1. KL between configurations at prompt positions: prompt_logprobs forces the logits of every prompt position
     through model.compute_logits; this driver captures the fp32 log-softmax at fixed positions (every --stride-th)
     of --prompts real-text windows. --save stores them as the reference arm; --ref <dir> compares against a stored
     reference: KL(ref || this) mean / p50 / p95 / p99 / max, top-1 agreement, mean NLL of the true next token.
  2. (--stats 1, needs GLM53_DENSE_W8A8 installed) per dense-FP8 layer, first served prefill call: the added error of
     the W8A8 path relative to the error production already has against the BF16 weight (the weights' e4m3
     rounding): e_w = ||Y_prod - Y_bf16||, e_a = ||Y_w8a8 - Y_prod||, r = e_a^2 / e_w^2, plus per-token activation
     statistics (amax/rms, e4m3-subnormal fraction after per-token scaling, worst-token relative error).
Deterministic in-process: the sparse indexer's prefill top-k is replaced by a stable full sort (as the handoff
harness's --topk full); the same prompts and cache salts in every arm.
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
ap.add_argument("--model", default=os.environ.get("KL_MODEL", os.path.join(os.environ.get("TF_EXL3_MODELS") or os.path.expanduser("~/models"), "GLM-5.3-Flash-handoff-mini-w8a8")))
ap.add_argument("--prompts", default="4000,4000,4000,4000")
ap.add_argument("--stride", type=int, default=16)
ap.add_argument("--save", type=int, default=0)
ap.add_argument("--ref", default="")
ap.add_argument("--stats", type=int, default=0)
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
    s = random.Random("w8a82-kl-%d" % i).randrange(0, len(ids) - n - 1)
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


# ---------------------------------------------------------------- per-layer error statistics (W8A8 vs production)
ST = {"bf16": {}, "seen": set(), "rows": []}


def install_stats_pwal():
    import vllm.model_executor.layers.quantization.exl3 as Q
    cls = Q.Glm53DenseFp8Method
    orig = cls.process_weights_after_loading

    def pwal(self, layer):
        w = layer.weight.data
        if w.dtype in (torch.bfloat16, torch.float16):
            ST["bf16"][id(layer)] = w.detach().to("cpu", copy=True)
        return orig(self, layer)

    cls.process_weights_after_loading = pwal


def install_stats_apply():
    import vllm.model_executor.layers.quantization.exl3 as Q
    import fp8_w8a8 as W
    cls = Q.Glm53DenseFp8Method
    cur = cls.apply

    def apply(self, layer, x, bias=None):
        y = cur(self, layer, x, bias)
        try:
            pre = getattr(self, "prefix", "?")
            if (W.STATE.enabled and pre not in ST["seen"] and x.dim() >= 2 and x.numel() // x.shape[-1] >= 2048
                    and not torch.cuda.is_current_stream_capturing() and id(layer) in ST["bf16"]):
                ST["seen"].add(pre)
                one(self, layer, x, bias, pre, y, W)
        except Exception as e:  # noqa: BLE001
            log(f"stats {getattr(self, 'prefix', '?')}: {e!r}")
        return y

    cls.apply = apply


def one(self, layer, x, bias, pre, y_served, W):
    k = x.shape[-1]
    x2 = x.reshape(-1, k)[:4096].contiguous()
    n, kk = int(layer.glm53_fp8_n), int(layer.glm53_fp8_k)
    wb = ST["bf16"][id(layer)].to(x.device)
    y_bf = (x2.float() @ wb.float().t())
    y_p = W.STATE.orig_apply(self, layer, x2, bias).float()            # production's path (fp8_gemv wrapper)
    y_w = W.w8a8_forward(x2, layer.weight, layer.weight_scale, n, kk, bias).float()
    if bias is not None:
        y_bf = y_bf + bias.float()
    e_w = (y_p - y_bf).norm().item()
    e_a = (y_w - y_p).norm().item()
    nrm = y_bf.norm().item()
    # per-token activation statistics
    xf = x2.float()
    amax = xf.abs().amax(1).clamp_min(1e-30)
    rms = xf.pow(2).mean(1).sqrt().clamp_min(1e-30)
    ratio = (amax / rms)
    scaled = xf.abs() / (amax[:, None] / 448.0)
    sub = ((scaled < 2 ** -6) & (xf != 0)).float().mean().item()
    # energy share of the subnormal elements (what matters for the dot products)
    sub_e = (xf.pow(2) * ((scaled < 2 ** -6) & (xf != 0)).float()).sum().item() / max(xf.pow(2).sum().item(), 1e-30)
    tok_a = (y_w - y_p).norm(dim=1) / y_p.norm(dim=1).clamp_min(1e-30)
    tok_w = (y_p - y_bf).norm(dim=1) / y_bf.norm(dim=1).clamp_min(1e-30)
    # activation-only rounding error of x itself
    q, s = W.quant_per_token(x2)
    xq = q.float() * s
    ex = ((xq - xf).norm() / xf.norm()).item()
    rec = {"prefix": pre, "group": self.group, "n": n, "k": kk, "rows": x2.shape[0],
           "rel_w": e_w / nrm, "rel_a": e_a / nrm, "r": (e_a / max(e_w, 1e-30)) ** 2, "rel_x_quant": ex,
           "amax_rms_p50": ratio.median().item(), "amax_rms_max": ratio.max().item(),
           "subnormal_frac": sub, "subnormal_energy": sub_e,
           "tok_rel_a_p99": tok_a.quantile(0.99).item(), "tok_rel_a_max": tok_a.max().item(),
           "tok_rel_w_p99": tok_w.quantile(0.99).item(), "served_eq": bool(torch.equal(y_served.reshape(-1, n)[:4096]
                                                                                       .float(), y_w))}
    ST["rows"].append(rec)
    log("STAT " + json.dumps(rec))


def main():
    from vllm import LLM, SamplingParams
    install_topk()
    install_logits_capture()
    if ARGS.stats:
        install_stats_pwal()
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
    try:
        import fp8_w8a8 as W
        log(f"w8a8: enabled {W.STATE.enabled} reason {W.STATE.disabled_reason}; {W.summary()}")
    except Exception as e:  # noqa: BLE001
        W = None
        log("fp8_w8a8 not importable:", repr(e))
    if ARGS.stats:
        install_stats_apply()
    sp = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True, prompt_logprobs=1, detokenize=False, seed=0)
    prompts = []
    for i, n in enumerate(int(x) for x in ARGS.prompts.split(",") if x.strip()):
        ids = prompt_ids(n, i)
        prompts.append(ids)
        CAP["cur"] = i
        llm.generate([{"prompt_token_ids": ids, "cache_salt": f"w8a82-kl-{i}"}], sp, use_tqdm=False)
        CAP["cur"] = None
        log(f"prompt {i}: {n} tokens, captured {len(CAP['rows'].get(i, {}))} positions")
    if W is not None:
        log(f"w8a8 counters after the prompts: {W.summary()}")
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
    if ARGS.stats:
        json.dump(ST["rows"], open(os.path.join(ARGS.out, "stats.json"), "w"))
        if ST["rows"]:
            import statistics as S
            by = {}
            for r in ST["rows"]:
                by.setdefault(r["group"], []).append(r)
            for g, rs in sorted(by.items()):
                log(f"STATSUM {g}: layers {len(rs)} r median {S.median(x['r'] for x in rs):.3f} "
                    f"[{min(x['r'] for x in rs):.3f}..{max(x['r'] for x in rs):.3f}] rel_w "
                    f"{S.median(x['rel_w'] for x in rs):.2e} rel_a {S.median(x['rel_a'] for x in rs):.2e} "
                    f"subnormal energy max {max(x['subnormal_energy'] for x in rs):.1e}")


if __name__ == "__main__":
    main()
