#!/usr/bin/env python3
"""FKDA2 end-to-end: prompt logprobs (top-20 per position, the production quality probe's 6 PREFILL texts) of the
handoff mini model (real GLM-5.3-Flash KDA weights, production-composed container) with the KDA chunked prefill run
by one of:
  FKDA2_MODE=triton  production's Triton chain (glm53_flashkda.STATE["off"] = True -> kda.py's else branch)
  FKDA2_MODE=fk      the FlashKDA build found first on sys.path (FKDA2_EXT_DIR, default the installed overlay one)
  FKDA2_MODE=exact   an fp64 torch implementation of the exact KDA recurrence (tests/fkda2/short_accuracy.py
                     ref_fp64), output rounded to bf16 once, final state fp32 -- for calls up to 4096 tokens
                     (boot/profile dummy runs above that use FlashKDA; their content is never read)
Everything else in the forward is production's code, deterministic in-process (handoff shadow control IDENTICAL),
so the per-position KL between two modes isolates the KDA prefill kernel. Writes /out/<label>/prompt_logprobs.json.
Run: tests/handoff/run.sh <out> <label> HANDOFF_DRIVER=/w/tests/fkda2/e2e_kl.py FKDA2_MODE=... GLM53_KDA_FLASHKDA=1 ...
"""
import json
import os
import sys

EXT = os.environ.get("FKDA2_EXT_DIR", "")
if EXT:
    sys.path.insert(0, EXT)          # before anything imports _flashkda_fp32_C
sys.path.insert(0, "/w/tests/handoff")
sys.path.insert(0, "/w/tools/prodcheck")
sys.path.insert(0, "/w/tests/fkda2")
import run_engine as RE  # noqa: E402
import quality_probe as QP  # noqa: E402

import torch  # noqa: E402

MODE = os.environ.get("FKDA2_MODE", "fk")
CNT = {"exact": 0, "fk": 0}


def install():
    import glm53_flashkda as F
    import _flashkda_fp32_C as X
    F.STATE["allow_any_ext"] = True   # this rig A/Bs builds other than the wrapper's pinned one
    RE.log(f"fkda2 e2e: mode {MODE}; _flashkda_fp32_C from {X.__file__}")
    if MODE == "triton":
        F.STATE["off"] = True
        return
    orig = F.chunk_prefill
    if MODE == "fk":
        def counted(*a, **k):
            CNT["fk"] += 1
            return orig(*a, **k)
        F.chunk_prefill = counted
        return
    from short_accuracy import ref_fp64

    def exact(layer, q, k, v, g, beta, initial_state, cu_seqlens):
        T = q.shape[1]
        if T > 4096 or torch.cuda.is_current_stream_capturing():
            return orig(layer, q, k, v, g, beta, initial_state, cu_seqlens)
        CNT["exact"] += 1
        D = layer.head_dim
        cu = cu_seqlens.tolist()
        o, fin = ref_fp64(q, k, v, g, beta, layer.A_log.float().reshape(-1), layer.dt_bias.float().reshape(-1, D),
                          float(layer.kda_lower_bound), initial_state, cu)
        return o.unsqueeze(0).to(q.dtype), fin.float()
    F.chunk_prefill = exact


def main():
    from vllm import LLM, SamplingParams
    from tokenizers import Tokenizer
    A = RE.ARGS
    k_set = sorted(int(x) for x in os.environ.get("GLM53_ADAPTIVE_K_SET", "4,5,7").split(",") if x.strip())
    sizes = {1, 2, 4, 8, 16, 24, 32}
    sizes.update(n * (k + 1) for n in range(1, 5) for k in k_set + [7])
    spec = {"method": "dflash", "model": A.model + "-dflash2", "num_speculative_tokens": 7,
            "kv_cache_dtype": "auto", "draft_sample_method": "probabilistic", "rejection_sample_method": "standard"}
    RE._install_topk_determinizer()
    install()
    llm = LLM(model=A.model, skip_tokenizer_init=True, tensor_parallel_size=1, dtype="bfloat16",
              max_model_len=A.max_model_len, max_num_seqs=4, max_num_batched_tokens=A.mnbt,
              enable_prefix_caching=True, kv_cache_dtype="fp8", kv_cache_memory_bytes=A.kv_bytes,
              gpu_memory_utilization=A.gpu_util, speculative_config=spec, enable_flashinfer_autotune=False, seed=0,
              compilation_config={"cudagraph_capture_sizes": sorted(sizes)})
    tok = Tokenizer.from_file(os.path.join(os.environ.get("TF_EXL3_MODELS") or os.path.expanduser("~/models"), "GLM-OCR/tokenizer.json"))
    texts = list(QP.PREFILL)
    extra = int(os.environ.get("FKDA2_EXTRA_LEN", "0"))
    ids = [[int(x) for x in tok.encode(t).ids] for t in texts]
    if extra:
        ids.append(RE._orig_prompt_ids(extra, 99) if hasattr(RE, "_orig_prompt_ids") else RE.prompt_ids(extra, 99))
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=20, detokenize=False, seed=0)
    res = {"mode": MODE, "ext": EXT, "prefill": []}
    for i, p in enumerate(ids):
        o = llm.generate([{"prompt_token_ids": p, "cache_salt": f"fkda2-{i}"}], sp, use_tqdm=False)[0]
        pl = [None if d is None else {str(int(t)): float(v.logprob) for t, v in d.items()} for d in o.prompt_logprobs]
        res["prefill"].append(pl)
        RE.log(f"fkda2 e2e: prompt {i}: {len(p)} tokens")
    res["counts"] = CNT
    json.dump(res, open(os.path.join(A.out, "prompt_logprobs.json"), "w"))
    RE.log(f"fkda2 e2e: mode {MODE} done; calls {CNT}")


main()
