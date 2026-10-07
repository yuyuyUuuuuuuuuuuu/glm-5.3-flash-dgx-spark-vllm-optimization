#!/usr/bin/env python3
"""FKDA2: capture REAL KDA chunked-prefill inputs from the handoff mini model (real GLM-5.3-Flash KDA weights).

Run as the handoff driver (tests/handoff/run.sh ... HANDOFF_DRIVER=/w/tests/fkda2/dump_kda_inputs.py
GLM53_KDA_FLASHKDA=1 -- --prompts ...): it wraps glm53_flashkda.chunk_prefill (the ON tree's dispatch point, which
receives exactly what FlashKDA gets: q/k/v/g1 [1,T,H,D] bf16 views, RAW bf16 beta logits, fp32 initial state,
int32 cu_seqlens) and saves every call's inputs + the layer's A_log / dt_bias / lower_bound to
/out/<label>/kda_calls/<n>.pt. The forward itself continues unchanged.

Prompts: the 6 PREFILL texts of tools/prodcheck/quality_probe.py (the production "quality short" probe), tokenized
with the GLM-OCR tokenizer run_engine uses, then real-text prompts of the --prompts lengths.
"""
import os
import sys

sys.path.insert(0, "/w/tests/handoff")
sys.path.insert(0, "/w/tools/prodcheck")
import run_engine as RE  # noqa: E402  (parses argv, opens the log)
import quality_probe as QP  # noqa: E402

import torch  # noqa: E402

DUMP = os.path.join(RE.ARGS.out, "kda_calls")
os.makedirs(DUMP, exist_ok=True)
N_CALLS = [0]
_orig_prompt_ids = RE.prompt_ids


def prompt_ids(n, i):
    # i < 6 -> the quality-probe texts (n ignored); else real text of length n
    if i < len(QP.PREFILL):
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(os.environ.get("TF_EXL3_MODELS") or os.path.expanduser("~/models"), "GLM-OCR/tokenizer.json"))
        return [int(x) for x in tok.encode(QP.PREFILL[i]).ids]
    return _orig_prompt_ids(n, i)


RE.prompt_ids = prompt_ids


def install():
    import glm53_flashkda as F
    orig = F.chunk_prefill

    def chunk_prefill(layer, q, k, v, g, beta, initial_state, cu_seqlens):
        if not torch.cuda.is_current_stream_capturing():
            rec = {k_: t.detach().to("cpu", copy=True) for k_, t in
                   dict(q=q, k=k, v=v, g=g, beta=beta, initial_state=initial_state, cu_seqlens=cu_seqlens).items()}
            rec.update(A_log=layer.A_log.detach().float().cpu().reshape(-1),
                       dt_bias=layer.dt_bias.detach().float().cpu().reshape(-1, layer.head_dim),
                       lower_bound=float(layer.kda_lower_bound), prefix=getattr(layer, "prefix", ""),
                       call=N_CALLS[0])
            torch.save(rec, os.path.join(DUMP, f"{N_CALLS[0]:04d}.pt"))
            N_CALLS[0] += 1
        return orig(layer, q, k, v, g, beta, initial_state, cu_seqlens)

    F.chunk_prefill = chunk_prefill
    RE.log(f"fkda2 dump: wrapped glm53_flashkda.chunk_prefill -> {DUMP}")


install()
RE.main()
RE.log(f"fkda2 dump: {N_CALLS[0]} chunk_prefill calls saved")
