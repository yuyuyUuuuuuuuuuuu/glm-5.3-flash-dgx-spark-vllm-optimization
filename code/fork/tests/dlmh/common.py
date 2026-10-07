"""Shared helpers for the DEC_DLMH tests / benches: production-converted FP8 lm_head shards (glm53_runtime numerics)
and real-drafter hidden states."""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, "/w")
DRAFT = os.path.join(os.environ.get("TF_EXL3_MODELS") or os.path.expanduser("~/models"), "GLM-5.3-Flash-DFlash2-dc77ff1c")
TGT = os.path.join(os.environ.get("TF_EXL3_MODELS") or os.path.expanduser("~/models"), "GLM-5.3-Flash-EXL3-TR3-4bpw-partial")
RO = f"{DRAFT}:{TGT}:" + os.path.join(os.environ.get("TF_EXL3_MODELS") or os.path.expanduser("~/models"), "GLM-5.3-Flash-handoff-mini")
V, H = 154880, 4096


def load_lm_bf16(dev="cuda"):
    return torch.from_file(os.path.join(TGT, "lm_head", "lm_head.weight.bin"), shared=False, size=V * H,
                           dtype=torch.bfloat16).view(V, H).to(dev)


def fp8_holder(w):
    """exactly glm53_runtime.convert_lm_head_fp8's holder for one vocab shard (bf16 [n, k])."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_layer_for_marlin
    n, k = w.shape
    holder = nn.Module()
    fp8 = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=w.device)
    scales = torch.empty(n, dtype=torch.float32, device=w.device)
    for a in range(0, n, 8192):
        wf = w[a:a + 8192].float()
        sc = wf.abs().amax(dim=1).clamp(min=1e-12) / 448.0
        fp8[a:a + 8192] = (wf / sc[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        scales[a:a + 8192] = sc
        del wf
    holder.output_size_per_partition, holder.input_size_per_partition = n, k
    holder.orig_dtype = w.dtype
    holder.weight = nn.Parameter(fp8, requires_grad=False)
    holder.weight_scale = nn.Parameter(scales.to(w.dtype), requires_grad=False)
    holder.weight_block_size = None
    prepare_fp8_layer_for_marlin(holder, size_k_first=False)
    holder.glm53_fp8_n, holder.glm53_fp8_k = n, k
    return holder, fp8, scales.to(w.dtype)


def enable_fp8_gemv():
    import fp8_gemv as G
    G.STATE.ext = G.ext()
    G.STATE.enabled = True
    G.CFG.strict = True
    return G


def prod_logits(G, holder, x):
    """production's lm_head call for this shard: Glm53DenseFp8Method.apply -> fp8_gemv serve (try_new / Marlin)."""
    return G.linear(x, holder.weight, holder.weight_scale, holder.workspace, None, int(holder.glm53_fp8_n),
                    int(holder.glm53_fp8_k))


def probe_hidden(dev="cuda"):
    """{name: bf16 [448, 4096]} final drafter hidden states of the real DFlash2 drafter on a fixed feature set.
    The feature file and the stand-alone drafter forward that produced it are not part of this repository, so the
    benches that need real drafter states (bench_step.py, review_adv.py, test_dlmh.py part 2) cannot run here."""
    raise NotImplementedError("probe_hidden: the drafter hidden-state fixture is not shipped")

