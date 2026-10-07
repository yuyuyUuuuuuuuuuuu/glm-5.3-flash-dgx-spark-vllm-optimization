#!/usr/bin/env python3
"""[glm53-draft-lmhead-fp8] Give the DFlash2 drafter its own FP8 copy of the (shared) target lm_head shard for
candidate generation only (idempotent, fail closed, env-gated at runtime; default off = stock).

Today the drafter has no lm_head of its own: load_dflash_model (spec_decode/dflash/utils.py) points
dflash_model.lm_head at the target's BF16 ParallelLMHead, and DFlash2's compute_candidates
(qwen3_dflash2.py) runs a full-vocab projection with it for 7 rows per request every step.
With GLM53_DRAFT_LMHEAD_FP8=1, after that sharing, a per-rank FP8 e4m3 copy (per output channel,
Marlin; the same quantization as exl3.Glm53DenseFp8Method, computed in row chunks so the fp32 transient
stays ~128 MiB) is attached as dflash_model.glm53_candidate_head and compute_candidates uses it.
The target keeps its BF16 lm_head for verification, sampling and logprobs (same object, untouched).
Memory: +302.6 MiB per rank at TP=2 (77440 x 4096 FP8 + scales), measured on nodeC.

  GLM53_DRAFT_LMHEAD_FP8 = 0 (default: stock) | 1

The copy is taken once at drafter load; an in-place reload of the target weights (RL refit, sleep-mode
reload) would leave it stale, so do not combine with those. Fails closed when the drafter has no
compute_candidates (not DFlash2), when compute_candidates does not consume the copy (qwen3_dflash2.py
overwritten after this patch), or when an fp32 head_dtype is configured.

Usage: python3 patch_drafter_lmhead_fp8.py [--preflight]   (GLM53_SITE overrides the vLLM package root)
"""
from __future__ import annotations

import hashlib
import os
import stat
import sys
from pathlib import Path

MARK = "[glm53-draft-lmhead-fp8]"
ENV_NAME = "GLM53_DRAFT_LMHEAD_FP8"
SITE = Path(os.environ.get("GLM53_SITE", "/usr/local/lib/python3.12/dist-packages/vllm"))
UTILS = SITE / "v1/worker/gpu/spec_decode/dflash/utils.py"
MODEL = SITE / "model_executor/models/qwen3_dflash2.py"
STOCK_SHA256 = {
    "utils.py": "94106b29446769d71cd82297c7320715bcdefaea9b97341bc166ca3f14088690",
    "qwen3_dflash2.py": "81e5294543d584644572e5de18a792a0987fc45d6b12df4be23941387beeba8b",
}

U1_OLD = "def load_dflash_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:\n"
U1_NEW = '''# [glm53-draft-lmhead-fp8] -----------------------------------------------------------------
# GLM53_DRAFT_LMHEAD_FP8 = 0 (default) | 1 (see /opt/glm53/patch_drafter_lmhead_fp8.py).
def _glm53_draft_lmhead_fp8_enabled() -> bool:
    import os

    raw = os.environ.get("GLM53_DRAFT_LMHEAD_FP8", "0").strip()
    if raw not in ("0", "1"):
        raise ValueError("GLM53_DRAFT_LMHEAD_FP8 must be 0 or 1 (got: %r)" % (raw,))
    return raw == "1"


class _Glm53Fp8HeadApply:
    """apply() of exl3.Glm53DenseFp8Method for the already-repacked copy. Deliberately not a
    QuantizeMethodBase, so no loader pass ever re-processes the copy."""

    def __init__(self, method) -> None:
        self.method = method

    def apply(self, layer, x, bias=None):
        return self.method.apply(layer, x, bias)


class _Glm53Fp8CandidateHead(nn.Module):
    """[glm53-draft-lmhead-fp8] Drafter-only FP8 e4m3 copy of this rank's lm_head shard."""

    CHUNK_ROWS = 8192

    def __init__(self, src: nn.Module) -> None:
        super().__init__()
        import inspect

        import torch

        from vllm.model_executor.layers.quantization.exl3 import Glm53DenseFp8Method
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            prepare_fp8_layer_for_marlin,
        )

        w = src.weight.data
        if w.dtype not in (torch.bfloat16, torch.float16) or w.dim() != 2:
            raise RuntimeError(
                "[glm53-draft-lmhead-fp8] expected a 2-D BF16/FP16 lm_head weight, got %s %s"
                % (w.dtype, tuple(w.shape))
            )
        n, k = w.shape
        # Same numerics as Glm53DenseFp8Method.process_weights_after_loading (amax/448 per
        # output channel, e4m3); rows are independent, so row chunks give identical bytes.
        fp8 = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=w.device)
        scales = torch.empty(n, dtype=torch.float32, device=w.device)
        for a in range(0, n, self.CHUNK_ROWS):
            wf = w[a : a + self.CHUNK_ROWS].float()
            s = wf.abs().amax(dim=1).clamp(min=1e-12) / 448.0
            fp8[a : a + self.CHUNK_ROWS] = (wf / s[:, None]).clamp(-448.0, 448.0).to(
                torch.float8_e4m3fn
            )
            scales[a : a + self.CHUNK_ROWS] = s
            del wf
        self.output_size_per_partition, self.input_size_per_partition = n, k
        self.orig_dtype = w.dtype
        self.weight = nn.Parameter(fp8, requires_grad=False)
        self.weight_scale = nn.Parameter(scales.to(w.dtype), requires_grad=False)
        self.weight_block_size = None
        prepare_fp8_layer_for_marlin(self, size_k_first=False)
        self.glm53_fp8_n, self.glm53_fp8_k = n, k
        if "prefix" in inspect.signature(Glm53DenseFp8Method.__init__).parameters:
            method = Glm53DenseFp8Method("draft_lm_head", "draft.glm53_candidate_head")
        else:
            method = Glm53DenseFp8Method("draft_lm_head")
        method.ready = True
        self.quant_method = _Glm53Fp8HeadApply(method)
        self.tp_size = src.tp_size
        self.shard_indices = getattr(src, "shard_indices", None)
        torch.cuda.empty_cache()


def _glm53_attach_candidate_head(dflash_model: nn.Module, vllm_config: VllmConfig) -> None:
    if not _glm53_draft_lmhead_fp8_enabled():
        return
    import inspect

    from vllm.logger import init_logger

    compute_candidates = getattr(type(dflash_model), "compute_candidates", None)
    if compute_candidates is None:
        raise RuntimeError(
            "[glm53-draft-lmhead-fp8] GLM53_DRAFT_LMHEAD_FP8=1 needs a DFlash2 drafter "
            "(%s has no compute_candidates)" % type(dflash_model).__name__
        )
    if "glm53_candidate_head" not in inspect.getsource(compute_candidates):
        raise RuntimeError(
            "[glm53-draft-lmhead-fp8] compute_candidates does not use the FP8 copy "
            "(qwen3_dflash2.py was replaced after patch_drafter_lmhead_fp8.py)"
        )
    lm_head = getattr(dflash_model, "lm_head", None)
    if lm_head is None:
        raise RuntimeError("[glm53-draft-lmhead-fp8] drafter has no lm_head to copy")
    head_dtype = vllm_config.model_config.head_dtype
    if head_dtype is not None and head_dtype != lm_head.weight.dtype:
        raise RuntimeError(
            "[glm53-draft-lmhead-fp8] head_dtype %s != lm_head dtype %s is not supported"
            % (head_dtype, lm_head.weight.dtype)
        )
    head = _Glm53Fp8CandidateHead(lm_head)
    # Plain attribute (not a registered submodule): invisible to loader passes, state_dict, reloads.
    object.__setattr__(dflash_model, "glm53_candidate_head", head)
    nbytes = sum(
        t.numel() * t.element_size() for t in (head.weight, head.weight_scale, head.workspace)
    )
    init_logger(__name__).info(
        "[glm53-draft-lmhead-fp8] drafter candidate head: FP8 copy of lm_head shard %s "
        "(+%.1f MiB this rank); target lm_head stays %s",
        tuple(lm_head.weight.shape),
        nbytes / 2**20,
        lm_head.weight.dtype,
    )


def load_dflash_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
'''
U2_OLD = "        dflash_model.lm_head = target_lm_head\n\n    return dflash_model\n"
U2_NEW = ("        dflash_model.lm_head = target_lm_head\n\n"
          "    _glm53_attach_candidate_head(dflash_model, vllm_config)  # [glm53-draft-lmhead-fp8]\n"
          "    return dflash_model\n")
M1_OLD = "        logits = self.candidate_logits_processor(self.lm_head, hidden_states)\n"
M1_NEW = ("        # [glm53-draft-lmhead-fp8] drafter-only FP8 copy of the shared lm_head, if attached\n"
          "        head = getattr(self, \"glm53_candidate_head\", None)\n"
          "        logits = self.candidate_logits_processor(\n"
          "            self.lm_head if head is None else head, hidden_states\n"
          "        )\n")

FILES = ((UTILS, "utils.py", (("helpers", U1_OLD, U1_NEW), ("attach call", U2_OLD, U2_NEW))),
         (MODEL, "qwen3_dflash2.py", (("compute_candidates", M1_OLD, M1_NEW),)))


def parse_env(raw: str | None) -> bool:
    raw = "0" if raw is None else raw.strip()
    if raw not in ("0", "1"):
        raise ValueError(f"{ENV_NAME} must be 0 or 1 (got: {raw!r})")
    return raw == "1"


def prepare(source: str, edits) -> tuple[str, str]:
    states = []
    for label, old, new in edits:
        n_new = source.count(new)
        n_old = source.count(old) - (n_new if old in new else 0)
        if n_old == 1 and n_new == 0:
            states.append("stock")
        elif n_old == 0 and n_new == 1:
            states.append("patched")
        else:
            raise ValueError(f"{label}: anchor ambiguous or drifted (stock={n_old}, patched={n_new})")
    if all(s == "patched" for s in states):
        return source, "already present"
    if not all(s == "stock" for s in states):
        raise ValueError(f"partially patched source: {dict(zip((e[0] for e in edits), states))}")
    if MARK in source:
        raise ValueError(f"{MARK} present outside the known edits")
    out = source
    for _label, old, new in edits:
        out = out.replace(old, new, 1)
    return out, "patched"


def replace_file(target: Path, text: str) -> None:
    tmp = target.with_name(f".{target.name}.glm53-draft-lmhead-fp8.tmp")
    try:
        tmp.write_text(text)
        os.chmod(tmp, stat.S_IMODE(target.stat().st_mode))
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def clear_pyc(target: Path) -> None:
    cache = target.parent / "__pycache__"
    if cache.is_dir():
        for pyc in cache.glob(f"{target.stem}.*.pyc"):
            pyc.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    unknown = [a for a in argv[1:] if a != "--preflight"]
    if unknown:
        raise SystemExit(f"unknown arguments: {' '.join(unknown)}")
    preflight = "--preflight" in argv[1:]
    plans = []
    try:
        on = parse_env(os.environ.get(ENV_NAME))
        for path, name, edits in FILES:  # verify BOTH files before writing either
            if not path.is_file():
                raise ValueError(f"missing {path}")
            source = path.read_text()
            patched, action = prepare(source, edits)
            compile(patched, str(path), "exec")
            sha = hashlib.sha256(source.encode()).hexdigest()
            stock = "stock sha256" if sha == STOCK_SHA256[name] else f"sha256 {sha[:12]}"
            plans.append((path, name, source, patched, action, stock))
    except ValueError as exc:
        raise SystemExit(f"{MARK} preflight failed: {exc}") from exc
    mode = "runtime on (drafter candidate head FP8)" if on else "runtime off (stock)"
    summary = "; ".join(f"{name}: {action} ({stock})" for _p, name, _s, _t, action, stock in plans)
    if preflight:
        print(f"{MARK} preflight OK: {summary}; {mode}")
        return 0
    for path, _name, source, patched, _action, _stock in plans:
        if patched != source:
            replace_file(path, patched)
            clear_pyc(path)
    print(f"{MARK} {summary}; {mode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
