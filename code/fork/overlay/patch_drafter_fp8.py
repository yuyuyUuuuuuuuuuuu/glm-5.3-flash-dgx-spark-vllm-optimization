#!/usr/bin/env python3
"""[glm53-draft-fp8] Let the DFlash/DFlash2 drafter's own linears use the target's FP8 Marlin path
(idempotent, fail closed, env-gated at runtime; default off = stock BF16).

How the drafter gets its quant method today (image vLLM, verified on nodeC with the real config classes):
  DFlashQwen3Model.__init__ sets self.quant_config = get_draft_quant_config(vllm_config), i.e. the quant
  config of the *draft* ModelConfig. The launcher's speculative config has no "quantization" key and
  incoai/GLM-5.3-Flash-DFlash2 has no quantization_config, so it is None and every drafter linear gets
  UnquantizedLinearMethod (BF16). The drafter never reaches Exl3Config.get_quant_method, so exl3.py's
  `"draft" in prefix` exclusion (image exl3.py:1965; launcher overlay exl3.py:1813) is not what keeps it BF16;
  its prefixes are model.layers.<45+i>.{self_attn,mlp}.* and model.fc anyway.

Both exl3.py versions work: the image's Glm53DenseFp8Method(group) and the launcher overlay's
Glm53DenseFp8Method(group, prefix) (which production installs over the image's at container start; its
process_weights_after_loading also consults the TP world size for a TP=3-only KDA gate that never matches the
drafter's "draft" group). The constructor shape is detected with inspect.

This patch wraps that call: with GLM53_DRAFT_FP8 set, the draft quant config becomes a small
Glm53DraftFp8Config whose get_quant_method returns exl3.Glm53DenseFp8Method (the class GLM53_DENSE_FP8
uses for the target: per-output-channel e4m3 + Marlin, quantized in process_weights_after_loading) for
allow-listed prefixes and UnquantizedLinearMethod for every other linear. The target is untouched.

  GLM53_DRAFT_FP8 = off | 0 (default: stock) | 1 | layers | layers,fc
    layers  model.layers.*.self_attn.qkv_proj / o_proj, model.layers.*.mlp.gate_up_proj / down_proj
    fc      model.fc (aux-hidden combiner, replicated, runs eagerly each step)

Also fails closed if the context-K/V fusion (_build_context_kv_buffers, which slices qkv_proj.weight)
ever runs after the FP8 repack (it normally runs at the end of load_weights, before
process_weights_after_loading, so it keeps a BF16 copy of the K/V rows).

Compile caches: the value is written INTO the patched qwen3_dflash.py (_GLM53_DRAFT_FP8_INSTALLED) at
container start, and the runtime refuses to build the drafter if GLM53_DRAFT_FP8 differs from it.
vLLM's torch.compile caches do not see GLM53_* variables: the AOT key (compilation/decorators.py) is
sha256(envs.compile_factors() [known VLLM_* only] + vllm_config.compute_hash() [draft quantization is
None in every mode] + vllm version, forward qualname and first line), and a loaded artifact runs with
guards disabled. What the loader does check is the text of every traced source file
(_verify_source_unchanged; a mismatch logs "Compiling model again due to a load failure" and recompiles);
the VllmBackend cache hashes the same files. DFlashQwen3Model.forward is defined in this file, so a
different mode gives different file text and the drafter is recompiled instead of reusing a graph that
was traced for the other weight layout (BF16 weight vs Marlin-packed int32 + weight_scale + workspace).
Every toggle and every rollback therefore costs one drafter recompile at start.

Usage: python3 patch_drafter_fp8.py [--preflight]   (GLM53_SITE overrides the vLLM package root)
"""
from __future__ import annotations

import hashlib
import os
import stat
import sys
from pathlib import Path

MARK = "[glm53-draft-fp8]"
ENV_NAME = "GLM53_DRAFT_FP8"
SITE = Path(os.environ.get("GLM53_SITE", "/usr/local/lib/python3.12/dist-packages/vllm"))
TARGET = SITE / "model_executor/models/qwen3_dflash.py"
STOCK_SHA256 = "40b3a4c7b8893fe92b9e291b566d763a2c6e29712f3a4d56d1a5b246d1815745"
GROUPS = ("layers", "fc")

A1_OLD = "logger = init_logger(__name__)\n"
A1_TEMPLATE = '''logger = init_logger(__name__)


# [glm53-draft-fp8] ------------------------------------------------------------------------
# GLM53_DRAFT_FP8 = off | 0 (default) | 1 | layers | layers,fc (see /opt/glm53/patch_drafter_fp8.py).
# Drafter-only FP8 e4m3 per output channel on the Marlin path of exl3.Glm53DenseFp8Method.
# The mode is part of this file's text on purpose: vLLM's compile caches are keyed without GLM53_*
# variables but are invalidated when a traced source file changes, and DFlashQwen3Model.forward is
# traced from this file. The patch script writes it at container start; the runtime checks the env.
_GLM53_DRAFT_FP8_INSTALLED = "@MODE@"
_GLM53_DRAFT_FP8_SUFFIXES = {
    "layers": (
        ".self_attn.qkv_proj",
        ".self_attn.o_proj",
        ".mlp.gate_up_proj",
        ".mlp.down_proj",
    ),
    "fc": (".fc",),
}


def _glm53_draft_fp8_parse(raw: str) -> frozenset:
    raw = raw.strip().lower()
    if raw in ("", "off", "0", "no", "none"):
        return frozenset()
    if raw in ("1", "on"):
        return frozenset({"layers"})
    groups = frozenset(g.strip() for g in raw.split(",") if g.strip())
    unknown = groups - set(_GLM53_DRAFT_FP8_SUFFIXES)
    if unknown or not groups:
        raise ValueError(
            "GLM53_DRAFT_FP8 must be off | 0 | 1 | layers | layers,fc (got: %r)" % (raw,)
        )
    return groups


def _glm53_draft_fp8_groups() -> frozenset:
    import os

    groups = _glm53_draft_fp8_parse(os.environ.get("GLM53_DRAFT_FP8", "off"))
    installed = _glm53_draft_fp8_parse(_GLM53_DRAFT_FP8_INSTALLED)
    if groups != installed:
        raise RuntimeError(
            "[glm53-draft-fp8] GLM53_DRAFT_FP8=%r at runtime, but qwen3_dflash.py was patched "
            "for %r; run /opt/glm53/patch_drafter_fp8.py with the same value (the compile "
            "cache is invalidated by this file's text, not by the variable)"
            % (os.environ.get("GLM53_DRAFT_FP8"), _GLM53_DRAFT_FP8_INSTALLED)
        )
    return groups


def _glm53_draft_fp8_group(prefix: str, groups) -> str | None:
    for group in sorted(groups):
        for suffix in _GLM53_DRAFT_FP8_SUFFIXES[group]:
            if prefix.endswith(suffix) and (group != "layers" or ".layers." in prefix):
                return group
    return None


class Glm53DraftFp8Config(QuantizationConfig):
    """[glm53-draft-fp8] Drafter-only config: FP8 Marlin for allow-listed linears, BF16 elsewhere."""

    def __init__(self, groups) -> None:
        super().__init__()
        self.groups = frozenset(groups)

    def get_name(self):
        return "glm53_draft_fp8"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError("built from GLM53_DRAFT_FP8, not from a checkpoint")

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        import inspect

        from vllm.model_executor.layers.linear import (
            LinearBase,
            ReplicatedLinear,
            UnquantizedLinearMethod,
        )

        if not isinstance(layer, LinearBase):
            return None
        group = _glm53_draft_fp8_group(prefix, self.groups)
        if group is None:
            return UnquantizedLinearMethod()
        if isinstance(layer, ReplicatedLinear) and not hasattr(layer, "output_size_per_partition"):
            # model.fc is replicated: no TP partition sizes, which the FP8 method and Marlin read.
            layer.output_size_per_partition = layer.output_size
            layer.input_size_per_partition = layer.input_size
        from vllm.model_executor.layers.quantization.exl3 import Glm53DenseFp8Method

        # The launcher's overlay exl3.py takes (group, prefix); the image's takes (group).
        if "prefix" in inspect.signature(Glm53DenseFp8Method.__init__).parameters:
            return Glm53DenseFp8Method("draft", prefix)
        return Glm53DenseFp8Method("draft")


def _glm53_draft_fp8_quant_config(base):
    groups = _glm53_draft_fp8_groups()
    if not groups:
        return base
    if base is not None:
        raise RuntimeError(
            "[glm53-draft-fp8] GLM53_DRAFT_FP8 needs a BF16 drafter, but the draft model "
            "already has quantization %r; unset GLM53_DRAFT_FP8" % (base.get_name(),)
        )
    logger.info(
        "[glm53-draft-fp8] drafter linears -> FP8 e4m3 per output channel (Marlin): %s",
        ",".join(sorted(groups)),
    )
    return Glm53DraftFp8Config(groups)
'''

A2_OLD = "        self.quant_config = get_draft_quant_config(vllm_config)\n"
A2_NEW = ("        self.quant_config = _glm53_draft_fp8_quant_config(  # [glm53-draft-fp8]\n"
          "            get_draft_quant_config(vllm_config)\n"
          "        )\n")

A3_OLD = """        # KV projection weights: [num_layers * 2 * kv_size, hidden_size]
        kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]
"""
A3_NEW = """        # KV projection weights: [num_layers * 2 * kv_size, hidden_size]
        for a in layers_attn:  # [glm53-draft-fp8] must read the BF16 weight, not a Marlin repack
            if not a.qkv_proj.weight.is_floating_point() or a.qkv_proj.weight.element_size() < 2:
                raise RuntimeError(
                    "[glm53-draft-fp8] context K/V fusion ran after the FP8 repack of "
                    "qkv_proj (dummy or reloaded weights?); unset GLM53_DRAFT_FP8"
                )
        kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]
"""

# Canonical mode strings written into the patched file (one per distinct group set).
MODES = ("off", "layers", "fc", "layers,fc")


def a1_new(mode: str) -> str:
    assert mode in MODES, mode
    return A1_TEMPLATE.replace("@MODE@", mode)


def parse_env(raw: str | None) -> frozenset:
    raw = "off" if raw is None else raw.strip().lower()
    if raw in ("", "off", "0", "no", "none"):
        return frozenset()
    if raw in ("1", "on"):
        return frozenset({"layers"})
    groups = frozenset(g.strip() for g in raw.split(",") if g.strip())
    if not groups or groups - set(GROUPS):
        raise ValueError(f"{ENV_NAME} must be off | 0 | 1 | layers | layers,fc (got: {raw!r})")
    return groups


def mode_of(groups: frozenset) -> str:
    return ",".join(g for g in GROUPS if g in groups) or "off"


def prepare(source: str, mode: str) -> tuple[str, str]:
    """Return (patched source, action). Every anchor must be in exactly one state; a file patched
    for another mode gets only its mode line rewritten."""
    states, installed = [], None
    for label, old, new in (("helpers", A1_OLD, None), ("draft quant config", A2_OLD, A2_NEW),
                            ("kv fusion guard", A3_OLD, A3_NEW)):
        if new is None:  # A1: the patched form depends on the mode; A1_OLD is a prefix of it
            hits = {m: source.count(a1_new(m)) for m in MODES}
            n_new = sum(hits.values())
            n_old = source.count(old) - n_new
            if n_new == 1:
                installed = next(m for m, c in hits.items() if c)
        else:
            n_new = source.count(new)
            n_old = source.count(old)
        if n_old == 1 and n_new == 0:
            states.append("stock")
        elif n_old == 0 and n_new == 1:
            states.append("patched")
        else:
            raise ValueError(f"{label}: anchor ambiguous or drifted (stock={n_old}, patched={n_new})")
    if all(s == "patched" for s in states):
        if installed == mode:
            return source, "already present"
        return source.replace(a1_new(installed), a1_new(mode), 1), f"re-patched (mode {installed} -> {mode})"
    if not all(s == "stock" for s in states):
        raise ValueError(f"partially patched source: {dict(zip(('helpers', 'draft quant config', 'kv fusion guard'), states))}")
    if MARK in source:
        raise ValueError(f"{MARK} present outside the known edits")
    out = source
    for old, new in ((A1_OLD, a1_new(mode)), (A2_OLD, A2_NEW), (A3_OLD, A3_NEW)):
        out = out.replace(old, new, 1)
    return out, "patched"


def replace_file(target: Path, text: str) -> None:
    tmp = target.with_name(f".{target.name}.glm53-draft-fp8.tmp")
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
    if not TARGET.is_file():
        raise SystemExit(f"{MARK} missing {TARGET}")
    try:
        groups = parse_env(os.environ.get(ENV_NAME))
        source = TARGET.read_text()
        patched, action = prepare(source, mode_of(groups))
        compile(patched, str(TARGET), "exec")
    except ValueError as exc:
        raise SystemExit(f"{MARK} preflight failed: {exc}") from exc
    sha = hashlib.sha256(source.encode()).hexdigest()
    stock = "stock sha256 match" if sha == STOCK_SHA256 else f"sha256 {sha[:12]} (not the pinned stock file; anchors matched)"
    mode = f"mode {mode_of(groups)} (FP8 groups)" if groups else "mode off (stock BF16 drafter)"
    if preflight:
        print(f"{MARK} {TARGET.name}: preflight OK ({action}; {stock}; {mode})")
        return 0
    if patched != source:
        replace_file(TARGET, patched)
        clear_pyc(TARGET)
    print(f"{MARK} {TARGET.name}: {action}; {stock}; {mode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
