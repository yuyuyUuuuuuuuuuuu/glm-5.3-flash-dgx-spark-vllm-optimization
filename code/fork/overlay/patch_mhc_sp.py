#!/usr/bin/env python3
"""[glm53-mhc-sp] Sequence-parallel prefill for the mHC bookkeeping over TP (default OFF = production byte-identical).

The cost
--------
One 13,824-token prefill chunk spends 881.6 ms of rank-0 GPU kernel time in the mHC family
(docs/../prefill3000/step0/ANATOMY.md 2.3: `mhc_post_tilelang` 389.8 + `mhc_pre_big_fuse_with_norm` 267.6 +
`sm120_tf32_hc_prenorm_gemm` 195.8 + `qw_post_mean` 28.4), 12.1 % of the step. Every mHC op is per-token
(the mixing matrices, the sinkhorn, the 4-stream residual bookkeeping, the prenorm GEMM's reduction is along
the hidden dim), so at TP=2 two ranks compute all of it twice, once per rank, on the same 13,824 tokens.

The fix (GLM53_MHC_SP=1; unset/empty/0 = production byte-identical)
-------------------------------------------------------------------
Kindling's per-forward switch (experimental/fixes/model.py, compose/sp.yaml), not the image's static
`use_sequence_parallel_moe` (that is the DSv4 EP-style path: expert parallel, an all2all backend, replicated
dense MLP, fixed at construction - PLAN.md critique 8.x). For a forward of >= 1024 tokens (MHC_SP_MIN_TOKENS;
decode and every CUDA-graph capture stay plain TP) the residual stream is SHARDED across the TP group and each
rank runs the mHC family on its T/2 rows:

  embed all-reduce (unchanged) -> sp_shard -> per layer:
      hc_fused_post_pre (T/2)  -> sp_all_gather  -> attention (full T, o_proj.reduce_results=False)
                               -> sp_reduce_scatter (reduces the rank partial AND shards)
      hc_fused_post_pre (T/2)  -> sp_all_gather  -> MLP/MoE (full T; the MoE runner's late all-reduce and the
                               dense down_proj reduce are off) -> sp_reduce_scatter
  final layer: hc_post + hc_contract (T/2) -> sp_all_gather -> final norm / lm_head (unchanged)
  aux (DFlash2 drafter) layers: sp_all_gather + clone of the contracted value (the drafter reads full T)

At 2 ranks reduce-scatter + all-gather move exactly the bytes of the all-reduce they replace
((N-1)/N * S each = S together vs 2(N-1)/N * S = S for the all-reduce), so the collectives are wire-neutral
(91 x 113.2 MB -> 182 x 56.6 MB per chunk); the gain is the mHC compute that stops being replicated: measured
901 -> 470 ms per chunk on nodeC with the production kernels (tests/mhc_sp/bench_mhc_halving.py).

Numerics: the sharded mHC ops are BITWISE identical to the full-T run at 13,824 / 6,912 / 4,289 tokens
(same rows compared; `compute_num_split` is 1 on both sides at >= 3,072 tokens on the 48-SM GB10, so even the
tf32 prenorm GEMM's summation order is unchanged). Below ~3,072 tokens `compute_num_split` picks split_k > 1
and the fp32 partial-sum order moves (measured: <= 4e-5 abs on the fp32 mixes, 1 bf16 ulp on rare layer_input
elements at 1,536 tokens) - the TP path there is unchanged, the SP path carries the ordinary fp32-order
difference of a split-K GEMM. Reduce-scatter vs all-reduce is exact at 2 ranks (bf16 addition is commutative;
both reduce the same two rank partials).

What this overlay edits (three files, all preflighted before any is written)
---------------------------------------------------------------------------
1. ``vllm/models/glm5next/nvidia/model.py`` - the module-level switch (`MHC_SP_ACTIVE`, `_mhc_sp_now`,
   `_mhc_sp_begin`, `_mhc_sp_set_reductions`) and the SP branches in `Glm5NextDecoderLayer.forward`
   (gather/reduce-scatter around attention and MLP) and `Glm5NextModel.forward` (shard the residual stream,
   toggle the layer reductions at the decode<->prefill transitions, gather the aux and final states).
   The MoE's own reduce is `moe_config.skip_final_all_reduce` (moe_runner.py `_maybe_reduce_final_output`,
   read at every call: the runner hands back the rank partial and the layer's reduce-scatter reduces it).
2. ``glm53_prefill_quickwins.py`` - VERIFIED gains the fingerprints of the two SP-patched forwards
   (additive; the anchors of mhc_aux/mhc_mean are untouched, so the quickwins transplant still applies).
3. ``glm53_moeglue.py`` - WARM_VERIFIED["Glm5NextDecoderLayer.forward"] gains the SP-patched fingerprints
   (plain and quickwins-transplanted; without them the decode L2 warm would silently disarm on the drift
   check). The warm itself is decode-only (<= 64 tokens) and SP never activates there.

If either fingerprint file is missing or its anchor has drifted, the patch prints a loud line and CONTINUES
(the model.py edit is what carries the feature; the victim is a WARNING + a disarmed quickwins item / warm,
never a wrong output).

Installation states per file: pristine -> patched; patched (marker + exact patched text) -> no-op; anything
else -> SystemExit. Atomic replace, pyc cleared. ``patch_tf_bundle.py`` runs this only when GLM53_MHC_SP is
non-empty, so unset = stock; ``0`` reaches this file and returns without reading or writing anything; any
other value raises SystemExit. The switch must reach BOTH ranks (make_start_sh.py forwards it like every
GLM53_ knob).

Env overrides for tests: GLM53_GLM5NEXT_MODEL_PY, GLM53_QUICKWINS_PY, GLM53_MOEGLUE_PY.
"""
from __future__ import annotations

import ast
import hashlib
import os
import stat
import sys
import textwrap
from pathlib import Path

SITEPKG = Path(os.environ.get("GLM53_SITEPKG", "/usr/local/lib/python3.12/dist-packages"))
MODEL_PY = Path(
    os.environ.get("GLM53_GLM5NEXT_MODEL_PY", SITEPKG / "vllm/models/glm5next/nvidia/model.py")
)
QUICKWINS_PY = Path(os.environ.get("GLM53_QUICKWINS_PY", SITEPKG / "glm53_prefill_quickwins.py"))
MOEGLUE_PY = Path(os.environ.get("GLM53_MOEGLUE_PY", SITEPKG / "glm53_moeglue.py"))
ENV = "GLM53_MHC_SP"
TAG = "[glm53-mhc-sp]"
MARK = "# [glm53-mhc-sp] sequence-parallel mHC prefill"


# ---------------------------------------------------------------------------------------------------------------------
# 1. model.py: the per-forward switch and the SP branches
# ---------------------------------------------------------------------------------------------------------------------
HEADER_ANCHOR = """
from vllm.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_reduce_scatter,
    sp_shard,
)
""".lstrip("\n")

HEADER_PATCHED = """
from vllm.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_reduce_scatter,
    sp_shard,
)

@MARK@: sequence-parallel prefill for the mHC bookkeeping (docs/MHC_SP.md; overlay/patch_mhc_sp.py,
# GLM53_MHC_SP=1). For a PREFILL-sized forward (>= MHC_SP_MIN_TOKENS tokens, TP > 1, PP off, no CUDA-graph
# capture, every layer mHC) the residual stream is sharded across the TP group: each rank runs the per-token
# mHC family (hc_pre / hc_fused_post_pre / hc_post, the tf32 prenorm GEMM, qw_post_mean) on its T/2 rows,
# and the attention/MLP all-reduces become reduce-scatter + all-gather pairs (the same wire bytes at TP=2).
# Decode keeps its captured graphs and the plain TP path: the switch is re-evaluated on every forward and
# every capture-sized batch fails the token gate, so no graph ever records an SP branch.
MHC_SP_ACTIVE = False
MHC_SP_MIN_TOKENS = 1024


def _mhc_sp_now(full_num_tokens: int) -> bool:
    # Per-forward SP gate (see the block comment above). Both ranks evaluate the same predicate on the same
    # arguments, so the ranks cannot disagree about which path a step takes.
    if full_num_tokens < MHC_SP_MIN_TOKENS or torch.cuda.is_current_stream_capturing():
        return False
    from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size

    tp = get_tensor_model_parallel_world_size()
    return tp > 1 and full_num_tokens % tp == 0 and get_pp_group().world_size == 1


def _mhc_sp_set_reductions(model, sp: bool) -> None:
    # Under SP the decoder-layer reduce-scatters attention and MLP outputs themselves, so their own final
    # all-reduces must be off: every o_proj / dense down_proj (RowParallelLinear reads reduce_results at
    # each call) and the MoE runner's late all-reduce (moe_config.skip_final_all_reduce, likewise read at
    # each call). Non-mHC / MTP layers are not sharded; SP never activates while any layer lacks mHC.
    for layer in model._active_layers:
        if getattr(layer, "is_mtp_layer", False) or getattr(layer, "is_sequence_parallel", False):
            continue
        if not getattr(layer, "mhc", False):
            continue
        layer.self_attn.o_proj.reduce_results = not sp
        if getattr(layer, "_mlp_is_moe", False):
            for mod in layer.mlp.modules():
                cfg = getattr(mod, "moe_config", None)
                if cfg is not None and hasattr(cfg, "skip_final_all_reduce"):
                    cfg.skip_final_all_reduce = sp
        else:
            layer.mlp.down_proj.reduce_results = not sp


def _mhc_sp_ok(model) -> bool:
    # SP needs every active layer on the mHC path (a non-mHC layer would keep the residual stream whole and
    # desynchronize the shard state threading).
    return all(
        getattr(layer, "mhc", False) and not getattr(layer, "is_mtp_layer", False)
        for layer in model._active_layers
    )


def _mhc_sp_begin(model, full_num_tokens: int) -> None:
    # Flip the model between the SP (prefill) and TP (decode) reductions when a step crosses the switch
    # (rare: once per prefill<->decode transition; decode replays never run this - the captured graphs carry
    # the TP path). The flip decision is per-model (a second Glm5NextModel instance owns its reductions);
    # MHC_SP_ACTIVE is only the indicator the decoder layers read.
    global MHC_SP_ACTIVE
    sp = _mhc_sp_now(full_num_tokens) and not model.is_sequence_parallel and _mhc_sp_ok(model)
    if sp != getattr(model, "_mhc_sp_flag", False):
        _mhc_sp_set_reductions(model, sp)
        model._mhc_sp_flag = sp
        MHC_SP_ACTIVE = sp
        logger.info("glm53-mhc-sp: sequence-parallel mHC prefill %s (T=%d)", "ACTIVE" if sp else "off",
                    full_num_tokens)
""".lstrip("\n")

ATTN_ANCHOR = """
        if self.is_sequence_parallel:
            x = sp_all_gather(x)[: positions.shape[0]]

        x = self.self_attn(
            hidden_states=x,
            positions=positions,
        )

        if self.is_sequence_parallel:
            x = sp_reduce_scatter(x)
""".lstrip("\n")
ATTN_PATCHED = """
        if self.is_sequence_parallel or MHC_SP_ACTIVE:
            @MARK@: the mHC above ran on
            # this rank's token shard; attention needs the full sequence (the code below is unchanged: it
            # also reads the full positions / metadata). Static is_sequence_parallel (EP) shares the branch.
            x = sp_all_gather(x)[: positions.shape[0]]

        x = self.self_attn(
            hidden_states=x,
            positions=positions,
        )

        if self.is_sequence_parallel or MHC_SP_ACTIVE:
            x = sp_reduce_scatter(x)
""".lstrip("\n")

MLP_ANCHOR = """
        # Fully Connected
        if self._mlp_is_moe:
            x = self.mlp(x, already_sequence_parallel=self.is_sequence_parallel)
        else:
            x = self.mlp(x)
""".lstrip("\n")
MLP_PATCHED = """
        if MHC_SP_ACTIVE and not self.is_sequence_parallel:
            @MARK@: the mHC above ran on the
            # shard; the MLP/MoE needs the full sequence, and it runs with its own final all-reduce off
            # (_mhc_sp_set_reductions), so the reduce-scatter after it both reduces the rank partial and
            # leaves the next mHC block on this rank's token shard. Static is_sequence_parallel (EP) shards
            # inside the MoE instead and needs neither branch.
            x = sp_all_gather(x)[: positions.shape[0]]

""" + MLP_ANCHOR + """
        if MHC_SP_ACTIVE and not self.is_sequence_parallel:
            x = sp_reduce_scatter(x)
""".lstrip("\n")

SHARD_ANCHOR = """
        full_num_tokens = positions.shape[0]
        if self.is_sequence_parallel:
            hidden_states = sp_shard(hidden_states)
""".lstrip("\n")
SHARD_PATCHED = """
        full_num_tokens = positions.shape[0]
        if self.is_sequence_parallel:
            hidden_states = sp_shard(hidden_states)
        else:
            _mhc_sp_begin(self, full_num_tokens)
            if MHC_SP_ACTIVE:
                @MARK@: this rank keeps its
                # token shard of the residual stream for the whole stack (see the block comment above).
                hidden_states = sp_shard(hidden_states)
""".lstrip("\n")

AUX_ANCHOR = """
            if self.is_sequence_parallel:
                value = sp_all_gather(value)[:full_num_tokens]
            aux_hidden_states.append(value)
""".lstrip("\n")
AUX_PATCHED = """
            if self.is_sequence_parallel:
                value = sp_all_gather(value)[:full_num_tokens]
            elif MHC_SP_ACTIVE:
                @MARK@: the drafter reads the
                # full sequence; clone so the returned tensors never alias a buffer a later gather reuses.
                value = sp_all_gather(value)[:full_num_tokens].clone()
            aux_hidden_states.append(value)
""".lstrip("\n")

FINAL_ANCHOR = """
        if self.is_sequence_parallel:
            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
""".lstrip("\n")
FINAL_PATCHED = """
        if self.is_sequence_parallel or MHC_SP_ACTIVE:
            @MARK@: the contracted final mHC state
            # comes back together for the norm / lm_head.
            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
""".lstrip("\n")

HUNKS = (HEADER_ANCHOR, HEADER_PATCHED, ATTN_ANCHOR, ATTN_PATCHED, MLP_ANCHOR, MLP_PATCHED,
         SHARD_ANCHOR, SHARD_PATCHED, AUX_ANCHOR, AUX_PATCHED, FINAL_ANCHOR, FINAL_PATCHED)


def _mark(hunks: tuple[str, ...]) -> tuple[str, ...]:
    """The @MARK@ placeholders become the marker comment (a hunk is never applied twice: `prepare` looks for
    the marker text and the patched text must occur exactly once)."""
    return tuple(h.replace("@MARK@", MARK) for h in hunks)


MODEL_HUNKS = _mark(HUNKS)


# ---------------------------------------------------------------------------------------------------------------------
# fingerprint plumbing: the SP-patched forwards' fingerprints, for the quickwins and moeglue drift guards
# ---------------------------------------------------------------------------------------------------------------------
def _func_source(src: str, cls: str, fn: str) -> str:
    """The function's verbatim file text (what inspect.getsource returns), for source_fingerprint."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name == fn:
                    return "\n".join(src.split("\n")[sub.lineno - 1: sub.end_lineno])
    raise ValueError(f"{cls}.{fn} not found")


def fingerprint(func_src: str) -> str:
    """integrate.source_fingerprint: sha256(ast.dump(ast.parse(dedent(src))))[:16]."""
    return hashlib.sha256(ast.dump(ast.parse(textwrap.dedent(func_src))).encode()).hexdigest()[:16]


# glm53_prefill_quickwins.py MHC_FINAL/MHC_AUX edits (verbatim): what the quickwins transplant turns the
# SP-patched forwards into (the fingerprints glm53_moeglue must then accept).
QW_MHC_FINAL_OLD = """        if self.layer_idx == self.num_hidden_layers - 1:
            x = self.hc_post(x, residual, post, comb)
            x = hc_contract(x, self.n)
            return x, None, None, None
"""
QW_MHC_FINAL_NEW = """        if self.layer_idx == self.num_hidden_layers - 1:
            x = _glm53_qw_final_post(self, x, residual, post, comb, hc_contract)
            return x, None, None, None
"""


def sp_fingerprints(patched_model_src: str) -> dict:
    layer = _func_source(patched_model_src, "Glm5NextDecoderLayer", "forward")
    model_fwd = _func_source(patched_model_src, "Glm5NextModel", "forward")
    layer_qw = layer.replace(QW_MHC_FINAL_OLD, QW_MHC_FINAL_NEW)
    if layer_qw == layer:
        raise ValueError("quickwins MHC_FINAL anchor did not apply to the SP-patched layer forward")
    return {
        "model_forward_sp": fingerprint(model_fwd),
        "layer_forward_sp": fingerprint(layer),
        "layer_forward_sp_qw": fingerprint(layer_qw),
    }


# 2/3. the drift-guard tables (additive, marker-commented)
QW_FP_ANCHORS = (
    ('    ("mhc_aux", "Glm5NextModel.forward"): frozenset({"224750fda049837c"}),', "model_forward_sp"),
    ('    ("mhc_mean", "Glm5NextModel.forward"): frozenset({"224750fda049837c"}),', "model_forward_sp"),
    ('    ("mhc_mean", "Glm5NextDecoderLayer.forward"): frozenset({"9c0fe21938cdc177"}),', "layer_forward_sp"),
)
MOEGLUE_FP_ANCHOR = ('    "Glm5NextDecoderLayer.forward": frozenset({"9c0fe21938cdc177", "f543fb24079672d5"}),',
                     ("layer_forward_sp", "layer_forward_sp_qw"))


def prepare_table(src: str, anchors, fps: dict, label: str) -> tuple[str, list[str]]:
    """Add the SP fingerprints to a VERIFIED/WARM_VERIFIED table (the caller gates on the file-wide marker).
    A drifted anchor only prints a warning: the guard then disarms at run time with its own WARNING; nothing
    mis-executes."""
    out, notes = src, []
    for anchor, keys in anchors:
        keys = (keys,) if isinstance(keys, str) else keys
        if anchor not in out:
            notes.append(f"{label}: anchor drifted, fingerprints not added: {anchor.strip()[:64]!r}")
            continue
        add = ", ".join(f'"{fps[k]}"' for k in keys)
        # the anchor ends with '"}),': reopen the frozenset after the existing fingerprints
        out = out.replace(anchor, anchor[:-4] + '", ' + add + "}),  " + MARK, 1)
    return out, notes


# ---------------------------------------------------------------------------------------------------------------------
# the common install machinery (patch_kda_strided_qkv.py's contract)
# ---------------------------------------------------------------------------------------------------------------------
def prepare(src: str, hunks: tuple[str, ...]) -> str:
    """Idempotent, fail-closed: pristine -> patched; already-patched -> no-op."""
    pairs = list(zip(hunks[::2], hunks[1::2]))
    if MARK in src and "[glm53-sp-fp8ag]" in src:
        # opt-kdamhc GLM53_SP_FP8AG (patch_sp_fp8ag.py) edited this tree after us: a second bundle pass is a no-op when
        # it verifies as fully applied on a fully applied SP(/SP2) tree (fail-closed otherwise).
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import patch_sp_fp8ag
        if not patch_sp_fp8ag.verify(src):
            raise ValueError("partial patch: the sp-fp8ag marker is present but its tree is not fully applied")
        return src
    if MARK in src and "# [glm53-mhc-sp2]" in src:
        # GLM53_MHC_SP2 (patch_mhc_sp2.py) extended this tree after us (it rewrites some of our hunks): a second
        # bundle pass is a no-op when the SP2 tree verifies as fully applied (fail-closed otherwise).
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import patch_mhc_sp2
        if patch_mhc_sp2.prepare(src) != src:
            raise ValueError("partial patch: the sp2 marker is present but its tree is not fully applied")
        return src
    if MARK in src:
        for _, patched in pairs:
            if src.count(patched) != 1:
                raise ValueError("partial patch: marker present but a hunk is not fully applied")
        return src
    patched = src
    for anchor, new in pairs:
        if patched.count(anchor) != 1:
            raise ValueError(f"anchor count {patched.count(anchor)} != 1: {anchor.strip()[:70]!r}")
        patched = patched.replace(anchor, new, 1)
    for _, new in pairs:
        if patched.count(new) != 1:
            raise ValueError("post-patch verification failed (a patched hunk is missing/ambiguous)")
    if prepare(patched, hunks) != patched:
        raise ValueError("post-patch verification failed (not recognised as already-applied)")
    return patched


def replace_file(target: Path, source: str) -> None:
    tmp = target.with_name(f".{target.name}.glm53-mhc-sp.tmp")
    try:
        tmp.write_text(source)
        os.chmod(tmp, stat.S_IMODE(target.stat().st_mode))
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def clear_pyc(target: Path) -> None:
    cache = target.parent / "__pycache__"
    if cache.is_dir():
        for pyc in cache.glob(target.stem + "*.pyc"):
            pyc.unlink(missing_ok=True)


def main() -> int:
    val = os.environ.get(ENV, "")
    if val.strip() == "0":
        print(f"{TAG} {ENV}=0: stock, files untouched")
        return 0
    if val.strip() != "1":
        raise SystemExit(f"{TAG} {ENV} must be exactly 1 to install (unset/empty/0 = stock), got {val!r}")
    model_path = MODEL_PY
    if not model_path.is_file():
        raise SystemExit(f"{TAG} missing {model_path}")
    src = model_path.read_text()
    try:
        patched = prepare(src, MODEL_HUNKS)
        compile(patched, str(model_path), "exec")
    except ValueError as exc:
        raise SystemExit(f"{TAG} preflight failed for vllm/models/glm5next/nvidia/model.py: {exc}") from exc
    fps = sp_fingerprints(patched)

    notes, table_state = [], {}
    for path, anchors, label in ((QUICKWINS_PY, QW_FP_ANCHORS, "glm53_prefill_quickwins.py"),
                                 (MOEGLUE_PY, (MOEGLUE_FP_ANCHOR,), "glm53_moeglue.py")):
        if not path.is_file():
            notes.append(f"{label}: not present, fingerprints not added")
            table_state[label] = "absent"
            continue
        tsrc = path.read_text()
        if MARK in tsrc:
            table_state[label] = "already present"
            continue
        tnew, tnotes = prepare_table(tsrc, anchors, fps, label)
        notes.extend(tnotes)
        if tnew != tsrc:
            compile(tnew, str(path), "exec")
            replace_file(path, tnew)
            clear_pyc(path)
        table_state[label] = "updated" if tnew != tsrc else "unchanged"
    model_act = "already present" if patched == src else "patched"
    if patched != src:
        replace_file(model_path, patched)
        clear_pyc(model_path)

    print(
        f"{TAG} vllm/models/glm5next/nvidia/model.py: {model_act}; fingerprints "
        f"quickwins={table_state.get('glm53_prefill_quickwins.py')} moeglue={table_state.get('glm53_moeglue.py')}; "
        f"mhc_forward_sp={fps['layer_forward_sp'][:6]} model_forward_sp={fps['model_forward_sp'][:6]}; "
        f"sequence-parallel mHC prefill installed (GLM53_MHC_SP=1: >= 1024-token forwards shard the residual "
        f"stream; decode byte-identical)"
    )
    for n in notes:
        print(f"{TAG} WARNING {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
