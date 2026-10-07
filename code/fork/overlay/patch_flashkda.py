#!/usr/bin/env python3
"""Route GLM-5.3-Flash's KDA chunked prefill through FlashKDA 17a037d (fp32 recurrent state).

Upstream: FlashKDA @ ``17a037d98da546deb4591e967cf961a43c034d8b`` ("Keep the
recurrent state in fp32 between tiles", vLLM #58846), built for sm_121a in the
production image (pfkda's kill-test build; the kernel sources are untouched and
the registration shim's namespace is renamed -- see
tests/fkda/build_flashkda_fp32.sh). The image ships its own pre-fix bf16-state
``vllm/_flashkda_C`` (14-arg-era ABI, bf16 state): that one must never be used,
so this feature's extension is ``_flashkda_fp32_C`` and cannot collide with it.
Measured against the production Triton chain (pfkda Test A, capped rerun):
3.38-3.82x faster per layer at T=13,824/4,608/1,791 (~0.58 s per 13,824-token
chunk over 34 KDA layers), final-state error vs fp64 1.31x Triton's in the
long-memory gate regime; output/final-state agreement with the Triton chain
5.7e-3/4.8e-3 rel-RMS (bf16-level), bit-identical to the pfkda build (renamed
included, tests/fkda/check_rename.sh).

What this overlay does (``vllm/models/glm5next/nvidia/kda.py``, preflighted in
full before anything is written):
1. installs ``glm53_flashkda.py`` (the wrapper: ABI check, workspace-manager
   buffers, one-line boot marker) and ``_flashkda_fp32_C.abi3.so`` into
   site-packages -- only when ``GLM53_KDA_FLASHKDA=1``;
2. three anchors in ``kda.py``: import the wrapper, resolve/configure in
   ``Glm5NextLinearAttention.__init__``, and dispatch the chunked-prefill call.
   The Triton ``chunk_kda_with_fused_gate`` call is kept verbatim as the else
   branch. Inputs/outputs contract: same q/k/v slices, RAW g1 and RAW beta
   (FlashKDA applies the bounded gate and the beta sigmoid in-kernel, the
   Triton chain is given the pre-sigmoided fp32 beta), same initial state and
   ``cu_seqlens``, same ``[1, T, H, D]`` output and fp32 ``[N, H, D, D]`` final
   state that ``scatter_states`` copies for decode. Conv1d, o_norm/o_proj,
   spec-verify decode and plain decode are untouched.

The Triton chunk kernel writes its output in place into ``v`` (pfkda finding);
FlashKDA writes to its own workspace output, so ``v`` is no longer mutated.
Nothing downstream reads ``v`` after the call.

Installation states: pristine -> patched; patched (marker + exact text) ->
no-op; anything else -> SystemExit. Atomic replace, pyc cleared. Unset/empty
``GLM53_KDA_FLASHKDA`` never reaches this file (``patch_tf_bundle.py`` skips
it), so default = stock; ``0`` reaches it and returns without touching
anything; any other value raises SystemExit (the r16m start.sh validates
empty/0/1 and forwards the knob to BOTH ranks).

The FlashKDA BUILD is picked by the operator switch ``GLM53_KDA_FLASHKDA_V``
(env.r16 ``#switch``, r16z; only read here when ``GLM53_KDA_FLASHKDA=1``):
unset/empty/1 installs the SHIPPED r16x build exactly as production runs it
(the two overlay files at their production names, the kda.py call without
``out=`` -- the composed tree of a switch-unset rank is production's byte
for byte); ``2`` installs the fkda2 precision build (overlay/
glm53_flashkda2.py + overlay/_flashkda_fp32_C2.abi3.so, same kda.py text);
``3`` installs the fkda3 build (overlay/glm53_flashkda3.py +
overlay/_flashkda_fp32_C3.abi3.so) and patches the kda.py call with the
fkda3 direct-output contract (``out=core_attn_out[:, :num_actual_tokens]``:
FlashKDA writes the layer output directly, docs/KDA_FLASHKDA3.md). All three
stage under the SAME site-packages names glm53_flashkda.py /
_flashkda_fp32_C.abi3.so (each wrapper pins its own extension sha at boot),
so switching between builds on an installed tree is refused as a byte
mismatch (fail closed): set GLM53_KDA_FLASHKDA=0, restart once (the Triton
chain serves), then set =1 with the new GLM53_KDA_FLASHKDA_V and restart
again. Any other value raises SystemExit (the r16z start.sh validates
unset/1/2/3 and forwards the knob to BOTH ranks).

Env overrides for tests: GLM53_KDA_PY (the model file), GLM53_TF_OVERLAY
(where the wrapper + .so are staged).
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

SITEPKG = Path(os.environ.get("GLM53_SITEPKG", "/usr/local/lib/python3.12/dist-packages"))
OVERLAY = Path(os.environ.get("GLM53_TF_OVERLAY", os.environ.get("GLM53_OPT", "/opt/glm53") + "/tf/overlay"))
KDA_PY = Path(
    os.environ.get("GLM53_KDA_PY", SITEPKG / "vllm/models/glm5next/nvidia/kda.py")
)
ENV = "GLM53_KDA_FLASHKDA"
TAG = "[glm53-kda-flashkda]"
MARK = "# [glm53-kda-flashkda]"
VERSION_ENV = "GLM53_KDA_FLASHKDA_V"
# the three builds: overlay file -> (site-packages name, wrapper, extension). V unset/1 stages production's
# r16x bytes (byte for byte); 2 = the fkda2 precision build; 3 = the fkda3 build. All three install under the
# SAME site-packages names, so the wrapper and the kda.py call text are version-selected together (the fkda3
# chunk_prefill takes out=, the fkda/fkda2 ones do not).
WRAPPER_NAME = "glm53_flashkda.py"
EXT_NAME = "_flashkda_fp32_C.abi3.so"
WRAPPERS = {"fkda": ("glm53_flashkda.py", "_flashkda_fp32_C.abi3.so"),
            "fkda2": ("glm53_flashkda2.py", "_flashkda_fp32_C2.abi3.so"),
            "fkda3": ("glm53_flashkda3.py", "_flashkda_fp32_C3.abi3.so")}


def kda_version() -> str:
    """The FlashKDA build this pass installs (GLM53_KDA_FLASHKDA_V): unset/empty/1 = the shipped r16x build
    (the c286213f extension, production bytes); 2 = the fkda2 precision build; 3 = the fkda3 build (the
    direct-output kda.py). Anything else -> SystemExit."""
    v = os.environ.get(VERSION_ENV, "").strip()
    if v in ("", "1"):
        return "fkda"
    if v in ("2", "3"):
        return "fkda" + v
    raise SystemExit(f"{TAG} {VERSION_ENV} must be unset, 1, 2 or 3 (unset/1 = the shipped r16x build), got {v!r}")

# --------------------------------------------------------------------- anchor 1
IMPORT_ANCHOR = """from vllm.third_party.flash_linear_attention.ops.kda import (
    FusedRMSNormGated,
    chunk_kda_with_fused_gate,
    fused_recurrent_kda,
)
"""
IMPORT_PATCHED = IMPORT_ANCHOR + f"""
{MARK} FlashKDA 17a037d chunked prefill (GLM53_KDA_FLASHKDA=1): the wrapper is
# installed next to this file by overlay/patch_flashkda.py; without the patch
# this import does not exist and the tree is stock.
import glm53_flashkda
"""

# --------------------------------------------------------------------- anchor 2
INIT_ANCHOR = """        # Process-global conv-state layout, resolved once here instead of on
        # every _forward call (it reads an env-derived flag each time).
        self._conv_state_dim_first = is_conv_state_dim_first()
"""
INIT_PATCHED = INIT_ANCHOR + f"""
        {MARK} FlashKDA chunked prefill (GLM53_KDA_FLASHKDA=1): resolve the
        # fp32-state extension's ABI once and size its workspace buffers for
        # this rank's max_num_batched_tokens / max_num_seqs. Unset/0 = the
        # Triton chain below, byte for byte upstream.
        if glm53_flashkda.ENV_ENABLED:
            glm53_flashkda.configure(self, vllm_config)
"""

# --------------------------------------------------------------------- anchor 3
CALL_ANCHOR = """            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = chunk_kda_with_fused_gate(
                q=_rearr(q_ns),
                k=_rearr(k_ns),
                v=_rearr(v_ns),
                raw_g=g1_ns,
                # Chunk path wants the pre-sigmoided fp32 beta (its kernels
                # don't sigmoid); beta_ns is raw bf16 from forward.
                beta=_cast_sigmoid(beta_ns.squeeze(0)).unsqueeze(0),
                A_log=self.A_log,
                g_bias=self.dt_bias,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=non_spec_query_start_loc,
                safe_gate=safe_gate,
                lower_bound=lower_bound,
            )
"""
CALL_PATCHED = f"""            if glm53_flashkda.enabled_for(self):
                # FlashKDA 17a037d (fp32 recurrent state): q/k l2norm and
                # the bounded gate happen in-kernel, so g1 stays RAW and beta
                # stays the RAW bf16 logits (FlashKDA sigmoids in-kernel, like
                # the decode path's fused_recurrent_kda(..., sigmoid_beta=True)).
                # Same [1, T, H, D] output (its own buffer: FlashKDA does NOT
                # alias v the way the Triton chunk kernel does) and the same
                # fp32 [N, H, D, D] final state for scatter_states/decode.
                core_attn_out_non_spec, last_recurrent_state = (
                    glm53_flashkda.chunk_prefill(
                        layer=self,
                        q=_rearr(q_ns),
                        k=_rearr(k_ns),
                        v=_rearr(v_ns),
                        g=g1_ns,
                        beta=beta_ns,
                        initial_state=initial_state,
                        cu_seqlens=non_spec_query_start_loc,
                    )
                )
            else:
                (
                    core_attn_out_non_spec,
                    last_recurrent_state,
                ) = chunk_kda_with_fused_gate(
                    q=_rearr(q_ns),
                    k=_rearr(k_ns),
                    v=_rearr(v_ns),
                    raw_g=g1_ns,
                    # Chunk path wants the pre-sigmoided fp32 beta (its kernels
                    # don't sigmoid); beta_ns is raw bf16 from forward.
                    beta=_cast_sigmoid(beta_ns.squeeze(0)).unsqueeze(0),
                    A_log=self.A_log,
                    g_bias=self.dt_bias,
                    initial_state=initial_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                    cu_seqlens=non_spec_query_start_loc,
                    safe_gate=safe_gate,
                    lower_bound=lower_bound,
                )
"""

HUNKS = (
    ("import", IMPORT_ANCHOR, IMPORT_PATCHED),
    ("init", INIT_ANCHOR, INIT_PATCHED),
    ("prefill-call", CALL_ANCHOR, CALL_PATCHED),
)
PYC = "kda*.pyc"

# --------------------------------------------------------------------- anchor 3, fkda3 (GLM53_KDA_FLASHKDA_V=3)
# The direct-output contract: without spec tokens the non-spec rows ARE the layer's first num_actual_tokens rows, so
# FlashKDA writes straight into core_attn_out and kda.py's merge copy becomes a same-storage no-op (fkda3's
# glm53_flashkda3.chunk_prefill accepts out=; the fkda2 wrapper does not, which is why the two CALL texts are
# version-selected -- a mixed install would be a TypeError at the first chunked prefill).
CALL_PATCHED3 = CALL_PATCHED.replace(
    """                # fp32 [N, H, D, D] final state for scatter_states/decode.
""",
    """                # fp32 [N, H, D, D] final state for scatter_states/decode.
                # fkda3: without spec tokens the non-spec rows ARE the
                # layer's first num_actual_tokens rows, so FlashKDA writes
                # straight into core_attn_out and the merge copy below
                # becomes a same-storage no-op (torch copy_ exits early).
""", 1).replace(
    """                        cu_seqlens=non_spec_query_start_loc,
                    )
""",
    """                        cu_seqlens=non_spec_query_start_loc,
                        out=(None if use_spec else core_attn_out[:, :num_actual_tokens]),
                    )
""", 1)
assert CALL_PATCHED3 != CALL_PATCHED, "CALL_PATCHED3 construction failed"
HUNKS3 = (
    ("import", IMPORT_ANCHOR, IMPORT_PATCHED),
    ("init", INIT_ANCHOR, INIT_PATCHED),
    ("prefill-call", CALL_ANCHOR, CALL_PATCHED3),
)


def hunks(ver: str) -> tuple:
    """The anchors of the SELECTED FlashKDA version (only fkda3's kda.py call text differs: it passes out=)."""
    return HUNKS3 if ver == "fkda3" else HUNKS

# --------------------------------------------------------------- the quickwins allowlist
# Production runs GLM53_PREFILL_QUICKWINS=all, whose kda_conv item transplants
# Glm5NextLinearAttention._forward only if the function's AST fingerprint is in
# glm53_prefill_quickwins.VERIFIED (the fingerprint of the image's stock
# _forward). This overlay edits the same function on disk BEFORE the plugin
# loads, so with GLM53_KDA_FLASHKDA=1 the fingerprint is different and kda_conv
# would be refused (-114 ms/chunk, boot_checks MISS). The fix: extend the
# installed VERIFIED["kda_conv"] set with the FlashKDA-patched fingerprint. The
# KDA_CONV edit composes with this patch (its conv block does not overlap the
# three anchors and it keeps the chunk call verbatim); the KDA_CONV edit is
# applied and compiled HERE as proof before the extended fingerprint is
# installed. Unset/0 never reaches this file, so OFF leaves the module stock.
QW_NAME = "glm53_prefill_quickwins.py"
QW_KEY = '("kda_conv", "Glm5NextLinearAttention._forward"):'
QW_PYC = "glm53_prefill_quickwins*.pyc"


def _forward_fingerprint(src: str, cls: str = "Glm5NextLinearAttention", fn: str = "_forward") -> str:
    """The quickwins recipe (glm53_prefill_quickwins.source_fingerprint):
    sha256(ast.dump(ast.parse(dedent(inspect.getsource(fn)))))[:16] -- decorators included."""
    import ast as _ast
    import hashlib as _hashlib
    import textwrap as _textwrap

    lines = src.splitlines(keepends=True)
    for node in _ast.walk(_ast.parse(src)):
        if isinstance(node, _ast.ClassDef) and node.name == cls:
            for m in node.body:
                if isinstance(m, _ast.FunctionDef) and m.name == fn:
                    start = min([d.lineno for d in m.decorator_list] + [m.lineno])
                    seg = "".join(lines[start - 1:m.end_lineno])
                    return _hashlib.sha256(_ast.dump(_ast.parse(_textwrap.dedent(seg))).encode()).hexdigest()[:16]
    raise ValueError(f"{cls}.{fn} not found")


def extend_quickwins_allowlist(stock_fp: str, patched_fp: str, qw_src_override: str = "") -> str:
    """Add the FlashKDA-patched _forward fingerprint to the installed module's
    VERIFIED kda_conv entry. Returns one of "already present" / "extended"."""
    qw_py = Path(qw_src_override) if qw_src_override else SITEPKG / QW_NAME
    if not qw_py.is_file():
        raise SystemExit(f"{TAG} missing {qw_py} (the bundle's site/ glm53_prefill_quickwins.py)")
    src = qw_py.read_text()
    lines = src.splitlines(keepends=True)
    hits = [i for i, ln in enumerate(lines) if QW_KEY in ln and "frozenset({" in ln]
    if len(hits) != 1:
        raise SystemExit(f"{TAG} {qw_py}: {QW_KEY} found {len(hits)}x (expected 1)")
    line = lines[hits[0]]
    if patched_fp in line:
        return "already present"
    if stock_fp not in line:
        raise SystemExit(
            f"{TAG} {qw_py}: the kda_conv VERIFIED set does not contain the stock _forward fingerprint "
            f"{stock_fp} -- the module and the image's kda.py are not the pair this patch was made for; refusing"
        )
    head, _, tail = line.partition("frozenset({")
    inner, _, rest = tail.partition("}")            # rest = "),\n" for the single-line VERIFIED entry
    if not rest.startswith(")"):
        raise SystemExit(f"{TAG} {qw_py}: unexpected kda_conv VERIFIED line format: {line!r}")
    fps = [x.strip().strip("\"'") for x in inner.split(",") if x.strip()]
    fps.append(patched_fp)
    lines[hits[0]] = (
        head + "frozenset({" + ", ".join(f'"{f}"' for f in fps) + "})"
        + rest[1:].rstrip("\n")                     # the dict entry's trailing comma
        + f"  # {MARK} the FlashKDA-patched _forward: the KDA_CONV edit composes with it\n"
    )
    joined = "".join(lines)
    compile(joined, str(qw_py), "exec")
    replace_file(qw_py, joined)
    return "extended"


def prepare(source: str, hks: tuple = HUNKS) -> str:
    if MARK in source:
        for _name, _old, new in hks:
            if source.count(new) != 1:
                raise ValueError("marker present but the patched text is partial or different")
        return source
    patched = source
    for name, old, new in hks:
        n = patched.count(old)
        if n != 1:
            raise ValueError(f"anchor {name} count={n} (expected 1)")
        patched = patched.replace(old, new, 1)
    return patched


def replace_file(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".flashkda.tmp")
    with open(tmp, "w") as fh:
        fh.write(content)
        fh.flush()
        os.fsync(fh.fileno())
    if path.exists():
        shutil.copymode(path, tmp)
    os.replace(tmp, path)
    cache = path.parent / "__pycache__"
    if cache.is_dir():
        for pyc in cache.glob(PYC):
            pyc.unlink(missing_ok=True)


def main() -> int:
    val = os.environ.get(ENV, "")
    if val.strip() == "0":
        print(f"{TAG} {ENV}=0: stock, files untouched")
        return 0
    if val.strip() != "1":
        raise SystemExit(f"{TAG} {ENV} must be exactly 1 to install (unset/empty/0 = stock), got {val!r}")

    ver = kda_version()
    hks = hunks(ver)
    wrapper_name, ext_name = WRAPPERS[ver]
    wrapper_src = OVERLAY / wrapper_name
    ext_src = OVERLAY / ext_name
    if not wrapper_src.is_file() or not ext_src.is_file():
        raise SystemExit(f"{TAG} missing {wrapper_src} or {ext_src} (the bundle's overlay/)")
    import hashlib

    for f in (wrapper_src, ext_src):
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        print(f"{TAG} staging {f.name} sha256 {h[:16]}")
    if ver == "fkda3":
        print(f"{TAG} {VERSION_ENV}=3: the fkda3 build + the direct-output kda.py "
              f"(installed under the same names as the shipped build)")

    source = KDA_PY.read_text()
    try:
        patched = prepare(source, hks)
    except ValueError as exc:
        raise SystemExit(f"{TAG} preflight failed for kda.py: {exc}") from exc
    compile(patched, str(KDA_PY), "exec")

    # the KDA_CONV composition proof: the quickwins edit the plugin will apply
    # to THIS _forward must still anchor exactly once and compile (with the
    # kda_conv old text present exactly once, the transplant cannot fail at
    # runtime either; the plugin re-verifies the fingerprint itself)
    qw_src_text = (SITEPKG / QW_NAME).read_text() if (SITEPKG / QW_NAME).is_file() else ""
    if "KDA_CONV = (" in qw_src_text:
        old_conv = qw_src_text[qw_src_text.index("KDA_CONV = ("):]
        # KDA_CONV = (old, new): take the two triple-quoted chunks
        import re as _re
        chunks = _re.findall(r'"""(.*?)"""', old_conv[:4000], _re.S)
        if len(chunks) >= 2:
            if patched.count(chunks[0]) != 1:
                raise SystemExit(f"{TAG} preflight: the quickwins KDA_CONV anchor does not anchor once "
                                 f"({patched.count(chunks[0])}) on the FlashKDA-patched kda.py")
            compile(patched.replace(chunks[0], chunks[1], 1), str(KDA_PY), "exec")

    # install the wrapper + extension (idempotent, content-addressed refuse)
    for src, dst in ((wrapper_src, SITEPKG / WRAPPER_NAME), (ext_src, SITEPKG / EXT_NAME)):
        if dst.is_file():
            same = hashlib.sha256(dst.read_bytes()).hexdigest() == hashlib.sha256(src.read_bytes()).hexdigest()
            if not same:
                raise SystemExit(f"{TAG} {dst.name} exists with different bytes; refusing")
        else:
            shutil.copy2(src, dst)
            os.chmod(dst, 0o644)
    if patched != source:
        replace_file(KDA_PY, patched)

    # the quickwins allowlist: keep kda_conv installable on the patched _forward
    # (the bundle's site/ always installs glm53_prefill_quickwins.py; a tree
    # without it has no kda_conv to break, so there the extension is a no-op)
    qw_override = os.environ.get("GLM53_QUICKWINS_PY", "")
    if qw_override or (SITEPKG / QW_NAME).is_file():
        try:
            # Second pass on the same filesystem: kda.py already carries the marker, so `source` IS the patched
            # text and _forward_fingerprint(source) would be the PATCHED fingerprint -- extend_quickwins_allowlist
            # would then look for it as the "stock" fingerprint in the kda_conv VERIFIED set and refuse
            # (r16n review: a second bundle pass fails). Un-patch back through the same three anchors to recover
            # the image's stock _forward before computing it.
            stock_src = source
            if MARK in source:
                for _name, old, new in hks:
                    n = stock_src.count(new)
                    if n != 1:
                        raise ValueError(f"cannot un-patch anchor {_name}: {new[:48]!r} found {n}x (expected 1)")
                    stock_src = stock_src.replace(new, old, 1)
                if MARK in stock_src:
                    raise ValueError("the un-patched kda.py still carries the marker")
            stock_fp = _forward_fingerprint(stock_src)
            patched_fp = _forward_fingerprint(patched)
            qw_act = extend_quickwins_allowlist(stock_fp, patched_fp, qw_override)
            cache = (SITEPKG / "__pycache__")
            if cache.is_dir():
                for pyc in cache.glob(QW_PYC):
                    pyc.unlink(missing_ok=True)
        except ValueError as exc:
            raise SystemExit(f"{TAG} quickwins allowlist: {exc}") from exc
        qw_line = (f"{TAG} glm53_prefill_quickwins.py: kda_conv VERIFIED + {patched_fp} ({qw_act}; "
                   f"the FlashKDA-patched _forward)")
    else:
        qw_line = f"{TAG} glm53_prefill_quickwins.py: not installed here (no kda_conv to keep compatible)"
    print(
        f"{TAG} kda.py: {'already present' if patched == source else 'patched'}; "
        f"glm53_flashkda.py + {EXT_NAME} installed; the KDA chunked prefill runs FlashKDA 17a037d "
        f"(fp32 recurrent state) -- decode and every other path untouched"
        + (f" ({ver} build, {VERSION_ENV}=3)" if ver == "fkda3" else "")
    )
    print(qw_line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
