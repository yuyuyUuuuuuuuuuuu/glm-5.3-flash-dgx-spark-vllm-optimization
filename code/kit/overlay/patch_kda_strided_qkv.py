#!/usr/bin/env python3
"""Backport the KDA half of vLLM PR #55736 (merged 2026-09-10, Apache-2.0) onto vLLM 487ecf187.

The waste
---------
Production's GLM-5.3-Flash KDA decode calls ``fused_recurrent_kda`` (vLLM
``487ecf187``, ``vllm/third_party/flash_linear_attention/ops/kda.py``) once per
KDA layer per step -- 34 KDA layers of 45, every rank. The layer hands the
kernel column slices of two merged buffers (``qkv_spec.split`` of the merged
short-conv output, ``beta`` a slice of the fused ``qkvbfg_a`` projection), and
the wrapper copies each of q/k/v/beta with ``.contiguous()`` first: four small
D2D copy kernels per layer per step whose only job is to undo a stride the
kernel could read in place. At the production verify shape (batch 1,
8 tokens per request) each copy moves <= 128 KiB, so this is pure launch /
latency floor, in every captured graph, forever.

The fix (upstream vLLM PR #55736, the ``fused_recurrent`` + ``kernels`` halves;
its MLA halves do NOT apply here -- docs/KDA_STRIDED_QKV.md section 3)
--------------------------------------------------------------------------
The kernel takes the token stride of q/k/v/beta as runtime arguments and walks
tokens with it, so column slices of a wider per-token buffer are consumed in
place. ``token_stride`` documents and enforces the layouts the addressing can
express (per-token [H, D] or [H] block contiguous, non-overlapping tokens, one
sequence per program when B > 1) and fails loudly on anything else, instead of
silently reading the wrong tokens. ``o`` becomes an explicit dense allocation:
with ``k`` strided, ``torch.empty_like(k)`` would inherit a layout the kernel's
dense ``o`` indexing does not describe (only reachable when the caller passes
no ``out``, i.e. mixed spec/non-spec steps).

The other launcher of the same kernel in ``fused_recurrent.py``
(``fused_recurrent_gated_delta_rule_fwd``, the non-KDA wrapper) passes the
strides of its (contiguous) inputs, so its addressing is bit-identical to
before; only the KDA wrapper stops copying.

Numerics: pure layout change -- the kernel loads the same values and computes
the same fp32 expression per token, so outputs and final states are bitwise
equal to production's path (asserted at rtol=0/atol=0 in
tests/test_kda_strided_qkv.py, plus a full CUDA-graph capture/replay).

Why the PR's other halves are not ported here
---------------------------------------------
* duplicate router GEMM (#55736's third part): production already removed it at
  runtime -- ``glm53_gemv_install.py`` (GLM53_BF16_GEMV_DEDUP_ROUTER=1, default)
  clears the MoE runner's ``gate`` so Glm5NextMoE.forward's pre-computed logits
  are the only router GEMM. Nothing to port.
* MLA absorbed-query token-major bmm + NoPE concat skip (#55736's second part):
  production serves MLA through ``flashinfer_mla_sparse_sm90``, whose
  ``forward_mqa`` *requires* the ``(q_nope, q_pe)`` tuple and passes the two
  halves to the kernel separately -- the ``torch.cat(q)`` the PR skips does not
  exist on this path (and the cat that could happen in mla_attention.py only
  fires with DCP > 1). Not applicable.

What this overlay edits (two files, both preflighted before any is written)
---------------------------------------------------------------------------
1. ``third_party/flash_linear_attention/ops/fused_recurrent.py`` --
   ``token_stride`` helper, the four stride kernel arguments, strided pointer
   bases/advance in the kernel body, and the stride arguments at the
   non-KDA wrapper's launch.
2. ``third_party/flash_linear_attention/ops/kda.py`` -- import ``token_stride``,
   pass the strides in ``fused_recurrent_kda_fwd``'s launch, drop the four
   ``.contiguous()`` copies in ``fused_recurrent_kda`` (g keeps its copy: the
   kernel reads the gate through a dense ``[H, K]`` bias table and its own
   layout, which production guarantees upstream of the call), dense ``o``.

Installation states per file: pristine -> patched; patched (marker + exact
patched text) -> no-op ("already present"); anything else (partial marker,
drifted anchor) -> SystemExit. Atomic replace, pyc cleared. Unset/empty
``GLM53_KDA_STRIDED_QKV`` never reaches this file (``patch_tf_bundle.py``
skips it), so default = stock; ``0`` reaches it and returns without reading or
writing anything (the r16k start.sh validates the value as empty/0/1); any
other value raises SystemExit.

Env overrides for tests: GLM53_FLA_FUSED_RECURRENT_PY, GLM53_FLA_KDA_PY.
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

SITEPKG = Path(os.environ.get("GLM53_SITEPKG", "/usr/local/lib/python3.12/dist-packages"))
FREC_PY = Path(
    os.environ.get(
        "GLM53_FLA_FUSED_RECURRENT_PY",
        SITEPKG / "vllm/third_party/flash_linear_attention/ops/fused_recurrent.py",
    )
)
KDA_PY = Path(
    os.environ.get(
        "GLM53_FLA_KDA_PY", SITEPKG / "vllm/third_party/flash_linear_attention/ops/kda.py"
    )
)
ENV = "GLM53_KDA_STRIDED_QKV"
TAG = "[glm53-kda-strided-qkv]"
MARK = "# [glm53-kda-strided-qkv] vLLM #55736"


# ---------------------------------------------------------------------------------------------------------------------
# 1. ops/fused_recurrent.py: the kernel reads q/k/v/beta with a token stride
# ---------------------------------------------------------------------------------------------------------------------
FREC_IMPORT_ANCHOR = "from .op import exp, log\n"
FREC_IMPORT_PATCHED = (
    "from .op import exp, log\n"
    "\n"
    "\n"
    + MARK
    + ": the token stride (elements) of a\n"
    "# ``[B, T, H, D]`` or ``[B, T, H]`` tensor.\n"
    "def token_stride(x):\n"
    '    """Token stride (elements) of a ``[B, T, H, D]`` or ``[B, T, H]`` tensor.\n'
    "\n"
    "    The recurrent kernel walks tokens with this stride and addresses heads\n"
    "    densely inside a token, so each token's ``[H, D]`` (or ``[H]``) block must\n"
    "    be contiguous, tokens must not overlap, and with ``B > 1`` sequence ``n``\n"
    "    must start at token ``n * T`` (dense batch). Column slices of a wider\n"
    "    per-token projection buffer satisfy this and are consumed in place.\n"
    '    """\n'
    "    st = x.stride()\n"
    "    assert x.dim() in (3, 4) and st[-1] == 1, (x.shape, st)\n"
    "    assert x.dim() == 3 or st[2] == x.shape[3], (x.shape, st)\n"
    "    assert st[1] >= x.shape[2] * (x.shape[3] if x.dim() == 4 else 1), (x.shape, st)\n"
    "    assert x.shape[0] == 1 or st[0] == x.shape[1] * st[1], (x.shape, st)\n"
    "    return st[1]\n"
)

FREC_SIG_ANCHOR = """    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
"""
FREC_SIG_PATCHED = (
    "    stride_indices_seq: tl.constexpr,\n"
    "    stride_indices_tok: tl.constexpr,\n"
    "    "
    + MARK
    + ": token strides of\n"
    "    # q/k/v/beta (elements), see `token_stride` -- tokens are walked with\n"
    "    # them instead of assuming the dense layout the .contiguous() copies\n"
    "    # used to guarantee.\n"
    "    stride_q_t,\n"
    "    stride_k_t,\n"
    "    stride_v_t,\n"
    "    stride_beta_t,\n"
    "    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state\n"
)

FREC_PTR_ANCHOR = """    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + (bos * HV + i_hv) * V + o_v
    else:
        p_beta = beta + bos * HV + i_hv
"""
FREC_PTR_PATCHED = """    p_q = q + bos * stride_q_t + i_h * K + o_k
    p_k = k + bos * stride_k_t + i_h * K + o_k
    p_v = v + bos * stride_v_t + i_hv * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + bos * stride_beta_t + i_hv * V + o_v
    else:
        p_beta = beta + bos * stride_beta_t + i_hv
"""

FREC_STEP_ANCHOR = """        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        if not IS_KDA:
            p_g += HV
        else:
            p_gk += HV * K
        p_beta += HV * (V if IS_BETA_HEADWISE else 1)
"""
FREC_STEP_PATCHED = """        p_q += stride_q_t
        p_k += stride_k_t
        p_o += HV * V
        p_v += stride_v_t
        if not IS_KDA:
            p_g += HV
        else:
            p_gk += HV * K
        p_beta += stride_beta_t
"""

FREC_NONKDA_LAUNCH_ANCHOR = """        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        INPLACE_FINAL_STATE=inplace_final_state,
        IS_KDA=False,
"""
FREC_NONKDA_LAUNCH_PATCHED = (
    "        stride_indices_seq=stride_indices_seq,\n"
    "        stride_indices_tok=stride_indices_tok,\n"
    + MARK
    + ": this wrapper's\n"
    "        # inputs arrive contiguous, so these reproduce the previous dense\n"
    "        # addressing exactly.\n"
    "        stride_q_t=token_stride(q),\n"
    "        stride_k_t=token_stride(k),\n"
    "        stride_v_t=token_stride(v),\n"
    "        stride_beta_t=token_stride(beta),\n"
    "        IS_BETA_HEADWISE=beta.ndim == v.ndim,\n"
    "        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,\n"
    "        INPLACE_FINAL_STATE=inplace_final_state,\n"
    "        IS_KDA=False,\n"
)


# ---------------------------------------------------------------------------------------------------------------------
# 2. ops/kda.py: no .contiguous() on the KDA decode inputs, strides to the kernel, dense o
# ---------------------------------------------------------------------------------------------------------------------
KDA_IMPORT_ANCHOR = (
    "from .fused_recurrent import fused_recurrent_gated_delta_rule_fwd_kernel\n"
)
KDA_IMPORT_PATCHED = (
    "from .fused_recurrent import (\n"
    "    fused_recurrent_gated_delta_rule_fwd_kernel,\n"
    "    token_stride,  "
    + MARK
    + ": token strides (see token_stride),\n"
    ")\n"
)

KDA_O_ANCHOR = """    if out is None:
        o = torch.empty_like(k)
"""
KDA_O_PATCHED = (
    "    if out is None:\n"
    + MARK
    + ": k may now be a strided\n"
    "        # view, so o is always the dense layout the kernel's o indexing\n"
    "        # assumes.\n"
    "        o = torch.empty(k.shape, dtype=k.dtype, device=k.device)\n"
)

KDA_LAUNCH_ANCHOR = """        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
"""
KDA_LAUNCH_PATCHED = (
    "        stride_indices_seq=stride_indices_seq,\n"
    "        stride_indices_tok=stride_indices_tok,\n"
    + MARK
    + ": q/k/v/beta are\n"
    "        # consumed in place with an explicit token stride, so column slices\n"
    "        # of the fused projection / conv buffers need no copy; layouts the\n"
    "        # kernel cannot address fail loudly in `token_stride`.\n"
    "        stride_q_t=token_stride(q),\n"
    "        stride_k_t=token_stride(k),\n"
    "        stride_v_t=token_stride(v),\n"
    "        stride_beta_t=token_stride(beta),\n"
    "        IS_BETA_HEADWISE=beta.ndim == v.ndim,\n"
)

KDA_CONTIG_ANCHOR = """    o, final_state = fused_recurrent_kda_fwd(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
"""
KDA_CONTIG_PATCHED = (
    "    "
    + MARK
    + ": q/k/v/beta are consumed in\n"
    "    # place with an explicit token stride (upstream vLLM #55736), so the\n"
    "    # decode path's column slices of the merged qkv conv output and of the\n"
    "    # fused qkvbfg_a projection no longer pay four .contiguous() copies per\n"
    "    # layer. g keeps its copy (the kernel reads it through a dense layout\n"
    "    # the caller does not guarantee). Layouts the kernel cannot address\n"
    "    # raise in `token_stride`.\n"
    "    o, final_state = fused_recurrent_kda_fwd(\n"
    "        q=q,\n"
    "        k=k,\n"
    "        v=v,\n"
    "        g=g.contiguous(),\n"
    "        beta=beta,\n"
)

FREC_HUNKS = (
    FREC_IMPORT_ANCHOR,
    FREC_IMPORT_PATCHED,
    FREC_SIG_ANCHOR,
    FREC_SIG_PATCHED,
    FREC_PTR_ANCHOR,
    FREC_PTR_PATCHED,
    FREC_STEP_ANCHOR,
    FREC_STEP_PATCHED,
    FREC_NONKDA_LAUNCH_ANCHOR,
    FREC_NONKDA_LAUNCH_PATCHED,
)
KDA_HUNKS = (
    KDA_IMPORT_ANCHOR,
    KDA_IMPORT_PATCHED,
    KDA_O_ANCHOR,
    KDA_O_PATCHED,
    KDA_LAUNCH_ANCHOR,
    KDA_LAUNCH_PATCHED,
    KDA_CONTIG_ANCHOR,
    KDA_CONTIG_PATCHED,
)


def prepare(source: str, hunks: tuple[str, ...]) -> str:
    """Idempotent, fail-closed: pristine -> patched; already-patched -> no-op."""
    pairs = list(zip(hunks[::2], hunks[1::2]))
    if MARK in source:
        for _, patched in pairs:
            if source.count(patched) != 1:
                raise ValueError("partial patch: marker present but a hunk is not fully applied")
        return source
    patched = source
    for anchor, new in pairs:
        if patched.count(anchor) != 1:
            raise ValueError(f"anchor count {patched.count(anchor)} != 1: {anchor.strip()[:70]!r}")
        patched = patched.replace(anchor, new, 1)
    # Post-patch verification: every patched hunk present exactly once and the
    # result recognised as already-applied (the idempotency contract).
    for _, new in pairs:
        if patched.count(new) != 1:
            raise ValueError("post-patch verification failed (a patched hunk is missing/ambiguous)")
    if prepare(patched, hunks) != patched:
        raise ValueError("post-patch verification failed (not recognised as already-applied)")
    return patched


TARGETS = (
    ("ops/fused_recurrent.py", lambda: FREC_PY, FREC_HUNKS, "fused_recurrent*.pyc"),
    ("ops/kda.py", lambda: KDA_PY, KDA_HUNKS, "kda*.pyc"),
)


def replace_file(target: Path, source: str) -> None:
    tmp = target.with_name(f".{target.name}.glm53-kda-strided-qkv.tmp")
    try:
        tmp.write_text(source)
        os.chmod(tmp, stat.S_IMODE(target.stat().st_mode))
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def clear_pyc(target: Path, pattern: str) -> None:
    cache = target.parent / "__pycache__"
    if cache.is_dir():
        for pyc in cache.glob(pattern):
            pyc.unlink(missing_ok=True)


def main() -> int:
    val = os.environ.get(ENV, "")
    if val.strip() == "0":
        # explicit off (the r16k start.sh accepts empty/0/1): stock, nothing read or written
        print(f"{TAG} {ENV}=0: stock, files untouched")
        return 0
    if val.strip() != "1":
        raise SystemExit(
            f"{TAG} {ENV} must be exactly 1 to install (unset/empty/0 = stock), got {val!r}"
        )
    plans = []
    for label, get_path, hunks, pyc in TARGETS:
        path = get_path()
        if not path.is_file():
            raise SystemExit(f"{TAG} missing {path}")
        source = path.read_text()
        try:
            patched = prepare(source, hunks)
        except ValueError as exc:
            raise SystemExit(f"{TAG} preflight failed for {label}: {exc}") from exc
        compile(patched, str(path), "exec")
        plans.append((label, path, source, patched, pyc))
    for label, path, source, patched, pyc in plans:
        if patched != source:
            replace_file(path, patched)
            clear_pyc(path, pyc)
    act = {label: ("already present" if p == s else "patched") for label, _, s, p, _ in plans}
    # one f-string (tools/deploy16/check_boot_strings.py must find boot_checks.sh's marker in the shipped code)
    print(
        f"{TAG} ops/fused_recurrent.py: {act['ops/fused_recurrent.py']}; ops/kda.py: {act['ops/kda.py']}; "
        f"the KDA recurrent decode reads q/k/v/beta token-strided (vLLM #55736)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
