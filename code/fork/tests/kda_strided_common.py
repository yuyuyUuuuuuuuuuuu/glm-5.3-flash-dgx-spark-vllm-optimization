"""Shared rig for the KDA strided-decode-inputs backport (overlay/patch_kda_strided_qkv.py, vLLM #55736).

Loads TWO instances of the KDA recurrent op from the running image's OWN site-packages text:
  stock = the pristine files (production's path: wrapper copies q/k/v/beta with .contiguous())
  patched = the same files with the shipped overlay's `prepare()` applied (wrapper reads them strided)
Each lands in its own importable package (`fla_stock`, `fla_patched`) so one process can A/B them; the
rest of vLLM (triton, math_utils, ...) is the image's own.

Inputs are built exactly like the production decode call (vllm/models/glm5next/nvidia/kda.py:546):
q/k/v are column slices of the merged short-conv output (token stride 3*proj), beta a column slice of
the fused qkvbfg_a projection, g contiguous [1, T, H, K], fp32 recurrent states paged per token
(spec-decode layout: ssm_state_indices [num_seqs, query_len] + num_accepted_tokens).
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

SITE_VLLM = Path("/usr/local/lib/python3.12/dist-packages/vllm")
OPS_REL = "third_party/flash_linear_attention/ops"
# test-time file overrides (same names as the overlay's own), so the rig can be pointed at a copy
ENV_FREC = "GLM53_FLA_FUSED_RECURRENT_PY"
ENV_KDA = "GLM53_FLA_KDA_PY"


def _load_package(tag: str, texts: dict[str, str], tmp: Path, tolerate_dup_ops: bool = False):
    """Copy the image's flash_linear_attention.ops tree into a fresh top-level package.

    `texts` overrides file contents (the patched files). Sibling modules and every vllm.* import
    resolve as usual; only this package's own files come from the copy. `tolerate_dup_ops` lets a
    SECOND instance load next to the stock one: vLLM's CustomOp registry refuses a repeated
    `@CustomOp.register(name)` (kda.py registers `fused_rms_norm_gated`), which the rig does not need
    — it exercises fused_recurrent_kda, not the RMSNormGated op — so the duplicate registration is
    skipped and the stock instance's op stays the registered one. Restored on exit.
    """
    src = SITE_VLLM / OPS_REL
    pkg = tmp / tag
    (pkg / "ops").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "ops" / "__init__.py").write_text("")
    for f in sorted(src.glob("*.py")):
        (pkg / "ops" / f.name).write_text(texts.get(f.name, f.read_text()))
    unpatch = None
    if tolerate_dup_ops:
        from vllm.model_executor import custom_op as _co

        orig_register = _co.CustomOp.register.__func__

        def tolerant(cls, name, *a, **kw):
            dec = orig_register(cls, name, *a, **kw)

            def safe(op_cls):
                # the Duplicate op assert lives inside the decorator, not in register()
                try:
                    return dec(op_cls)
                except AssertionError as exc:
                    if "Duplicate op name" not in str(exc):
                        raise
                    return op_cls

            return safe

        _co.CustomOp.register = classmethod(tolerant)

        def unpatch():
            _co.CustomOp.register = classmethod(orig_register)

    sys.path.insert(0, str(tmp))
    try:
        importlib.import_module(f"{tag}.ops")
        mod = importlib.import_module(f"{tag}.ops.kda")  # the tolerant window must stay open here
    finally:
        sys.path.pop(0)
        if unpatch is not None:
            unpatch()
    return mod


def load_stock_and_patched():
    """(stock module, patched module) — the patched text is produced by the shipped overlay's own
    `prepare()`, so the test exercises exactly the bytes the overlay would write to site-packages,
    and its preflight must accept the image's file as pristine."""
    overlay_path = Path(__file__).resolve().parent.parent / "overlay" / "patch_kda_strided_qkv.py"
    spec = importlib.util.spec_from_file_location("patch_kda_strided_qkv", overlay_path)
    pk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pk)
    texts = {}
    for fname, hunks, env in (
        ("fused_recurrent.py", pk.FREC_HUNKS, ENV_FREC),
        ("kda.py", pk.KDA_HUNKS, ENV_KDA),
    ):
        real = Path(os.environ.get(env, SITE_VLLM / OPS_REL / fname))
        text = real.read_text()
        prepared = pk.prepare(source=text, hunks=hunks)
        assert prepared != text, f"{fname}: the overlay did not change the image's text (drift?)"
        assert pk.prepare(source=prepared, hunks=hunks) == prepared, f"{fname}: not idempotent"
        compile(prepared, fname, "exec")
        texts[fname] = (text, prepared)
    tmp = Path(tempfile.mkdtemp(prefix="kdaqkv-"))
    # both loads tolerate an already-registered duplicate op (if vLLM itself imported
    # flash_linear_attention.ops.kda earlier in the process, the stock shadow hits it too)
    stock = _load_package(
        "fla_stock", {k: v[0] for k, v in texts.items()}, tmp, tolerate_dup_ops=True
    )
    patched = _load_package(
        "fla_patched", {k: v[1] for k, v in texts.items()}, tmp, tolerate_dup_ops=True
    )
    return stock, patched, pk, tmp


# ---------------------------------------------------------------------------------------------------------------------
# production-shaped decode inputs
# ---------------------------------------------------------------------------------------------------------------------
import torch  # noqa: E402


def make_decode_inputs(
    num_seqs: int,
    query_len: int,
    num_heads: int,
    head_dim: int,
    device="cuda",
    seed: int = 0,
):
    """Token-strided q/k/v/beta as glm5next/nvidia/kda.py hands them to fused_recurrent_kda."""
    g_cpu = torch.Generator(device="cpu").manual_seed(seed)
    H, K, T = num_heads, head_dim, num_seqs * query_len
    proj = H * K  # per-part width of the merged q|k|v conv output (total 3*proj)
    # the conv update's merged output: [T, q|k|v]; the splits are the strided views production passes
    conv_out = (
        torch.randn(T, 3 * proj, generator=g_cpu, dtype=torch.float32)
        .to(dtype=torch.bfloat16)
        .to(device)
    )
    q, k, v = (p.view(1, T, H, K) for p in conv_out.split(proj, dim=-1))
    # fused qkvbfg_a projection: [T, qkv | beta | f_a | g_a]; beta is its first extra column block
    projected = (
        torch.randn(T, 3 * proj + H + 2 * K, generator=g_cpu, dtype=torch.float32)
        .to(dtype=torch.bfloat16)
        .to(device)
    )
    beta = projected[:, 3 * proj : 3 * proj + H].unsqueeze(0)  # [1, T, H]
    assert T == 1 or (q.stride(1) == 3 * proj and beta.stride(1) == projected.stride(0)), (
        q.stride(),
        beta.stride(),
    )
    g = (
        torch.randn(1, T, H, K, generator=g_cpu, dtype=torch.float32)
        .to(dtype=torch.bfloat16)
        .to(device)
    )
    a_log = (0.5 * torch.randn(H, generator=g_cpu)).to(dtype=torch.float32, device=device)
    g_bias = (0.1 * torch.randn(H * K, generator=g_cpu)).to(dtype=torch.float32, device=device)
    # slot 0 is NULL_BLOCK_ID; every token owns a random distinct slot (spec-decode layout)
    slots = torch.randperm(T, generator=g_cpu).to(device).to(torch.int32) + 1
    inp = dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        a_log=a_log,
        g_bias=g_bias,
        cu_seqlens=torch.arange(0, T + 1, query_len, dtype=torch.int32, device=device),
        initial_state=(
            torch.randn(T + 1, H, K, K, generator=g_cpu, dtype=torch.float32).to(device)
        ),
        ssm_state_indices=slots if query_len == 1 else slots.view(num_seqs, query_len),
    )
    if query_len > 1:
        inp["num_accepted_tokens"] = (
            torch.randint(1, query_len + 1, (num_seqs,), generator=g_cpu, dtype=torch.int32)
            .to(device)
        )
    return inp


def run_kda(mod, inp: dict, out=None):
    return mod.fused_recurrent_kda(
        q=inp["q"],
        k=inp["k"],
        v=inp["v"],
        g=inp["g"],
        beta=inp["beta"],
        initial_state=inp["initial_state"],
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=inp["cu_seqlens"],
        ssm_state_indices=inp["ssm_state_indices"],
        num_accepted_tokens=inp.get("num_accepted_tokens"),
        out=out,
        sigmoid_beta=True,
        a_log=inp["a_log"],
        g_bias=inp["g_bias"],
        compute_gate=True,
        lower_bound=-5.0,
    )
