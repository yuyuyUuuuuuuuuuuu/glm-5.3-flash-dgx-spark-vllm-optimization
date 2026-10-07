"""GLM53_DEC_SMALLOPS kernels: thin Python wrappers over kernels/smallops.cu (docs/DEC_SMALLOPS.md).

No side effects on import. The production wiring (env switch, hooks, self-tests, fallbacks) is in
glm53_smallops_install.py. Extension ``glm53_smallops_ext``: AOT via setup.py; JIT build only when GLM53_SMALLOPS_JIT
or TF_EXL3_JIT is set (nodeC tests).

    dconv(x, delta, base, gs, block_size)      DFlash2 _grouped_conv, bit-identical to the PyTorch eager op chain
    mhc_fused(comb, post, res_in, x, w, S)     mhc_fused_tilelang (M <= 16), bit-identical; w fp32 or its exact
                                               bf16 copy; returns (yp [S, M, 24], rp [S, M], residual_out)
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import torch

_TRUE = frozenset({"1", "on", "true", "yes"})
_EXT: Any = None
_EXT_LOCK = threading.Lock()
EXT_SOURCE = None
MHC_M_MAX = 16


def load_ext():
    global _EXT, EXT_SOURCE
    if _EXT is not None:
        return _EXT
    with _EXT_LOCK:
        if _EXT is not None:
            return _EXT
        try:
            import glm53_smallops_ext as m  # AOT
            _EXT, EXT_SOURCE = m, f"aot:{getattr(m, '__file__', '?')}"
            return _EXT
        except ImportError as e:
            err = e
        jit = any((os.environ.get(k, "0") or "0").strip().lower() in _TRUE for k in ("GLM53_SMALLOPS_JIT", "TF_EXL3_JIT"))
        if not jit:
            raise ImportError(f"glm53_smallops_ext (AOT) not importable and GLM53_SMALLOPS_JIT is off: {err!r}")
        from torch.utils.cpp_extension import load

        from tf_exl3_moe import _cuda_include_shim

        import hashlib
        here = Path(__file__).resolve().parent / "kernels"
        inc = _cuda_include_shim()
        tag = hashlib.sha256(b"".join((here / f).read_bytes() for f in ("smallops.cu", "smallops_kernels.cuh",
                                                                         "smallops_tc.cuh")))
        m = load(name=f"glm53_smallops_ext_jit_{tag.hexdigest()[:10]}", sources=[str(here / "smallops.cu")],
                 extra_cuda_cflags=["-O3", *inc], extra_cflags=["-O3", *inc], verbose=False)
        _EXT, EXT_SOURCE = m, f"jit:{getattr(m, '__file__', '?')}"
        return _EXT


def dconv(x: torch.Tensor, delta: torch.Tensor, base: torch.Tensor, gs: int, block_size: int,
          out: torch.Tensor | None = None) -> torch.Tensor:
    if out is None:
        out = torch.empty((x.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
    load_ext().dconv(x, delta, base, out, int(gs), int(block_size))
    return out


def mhc_splits(M: int) -> int:
    """Production's split count on the small-M path (mhc_fused_post_pre_tilelang, hidden_size <= 4096)."""
    return 8 if M < 8 else 4


def mhc_fused(comb: torch.Tensor, post: torch.Tensor, res_in: torch.Tensor, x: torch.Tensor, w: torch.Tensor,
              S: int | None = None, yp=None, rp=None, res_out=None):
    M = x.shape[0]
    S = mhc_splits(M) if S is None else S
    dev = x.device
    if yp is None:
        yp = torch.empty((S, M, 24), dtype=torch.float32, device=dev)
    if rp is None:
        rp = torch.empty((S, M), dtype=torch.float32, device=dev)
    if res_out is None:
        res_out = torch.empty_like(res_in)
    load_ext().mhc_fused(comb, post, res_in, x, w, yp, rp, res_out)
    return yp, rp, res_out


TC_CFG_DEFAULT = 406        # warps * 100 + stages


def tc_gemm(w: torch.Tensor, x: torch.Tensor, y: torch.Tensor, cfg: int = TC_CFG_DEFAULT) -> torch.Tensor:
    """y[b, m, n] = bf16(sum_k w[b, n, k] x[b, m, k]) (3-d views; cuBLAS-wmma bitwise order, M <= 16)."""
    load_ext().tc_gemm(w, x, y, int(cfg))
    return y


_SINKS: dict = {}


def l2_prefetch(t: torch.Tensor, ctas: int = 16) -> None:
    """Pull t's bytes into L2 (evict_last) on the current stream (loads whose values are discarded)."""
    sink = _SINKS.get(t.device)
    if sink is None:
        sink = _SINKS[t.device] = torch.zeros(1, dtype=torch.int32, device=t.device)
    load_ext().l2_prefetch(t, sink, int(ctas))
