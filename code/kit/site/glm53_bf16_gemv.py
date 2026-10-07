"""Small-M dense GEMMs of GLM-5.3 decode: kernels + launch plans (docs/BF16_GEMV.md).

Kernels: kernels/gemv_bf16.cu, extension ``glm53_gemv_ext`` (AOT via setup.py; JIT build only when
GLM53_GEMV_JIT or TF_EXL3_JIT is set, nodeC tests). This module has no side effects on import; the production
wiring (module wrappers, self-tests, env switch GLM53_BF16_GEMV) is in glm53_gemv_install.py.

    gemm(x, w, ctx, out_mode)  y = x @ w.T, x [M, K] bf16 (M <= 64), w [N, K] bf16; out_mode 0 bf16 / 1 fp32 /
                               2 fp32 of the bf16-rounded value (what F.linear(x, w).to(float32) returns)
    gemm_f32(x, w, ctx)        y = x.float() @ w.float().T in IEEE fp32 on CUDA cores, N <= 32
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
M_MAX = 64


def load_ext():
    global _EXT, EXT_SOURCE
    if _EXT is not None:
        return _EXT
    with _EXT_LOCK:
        if _EXT is not None:
            return _EXT
        try:
            import glm53_gemv_ext as m  # AOT
            _EXT, EXT_SOURCE = m, f"aot:{getattr(m, '__file__', '?')}"
            return _EXT
        except ImportError as e:
            err = e
        jit = any((os.environ.get(k, "0") or "0").strip().lower() in _TRUE for k in ("GLM53_GEMV_JIT", "TF_EXL3_JIT"))
        if not jit:
            raise ImportError(f"glm53_gemv_ext (AOT) not importable and GLM53_GEMV_JIT is off: {err!r}")
        from torch.utils.cpp_extension import load

        from tf_exl3_moe import _cuda_include_shim

        here = Path(__file__).resolve().parent / "kernels"
        inc = _cuda_include_shim()
        m = load(name="glm53_gemv_ext_jit", sources=[str(here / "gemv_bf16.cu")],
                 extra_cuda_cflags=["-O3", *inc], extra_cflags=["-O3", *inc], verbose=False)
        _EXT, EXT_SOURCE = m, f"jit:{getattr(m, '__file__', '?')}"
        return _EXT


# ---------------------------------------------------------------------------------------------------------
# launch plans: (S, wn, kch) per weight shape and M bucket. S = K splits over CTAs, wn = warps (8 weight rows each)
# per CTA, kch = k per shared-memory chunk. Measured on GB10 with cold weights (tests/bench_bf16_gemv.py --sweep,
# docs/BF16_GEMV.md); a shape not in the table gets the generic rule.

M_BUCKETS = (1, 8, 16, 32, 48, 64)


def m_bucket(M: int) -> int:
    for b in M_BUCKETS:
        if M <= b:
            return b
    raise ValueError(f"M={M} > {M_MAX}")


# (N, K) -> {M bucket: plan, or None = keep production's cuBLAS call}. From tests/bench_bf16_gemv.py --sweep
# (docs/logs/bf16_gemv/): the plan with the lowest median at the bucket's largest M, used only where it beat
# production's op there. Shapes not listed are never served by the new kernel.
# Bucket b holds the M values above the previous bucket up to b (1 | 2..8 | 9..16 | 17..32 | 33..48 | 49..64).
PLANS: dict[tuple[int, int], dict[int, tuple[int, int, int] | None]] = {
    (288, 4096): {1: (4, 2, 256), 8: (4, 2, 256), 16: (4, 2, 256), 32: (4, 2, 256), 48: (4, 4, 256), 64: None},
    (160, 4096): {1: None, 8: (8, 2, 256), 16: (8, 2, 256), 32: None, 48: None, 64: None},   # indexer wk_weights_proj
    (128, 4096): {1: None, 8: (8, 2, 256), 16: (8, 2, 256), 32: None, 48: None, 64: None},   # indexer kpool gate
    (1024, 4096): {1: (1, 2, 256), 8: (1, 2, 256), 16: (1, 2, 256), 32: None, 48: None, 64: None},  # DFlash2 conv
}
# fp32 head gate (gemm_f32): largest M served (production's gemmSN is 72-77 us at M <= 16, cuBLAS 12-16 us above;
# the kernel is 1.2-1.3x at M = 17..32 and within noise at 33..48)
F32_MAX_M = 32


def smem_bytes(M: int, K: int, S: int, kch: int) -> int:
    """Dynamic shared memory of one gemm_bf16 CTA (kernels/gemv_bf16.cu launch_t)."""
    rows = 8 if M <= 8 else 16 * ((M + 15) // 16)
    nch = (K // S) // kch
    chunk = rows * (kch + 32) * 2
    if nch > 1 and nch * chunk <= 64 * 1024:
        return nch * chunk
    return (2 if nch > 1 else 1) * chunk


def _valid(N: int, K: int, S: int, wn: int, kch: int, M: int = M_MAX) -> bool:
    return (S >= 1 and K % S == 0 and (K // S) % kch == 0 and N % 8 == 0
            and smem_bytes(M, K, S, kch) <= 96 * 1024)


def generic_plan(N: int, K: int, M: int) -> tuple[int, int, int]:
    """Enough CTAs to cover 48 SMs a few times over, K per CTA >= kch."""
    mb = m_bucket(M)
    kch = 512 if mb <= 16 else 256
    wn = 4
    groups = max(1, (N // 8 + wn - 1) // wn)
    S = 1
    while groups * S < 96 and K % (S * 2) == 0 and (K // (S * 2)) % kch == 0 and (K // (S * 2)) >= kch:
        S *= 2
    return S, wn, kch


def plan_for(N: int, K: int, M: int) -> tuple[int, int, int]:
    """The launch plan for any shape (tests, sweeps): the table's plan if there is one, else the generic rule."""
    p = PLANS.get((N, K), {}).get(m_bucket(M)) if 1 <= M <= M_MAX else None
    if p is not None and _valid(N, K, *p, M=M):
        return p
    return generic_plan(N, K, M)


def serve_plan(N: int, K: int, M: int) -> tuple[int, int, int] | None:
    """The plan production uses for this call, or None: keep production's own GEMM (shape not measured, M outside
    1..64, or the new kernel was not faster at this M bucket)."""
    if not 1 <= M <= M_MAX:
        return None
    p = PLANS.get((N, K), {}).get(m_bucket(M))
    return p if p is not None and _valid(N, K, *p, M=M) else None


def plans_digest() -> str:
    import hashlib
    return hashlib.sha256(repr((sorted((k, sorted(v.items())) for k, v in PLANS.items()), F32_MAX_M)).encode()).hexdigest()[:12]


class GemmCtx:
    """Per call-site scratch: split-K partials and the CTA ticket counters (zero between calls). One ctx per module
    that owns the weight; calls on one ctx must not run concurrently (they are stream-ordered in production)."""

    def __init__(self, N: int, K: int, device, f32: bool = False, max_splits: int | None = None,
                 served_only: bool = False) -> None:
        self.N, self.K, self.f32 = N, K, f32
        if max_splits is not None:   # any plan with S <= max_splits (tests / sweeps)
            ws, cnt = max_splits * M_MAX * N, max(1, N // 8)
        elif f32:
            s_max = K // 128
            ws = s_max * M_MAX * N
            cnt = 1
        else:   # every plan gemm(plan=None) may pick (served_only: only the ones serve_plan returns, production)
            s_max, cnt = 1, 1
            for mb in M_BUCKETS:
                p = serve_plan(N, K, mb) if served_only else plan_for(N, K, mb)
                if p is None:
                    continue
                S, wn, _ = p
                s_max = max(s_max, S)
                cnt = max(cnt, (N // 8 + wn - 1) // wn)
            ws = s_max * M_MAX * N if s_max > 1 else 1
        self.ws = torch.zeros(ws, dtype=torch.float32, device=device)
        self.counters = torch.zeros(cnt, dtype=torch.int32, device=device)


def gemm(x: torch.Tensor, w: torch.Tensor, ctx: GemmCtx, out_mode: int = 0, plan=None, use_pol: int = 1,
         out: torch.Tensor | None = None) -> torch.Tensor:
    M, N = x.shape[0], w.shape[0]
    S, wn, kch = plan or plan_for(N, w.shape[1], M)
    if out is None:
        out = torch.empty((M, N), dtype=torch.bfloat16 if out_mode == 0 else torch.float32, device=x.device)
    load_ext().gemm_bf16(x, w, out, ctx.ws, ctx.counters, S, wn, kch, out_mode, use_pol)
    return out


def gemm_f32(x: torch.Tensor, w: torch.Tensor, ctx: GemmCtx, out: torch.Tensor | None = None) -> torch.Tensor:
    if out is None:
        out = torch.empty((x.shape[0], w.shape[0]), dtype=torch.float32, device=x.device)
    load_ext().gemm_f32(x, w, out, ctx.ws, ctx.counters)
    return out
