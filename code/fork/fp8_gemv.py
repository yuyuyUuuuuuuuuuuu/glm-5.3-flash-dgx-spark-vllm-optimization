"""FP8 (e4m3, per-output-channel) small-M decode GEMM on production's Marlin-packed weights (kernels/fp8_gemv.cu).

Production (launcher overlay exl3.py ``Glm53DenseFp8Method``; the drafter overlays patch_drafter_fp8.py and
patch_drafter_lmhead_fp8.py use the same class) quantizes BF16 linears per output channel to e4m3 and repacks them for
Marlin (``prepare_fp8_layer_for_marlin``): ``layer.weight`` = int32 [K/16, 4*Npad], ``layer.weight_scale`` = bf16
[1, Npad] (permuted, x 2^120), ``layer.workspace``. This module reads those tensors in place (no second copy) and
computes the same linear with fp32 accumulation for M = 1..64 rows, where decode spends its time.

Enable: ``GLM53_FP8_GEMV`` in {1, on, true, yes} (read once, when vLLM loads general plugins: integrate.plugin_register).
Unset -> nothing is patched and the extension is not imported (the custom op ``tf_fp8::linear`` is still registered so
a compiled graph that references it keeps loading; its body then runs Marlin). Optional ``GLM53_FP8_GEMV_MAX_M``
(1..64, default 64) lowers the M limit below the tuned table (e.g. 8 = batch-1 decode only).

install() wraps two methods of production's ``Glm53DenseFp8Method`` (image or launcher-overlay exl3.py):
  - ``process_weights_after_loading``: production's method first (exceptions propagate unchanged), then a per-layer
    self-test (the new kernel vs production's Marlin on random inputs at every enabled M bucket); a failing layer
    stays on Marlin (WARNING). Layers built elsewhere (the drafter's FP8 lm_head copy) are self-tested at their first
    eager call instead; until then (or when first seen during CUDA-graph capture) they run Marlin.
  - ``apply``: eager (the target model: not torch.compiled) -> ``try_new()``, and production's own apply whenever it
    declines (so every fallback is exactly production's path, observability notes included); under torch.compile
    (the DFlash drafter) -> the custom op ``tf_fp8::linear`` whose body is ``linear()`` = ``try_new()`` or
    ``apply_fp8_marlin_linear``, so the M-dependent choice is made at run time, never baked into a traced graph.
    ``try_new()`` runs the new kernel when M <= the table's limit for this (Npad, K), the layer passed its self-test
    and every layout / dtype / alignment check holds; otherwise (and on any non-CUDA error before the launch) it
    declines. Layers production does not route to Marlin (``ready`` False, the KDA large-M BF16 copy present) go to
    production's own apply unchanged. Nothing raises into vLLM except CUDA errors.

Large-M (prefill) path, ``GLM53_FP8_LARGE_M`` in {1, on, true, yes} (independent of GLM53_FP8_GEMV; either one
installs the same two wrappers, each path runs only when its own variable is set; unset -> exactly production):
  For a call with M >= LARGE_TABLE[(Npad, K)] rows (the measured crossover against Marlin, docs/FP8_LARGE_M.md),
  outside CUDA-graph capture, on a layer that passed its large-M self-test: for each chunk of output columns the
  Marlin-packed e4m3 weight is dequantized EXACTLY to a transient BF16 [nc, K] (kernels/fp8_large_m.cu: the scale is
  not folded in), then y = bf16(fp32 sum * scale) (one rounding, as Marlin's bf16(sum * s)) by a TileLang kernel with
  the scale in its epilogue (GLM53_FP8_LARGE_M_GEMM=auto, default) or by torch.mm(out_dtype=float32) per row chunk +
  scale_cast (=cublas, and the fallback when TileLang is unusable; bitwise the same results in every test), + bias
  exactly as Marlin (bf16(y + b)). The only difference class against Marlin is the fp32 summation order. Transient
  memory is bounded by GLM53_FP8_LARGE_M_TEMP_MIB (default 128: the BF16 weight chunk, and for cublas the fp32 rows); the
  output is written directly at [M, N] (Marlin writes [M, Npad] and copies to [M, N] for a padded N). Persistent
  memory: the un-permuted fp32 scale, 4 * N bytes per served layer. Everything else (smaller M, other shapes, capture,
  a failed self-test, any non-CUDA error) is production's Marlin apply.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

import torch

HERE = Path(__file__).resolve().parent
ENV = "GLM53_FP8_GEMV"
ENV_MAX_M = "GLM53_FP8_GEMV_MAX_M"
PROD_MODULE = "vllm.model_executor.layers.quantization.exl3"
_TRUE = frozenset({"1", "on", "true", "yes"})
_CUDA_ERR = re.compile(r"CUDA error|cudaError|illegal|capture")
_log = logging.getLogger("vllm.tf_fp8_gemv")

# ---------------------------------------------------------------------------------------------------------------
# Tuned configuration table: (Npad, K) -> {MB bucket: (warps, kw, u, evict_first)}. MB bucket of M: 1 = M 1..8,
# 2 = 9..16, 3 = 17..24, 4 = 25..32, 6 = 33..48, 8 = 49..64. A missing bucket (or shape) = Marlin. Source: paired
# interleaved CUDA-graph benchmarks on nodeC (tests/bench_fp8_gemv.py --tune: docs/logs/fp8_gemv/tune_shapes_*.log,
# tune_w4_mb3-8.log). An entry exists only where the kernel beat Marlin by >= 2 % in tuning (geometric mean over the
# measured M of the bucket) and then stayed >= 1 % faster at every M of the bucket in an independent verification
# run (docs/logs/fp8_gemv/bench_final.log for MB 1-2 -> (3072, 4096) MB 2 dropped at 0.997; verify_mb3-8.log for
# MB 3-8, lowest kept 1.026). The shipped table measured end to end: docs/logs/fp8_gemv/run/bench_fp8_gemv.log.
TABLE: dict[tuple[int, int], dict[int, tuple[int, int, int, bool]]] = {
    (77440, 4096): {1: (8, 4, 2, False), 2: (16, 4, 2, True), 8: (8, 4, 2, True)},
    (12608, 4096): {1: (8, 4, 2, False), 2: (8, 4, 2, True), 3: (4, 2, 2, True), 4: (4, 2, 2, True)},
    (12288, 4096): {1: (16, 4, 2, True), 2: (16, 4, 2, True), 3: (4, 2, 2, True), 6: (8, 2, 2, True),
                    8: (4, 2, 2, True)},
    (4096, 8192): {1: (8, 8, 2, False), 2: (16, 8, 2, True), 3: (8, 4, 2, True), 6: (8, 4, 2, True),
                   8: (8, 4, 2, True)},
    (4096, 6144): {1: (8, 8, 2, True), 2: (16, 8, 2, True), 3: (8, 4, 2, True), 6: (8, 4, 2, True),
                   8: (4, 4, 2, True)},
    (4096, 4096): {1: (16, 8, 2, True), 2: (16, 8, 2, True), 3: (8, 8, 2, True), 8: (8, 4, 2, True)},
    (8192, 1536): {1: (16, 4, 2, True), 2: (8, 4, 2, True), 3: (4, 4, 2, True), 4: (4, 4, 2, True)},
    (3072, 4096): {1: (16, 16, 2, True), 3: (8, 8, 2, True)},
    (4096, 2048): {1: (16, 8, 2, True), 2: (8, 8, 2, True), 3: (8, 8, 2, True), 4: (8, 8, 2, True),
                   6: (4, 4, 2, True), 8: (8, 4, 2, True)},
    (2048, 4096): {1: (16, 16, 2, True), 2: (16, 16, 2, True)},
    (4096, 1024): {1: (16, 8, 2, True), 2: (8, 8, 2, True), 3: (8, 8, 2, True), 4: (8, 8, 2, True),
                   6: (4, 4, 2, True), 8: (4, 4, 2, True)},
    (4096, 128): {1: (8, 8, 2, True), 2: (8, 8, 2, True), 3: (4, 4, 2, True), 4: (4, 4, 2, True),
                  6: (4, 4, 2, True), 8: (4, 4, 2, True)},
}
BUCKETS = (1, 2, 3, 4, 6, 8)
_BUCKET_OF_MB = {1: 1, 2: 2, 3: 3, 4: 4, 5: 6, 6: 6, 7: 8, 8: 8}
MAX_M = 64
SELFTEST_TOL = 2e-3      # rel_l2(new, Marlin); both are fp32-accumulated, bf16-rounded: measured <= ~1e-4

# ---------------------------------------------------------------------------------------------------------------
# Large-M path (GLM53_FP8_LARGE_M): (Npad, K) -> smallest M served. Only shapes where Marlin's large-M kernel is slow
# on GB10 and the new path measured >= 1.15x faster from that M on nodeC in two A/B runs with either GEMM backend
# (tests/bench_fp8_large_m_crossover.py, docs/logs/fp8_large_m/crossover_{run,tilelang_run}{1,2}.log;
# docs/FP8_LARGE_M.md):
#   N > 4096: Marlin splits M into 1024-row launches, 25-28 TFLOPS (nodeC's per-launch times match production's R12
#   trace, tests/probe_marlin_launches.py); the drafter fc (N 4096, K 20480): below 8192 rows the gain over Marlin's
#   noisy baseline varied 1.00-1.65x between runs; from 8192 rows (Marlin's slow 8192-row launch, also in production's
#   trace) it was 1.57-1.72x in every run -> served from 8192 only.
# Anything else stays Marlin (MLA o_proj 4096x8192 and q_b 8192x1536: production's Marlin is already 70+ TFLOPS).
ENV_LARGE = "GLM53_FP8_LARGE_M"
ENV_LARGE_TEMP = "GLM53_FP8_LARGE_M_TEMP_MIB"
LARGE_TABLE: dict[tuple[int, int], int] = {
    (12608, 4096): 640,     # KDA in_proj_qkvbfg_a (N 12576, Marlin-padded to 12608), 34 layers
    (12288, 4096): 640,     # dense mlp gate_up (3 layers) and the drafter's mlp gate_up
    (4096, 20480): 8192,    # drafter fc (GLM53_DRAFT_FP8=layers,fc), runs eagerly on the target's tokens
}
LARGE_TEMP_MIB = 128
LARGE_SELFTEST_M = 320      # not a multiple of 64 (tails); the chunk loops are forced to iterate in the self-test
# GEMM backend of the large-M path (GLM53_FP8_LARGE_M_GEMM = auto (default) | tilelang | cublas):
#   tilelang: one TileLang kernel per K (symbolic M, N and output row stride; JIT-compiled at the first large-M
#     self-test, i.e. at load) computing bf16(sum * scale) in its epilogue: no fp32 temp, bitwise equal to the cublas
#     backend in every test (same fp32 accumulation order), 1.2-1.5x faster than it (docs/FP8_LARGE_M.md).
#   cublas: torch.mm(out_dtype=float32) per row chunk + scale_cast. auto = tilelang if it imports, compiles and passes
#     the self-test, else cublas (WARNING).
ENV_LARGE_GEMM = "GLM53_FP8_LARGE_M_GEMM"
TL_CONFIG = (128, 256, 32, 4, 256, 10)   # block M, block N, block K, pipeline stages, threads, swizzle panel
LARGE_ALPHA_PAD = 256                    # zero tail on the fp32 scale: the kernel's last N tile may read past N


def bucket(m: int) -> Optional[int]:
    return _BUCKET_OF_MB.get((m + 7) // 8) if 1 <= m <= MAX_M else None


def bucket_max_m(b: int) -> int:
    return {1: 8, 2: 16, 3: 24, 4: 32, 6: 48, 8: 64}[b]


def select_config(npad: int, k: int, m: int):
    """(warps, kw, u, evict_first) for this Marlin weight shape and M, or None -> Marlin."""
    if m > CFG.max_m:
        return None
    b = bucket(m)
    return None if b is None else TABLE.get((npad, k), {}).get(b)


def enabled_buckets(npad: int, k: int) -> list[int]:
    return [b for b in TABLE.get((npad, k), {}) if bucket_max_m(b) - (8 if b <= 4 else 16) < CFG.max_m]


# ---------------------------------------------------------------------------------------------------------------
class _Cfg:
    def __init__(self) -> None:
        self.max_m = MAX_M
        self.strict = False          # tests: raise instead of falling back to Marlin
        self.large_temp_mib = LARGE_TEMP_MIB


class _State:
    def __init__(self) -> None:
        self.enabled = False
        self.disabled_reason = "not installed"
        self.ext = None
        self.orig_apply = None
        self.orig_pwal = None
        self.cls = None
        self.large = False           # GLM53_FP8_LARGE_M path active
        self.large_ext = None
        self.large_gemm = "cublas"   # backend in use: "tilelang" or "cublas"
        self.large_gemm_req = "auto"
        self.tl_kernels = {}         # K -> compiled TileLang kernel


CFG = _Cfg()
STATE = _State()
VERDICT: dict[tuple, tuple[bool, str]] = {}   # (weight ptr, scale ptr, Npad, K) -> (passed, detail)
LVERDICT: dict[tuple, tuple[bool, str]] = {}  # same key -> large-M self-test (passed, detail)
LALPHA: dict[tuple, torch.Tensor] = {}        # same key -> un-permuted fp32 per-channel scale [N] (large-M path)
_PERM: dict[tuple, torch.Tensor] = {}         # (Npad, device) -> Marlin scale/bias permutation index [Npad]
COUNTERS: dict[str, int] = {}
_LOGGED: set = set()
ROOF_HOOK = None   # fp8_roof module while GLM53_DEC_FP8ROOF is installed (fp8_roof.install), else None


def _count(key: str, n: int = 1) -> None:
    COUNTERS[key] = COUNTERS.get(key, 0) + n


def _log_once(key: str, level: int, msg: str, *a) -> None:
    if key not in _LOGGED:
        _LOGGED.add(key)
        _log.log(level, msg, *a)


def env_enabled(environ: dict | None = None) -> bool:
    v = (os.environ if environ is None else environ).get(ENV)
    return v is not None and v.strip().lower() in _TRUE


def env_large_enabled(environ: dict | None = None) -> bool:
    v = (os.environ if environ is None else environ).get(ENV_LARGE)
    return v is not None and v.strip().lower() in _TRUE


def _include_shim() -> list[str]:
    base = Path("/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include")
    tk = Path("/usr/local/cuda/include")
    if not base.is_dir():
        return []
    shim = Path(os.environ.get("TF_EXL3_SHIM", "/tmp/tf_exl3_shim"))
    shim.mkdir(parents=True, exist_ok=True)
    for f in list(base.glob("*.h")) + list(base.glob("*.hpp")):
        if not (tk / f.name).exists():
            link = shim / f.name
            if not link.exists():
                try:
                    link.symlink_to(f)
                except OSError:
                    pass
    return [f"-I{shim}"]


def ext():
    """AOT module tf_fp8_gemv_ext (setup.py) or, with TF_EXL3_JIT=1 (nodeC tests), a JIT build of the .cu."""
    if STATE.ext is not None:
        return STATE.ext
    try:
        import tf_fp8_gemv_ext as m
    except ImportError as err:
        if os.environ.get("TF_EXL3_JIT", "0").strip().lower() not in _TRUE:
            raise ImportError(f"tf_fp8_gemv_ext (AOT) not importable and TF_EXL3_JIT is off: {err!r}")
        from torch.utils.cpp_extension import load
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1a")
        inc = _include_shim()
        m = load(name="tf_fp8_gemv_jit", sources=[str(HERE / "kernels" / "fp8_gemv.cu")],
                 extra_cuda_cflags=["-O3", *inc], extra_cflags=["-O3", *inc], verbose=False)
    STATE.ext = m
    return m


def ext_large():
    """AOT module tf_fp8_large_m_ext (setup.py) or, with TF_EXL3_JIT=1 (nodeC tests), a JIT build of the .cu."""
    if STATE.large_ext is not None:
        return STATE.large_ext
    try:
        import tf_fp8_large_m_ext as m
    except ImportError as err:
        if os.environ.get("TF_EXL3_JIT", "0").strip().lower() not in _TRUE:
            raise ImportError(f"tf_fp8_large_m_ext (AOT) not importable and TF_EXL3_JIT is off: {err!r}")
        from torch.utils.cpp_extension import load
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1a")
        inc = _include_shim()
        m = load(name="tf_fp8_large_m_jit", sources=[str(HERE / "kernels" / "fp8_large_m.cu")],
                 extra_cuda_cflags=["-O3", *inc], extra_cflags=["-O3", *inc], verbose=False)
    STATE.large_ext = m
    return m


# ---------------------------------------------------------------------------------------------------------------
def marlin(x, weight, weight_scale, workspace, bias, n: int, k: int):
    """Production's FP8 Marlin linear (what Glm53DenseFp8Method.apply runs for these layers)."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import apply_fp8_marlin_linear
    return apply_fp8_marlin_linear(input=x, weight=weight, weight_scale=weight_scale, workspace=workspace,
                                   size_n=n, size_k=k, bias=bias)


def _ineligible(x2d, weight, weight_scale, bias, n: int, k: int) -> Optional[str]:
    """None if the new kernel can take this call, else the reason (the C++ side re-checks everything)."""
    if x2d.dtype != torch.bfloat16 or not x2d.is_cuda:
        return "dtype"
    if weight.dtype != torch.int32 or weight.dim() != 2 or not weight.is_contiguous() or weight.shape[0] * 16 != k:
        return "weight_layout"
    npad = weight.shape[1] // 4
    if npad % 64 or not (0 <= npad - n < 128) or weight.data_ptr() % 32:
        return "weight_layout"
    if weight_scale.dtype != torch.bfloat16 or weight_scale.numel() != npad or not weight_scale.is_contiguous():
        return "scale_layout"
    if bias is not None and (bias.dtype != torch.bfloat16 or bias.numel() != npad or not bias.is_contiguous()):
        return "bias_layout"
    if x2d.stride(1) != 1 or x2d.data_ptr() % 4 or (x2d.shape[0] > 1 and (x2d.stride(0) % 2 or x2d.stride(0) < k)):
        return "x_layout"
    if x2d.device != weight.device:
        return "device"
    return None


def _key(weight, weight_scale, k: int):
    return (weight.data_ptr(), weight_scale.data_ptr(), weight.shape[1] // 4, k)


def selftest(weight, weight_scale, workspace, n: int, k: int, bias=None, label: str = "") -> tuple[bool, str]:
    """New kernel vs production's Marlin on seeded random inputs at every enabled M bucket (+ M = 5, a partial
    8-row block). Runs kernels and synchronizes: call eagerly only (never while capturing). Records the verdict."""
    key = _key(weight, weight_scale, k)
    npad = weight.shape[1] // 4
    buckets = enabled_buckets(npad, k)
    if not buckets:
        VERDICT[key] = (False, "no tuned configuration for this shape")
        return VERDICT[key]
    ms = sorted({5, *[min(bucket_max_m(b), CFG.max_m) for b in buckets]})
    E = ext()
    g = torch.Generator(device=weight.device).manual_seed(1234)
    worst, detail = 0.0, []
    try:
        for m in ms:
            cfg = select_config(npad, k, m)
            if cfg is None:
                continue
            x = torch.randn(m, k, device=weight.device, generator=g).to(torch.bfloat16)
            ref = marlin(x, weight, weight_scale, workspace, bias, n, k)
            y = E.fp8_gemv(x, weight, weight_scale.view(-1), bias, n, k, *cfg)
            d = (y.float() - ref.float()).norm().item() / max(ref.float().norm().item(), 1e-30)
            fin = bool(torch.isfinite(y).all().item()) or not bool(torch.isfinite(ref).all().item())
            if not fin:
                d = float("inf")
            worst = max(worst, d)
            detail.append(f"M{m}:{d:.1e}")
    except Exception as exc:  # noqa: BLE001 - a self-test that cannot run is a failed self-test
        if _CUDA_ERR.search(str(exc)):
            raise
        VERDICT[key] = (False, f"self-test raised {type(exc).__name__}: {str(exc)[:160]}")
        return VERDICT[key]
    ok = worst <= SELFTEST_TOL
    VERDICT[key] = (ok, f"rel_l2 vs Marlin max {worst:.2e} (tol {SELFTEST_TOL:.0e}) [{' '.join(detail)}]")
    if ok:
        _log_once(f"st:{npad}x{k}", logging.INFO, "tf_fp8_gemv: %s [%dx%d] Npad=%d self-test passed, M buckets %s: %s",
                  label or "layer", n, k, npad, buckets, VERDICT[key][1])
    else:
        _log.warning("tf_fp8_gemv: %s [%dx%d] self-test FAILED, this layer stays on Marlin: %s", label or "layer",
                     n, k, VERDICT[key][1])
    _count("selftest_passed" if ok else "selftest_failed")
    return VERDICT[key]


def try_new(x, weight, weight_scale, workspace, bias, n: int, k: int):
    """The new kernel's result for this call, or None when it declines (the caller then runs production's path)."""
    if not STATE.enabled or x.shape[-1] != k:
        return None
    x2d = x.reshape(-1, k)
    m = x2d.shape[0]
    cfg = select_config(weight.shape[1] // 4, k, m) if m >= 1 else None
    if cfg is None:
        _count("marlin_m")
        return None
    why = _ineligible(x2d, weight, weight_scale, bias, n, k)
    if why is not None:
        _count("marlin_" + why)
        _log_once("inel:" + why, logging.WARNING, "tf_fp8_gemv: a call is not eligible (%s: x %s %s stride %s, "
                  "weight %s %s): Marlin serves it (counted, not repeated)", why, tuple(x2d.shape), x2d.dtype,
                  tuple(x2d.stride()), tuple(weight.shape), weight.dtype)
        return None
    key = _key(weight, weight_scale, k)
    v = VERDICT.get(key)
    capturing = torch.cuda.is_current_stream_capturing()
    if v is None:
        if capturing:
            _count("marlin_untested_in_capture")
            _log_once("untested_capture", logging.WARNING, "tf_fp8_gemv: a layer was first seen during CUDA-graph "
                      "capture (no eager call before): Marlin serves it (counted, not repeated)")
            return None
        v = selftest(weight, weight_scale, workspace, n, k, bias, label="lazily tested layer")
    if not v[0]:
        _count("marlin_selftest_failed")
        return None
    try:
        y = STATE.ext.fp8_gemv(x2d, weight, weight_scale.view(-1), bias, n, k, *cfg)
    except Exception as exc:
        if CFG.strict or not isinstance(exc, (RuntimeError, ValueError, TypeError, IndexError)) or \
                _CUDA_ERR.search(str(exc)):
            raise
        _count("marlin_error")
        _log_once("err:" + str(exc)[:80], logging.WARNING, "tf_fp8_gemv: kernel pre-check failed (%s: %s): Marlin "
                  "serves this call (identical failures are counted, not repeated)", type(exc).__name__, str(exc)[:200])
        return None
    _count("new_captured" if capturing else "new_eager")
    return y.reshape(x.shape[:-1] + (n,))


# ---------------------------------------------------------------------------------------------------------------
# Large-M path
def _perm(npad: int, device) -> torch.Tensor:
    """Stored position of each logical column in a Marlin-permuted per-channel vector (scale_perm_single)."""
    key = (npad, str(device))
    p = _PERM.get(key)
    if p is None:
        c = torch.arange(npad, device=device)
        r = c % 32
        p = (c - r) + 8 * ((r % 8) // 2) + (r % 2) + 2 * (r // 8)
        _PERM[key] = p
    return p


def large_alpha(weight_scale, n: int) -> torch.Tensor:
    """The per-channel scale Marlin multiplies by, un-permuted, as fp32 [N] (+ LARGE_ALPHA_PAD zeros): stored bf16
    (scale * 2^120) * 2^-120, exact (a power-of-two shift of a bf16 value; the scales are normal fp32 numbers)."""
    npad = weight_scale.numel()
    a = torch.zeros(n + LARGE_ALPHA_PAD, dtype=torch.float32, device=weight_scale.device)
    a[:n] = weight_scale.reshape(-1)[_perm(npad, weight_scale.device)[:n]].float() * 2.0 ** -120
    return a


def large_plan(n: int, k: int, m: int, temp_mib: int | None = None, backend: str | None = None) -> tuple[int, int]:
    """(columns per weight chunk, rows per GEMM call) within the transient budget. tilelang: the BF16 weight chunk
    [nc, K] may take the whole budget (no fp32 temp), all rows in one call. cublas: the weight chunk takes at most 3/4
    of it, the fp32 result chunk [mc, nc] the rest; fewest column chunks such that a row chunk still has >= 2048 rows
    (both a narrow and a short GEMM are slower; tests/probe_large_m7.py); at least 256 rows. nc is a multiple of 64,
    so every chunk start stays 16-byte aligned."""
    budget = (CFG.large_temp_mib if temp_mib is None else temp_mib) * 2 ** 20
    if (backend or STATE.large_gemm) == "tilelang":
        nc_max = max(64, budget // (2 * k) // 64 * 64)
        nchunks = -(-n // nc_max)
        return min(-(-(-(-n // nchunks)) // 64) * 64, -(-n // 64) * 64), m
    nc_max = max(64, budget * 3 // 4 // (2 * k) // 64 * 64)
    nchunks = -(-n // nc_max)
    while True:
        nc = min(-(-(-(-n // nchunks)) // 64) * 64, -(-n // 64) * 64)
        mc = (budget - nc * k * 2) // (4 * nc) // 64 * 64
        if mc >= 2048 or nc <= 1024:
            break
        nchunks += 1
    return nc, min(max(256, mc), m)


def _tl_build(k: int):
    """The TileLang kernel for this K (fp8_large_m_tl.py): symbolic M, N and output row stride, so one compile per K
    serves every prefill size and column chunk."""
    import fp8_large_m_tl
    return fp8_large_m_tl.build(k, *TL_CONFIG)


def _tl_kernel(k: int):
    kern = STATE.tl_kernels.get(k)
    if kern is None:
        kern = _tl_build(k)
        STATE.tl_kernels[k] = kern
    return kern


def large_forward(x2d, weight, alpha, bias, n: int, k: int, nc: int | None = None, mc: int | None = None,
                  backend: str | None = None):
    """The large-M linear for x2d [M, K] bf16 (any row stride with unit column stride): bf16 [M, N]. alpha: the
    padded fp32 scale (large_alpha). bias, if given, is production's Marlin-permuted [Npad] bf16 vector."""
    L = STATE.large_ext
    backend = backend or STATE.large_gemm
    m = x2d.shape[0]
    pn, pm = large_plan(n, k, m, backend=backend)
    nc, mc = nc or pn, mc or pm
    out = torch.empty(m, n, dtype=torch.bfloat16, device=x2d.device)
    bu = None
    if bias is not None:
        bu = bias.reshape(-1)[_perm(bias.numel(), bias.device)[:n]].contiguous()
    wtmp = torch.empty(min(nc, n), k, dtype=torch.bfloat16, device=x2d.device)
    if backend == "tilelang":
        kern = _tl_kernel(k)
        if x2d.stride(0) != k or x2d.data_ptr() % 16:        # the kernel's TMA loads want a dense, aligned A
            x2d = x2d.clone(memory_format=torch.contiguous_format)
    for n0 in range(0, n, nc):
        c = min(nc, n - n0)
        wv = wtmp[:c]
        L.dequant(wv, weight, n0, c, k)
        if backend == "tilelang":
            for m0 in range(0, m, mc):
                r = min(mc, m - m0)
                kern(x2d[m0:m0 + r], wv, alpha[n0:n0 + c], out[m0:m0 + r, n0:n0 + c])
            continue
        wt = wv.t()
        for m0 in range(0, m, mc):
            r = min(mc, m - m0)
            y32 = torch.mm(x2d[m0:m0 + r], wt, out_dtype=torch.float32)
            L.scale_cast(out[m0:m0 + r, n0:n0 + c], y32, alpha[n0:n0 + c], None if bu is None else bu[n0:n0 + c])
            del y32
    if bu is not None and backend == "tilelang":
        out += bu                                            # bf16(bf16(sum * s) + b): Marlin's epilogue order
    return out


def _ineligible_large(x2d, weight, weight_scale, bias, n: int, k: int) -> Optional[str]:
    if x2d.dtype != torch.bfloat16 or not x2d.is_cuda:
        return "dtype"
    if weight.dtype != torch.int32 or weight.dim() != 2 or not weight.is_contiguous() or weight.shape[0] * 16 != k:
        return "weight_layout"
    npad = weight.shape[1] // 4
    if npad % 64 or not (0 <= npad - n < 128) or n % 8 or k % 16:
        return "weight_layout"
    if weight_scale.dtype != torch.bfloat16 or weight_scale.numel() != npad or not weight_scale.is_contiguous():
        return "scale_layout"
    if bias is not None and (bias.dtype != torch.bfloat16 or bias.numel() != npad or not bias.is_contiguous()):
        return "bias_layout"
    if x2d.stride(1) != 1 or x2d.device != weight.device:
        return "x_layout"
    return None


def selftest_large(weight, weight_scale, workspace, n: int, k: int, bias=None, label: str = "") -> tuple[bool, str]:
    """Large-M path vs production's Marlin on seeded random rows (M = LARGE_SELFTEST_M; the column and row chunk loops
    are forced to iterate): rel_l2 <= SELFTEST_TOL, >= 99 % of elements bitwise equal, finite. Caches the fp32 scale.
    Runs kernels and synchronizes: eager only (never while capturing)."""
    key = _key(weight, weight_scale, k)
    try:
        why = _ineligible_large(torch.empty(1, k, dtype=torch.bfloat16, device=weight.device), weight, weight_scale,
                                bias, n, k)
        if why is not None:
            LVERDICT[key] = (False, f"not eligible ({why})")
            return LVERDICT[key]
        rp = _large_runtime_problem()
        if rp is not None:
            LVERDICT[key] = (False, rp)
            return LVERDICT[key]
        alpha = large_alpha(weight_scale, n)
        if not bool(torch.isfinite(alpha).all().item()) or not bool((alpha > 0).any().item()):
            LVERDICT[key] = (False, "per-channel scale not finite / all zero")
            return LVERDICT[key]
        g = torch.Generator(device=weight.device).manual_seed(4321)
        m = LARGE_SELFTEST_M
        x = torch.randn(m, k, device=weight.device, generator=g).to(torch.bfloat16)
        ref = marlin(x, weight, weight_scale, workspace, bias, n, k)
        nc = max(64, (-(-n // 3) + 63) // 64 * 64)          # >= 2 column chunks, unaligned last chunk
        vs_cublas = ""
        if STATE.large_gemm == "tilelang":
            try:
                y = large_forward(x, weight, alpha, bias, n, k, nc=nc, mc=128)
            except Exception as exc:  # noqa: BLE001 - TileLang unusable here: the cuBLAS backend serves instead
                if _CUDA_ERR.search(str(exc)):
                    raise
                STATE.large_gemm = "cublas"
                _log.warning("tf_fp8_gemv: the TileLang GEMM of the large-M path failed (%s: %s); the cuBLAS backend "
                             "serves the large-M path instead", type(exc).__name__, str(exc)[:200])
            else:
                yc = large_forward(x, weight, alpha, bias, n, k, nc=nc, mc=128, backend="cublas")
                vs_cublas = f", bitwise == cuBLAS backend {(y == yc).float().mean().item() * 100:.2f}%"
                del yc
        if STATE.large_gemm == "cublas":
            y = large_forward(x, weight, alpha, bias, n, k, nc=nc, mc=128)
        yf, rf = y.float(), ref.float()
        d = (yf - rf).norm().item() / max(rf.norm().item(), 1e-30)
        same = (y == ref).float().mean().item()
        fin = bool(torch.isfinite(y).all().item()) or not bool(torch.isfinite(ref).all().item())
        del x, ref, y, yf, rf
    except Exception as exc:  # noqa: BLE001 - a self-test that cannot run is a failed self-test
        if _CUDA_ERR.search(str(exc)):
            raise
        LVERDICT[key] = (False, f"large-M self-test raised {type(exc).__name__}: {str(exc)[:160]}")
        _log.warning("tf_fp8_gemv: %s [%dx%d] large-M self-test raised, layer stays on Marlin for large M: %s",
                     label or "layer", n, k, LVERDICT[key][1])
        _count("large_selftest_failed")
        return LVERDICT[key]
    ok = fin and d <= SELFTEST_TOL and same >= 0.99
    detail = (f"M{LARGE_SELFTEST_M} ({STATE.large_gemm}): rel_l2 vs Marlin {d:.2e} (tol {SELFTEST_TOL:.0e}), bitwise "
              f"equal {same * 100:.2f}%{vs_cublas}")
    LVERDICT[key] = (ok, detail)
    if ok:
        LALPHA[key] = alpha
        npad = weight.shape[1] // 4
        _log_once(f"lst:{npad}x{k}", logging.INFO, "tf_fp8_gemv: %s [%dx%d] large-M self-test passed (M >= %d served): "
                  "%s", label or "layer", n, k, LARGE_TABLE.get((npad, k), 0), detail)
    else:
        _log.warning("tf_fp8_gemv: %s [%dx%d] large-M self-test FAILED, this layer stays on Marlin for large M: %s",
                     label or "layer", n, k, detail)
    _count("large_selftest_passed" if ok else "large_selftest_failed")
    return LVERDICT[key]


def try_large(x, weight, weight_scale, workspace, bias, n: int, k: int):
    """The large-M path's result for this call, or None when it declines (the caller then runs production's path)."""
    if not STATE.large or x.shape[-1] != k:
        return None
    min_m = LARGE_TABLE.get((weight.shape[1] // 4, k)) if weight.dim() == 2 else None
    if min_m is None:
        return None
    m = x.numel() // k
    if m < min_m:
        _count("large_marlin_m")
        return None
    if torch.cuda.is_current_stream_capturing():     # decode graphs are <= 64 tokens; never serve a capture
        _count("large_marlin_capture")
        return None
    x2d = x.reshape(-1, k)
    why = _ineligible_large(x2d, weight, weight_scale, bias, n, k)
    if why is not None:
        _count("large_marlin_" + why)
        _log_once("linel:" + why, logging.WARNING, "tf_fp8_gemv: a large-M call is not eligible (%s: x %s %s stride %s, "
                  "weight %s %s): Marlin serves it (counted, not repeated)", why, tuple(x2d.shape), x2d.dtype,
                  tuple(x2d.stride()), tuple(weight.shape), weight.dtype)
        return None
    key = _key(weight, weight_scale, k)
    v = LVERDICT.get(key)
    if v is None:
        v = selftest_large(weight, weight_scale, workspace, n, k, bias, label="lazily tested layer")
    if not v[0]:
        _count("large_marlin_selftest_failed")
        return None
    try:
        y = large_forward(x2d, weight, LALPHA[key], bias, n, k)
    except Exception as exc:
        if CFG.strict or not isinstance(exc, (RuntimeError, ValueError, TypeError, IndexError)) or \
                _CUDA_ERR.search(str(exc)):
            raise
        _count("large_marlin_error")
        _log_once("lerr:" + str(exc)[:80], logging.WARNING, "tf_fp8_gemv: large-M path failed before launch (%s: %s): "
                  "Marlin serves this call (identical failures are counted, not repeated)", type(exc).__name__,
                  str(exc)[:200])
        return None
    _count("large_eager")
    _count("large_rows", m)
    _log_once(f"lfirst:{weight.shape[1] // 4}x{k}", logging.INFO, "tf_fp8_gemv: first large-M call served for "
              "%dx%d (M=%d, %s GEMM; M >= %d goes this way, logged once per shape)", n, k, m, STATE.large_gemm, min_m)
    return y.reshape(x.shape[:-1] + (n,))


def linear(x, weight, weight_scale, workspace, bias, n: int, k: int):
    """Drop-in for apply_fp8_marlin_linear(input=x, ...): same output (dtype of x, shape x.shape[:-1] + (n,))."""
    y = try_new(x, weight, weight_scale, workspace, bias, n, k)
    if y is None:
        y = try_large(x, weight, weight_scale, workspace, bias, n, k)
    return y if y is not None else marlin(x, weight, weight_scale, workspace, bias, n, k)


# The custom op (torch.compile path). Registered at import so compiled graphs that reference it always load.
@torch.library.custom_op("tf_fp8::linear", mutates_args=())
def _linear_op(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, workspace: torch.Tensor,
               bias: Optional[torch.Tensor], size_n: int, size_k: int) -> torch.Tensor:
    return linear(x, weight, weight_scale, workspace, bias, size_n, size_k)


@_linear_op.register_fake
def _(x, weight, weight_scale, workspace, bias, size_n, size_k):
    return x.new_empty(tuple(x.shape[:-1]) + (size_n,))


# ---------------------------------------------------------------------------------------------------------------
def _make_apply(orig_apply):
    def serve(self, layer, x, bias):
        # production's own apply for everything it does not send to Marlin (not ready, KDA large-M BF16 copy)
        if not (STATE.enabled or STATE.large) or not getattr(self, "ready", False) or \
                getattr(layer, "glm53_bf16_lm_w", None) is not None:
            return orig_apply(self, layer, x, bias)
        n, k = int(layer.glm53_fp8_n), int(layer.glm53_fp8_k)
        if x.numel() > CFG.max_m * k:          # prefill: the large-M path (GLM53_FP8_LARGE_M) or production's apply
            if STATE.large:
                y = try_large(x, layer.weight, layer.weight_scale, layer.workspace, bias, n, k)
                if y is not None:
                    return y
            _count("marlin_m")
            return orig_apply(self, layer, x, bias)
        y = try_new(x, layer.weight, layer.weight_scale, layer.workspace, bias, n, k)
        return y if y is not None else orig_apply(self, layer, x, bias)

    def apply(self, layer, x, bias=None):
        if torch.compiler.is_compiling():
            if (STATE.enabled or STATE.large) and getattr(self, "ready", False) and \
                    getattr(layer, "glm53_bf16_lm_w", None) is None:   # the choice must happen at run time, in the op
                return torch.ops.tf_fp8.linear(x, layer.weight, layer.weight_scale, layer.workspace, bias,
                                               int(layer.glm53_fp8_n), int(layer.glm53_fp8_k))
            return orig_apply(self, layer, x, bias)
        y = serve(self, layer, x, bias)
        if ROOF_HOOK is not None:              # GLM53_DEC_FP8ROOF (fp8_roof.py): this linear's kernels are enqueued
            ROOF_HOOK.after_linear(layer, x, self)
        return y

    apply.__doc__ = getattr(orig_apply, "__doc__", None)
    apply._tf_fp8_hook = True
    apply._tf_fp8_orig = orig_apply
    return apply


def _make_pwal(orig_pwal):
    def process_weights_after_loading(self, layer):
        orig_pwal(self, layer)                     # production first; its exceptions propagate unchanged
        try:
            if (STATE.enabled or STATE.large) and getattr(self, "ready", False) and \
                    getattr(layer, "glm53_bf16_lm_w", None) is None and hasattr(layer, "workspace"):
                n, k = int(layer.glm53_fp8_n), int(layer.glm53_fp8_k)
                label = f"{getattr(self, 'group', '?')} {getattr(self, 'prefix', '')}".strip()
                if STATE.enabled:
                    selftest(layer.weight, layer.weight_scale, layer.workspace, n, k, getattr(layer, "bias", None),
                             label)
                if STATE.large and (layer.weight.shape[1] // 4, k) in LARGE_TABLE:
                    selftest_large(layer.weight, layer.weight_scale, layer.workspace, n, k,
                                   getattr(layer, "bias", None), label)
                if ROOF_HOOK is not None:          # GLM53_DEC_FP8ROOF: L2-prefetch registry (fp8_roof.py)
                    ROOF_HOOK.register(self, layer)
        except Exception as exc:  # noqa: BLE001 - never fail the load; the layer stays on Marlin
            if _CUDA_ERR.search(str(exc)):
                raise
            _log.warning("tf_fp8_gemv: self-test at load raised %r (layer stays on Marlin)", exc)

    process_weights_after_loading.__doc__ = getattr(orig_pwal, "__doc__", None)
    process_weights_after_loading._tf_fp8_hook = True
    process_weights_after_loading._tf_fp8_orig = orig_pwal
    return process_weights_after_loading


def _refuse(report: dict, reason: str) -> dict:
    report["reason"] = reason
    STATE.disabled_reason = reason
    _log.warning("tf_fp8_gemv enabled (%s) but NOT installed: %s; production's Marlin FP8 path unchanged", ENV, reason)
    return report


def _large_problem() -> Optional[str]:
    """None if the large-M path can be enabled here, else the reason. Loads its extension."""
    raw = os.environ.get(ENV_LARGE_TEMP, "").strip()
    if raw:
        try:
            v = int(raw)
        except ValueError:
            return f"{ENV_LARGE_TEMP}={raw!r} is not an integer"
        if not 16 <= v <= 4096:
            return f"{ENV_LARGE_TEMP}={raw!r} is not in 16..4096"
        CFG.large_temp_mib = v
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (12, 0):
        return "needs an SM12x GPU (GB10: sm_121a)"
    req = os.environ.get(ENV_LARGE_GEMM, "").strip().lower() or "auto"
    if req not in ("auto", "tilelang", "cublas"):
        return f"{ENV_LARGE_GEMM}={req!r} is not auto | tilelang | cublas"
    STATE.large_gemm_req = req
    STATE.large_gemm = "cublas"
    if req != "cublas":
        # find_spec only: importing tilelang (TVM) here would cost every process that loads vLLM's general plugins
        # (API server, engine core) ~0.3 GB RSS; it is imported and compiled at the first large-M self-test (workers)
        import importlib.util
        if importlib.util.find_spec("tilelang") is not None:
            STATE.large_gemm = "tilelang"
        else:
            _log.warning("tf_fp8_gemv: %s=%s but tilelang is not installed: the large-M path uses cuBLAS",
                         ENV_LARGE_GEMM, req)
    try:
        L = ext_large()
    except Exception as exc:  # noqa: BLE001
        return f"extension not importable: {exc!r}"
    if getattr(L, "VERSION", None) != 1:
        return f"large-M extension version {getattr(L, 'VERSION', None)!r} != 1"
    # the torch.mm(out_dtype=float32) probe runs at the first large-M self-test (_large_runtime_problem): running it
    # here would create a CUDA context (~0.5 GB of unified memory) in every process that loads general plugins
    return None


_RUNTIME_PROBLEM: list = []


def _large_runtime_problem() -> Optional[str]:
    """torch.mm with a bf16 x bf16 -> fp32 output (torch >= 2.9 on CUDA); checked once, in a process that has a CUDA
    context already (the first large-M self-test, i.e. a worker at load)."""
    if _RUNTIME_PROBLEM:
        return _RUNTIME_PROBLEM[0]
    why = None
    try:
        a = torch.ones(8, 16, dtype=torch.bfloat16, device="cuda")
        y = torch.mm(a, a.t(), out_dtype=torch.float32)
        if y.dtype != torch.float32 or float(y[0, 0].item()) != 16.0:
            why = "torch.mm(out_dtype=float32) returned an unexpected result"
    except Exception as exc:  # noqa: BLE001
        if _CUDA_ERR.search(str(exc)):
            raise
        why = f"torch.mm(out_dtype=float32) unsupported: {exc!r}"
    _RUNTIME_PROBLEM.append(why)
    if why is not None:
        _log.warning("tf_fp8_gemv: the large-M path is NOT usable here: %s; production's Marlin serves every large-M "
                     "call", why)
    return why


def install(prodmod=None, *, force: bool | None = None, force_large: bool | None = None) -> dict:
    """Wrap Glm53DenseFp8Method.apply / process_weights_after_loading. No-op unless GLM53_FP8_GEMV (force) or
    GLM53_FP8_LARGE_M (force_large) is enabled; each path is enabled only by its own variable. Idempotent. Never
    raises: problems are logged as a WARNING and returned in report["reason"] / report["large_reason"]."""
    report: dict = {"installed": False, "reason": None}
    gemv_on = env_enabled() if force is None else bool(force)
    large_on = env_large_enabled() if force_large is None else bool(force_large)
    if not gemv_on and not large_on:
        report["reason"] = f"{ENV} not enabled"
        return report
    try:
        if prodmod is None:
            import importlib
            prodmod = importlib.import_module(PROD_MODULE)
        cls = getattr(prodmod, "Glm53DenseFp8Method", None)
        if cls is None or not callable(getattr(cls, "apply", None)) or \
                not callable(getattr(cls, "process_weights_after_loading", None)):
            return _refuse(report, f"{getattr(prodmod, '__file__', prodmod)} has no Glm53DenseFp8Method")
        gemv_ok = False
        if gemv_on:
            why = None
            raw = os.environ.get(ENV_MAX_M, "").strip()
            if raw:
                try:
                    v = int(raw)
                except ValueError:
                    v = 0
                if not 1 <= v <= MAX_M:
                    why = f"{ENV_MAX_M}={raw!r} is not in 1..{MAX_M}"
                else:
                    CFG.max_m = v
            if why is None:
                try:
                    E = ext()
                    if getattr(E, "VERSION", None) != 1:
                        why = f"extension version {getattr(E, 'VERSION', None)!r} != 1"
                except Exception as exc:  # noqa: BLE001
                    why = f"extension not importable: {exc!r}"
            if why is None and (not torch.cuda.is_available() or torch.cuda.get_device_capability() < (12, 0)):
                why = "needs an SM12x GPU (GB10: sm_121a)"
            gemv_why = why
            gemv_ok = why is None
        large_ok = False
        if large_on:
            why = _large_problem()
            if why is not None:
                report["large_reason"] = why
                _log.warning("tf_fp8_gemv: %s is set but the large-M path is NOT enabled: %s; production's Marlin "
                             "serves every large-M call", ENV_LARGE, why)
            large_ok = why is None
        if gemv_on and not gemv_ok:
            if large_ok:
                report["reason"] = gemv_why
                _log.warning("tf_fp8_gemv: %s is set but the small-M path is NOT enabled: %s (the large-M path "
                             "installs)", ENV, gemv_why)
            else:
                return _refuse(report, gemv_why)
        if not gemv_ok and not large_ok:
            return report
        if getattr(cls.apply, "_tf_fp8_hook", False):
            STATE.enabled = STATE.enabled or gemv_ok
            STATE.large = STATE.large or large_ok
            report.update(installed=True, reason="already installed", large=STATE.large)
            return report
        STATE.cls, STATE.orig_apply, STATE.orig_pwal = cls, cls.apply, cls.process_weights_after_loading
        cls.apply = _make_apply(cls.apply)
        cls.process_weights_after_loading = _make_pwal(cls.process_weights_after_loading)
        STATE.enabled, STATE.large = gemv_ok, large_ok
        STATE.disabled_reason = None if gemv_ok else STATE.disabled_reason
    except Exception as exc:  # noqa: BLE001
        return _refuse(report, f"install raised {exc!r}")
    report.update(installed=True, reason="ok" if gemv_ok or not gemv_on else report["reason"], max_m=CFG.max_m,
                  large=large_ok)
    if gemv_ok:
        _log.info("tf_fp8_gemv installed: Glm53DenseFp8Method.apply of %s now serves M <= %d from the Marlin weights "
                  "in place (tuned shapes: %d; everything else stays Marlin)", getattr(prodmod, "__file__", prodmod),
                  CFG.max_m, len(TABLE))
    if large_ok:
        _log.info("tf_fp8_gemv large-M path installed (%s): Glm53DenseFp8Method.apply of %s serves %s (Npad x K: min "
                  "M) by exact BF16 dequant + %s GEMM with the per-channel scale in fp32, transient <= %d MiB; "
                  "everything else stays Marlin", ENV_LARGE, getattr(prodmod, "__file__", prodmod),
                  ", ".join(f"{a}x{b}: {m}" for (a, b), m in LARGE_TABLE.items()),
                  "TileLang (fused scale)" if STATE.large_gemm == "tilelang" else "cuBLAS fp32-out", CFG.large_temp_mib)
    return report


def uninstall() -> dict:
    """Restore only what install() replaced; idempotent. Already-captured CUDA graphs keep what they captured."""
    rep = {"restored": False}
    cls = STATE.cls
    if cls is not None and getattr(cls.apply, "_tf_fp8_hook", False):
        cls.apply = cls.apply._tf_fp8_orig
        rep["restored"] = True
    if cls is not None and getattr(cls.process_weights_after_loading, "_tf_fp8_hook", False):
        cls.process_weights_after_loading = cls.process_weights_after_loading._tf_fp8_orig
    STATE.enabled, STATE.large, STATE.disabled_reason = False, False, "uninstalled"
    STATE.large_gemm = "cublas"
    STATE.cls = STATE.orig_apply = STATE.orig_pwal = None
    VERDICT.clear()
    LVERDICT.clear()
    LALPHA.clear()
    return rep


def summary() -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(COUNTERS.items())) or "no calls"


def plugin_register() -> None:
    """Called from integrate.plugin_register (vllm.general_plugins, every vLLM process); never raises."""
    try:
        on, large = env_enabled(), env_large_enabled()
        _log.info("tf_fp8_gemv plugin loaded (pid %d): %s=%r, %s=%r -> %s", os.getpid(), ENV, os.environ.get(ENV),
                  ENV_LARGE, os.environ.get(ENV_LARGE),
                  "installing" if on or large else "off, production's Marlin FP8 path unchanged")
        if on or large:
            install()
    except Exception as exc:  # noqa: BLE001
        _log.warning("tf_fp8_gemv plugin install failed (Marlin FP8 path unchanged): %r", exc)
