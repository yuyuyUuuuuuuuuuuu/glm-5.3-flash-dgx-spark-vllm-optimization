"""W8A8 (per-token fp8 activations x the existing per-channel fp8 weights) for the DENSE + shared-expert
FP8 linears on the PREFILL path only (pf3000 plan step 3, docs/PF3000_KILLTESTS_FP8.md + its review).

Production (launcher overlay exl3.py ``Glm53DenseFp8Method``; GLM53_DENSE_FP8=dense,kda,mla,shared) stores each
dense projection ONLY as the Marlin repack (int32 [K/16, 4*Npad] + bf16 permuted scale x 2^120): at prefill the
weights are served by the exact BF16-dequant + TileLang W8A16 path (GLM53_FP8_LARGE_M shapes) or by Marlin. The
kill test measured cutlass W8A8 at our shapes 1.14-2.41x faster than that per call (about -530 ms per
13,824-token chunk per rank) but the standard-layout fp8 operand cutlass needs exists nowhere on the device:
a resident copy costs 3.50 GiB/rank (KV pool 2.00M -> ~1.58M, rejected), and the bf16-dequant + fp8 cast
re-layout costs 106.9 ms/chunk (its own DRAM floor, 6*N*K bytes per chunk).

This module serves the GEMM from a TRANSIENT standard-layout copy rebuilt per call by the byte-exact un-permutation
kernel ``fp8_marlin_to_std`` (kernels/fp8_w8a8.cu, 2*N*K bytes): the fp8 values it writes are the same bytes the
Marlin payload holds (production's own e4m3 quantization), so the weights introduce no rounding of their own and
the per-channel scale cutlass multiplies by is the same scale Marlin multiplies by (the un-permuted stored bf16
scale, exactly as fp8_gemv.large_alpha). The only new rounding against production is the per-token e4m3
activation quantization (measured 2.42e-2..2.65e-2 rel_l2 per GEMM, docs/PF3000_KILLTESTS_FP8.md 2).

Enable: ``GLM53_DENSE_W8A8`` in {1, on, true, yes}, read once when vLLM loads general plugins
(integrate.plugin_register, AFTER fp8_gemv so this wrapper is the outer one and delegates to it). Unset -> nothing
is patched, byte-identical to production. Optional knobs (read once at install):
  GLM53_DENSE_W8A8_MIN_M      smallest M (rows) served, default 512 (decode stays Marlin: decode calls are
                              M <= 64 and CUDA-graph captured, which this path always declines anyway)
  GLM53_DENSE_W8A8_PIECE_MIB  2048-row pieces are used when the fp8 weight exceeds this many MiB, else the whole
                              M in one cutlass call (the measured per-shape crossover: pieces win on the big
                              weights, one call on the small ones); default 16, 0 = always pieces. Only for the
                              image's cutlass_scaled_mm (GLM53_DENSE_W8A8_GEMM=cutlass_mm)
  GLM53_DENSE_W8A8_ONLY       w8a82: comma list of "<group>.<projection>" (the last component of the vLLM prefix, e.g.
                              kda.in_proj_qkvbfg_a, mla.o_proj, shared.down_proj); when set ONLY those projections are
                              served (a quality/speed dial for the production A/B: kda.in_proj_qkvbfg_a + mla.o_proj
                              carry ~65 % of the saving through 45 of the 192 GEMMs per chunk). Unset = every
                              projection of the served groups (w8a8's behaviour)
  GLM53_DENSE_W8A8_GEMM       w8a82: "custom" (default when the extension has it) = this extension's CUTLASS SM120
                              persistent GEMM with a per-shape tile/schedule/swizzle/raster (GEMM_TABLE), one call over
                              the whole M, epilogue arithmetic identical to cutlass_scaled_mm (bitwise-equal output,
                              checked per layer at load); "cutlass_mm" = the image's cutlass_scaled_mm in pieces (w8a8)
  GLM53_DENSE_W8A8_SKIP_LAYERS opt-w8a8layers: comma list of "<layers>[:<name>]" excluded from the W8A8 path (they stay
                              on production's path), applied after ONLY; <layers> = N or A-B (model layer indices of
                              the vLLM prefix "...layers.N..."), <name> = a PROJ_NAMES entry, kda.f_b_proj /
                              kda.g_b_proj (served when ONLY is unset), or a whole group (kda, mla, dense, shared);
                              no <name> = every projection of those layers. E.g. "0-2" (the three dense layers) or
                              "0-2,3:mla.o_proj". Unset/empty = no layer excluded (byte-identical behaviour). A
                              malformed value refuses the whole W8A8 install (production's path), like ONLY

Served: eager prefill calls (never CUDA-graph capture, never torch.compile) of layers production routed to a dense
FP8 group (self.group in the module's own _glm53_dense_fp8_groups(); the drafter's "draft" group and the KDA
BF16-large-M layers are not served), whose weights are the Marlin layout this module reads. Every layer runs a
load-time self-test (below); a failing layer stays on production's path. Any non-CUDA error declines to
production's path (counted, not repeated); CUDA errors propagate like production's.

Per-call cost: repack kernel (2*N*K bytes) + per-token quant of each piece (the image's own
dynamic_per_token_scaled_fp8_quant) + one torch.ops._C.cutlass_scaled_mm per piece (out = bf16(sum *
scale_token * scale_channel), fp32 accumulation: one rounding, the same class as production's bf16(sum * s)).
Persistent state per served layer: the un-permuted fp32 scale [N] (4*N bytes) - the repack itself lives in one
scratch buffer per device, sized by the largest weight seen (<= 52 MiB at TP=2: the KDA in_proj), and the
transient per-token fp8 + bf16 bias are torch allocations.

Self-tests (load time, per layer: eager, synchronizing; never while capturing):
  - layout (once per (Npad, K) shape): every byte of the repack equals the exact bf16 dequant
    (tf_fp8_large_m_ext.dequant, the kernel production's large-M path runs) cast back to fp8 - the same
    identity tests/pf3000/bench_test_c.py measured at 100.00% on real weights;
  - GEMM (every layer): per-token quantized random rows through cutlass vs production's Marlin on the same
    weights: rel_l2 <= 6e-2 (the measured W8A8-vs-Marlin class is 2.4e-2..2.7e-2; a wrong scale or a transposed
    operand is far above), every element finite.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

import torch

HERE = Path(__file__).resolve().parent
ENV = "GLM53_DENSE_W8A8"
ENV_MIN_M = "GLM53_DENSE_W8A8_MIN_M"
ENV_PIECE_MIB = "GLM53_DENSE_W8A8_PIECE_MIB"
ENV_GEMM = "GLM53_DENSE_W8A8_GEMM"
ENV_ONLY = "GLM53_DENSE_W8A8_ONLY"
ENV_FP8AG = "GLM53_DENSE_W8A8_FP8AG"     # opt-dense: fp8 sequence-parallel all-gather into a served KDA in_proj
ENV_HILO = "GLM53_DENSE_W8A8_HILO"       # opt-dense PROTOTYPE: "<group>.<proj>:<channels>,..." outlier-channel hi+lo
ENV_HILO_SEL = "GLM53_DENSE_W8A8_HILO_SEL"  # "call" (per call, default) | "first" (frozen at the first served call)
ENV_SKIP = "GLM53_DENSE_W8A8_SKIP_LAYERS"  # opt-w8a8layers: "<layer>[:<name>],..." exclusions, applied after ONLY
PROJ_NAMES = frozenset({"kda.in_proj_qkvbfg_a", "kda.o_proj", "mla.fused_qkv_a_proj", "mla.q_b_proj", "mla.o_proj",
                        "shared.gate_up_proj", "shared.down_proj", "dense.gate_up_proj", "dense.down_proj",
                        "draft.fc"})
# opt-dense: "draft.fc" = the DFlash drafter's aux-hidden combiner (GLM53_DRAFT_FP8=...,fc: group "draft", prefix
# model.fc, 4096 x 20480, run eagerly once per step on the step's tokens). Served ONLY when named in
# GLM53_DENSE_W8A8_ONLY (never by "all"): its numerics reach only the drafter's context features (acceptance), never
# the target's logits. The draft group joins STATE.groups only then.
OPT_IN = frozenset({"draft.fc"})
SKIP_GROUPS = frozenset({"kda", "mla", "dense", "shared"})
SKIP_EXTRA_NAMES = frozenset({"kda.f_b_proj", "kda.g_b_proj"})   # served when ONLY is unset, not ONLY-selectable
SKIP_MAX_LAYER = 4095
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
PROD_MODULE = "vllm.model_executor.layers.quantization.exl3"
_TRUE = frozenset({"1", "on", "true", "yes"})
_CUDA_ERR = re.compile(r"CUDA error|cudaError|illegal|capture")
DEFAULT_MIN_M = 512
DEFAULT_PIECE_MIB = 16
PIECE_ROWS = 2048            # measured optimum at M = 13824 on the big weights (docs/PF3000_KILLTESTS_FP8.md 2)
SELFTEST_TOL = 6e-2          # rel_l2(cutlass W8A8, production Marlin) at the self-test M
SELFTEST_M = 64
_log = logging.getLogger("vllm.tf_fp8_w8a8")


class _Cfg:
    def __init__(self) -> None:
        self.min_m = DEFAULT_MIN_M
        self.piece_bytes = DEFAULT_PIECE_MIB * 2 ** 20
        self.strict = False          # tests: raise instead of falling back to production's path
        self.gemm = "custom"         # w8a82: "custom" (this extension's CUTLASS GEMM) or "cutlass_mm"
        self.only = None             # w8a82: None = every projection, else frozenset of "<group>.<projection>"
        self.fp8ag = False           # opt-dense: GLM53_DENSE_W8A8_FP8AG (see the FP8 all-gather section)
        self.hilo = {}               # opt-dense prototype: "<group>.<proj>" -> residual channels (GLM53_DENSE_W8A8_HILO)
        self.hilo_sel = "call"
        self.skip = None             # opt-w8a8layers: None = no layer excluded, else tuple of (lo, hi, name|group|None)


class _State:
    def __init__(self) -> None:
        self.enabled = False
        self.disabled_reason = "not installed"
        self.ext = None
        self.orig_apply = None
        self.orig_pwal = None
        self.cls = None
        self.groups = frozenset()
        self.scratch = {}            # (device, k) -> fp8 [n_max, k] buffer (the repack target, reused per call)
        self.ws = {}                 # device -> uint8 workspace of the custom GEMM (persistent scheduler)
        self.custom = False          # w8a82: the custom GEMM is usable (extension VERSION >= 2 and not disabled)


CFG = _Cfg()
STATE = _State()
WVERDICT: dict[tuple, tuple[bool, str]] = {}   # (weight ptr, scale ptr, Npad, K) -> (passed, detail)
ALPHA: dict[tuple, torch.Tensor] = {}          # same key -> un-permuted fp32 scale [N] (+256 zeros)
BYTECHECK: dict[tuple[int, int], bool] = {}    # (Npad, K) -> layout self-test already passed on this device
GEMMCHECK: dict[tuple[int, int], bool] = {}    # (N, K) -> custom GEMM bitwise == cutlass_scaled_mm (else cutlass_mm)
WS_BYTES = 1 << 20

# w8a82 per-shape GEMM choice (N, K) -> (cfg, swizzle, raster) for M >= GEMM_BIG_M, measured on nodeC at M=13824
# (tests/w8a82/cutlass_bench.py; cfg 0 coop 128x128x128, 1 pingpong 128x128x128, 2 coop 128x256x64, 3 coop
# 256x128x64; raster 0 heuristic, 1 along M, 2 along N). Unlisted shapes: DEFAULT_GEMM.
GEMM_TABLE = {
    (12576, 4096): (3, 4, 2), (4096, 4096): (2, 8, 1), (2048, 4096): (3, 4, 0), (8192, 1536): (1, 1, 2),
    (4096, 8192): (0, 8, 1), (4096, 1024): (2, 4, 0), (12288, 4096): (1, 8, 1), (4096, 6144): (2, 8, 1),
    (4096, 20480): (3, 4, 2),                   # opt-dense draft.fc: sweep 13.90 ms @13824 / 4.55 @4289 (167/158 TF)
    # opt-dense hi+lo K + C operands (tests/optdense/sweep_hilo_gemm.py, best at M=13824)
    (4096, 4352): (2, 8, 1), (4096, 8704): (2, 4, 1), (8192, 1792): (2, 8, 1), (4096, 6656): (2, 4, 1),
    (4096, 1152): (1, 4, 2),
}
GEMM_TABLE_SMALL = {                            # M < GEMM_BIG_M: pingpong 128x128x128 (sweep at M 512/1791)
    (12576, 4096): (1, 2, 0), (4096, 4096): (1, 2, 1), (2048, 4096): (1, 8, 0), (8192, 1536): (1, 4, 1),
    (4096, 8192): (1, 1, 0), (4096, 1024): (1, 4, 2), (12288, 4096): (1, 2, 0), (4096, 6144): (1, 4, 1),
}
GEMM_BIG_M = 3072
DEFAULT_GEMM = (0, 8, 1)
COUNTERS: dict[str, int] = {}
_LOGGED: set = set()


def _count(key: str, n: int = 1) -> None:
    COUNTERS[key] = COUNTERS.get(key, 0) + n


def _log_once(key: str, level: int, msg: str, *a) -> None:
    if key not in _LOGGED:
        _LOGGED.add(key)
        _log.log(level, msg, *a)


def env_enabled(environ: dict | None = None) -> bool:
    v = (os.environ if environ is None else environ).get(ENV)
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
    """AOT module tf_fp8_w8a8_ext (setup.py) or, with TF_EXL3_JIT=1 (nodeC tests), a JIT build of the .cu."""
    if STATE.ext is not None:
        return STATE.ext
    try:
        import tf_fp8_w8a8_ext as m
    except ImportError as err:
        if os.environ.get("TF_EXL3_JIT", "0").strip().lower() not in _TRUE:
            raise ImportError(f"tf_fp8_w8a8_ext (AOT) not importable and TF_EXL3_JIT is off: {err!r}")
        from torch.utils.cpp_extension import load
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1a")
        inc = _include_shim()
        cl = "/usr/local/lib/python3.12/dist-packages/flashinfer/data/cutlass"
        cinc = [f"-I{cl}/include", f"-I{cl}/tools/util/include"]
        m = load(name="tf_fp8_w8a8_jit2", sources=[str(HERE / "kernels" / "fp8_w8a8.cu")],
                 extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-DNDEBUG", *inc, *cinc],
                 extra_cflags=["-O3", *inc, *cinc], verbose=False)
    STATE.ext = m
    return m


def _scratch(n: int, k: int, device) -> torch.Tensor:
    """The repack target for this call: one fp8 [>= n, k] buffer per (device, K), grown to the largest weight of
    that K (<= 52 MiB at TP=2: the KDA in_proj; ~0.12 GiB over the five K values). Serial per stream: the buffer is
    written and consumed within one call, before the next one is enqueued."""
    key = (str(device), k)
    buf = STATE.scratch.get(key)
    if buf is None or buf.shape[0] < n:
        buf = torch.empty(max(n, 128), k, dtype=torch.float8_e4m3fn, device=device)
        STATE.scratch[key] = buf
    return buf[:n]


def quant_per_token(x2d: torch.Tensor):
    """The image's own per-token e4m3 quantization (dynamic_per_token_scaled_fp8_quant) -> (fp8 [m, k], fp32 [m, 1])."""
    import vllm._custom_ops as ops
    q, s = ops.scaled_fp8_quant(x2d, use_per_token_if_dynamic=True)
    return q, (s if s.dim() == 2 else s.unsqueeze(1)).float().contiguous()


def large_alpha(weight_scale, n: int) -> torch.Tensor:
    """The per-channel scale Marlin multiplies by, un-permuted, as fp32 [N] (+256 zeros) - fp8_gemv.large_alpha."""
    import fp8_gemv as G
    return G.large_alpha(weight_scale, n)


def _ineligible(layer, n: int, k: int) -> Optional[str]:
    """None if this layer's stored state is what this module reads, else the reason."""
    w = layer.weight
    if w.dtype != torch.int32 or w.dim() != 2 or not w.is_contiguous() or w.shape[0] * 16 != k:
        return "weight_layout"
    npad = w.shape[1] // 4
    if npad % 64 or not (0 <= npad - n < 128):
        return "weight_layout"
    if n % 16 or k % 16:                       # cutlass_scaled_mm's operand alignment (every production shape has it)
        return "gemm_alignment"
    ws = layer.weight_scale
    if ws.dtype != torch.bfloat16 or ws.numel() != npad or not ws.is_contiguous():
        return "scale_layout"
    if not hasattr(layer, "workspace"):
        return "no_workspace"
    return None


def _key(weight, weight_scale, k: int):
    return (weight.data_ptr(), weight_scale.data_ptr(), weight.shape[1] // 4, k)


def repack(out: torch.Tensor, weight, n: int, k: int) -> None:
    """Byte-exact Marlin -> standard row-major fp8 into out [n, k]."""
    ext().fp8_marlin_to_std(out, weight, n, k)


def _dequant_reference(weight, n: int, k: int, npad: int) -> torch.Tensor:
    """The exact bf16 dequant of the Marlin payload (production's large-M kernel) cast back to fp8: the
    identity the layout self-test compares against (100.00% on real weights, tests/pf3000/bench_test_c.py)."""
    import fp8_gemv as G
    L = G.ext_large()
    wtmp = torch.empty(n, k, dtype=torch.bfloat16, device=weight.device)
    for n0 in range(0, n, 2048):
        c = min(2048, n - n0)
        L.dequant(wtmp[n0:n0 + c], weight, n0, c, k)
    return wtmp.to(torch.float8_e4m3fn)


def selftest(layer, n: int, k: int, bias=None, label: str = "") -> tuple[bool, str]:
    """Layout (once per shape) + GEMM self-test vs production's Marlin on seeded random rows. Eager only."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        apply_fp8_marlin_linear,
    )

    weight, weight_scale = layer.weight, layer.weight_scale
    key = _key(weight, weight_scale, k)
    npad = weight.shape[1] // 4
    detail = []
    try:
        repack_ok = BYTECHECK.get((npad, k))
        if repack_ok is None:
            ref = _dequant_reference(weight, n, k, npad)
            mine = _scratch(n, k, weight.device)
            repack(mine, weight, n, k)
            same = (mine.view(torch.uint8) == ref.view(torch.uint8)).all().item()
            del ref
            repack_ok = bool(same)
            BYTECHECK[(npad, k)] = repack_ok
            if not repack_ok:
                WVERDICT[key] = (False, "repack bytes differ from the exact bf16 dequant")
                return WVERDICT[key]
            detail.append("layout ok")
        alpha = ALPHA.get(key)
        if alpha is None:
            alpha = large_alpha(weight_scale, n)
            if not bool(torch.isfinite(alpha).all().item()) or not bool((alpha[:n] > 0).all().item()):
                WVERDICT[key] = (False, "per-channel scale not finite / not positive")
                return WVERDICT[key]
            ALPHA[key] = alpha
        g = torch.Generator(device=weight.device).manual_seed(5678)
        if STATE.custom and (n, k) not in GEMMCHECK:
            GEMMCHECK[(n, k)] = _gemm_bitwise(weight, alpha, n, k, g)
            detail.append("custom GEMM " + ("== cutlass_scaled_mm" if GEMMCHECK[(n, k)] else
                                            "!= cutlass_scaled_mm -> cutlass_mm for this shape"))
            if not GEMMCHECK[(n, k)]:
                _log.warning("tf_fp8_w8a8: [%dx%d] custom GEMM output differs from cutlass_scaled_mm; this shape uses "
                             "cutlass_scaled_mm", n, k)
        x = torch.randn(SELFTEST_M, k, device=weight.device, generator=g).to(torch.bfloat16)
        ref = apply_fp8_marlin_linear(input=x, weight=weight, weight_scale=weight_scale,
                                      workspace=layer.workspace, size_n=n, size_k=k, bias=bias)
        y = w8a8_forward(x.reshape(-1, k), weight, weight_scale, n, k, bias, layer_key=key)
        yf, rf = y.float(), ref.float()
        d = (yf - rf).norm().item() / max(rf.norm().item(), 1e-30)
        fin = bool(torch.isfinite(yf).all().item())
        del x, ref, y, yf, rf
    except Exception as exc:  # noqa: BLE001 - a self-test that cannot run is a failed self-test
        if _CUDA_ERR.search(str(exc)):
            raise
        WVERDICT[key] = (False, f"self-test raised {type(exc).__name__}: {str(exc)[:160]}")
        _log.warning("tf_fp8_w8a8: %s [%dx%d] self-test raised, layer stays on production's path: %s",
                     label or "layer", n, k, WVERDICT[key][1])
        _count("selftest_failed")
        return WVERDICT[key]
    ok = fin and d <= SELFTEST_TOL
    detail.append(f"rel_l2 vs Marlin {d:.2e} (tol {SELFTEST_TOL:.0e})")
    WVERDICT[key] = (ok, ", ".join(detail))
    if ok:
        _log_once(f"st:{npad}x{k}", logging.INFO, "tf_fp8_w8a8: %s [%dx%d] self-test passed: %s", label or "layer",
                  n, k, WVERDICT[key][1])
    else:
        _log.warning("tf_fp8_w8a8: %s [%dx%d] self-test FAILED, this layer stays on production's path: %s",
                     label or "layer", n, k, WVERDICT[key][1])
    _count("selftest_passed" if ok else "selftest_failed")
    return WVERDICT[key]


def gemm_choice(n: int, k: int, m: int) -> tuple[int, int, int]:
    if m < GEMM_BIG_M and (n, k) in GEMM_TABLE_SMALL:
        return GEMM_TABLE_SMALL[(n, k)]
    return GEMM_TABLE.get((n, k), DEFAULT_GEMM)


def _ws(device) -> torch.Tensor:
    key = str(device)
    buf = STATE.ws.get(key)
    if buf is None:
        buf = torch.empty(WS_BYTES, dtype=torch.uint8, device=device)
        STATE.ws[key] = buf
    return buf


def custom_gemm(out, q, w8, sa, alpha, n: int, k: int, m: int) -> None:
    cfg, sw, ro = gemm_choice(n, k, m)
    ext().fp8_w8a8_gemm(out, q, w8, sa.view(-1), alpha, cfg, sw, ro, _ws(q.device))


def use_custom(n: int, k: int) -> bool:
    return STATE.custom and GEMMCHECK.get((n, k), False)


def _gemm_bitwise(weight, alpha, n: int, k: int, g) -> bool:
    """The custom GEMM must reproduce cutlass_scaled_mm bit for bit on this layer's weights (same epilogue arithmetic,
    same fp32 accumulation order) at a small and a multi-tile M with the shape's configured cfg; any raise or
    difference -> False (the shape stays on cutlass_scaled_mm)."""
    try:
        w8 = _scratch(n, k, weight.device)
        repack(w8, weight, n, k)
        for mm in (SELFTEST_M, GEMM_BIG_M + 136):
            xs = torch.randn(mm, k, device=weight.device, generator=g).to(torch.bfloat16)
            q, sa = quant_per_token(xs)
            ref = torch.empty(mm, n, dtype=torch.bfloat16, device=weight.device)
            torch.ops._C.cutlass_scaled_mm(ref, q, w8.t(), sa, alpha[:n].view(n, 1), None)
            out = torch.empty_like(ref)
            custom_gemm(out, q, w8, sa, alpha, n, k, mm)
            if not torch.equal(out, ref):
                return False
        return True
    except Exception as exc:  # noqa: BLE001
        if _CUDA_ERR.search(str(exc)) and "can_implement" not in str(exc):
            raise
        _log.warning("tf_fp8_w8a8: [%dx%d] custom GEMM check raised %r", n, k, exc)
        return False


def piece_rows(n: int, k: int, m: int) -> int:
    """Rows per cutlass call: the whole M for a small weight, PIECE_ROWS for a big one."""
    if n * k > CFG.piece_bytes:
        return PIECE_ROWS
    return max(m, 1)


def w8a8_forward(x2d: torch.Tensor, weight, weight_scale, n: int, k: int, bias=None,
                 layer_key: tuple | None = None, pre: tuple | None = None) -> torch.Tensor:
    """The W8A8 linear for x2d [M, K] bf16 contiguous: repack -> per-token quant per piece -> cutlass_scaled_mm.
    pre = (fp8 [M, K], fp32 [M, 1]): x2d's per-token quantization, already done (the FP8 all-gather); x2d is then
    only a shape/device carrier and is never read."""
    m = x2d.shape[0]
    key = layer_key or _key(weight, weight_scale, k)
    alpha = ALPHA[key]
    w8 = _scratch(n, k, x2d.device)
    repack(w8, weight, n, k)
    wt = w8.t()
    sb = alpha[:n].view(n, 1)
    out = torch.empty(m, n, dtype=torch.bfloat16, device=x2d.device)
    if use_custom(n, k):
        q, sa = pre if pre is not None else quant_per_token(x2d)
        custom_gemm(out, q, w8, sa, alpha, n, k, m)
        if bias is not None:
            import fp8_gemv as G
            out += bias.reshape(-1)[G._perm(bias.numel(), bias.device)[:n]]
        return out
    pr = piece_rows(n, k, m)
    for r0 in range(0, m, pr):
        r = min(pr, m - r0)
        if pre is not None:
            q, sa = pre[0][r0:r0 + r], pre[1][r0:r0 + r]
        else:
            q, sa = quant_per_token(x2d[r0:r0 + r])
        torch.ops._C.cutlass_scaled_mm(out[r0:r0 + r], q, wt, sa, sb, None)
    if bias is not None:
        import fp8_gemv as G
        out += bias.reshape(-1)[G._perm(bias.numel(), bias.device)[:n]]
    return out


HILO_WN2: dict[tuple, torch.Tensor] = {}       # layer key -> ||W_k||^2 per input channel (fp32 [K])
HILO_SEL: dict[tuple, torch.Tensor] = {}       # layer key -> frozen channel set (GLM53_DENSE_W8A8_HILO_SEL=first)


def hilo_forward(x2d, weight, weight_scale, n: int, k: int, bias, key, c: int) -> torch.Tensor:
    """PROTOTYPE (quality measurement only, unoptimized torch ops): W8A8 plus the e4m3 residual of the c input
    channels with the largest residual error energy sum_t r_tk^2 ||W_k||^2, as c extra K columns of ONE GEMM:
        y = s_tok * s_ch * ([q(x) | q(r_S)] . [W8 | W8_S])     (r = x - q(x) s_tok, q(r_S) = e4m3(r_S / s_tok))
    The same per-token scale serves both halves, so the existing epilogue is unchanged."""
    alpha = ALPHA[key]
    w8 = _scratch(n, k, x2d.device)
    repack(w8, weight, n, k)
    wn2 = HILO_WN2.get(key)
    if wn2 is None:
        wn2 = HILO_WN2[key] = (w8.float() * alpha[:n, None]).pow(2).sum(0)
    q, sa = quant_per_token(x2d)
    r = x2d.float() - q.float() * sa
    S = HILO_SEL.get(key) if CFG.hilo_sel == "first" else None
    if S is None:
        S = torch.sort(torch.topk(r.pow(2).sum(0) * wn2, c).indices).values
        if CFG.hilo_sel == "first" and _real_call():       # never freeze on the profile run's dummy rows
            HILO_SEL[key] = S.to(torch.int32)
    S = S.long()
    qr = (r[:, S] / sa).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    a2 = torch.cat([q, qr], 1).contiguous()
    w2 = torch.cat([w8, w8.view(torch.uint8)[:, S].view(torch.float8_e4m3fn)], 1).contiguous()
    out = torch.empty(x2d.shape[0], n, dtype=torch.bfloat16, device=x2d.device)
    torch.ops._C.cutlass_scaled_mm(out, a2, w2.t(), sa, alpha[:n].view(n, 1), None)
    if bias is not None:
        import fp8_gemv as G
        out += bias.reshape(-1)[G._perm(bias.numel(), bias.device)[:n]]
    _count("hilo_calls")
    return out


_HILO_KERNEL = {}


def _hilo_kernel():
    """Triton: per row, the per-token e4m3 quantization of x (scale = max(amax/448, 1/(448*512)), q = e4m3(clamp(x /
    scale))) into a2[:, :K] and the e4m3 residual of the selected channels, e4m3(clamp((x_S - q_S scale) / scale)),
    into a2[:, K:K+C] - the GEMM operand [q(x) | q(r_S)] in one pass over x."""
    if "k" in _HILO_KERNEL:
        return _HILO_KERNEL["k"]
    import triton
    import triton.language as tl

    @triton.jit
    def hilo_quant(X, S, A, SC, K: tl.constexpr, C: tl.constexpr, LDA: tl.constexpr, BK: tl.constexpr,
                   BC: tl.constexpr):
        # IEEE divisions (div_rn) throughout: bit for bit the image's dynamic per-token quant (scale = amax / 448,
        # q = e4m3(clamp(x / scale)); tests/optdense/probe_quant_bits*.py) and the prototype's residual formula.
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BK)
        msk = offs < K
        x = tl.load(X + row * K + offs, mask=msk, other=0.0).to(tl.float32)
        amax = tl.max(tl.abs(x), axis=0)
        sc = tl.maximum(tl.math.div_rn(amax, 448.0), 1.0 / (448.0 * 512.0))
        y = tl.math.div_rn(x, tl.zeros_like(x) + sc)
        y = tl.minimum(tl.maximum(y, -448.0), 448.0)
        tl.store(A + row * LDA + offs, y.to(tl.float8e4nv), mask=msk)
        tl.store(SC + row, sc)
        jo = tl.arange(0, BC)
        jm = jo < C
        idx = tl.load(S + jo, mask=jm, other=0)
        xs = tl.load(X + row * K + idx, mask=jm, other=0.0).to(tl.float32)
        scv = tl.zeros_like(xs) + sc
        ys = tl.math.div_rn(xs, scv)
        qs = tl.minimum(tl.maximum(ys, -448.0), 448.0).to(tl.float8e4nv).to(tl.float32)
        r = xs - qs * sc
        qr = tl.minimum(tl.maximum(tl.math.div_rn(r, scv), -448.0), 448.0)
        tl.store(A + row * LDA + K + jo, qr.to(tl.float8e4nv), mask=jm)

    _HILO_KERNEL["k"] = hilo_quant
    return hilo_quant


def hilo_quant(x2d: torch.Tensor, S: torch.Tensor):
    """(a2 fp8 [M, K + C] = [q(x) | q(r_S)], scale fp32 [M, 1]) in one Triton pass."""
    import triton
    m, k = x2d.shape
    c = S.numel()
    a2 = torch.empty(m, k + c, dtype=torch.float8_e4m3fn, device=x2d.device)
    sa = torch.empty(m, 1, dtype=torch.float32, device=x2d.device)
    bk = triton.next_power_of_2(k)
    _hilo_kernel()[(m,)](x2d, S, a2, sa, K=k, C=c, LDA=k + c, BK=bk, BC=triton.next_power_of_2(c),
                         num_warps=8 if bk >= 4096 else 4, enable_fp_fusion=False)   # r = x - q*s: no FMA
    return a2, sa


def hilo_fast_ok() -> bool:
    try:
        return bool(getattr(ext(), "REPACK_LD", 0)) and STATE.custom
    except Exception:  # noqa: BLE001
        return False


def _real_call() -> bool:
    """False inside vLLM's dummy/profile forward (no attention metadata): never freeze a channel set on it."""
    try:
        from vllm.forward_context import get_forward_context, is_forward_context_available
        if not is_forward_context_available():
            return True
        return get_forward_context().attn_metadata is not None
    except Exception:  # noqa: BLE001
        return True


def hilo_select(x2d, w8n, alpha, n: int, k: int, c: int, key) -> torch.Tensor:
    """The c channels with the largest residual error energy sum_t r_tk^2 ||W_k||^2 on this call's rows (sorted
    int32), computed once per layer (row chunks keep the fp32 temporaries small)."""
    wn2 = HILO_WN2.get(key)
    if wn2 is None:
        wn2 = HILO_WN2[key] = (w8n.float() * alpha[:n, None]).pow(2).sum(0)
    e = torch.zeros(k, dtype=torch.float32, device=x2d.device)
    for r0 in range(0, x2d.shape[0], 2048):
        xs = x2d[r0:r0 + 2048]
        q, sa = quant_per_token(xs)
        e += (xs.float() - q.float() * sa).pow(2).sum(0)
    return torch.sort(torch.topk(e * wn2, c).indices).values.to(torch.int32)


def _scratch2(n: int, k2: int, device) -> torch.Tensor:
    key = (str(device), "hilo", k2)
    buf = STATE.scratch.get(key)
    if buf is None or buf.shape[0] < n:
        buf = torch.empty(max(n, 128), k2, dtype=torch.float8_e4m3fn, device=device)
        STATE.scratch[key] = buf
    return buf[:n]


HILO_GEMMCHECK: dict[tuple[int, int], bool] = {}


def hilo_forward_fast(x2d, weight, weight_scale, n: int, k: int, bias, key, c: int):
    """The hi+lo W8A8 linear: repack straight into [W | .] (row stride K + C), W_S copied into the last C columns,
    one Triton pass for [q(x) | q(r_S)], one custom GEMM with K + C. The channel set is frozen at the layer's first
    real (non-profile) served call. Returns None to fall back to plain W8A8 (profile run before the freeze)."""
    S = HILO_SEL.get(key)
    k2 = k + c
    w2 = _scratch2(n, k2, x2d.device)
    repack(w2[:, :k], weight, n, k)
    if S is None:
        if not _real_call():
            return None
        S = HILO_SEL[key] = hilo_select(x2d, w2[:, :k], ALPHA[key], n, k, c, key)
        _log_once(f"hilo:{n}x{k}", logging.INFO, "tf_fp8_w8a8: hi+lo [%dx%d] channel set frozen at the first real "
                  "call (M=%d): %d of %d input channels carry their e4m3 residual", n, k, x2d.shape[0], c, k)
    alpha = ALPHA[key]
    w2u = w2.view(torch.uint8)
    w2u[:, k:].copy_(w2u[:, :k].index_select(1, S.long()))
    a2, sa = hilo_quant(x2d, S)
    m = x2d.shape[0]
    out = torch.empty(m, n, dtype=torch.bfloat16, device=x2d.device)
    kc = k2 if (n, k2) in GEMM_TABLE else k          # tile config: the K + C sweep, else the base shape's
    if (n, k2) not in HILO_GEMMCHECK:
        ref = torch.empty_like(out[:min(m, 4096)])
        mm = ref.shape[0]
        torch.ops._C.cutlass_scaled_mm(ref, a2[:mm], w2.t(), sa[:mm], alpha[:n].view(n, 1), None)
        tst = torch.empty_like(ref)
        custom_gemm(tst, a2[:mm], w2, sa[:mm], alpha, n, kc, mm)
        HILO_GEMMCHECK[(n, k2)] = bool(torch.equal(tst, ref))
    if HILO_GEMMCHECK[(n, k2)]:
        custom_gemm(out, a2, w2, sa, alpha, n, kc, m)
    else:
        torch.ops._C.cutlass_scaled_mm(out, a2, w2.t(), sa, alpha[:n].view(n, 1), None)
    if bias is not None:
        import fp8_gemv as G
        out += bias.reshape(-1)[G._perm(bias.numel(), bias.device)[:n]]
    _count("hilo_fast_calls")
    return out


def try_w8a8(x, layer, bias, n: int, k: int, pre: tuple | None = None, hilo: int = 0):
    """This call through the W8A8 path, or None when it declines (the caller then runs production's path).
    pre: x's per-token quantization done by the FP8 all-gather (x itself is then an unwritten placeholder)."""
    if not STATE.enabled or x.dim() == 0 or x.shape[-1] != k:
        return None
    m = x.numel() // k
    if m < CFG.min_m:
        _count("prod_m")
        return None
    if torch.cuda.is_current_stream_capturing():     # decode graphs are <= 64 tokens; never serve a capture
        _count("prod_capture")
        return None
    why = _ineligible(layer, n, k)
    if why is not None:
        _count("prod_" + why)
        _log_once("inel:" + why, logging.WARNING, "tf_fp8_w8a8: a call is not eligible (%s, layer [%dx%d]): "
                  "production's path serves it (counted, not repeated)", why, n, k)
        return None
    x2d = x.reshape(-1, k)
    if x2d.dtype != torch.bfloat16 or not x2d.is_cuda or x2d.device != layer.weight.device:
        _count("prod_x_layout")
        return None
    if not x2d.is_contiguous():
        if pre is not None:
            return None
        x2d = x2d.contiguous()
    if bias is not None and (bias.dtype != torch.bfloat16 or bias.numel() != layer.weight.shape[1] // 4):
        _count("prod_bias_layout")
        return None
    key = _key(layer.weight, layer.weight_scale, k)
    v = WVERDICT.get(key)
    if v is None:
        v = selftest(layer, n, k, bias, label="lazily tested layer")
    if not v[0]:
        _count("prod_selftest_failed")
        return None
    try:
        y = None
        if hilo and pre is None and hilo < k:
            if CFG.hilo_sel == "first" and hilo_fast_ok():
                y = hilo_forward_fast(x2d, layer.weight, layer.weight_scale, n, k, bias, key, hilo)
            else:
                y = hilo_forward(x2d, layer.weight, layer.weight_scale, n, k, bias, key, hilo)
        if y is None:
            y = w8a8_forward(x2d, layer.weight, layer.weight_scale, n, k, bias, layer_key=key, pre=pre)
    except Exception as exc:
        if CFG.strict or not isinstance(exc, (RuntimeError, ValueError, TypeError, IndexError)) or \
                _CUDA_ERR.search(str(exc)):
            raise
        _count("prod_error")
        _log_once("err:" + str(exc)[:80], logging.WARNING, "tf_fp8_w8a8: W8A8 path failed before launch (%s: %s): "
                  "production's path serves this call (identical failures are counted, not repeated)",
                  type(exc).__name__, str(exc)[:200])
        return None
    _count("w8a8_eager")
    _count("w8a8_rows", m)
    _log_once("first", logging.INFO, "tf_fp8_w8a8: first call served (M=%d, min M %d, %s; per-token quant; the "
              "repack is per call, no resident weight copy)", m, CFG.min_m,
              f"custom CUTLASS GEMM cfg {gemm_choice(n, k, m)}" if use_custom(n, k) else
              f"cutlass fp8 pieces of {piece_rows(n, k, m)} rows")
    return y.reshape(x.shape[:-1] + (n,))


# ---------------------------------------------------------------------------------------------------------------
def _proj_name(method) -> str:
    return f"{method.group}.{(getattr(method, 'prefix', '') or '').rsplit('.', 1)[-1]}"


def parse_skip(raw: str) -> Optional[tuple]:
    """GLM53_DENSE_W8A8_SKIP_LAYERS -> None (unset/empty) or a tuple of (lo, hi, sel) with sel None (every projection),
    a group name or a "<group>.<projection>" name. Raises ValueError with the reason (the value itself is not secret,
    but the message names only the offending item)."""
    raw = (raw or "").strip()
    if not raw:
        return None
    items = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        rng, sep, sel = tok.partition(":")
        m = re.fullmatch(r"(\d{1,4})(?:-(\d{1,4}))?", rng.strip())
        if not m:
            raise ValueError(f"item {tok!r}: the layer part must be N or A-B")
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2) is not None else lo
        if lo > hi or hi > SKIP_MAX_LAYER:
            raise ValueError(f"item {tok!r}: need 0 <= A <= B <= {SKIP_MAX_LAYER}")
        sel = sel.strip() if sep else None
        if sep and sel not in PROJ_NAMES and sel not in SKIP_EXTRA_NAMES and sel not in SKIP_GROUPS:
            raise ValueError(f"item {tok!r}: unknown projection/group (known: {sorted(PROJ_NAMES | SKIP_EXTRA_NAMES)} "
                             f"or a group {sorted(SKIP_GROUPS)})")
        items.append((lo, hi, sel))
    if not items:
        raise ValueError("no item")
    return tuple(items)


def layer_of(prefix: str) -> Optional[int]:
    """The model layer index of a vLLM prefix ("model.layers.12.self_attn.o_proj" -> 12), None if it has none."""
    m = _LAYER_RE.search(prefix or "")
    return int(m.group(1)) if m else None


def skipped(method) -> bool:
    """GLM53_DENSE_W8A8_SKIP_LAYERS filter: True if this projection's layer is excluded (False when unset)."""
    if CFG.skip is None:
        return False
    pre = getattr(method, "prefix", "") or ""
    lay = layer_of(pre)
    if lay is None:
        return False
    grp = getattr(method, "group", "")
    name = f"{grp}.{pre.rsplit('.', 1)[-1]}"
    for lo, hi, sel in CFG.skip:
        if lo <= lay <= hi and (sel is None or sel == name or sel == grp):
            return True
    return False


def selected(method) -> bool:
    """GLM53_DENSE_W8A8_ONLY filter (None = everything the groups select), then GLM53_DENSE_W8A8_SKIP_LAYERS."""
    if CFG.only is not None:
        pre = getattr(method, "prefix", "") or ""
        if f"{method.group}.{pre.rsplit('.', 1)[-1]}" not in CFG.only:
            return False
    return CFG.skip is None or not skipped(method)


def _make_apply(orig_apply):
    def serve(self, layer, x, bias):
        if not STATE.enabled or not getattr(self, "ready", False) or \
                getattr(layer, "glm53_bf16_lm_w", None) is not None or \
                self.group not in STATE.groups or not hasattr(layer, "glm53_fp8_n") or not selected(self):
            return orig_apply(self, layer, x, bias)
        n, k = int(layer.glm53_fp8_n), int(layer.glm53_fp8_k)
        hc = CFG.hilo.get(_proj_name(self), 0) if CFG.hilo else 0
        y = try_w8a8(x, layer, bias, n, k, hilo=hc)
        return y if y is not None else orig_apply(self, layer, x, bias)

    def apply(self, layer, x, bias=None):
        if torch.compiler.is_compiling():
            return orig_apply(self, layer, x, bias)      # never inside a traced graph (the drafter)
        if AG.pending is not None:                       # opt-dense FP8 all-gather: x may be its placeholder
            return _apply_pending(serve, orig_apply, self, layer, x, bias)
        return serve(self, layer, x, bias)

    apply.__doc__ = getattr(orig_apply, "__doc__", None)
    apply._tf_w8a8_hook = True
    apply._tf_w8a8_orig = orig_apply
    return apply


def _make_pwal(orig_pwal):
    def process_weights_after_loading(self, layer):
        orig_pwal(self, layer)                     # production first; its exceptions propagate unchanged
        if CFG.fp8ag and STATE.enabled and not AG.tried:
            fp8ag_install()
        try:
            if (STATE.enabled and getattr(self, "ready", False) and self.group in STATE.groups and
                    getattr(layer, "glm53_bf16_lm_w", None) is None and hasattr(layer, "glm53_fp8_n")):
                if selected(self):
                    n, k = int(layer.glm53_fp8_n), int(layer.glm53_fp8_k)
                    label = f"{getattr(self, 'group', '?')} {getattr(self, 'prefix', '')}".strip()
                    selftest(layer, n, k, getattr(layer, "bias", None), label)
                elif CFG.skip is not None and skipped(self):
                    _count("skip_layers_at_load")     # excluded by GLM53_DENSE_W8A8_SKIP_LAYERS (never served)
        except Exception as exc:  # noqa: BLE001 - never fail the load; the layer stays on production's path
            if _CUDA_ERR.search(str(exc)):
                raise
            _log.warning("tf_fp8_w8a8: self-test at load raised %r (layer stays on production's path)", exc)

    process_weights_after_loading.__doc__ = getattr(orig_pwal, "__doc__", None)
    process_weights_after_loading._tf_w8a8_hook = True
    process_weights_after_loading._tf_w8a8_orig = orig_pwal
    return process_weights_after_loading


def _refuse(report: dict, reason: str) -> dict:
    report["reason"] = reason
    STATE.disabled_reason = reason
    _log.warning("tf_fp8_w8a8 enabled (%s) but NOT installed: %s; production's dense FP8 path unchanged", ENV, reason)
    # opt-dense-rev: the FP8 all-gather is a PAIRED collective (a per-layer CPU MIN vote, then 2 fp8+scale gathers
    # instead of 1 bf16 gather). A rank whose W8A8 install is refused never wraps sp_all_gather, so it never votes:
    # a peer that did install it blocks in the vote at its first sequence-parallel forward (vLLM's profile run) =
    # a boot hang, not a silent fallback. Say so loudly; the kit's boot_checks must pair-gate on
    # "FP8 all-gather installed" appearing in BOTH ranks' logs (like mhcsp2).
    if os.environ.get(ENV_FP8AG, "").strip().lower() in _TRUE:
        report["fp8ag_refused"] = True
        _log.error("tf_fp8_w8a8: %s=1 but the W8A8 path is NOT installed on this rank (%s): this rank will not join "
                   "the FP8 all-gather vote; if the peer rank installed it, the first sequence-parallel forward HANGS. "
                   "Turn %s off on both ranks or fix the cause.", ENV_FP8AG, reason, ENV_FP8AG)
    return report


def _problem() -> Optional[str]:
    """None if the W8A8 path can be enabled here, else the reason. Loads its extension."""
    raw = os.environ.get(ENV_MIN_M, "").strip()
    if raw:
        try:
            v = int(raw)
        except ValueError:
            return f"{ENV_MIN_M}={raw!r} is not an integer"
        if not 1 <= v <= 2 ** 24:
            return f"{ENV_MIN_M}={raw!r} is not in 1..16777216"
        CFG.min_m = v
    raw = os.environ.get(ENV_PIECE_MIB, "").strip()
    if raw:
        try:
            v = int(raw)
        except ValueError:
            return f"{ENV_PIECE_MIB}={raw!r} is not an integer"
        if not 0 <= v <= 65536:
            return f"{ENV_PIECE_MIB}={raw!r} is not in 0..65536"
        CFG.piece_bytes = v * 2 ** 20
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (12, 0):
        return "needs an SM12x GPU (GB10: sm_121a)"
    try:
        E = ext()
    except Exception as exc:  # noqa: BLE001
        return f"extension not importable: {exc!r}"
    if getattr(E, "VERSION", None) not in (1, 2):
        return f"extension version {getattr(E, 'VERSION', None)!r} not in (1, 2)"
    raw = os.environ.get(ENV_GEMM, "").strip().lower()
    if raw not in ("", "custom", "cutlass_mm"):
        return f"{ENV_GEMM}={raw!r} is not custom|cutlass_mm"
    CFG.gemm = raw or "custom"
    raw = os.environ.get(ENV_ONLY, "").strip()
    if raw:
        names = frozenset(t.strip() for t in raw.split(",") if t.strip())
        bad = sorted(names - PROJ_NAMES)
        if bad or not names:
            return f"{ENV_ONLY}: unknown projection(s) {bad} (known: {sorted(PROJ_NAMES)})"
        CFG.only = names
    else:
        CFG.only = None
    raw = os.environ.get(ENV_FP8AG, "").strip().lower()
    if raw not in ("", "0", "off", "false", "no") and raw not in _TRUE:
        return f"{ENV_FP8AG}={raw!r} is not 0|1"
    CFG.fp8ag = raw in _TRUE
    raw = os.environ.get(ENV_HILO, "").strip()
    CFG.hilo = {}
    for tok in (t.strip() for t in raw.split(",") if t.strip()):
        name, _, c = tok.partition(":")
        if name not in PROJ_NAMES or not c.isdigit() or int(c) % 16 or not 16 <= int(c) <= 2048:
            return f"{ENV_HILO}: bad entry {tok!r} (<group>.<proj>:<channels, multiple of 16, 16..2048>)"
        CFG.hilo[name] = int(c)
    CFG.hilo_sel = os.environ.get(ENV_HILO_SEL, "call").strip().lower() or "call"
    if CFG.hilo_sel not in ("call", "first"):
        return f"{ENV_HILO_SEL}={CFG.hilo_sel!r} is not call|first"
    try:
        CFG.skip = parse_skip(os.environ.get(ENV_SKIP, ""))
    except ValueError as exc:
        CFG.skip = None
        return f"{ENV_SKIP}: {exc}"
    STATE.custom = CFG.gemm == "custom" and hasattr(E, "fp8_w8a8_gemm")
    try:
        import vllm._custom_ops as ops
        if not hasattr(ops, "scaled_fp8_quant"):
            return "vllm._custom_ops.scaled_fp8_quant missing"
        cap = torch.cuda.get_device_capability()
        if not torch.ops._C.cutlass_scaled_mm_supports_fp8(cap[0] * 10 + cap[1]):
            return f"cutlass_scaled_mm does not support fp8 on sm_{cap[0]}{cap[1]}"
    except Exception as exc:  # noqa: BLE001
        if _CUDA_ERR.search(str(exc)):
            raise
        return f"the cutlass fp8 path is not usable here: {exc!r}"
    try:
        prodmod_groups()
    except Exception as exc:  # noqa: BLE001
        return f"cannot read the dense FP8 groups: {exc!r}"
    return None


def prodmod_groups() -> frozenset:
    """The dense FP8 groups production routes (GLM53_DENSE_FP8), from production's own module."""
    import importlib
    mod = importlib.import_module(PROD_MODULE)
    return frozenset(mod._glm53_dense_fp8_groups())


def install(prodmod=None, *, force: bool | None = None) -> dict:
    """Wrap Glm53DenseFp8Method.apply / process_weights_after_loading. No-op unless GLM53_DENSE_W8A8 is enabled
    (or force=True). Install AFTER fp8_gemv (integrate.plugin_register order): this wrapper must be the outer one
    so a served call never reaches the large-M path. Idempotent. Never raises."""
    report: dict = {"installed": False, "reason": None}
    on = env_enabled() if force is None else bool(force)
    if not on:
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
        why = _problem()
        if why is not None:
            return _refuse(report, why)
        try:
            STATE.groups = prodmod_groups()
        except Exception as exc:  # noqa: BLE001
            return _refuse(report, f"cannot read the dense FP8 groups: {exc!r}")
        if CFG.only and any(n.startswith("draft.") for n in CFG.only):
            STATE.groups = STATE.groups | {"draft"}       # opt-dense: the drafter fc, only when named
        if not STATE.groups:
            return _refuse(report, "GLM53_DENSE_FP8 selects no group")
        if getattr(cls.apply, "_tf_w8a8_hook", False):
            STATE.enabled = True
            report.update(installed=True, reason="already installed")
            return report
        STATE.cls, STATE.orig_apply, STATE.orig_pwal = cls, cls.apply, cls.process_weights_after_loading
        cls.apply = _make_apply(cls.apply)
        cls.process_weights_after_loading = _make_pwal(cls.process_weights_after_loading)
        STATE.enabled = True
        STATE.disabled_reason = None
    except Exception as exc:  # noqa: BLE001
        return _refuse(report, f"install raised {exc!r}")
    report.update(installed=True, reason="ok", min_m=CFG.min_m, groups=",".join(sorted(STATE.groups)),
                  gemm="custom" if STATE.custom else "cutlass_mm",
                  only=",".join(sorted(CFG.only)) if CFG.only else "all")
    if CFG.skip is not None:
        report["skip_layers"] = skip_text()
    _log.info("tf_fp8_w8a8 installed: Glm53DenseFp8Method.apply of %s now serves eager prefill calls (M >= %d) of the "
              "dense FP8 groups [%s] (projections: %s) with per-token fp8 activations x the stored per-channel fp8 weights "
              "(%s); the standard-layout fp8 operand is repacked per call from the Marlin payload "
              "(no resident copy); everything else stays production's path", getattr(prodmod, "__file__", prodmod),
              CFG.min_m, ",".join(sorted(STATE.groups)), ",".join(sorted(CFG.only)) if CFG.only else "all",
              "custom CUTLASS SM120 GEMM, one call per M" if STATE.custom else "cutlass_scaled_mm")
    if CFG.skip is not None:
        _log.info("tf_fp8_w8a8 layer filter %s: excluded (stay on production's path): %s", ENV_SKIP, skip_text())
    return report


def skip_text() -> str:
    if CFG.skip is None:
        return "none"
    return ",".join((f"{lo}" if lo == hi else f"{lo}-{hi}") + (f":{sel}" if sel else "") for lo, hi, sel in CFG.skip)


def uninstall() -> dict:
    """Restore only what install() replaced; idempotent. Already-captured CUDA graphs keep what they captured."""
    rep = {"restored": False}
    cls = STATE.cls
    if cls is not None and getattr(cls.apply, "_tf_w8a8_hook", False):
        cls.apply = cls.apply._tf_w8a8_orig
        rep["restored"] = True
    if cls is not None and getattr(cls.process_weights_after_loading, "_tf_w8a8_hook", False):
        cls.process_weights_after_loading = cls.process_weights_after_loading._tf_w8a8_orig
    STATE.enabled = False
    STATE.disabled_reason = "uninstalled"
    STATE.cls = STATE.orig_apply = STATE.orig_pwal = None
    WVERDICT.clear()
    ALPHA.clear()
    STATE.scratch.clear()
    STATE.ws.clear()
    STATE.custom = False
    GEMMCHECK.clear()
    fp8ag_uninstall()
    return rep


# ---------------------------------------------------------------------------------------------------------------
# opt-dense: FP8 sequence-parallel all-gather into a served KDA in_proj (GLM53_DENSE_W8A8_FP8AG=1, default off)
# ---------------------------------------------------------------------------------------------------------------
# Under GLM53_MHC_SP the decoder layer runs the mHC on this rank's token shard and gathers the attention input:
#     x = sp_all_gather(x)[: positions.shape[0]]          (bf16 [T, 4096]: 56.6 MB per rank at T = 13,824)
#     x = self.self_attn(hidden_states=x, ...)
# For a KDA layer whose in_proj_qkvbfg_a this module serves, the gathered bf16 rows are consumed ONLY by that
# projection, which immediately re-quantizes them per token. The per-token quantization is row-local, so quantizing
# each rank's shard BEFORE the gather gives the very bytes the served call would compute on the gathered rows:
#     q, s = quant_per_token(x_shard); gather q (fp8 as uint8) and s (fp32 [rows, 1])
# -> the wire carries half the bytes and each rank quantizes half the rows; the in_proj output is BITWISE the
# GLM53_DENSE_W8A8 output (tests/optdense/test_fp8ag.py). The attention receives an unwritten bf16 placeholder of
# the gathered shape whose data pointer keys the pending (q, s); the in_proj call consumes it.
# Safety:
#  - only the attention gather of a decoder-layer forward is touched: the caller frame's code line must be the
#    `sp_all_gather(x)` line followed by `self.self_attn(` (computed per code object from its own source);
#  - the attention class's forward must use hidden_states only as the in_proj input or for .size/.shape/.dtype/
#    .device (AST check, per class); anything else -> that layer keeps the bf16 gather;
#  - the ranks must take the same collective path: each layer's decision is the MIN over the TP group of the
#    local verdicts (one CPU all-reduce per layer, once) and is never revisited by local state afterwards;
#  - any call that reaches Glm53DenseFp8Method.apply with a pending gather it cannot serve materializes the
#    placeholder as bf16(q * s) first (warned + counted; the rows are then e4m3-rounded, never garbage).
import sys as _sys  # noqa: E402


class _AgState:
    def __init__(self) -> None:
        self.tried = False
        self.installed = False
        self.reason = "not installed"
        self.mod = None
        self.orig = None             # the model module's sp_all_gather before the wrap
        self.lines: dict = {}        # code object -> line number of its attention gather (None: none found)
        self.cls_ok: dict = {}       # attention class -> its forward feeds hidden_states only to in_proj
        self.decision: dict = {}     # id(decoder layer) -> (layer, agreed bool)
        self.pending = None          # (placeholder ptr, rows, q, s, placeholder, in_proj layer)
        self.gather = None           # tests: replaces tensor_model_parallel_all_gather
        self.agree = None            # tests: replaces the TP MIN all-reduce of the local verdict


AG = _AgState()
MODEL_MOD = "vllm.models.glm5next.nvidia.model"
_ATTN_OK_ATTRS = frozenset({"size", "shape", "dtype", "device"})


def _attn_gather_line(code) -> Optional[int]:
    """Line number of `x = sp_all_gather(x)[...]` directly followed (comments/blank lines aside) by
    `x = self.self_attn(` in this code object's own source, else None. Exactly one such line or None."""
    import inspect
    try:
        lines, start = inspect.getsourcelines(code)
    except (OSError, TypeError):
        return None
    hits = []
    for i, ln in enumerate(lines):
        t = ln.strip()
        if not (t.startswith("x = sp_all_gather(x)") and t.endswith("]")):
            continue
        j = i + 1
        while j < len(lines) and (not lines[j].strip() or lines[j].strip().startswith("#") or
                                  lines[j].strip().startswith("if ")):
            if lines[j].strip().startswith("if "):
                j = len(lines)
                break
            j += 1
        if j < len(lines) and lines[j].strip().startswith("x = self.self_attn("):
            hits.append(start + i)
    return hits[0] if len(hits) == 1 else None


def _attn_forward_ok(cls) -> bool:
    """cls.forward(self, hidden_states, ...) uses hidden_states only as self.in_proj_qkvbfg_a(hidden_states) and
    through .size/.shape/.dtype/.device, and never rebinds it (AST of the live source)."""
    import ast
    import inspect
    import textwrap
    try:
        fn = ast.parse(textwrap.dedent(inspect.getsource(cls.forward))).body[0]
    except Exception:  # noqa: BLE001
        return False
    if not isinstance(fn, ast.FunctionDef) or len(fn.args.args) < 2 or fn.args.args[1].arg != "hidden_states":
        return False
    parent = {}
    for node in ast.walk(fn):
        for ch in ast.iter_child_nodes(node):
            parent[ch] = node
    uses = 0
    for node in ast.walk(fn):
        if not (isinstance(node, ast.Name) and node.id == "hidden_states"):
            continue
        if not isinstance(node.ctx, ast.Load):
            return False
        p = parent.get(node)
        if isinstance(p, ast.Attribute) and p.attr in _ATTN_OK_ATTRS:
            continue
        if (isinstance(p, ast.Call) and node in p.args and len(p.args) == 1 and not p.keywords and
                isinstance(p.func, ast.Attribute) and p.func.attr == "in_proj_qkvbfg_a" and
                isinstance(p.func.value, ast.Name) and p.func.value.id == "self"):
            uses += 1
            continue
        return False
    return uses == 1


def _in_proj_local_ok(dec, x) -> tuple[bool, object]:
    """This rank's verdict: the decoder layer's attention is a verified KDA class whose in_proj this module serves
    (selected, eligible, self-test passed) and x is the bf16 [rows, K] shard it would quantize."""
    attn = getattr(dec, "self_attn", None)
    lin = getattr(attn, "in_proj_qkvbfg_a", None)
    meth = getattr(lin, "quant_method", None)
    if attn is None or lin is None or meth is None or STATE.cls is None or not isinstance(meth, STATE.cls):
        return False, None
    c = type(attn)
    if c not in AG.cls_ok:
        AG.cls_ok[c] = _attn_forward_ok(c)
    if not AG.cls_ok[c]:
        return False, None
    if not (STATE.enabled and getattr(meth, "ready", False) and meth.group in STATE.groups and selected(meth) and
            hasattr(lin, "glm53_fp8_n") and getattr(lin, "glm53_bf16_lm_w", None) is None and
            getattr(lin, "bias", None) is None):
        return False, None
    n, k = int(lin.glm53_fp8_n), int(lin.glm53_fp8_k)
    if x.dtype != torch.bfloat16 or x.dim() != 2 or x.shape[1] != k or _ineligible(lin, n, k) is not None:
        return False, None
    key = _key(lin.weight, lin.weight_scale, k)
    v = WVERDICT.get(key)
    if v is None:
        v = selftest(lin, n, k, None, label="lazily tested layer")
    return bool(v[0]), lin


def _agree_min(local: bool) -> bool:
    if AG.agree is not None:
        return bool(AG.agree(local))
    from vllm.distributed import get_tp_group
    import torch.distributed as dist
    g = get_tp_group()
    if g.world_size == 1:
        return local
    t = torch.tensor([1 if local else 0], dtype=torch.int32)
    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=g.cpu_group)
    return bool(t.item())


def _gather0(t: torch.Tensor) -> torch.Tensor:
    if AG.gather is not None:
        return AG.gather(t)
    from vllm.distributed import tensor_model_parallel_all_gather
    return tensor_model_parallel_all_gather(t, 0)


def fp8ag_gather(x: torch.Tensor, lin) -> torch.Tensor:
    """Quantize this rank's shard per token, gather fp8 + scales, return the unwritten bf16 placeholder of the
    gathered shape (the pending (q, s) is keyed by its data pointer)."""
    q, s = quant_per_token(x.contiguous())
    qg = _gather0(q.view(torch.uint8)).view(torch.float8_e4m3fn)
    sg = _gather0(s)
    ph = torch.empty(qg.shape, dtype=torch.bfloat16, device=x.device)
    AG.pending = (ph.data_ptr(), qg.shape[0], qg, sg, ph, lin)
    _count("fp8ag_gathers")
    _count("fp8ag_rows", qg.shape[0])
    return ph


def _materialize(why: str) -> None:
    p, AG.pending = AG.pending, None
    if p is None:
        return
    _ptr, rows, qg, sg, ph, _lin = p
    ph.copy_((qg.float() * sg).to(torch.bfloat16))
    _count("fp8ag_materialized")
    _log_once("ag-mat:" + why, logging.WARNING, "tf_fp8_w8a8: FP8 all-gather placeholder materialized as bf16(q*s) "
              "(%s): that layer's attention input is e4m3-rounded for this call (counted, not repeated)", why)


def _apply_pending(serve, orig_apply, self, layer, x, bias):
    p = AG.pending
    if p[5] is layer and x.data_ptr() == p[0] and x.dim() >= 1 and x.numel() // max(x.shape[-1], 1) <= p[1]:
        AG.pending = None
        m = x.numel() // x.shape[-1]
        n, k = int(layer.glm53_fp8_n), int(layer.glm53_fp8_k)
        y = None
        if STATE.enabled and selected(self) and self.group in STATE.groups:
            y = try_w8a8(x, layer, bias, n, k, pre=(p[2][:m], p[3][:m]))
        if y is not None:
            _count("fp8ag_served")
            return y
        AG.pending = p
        _materialize("the in_proj call declined")
        return orig_apply(self, layer, x, bias)
    _materialize("another projection ran first")
    return serve(self, layer, x, bias)


def _sp_all_gather_fp8(x):
    orig = AG.orig
    try:
        f = _sys._getframe(1)
        code = f.f_code
        line = AG.lines.get(code, -1)
        if line == -1:
            line = AG.lines[code] = _attn_gather_line(code)
        if line is None or f.f_lineno != line or torch.cuda.is_current_stream_capturing():
            return orig(x)
        dec = f.f_locals.get("self")
        ent = AG.decision.get(id(dec))
        if ent is None or ent[0] is not dec:
            try:                                 # the local verdict never skips the agreement (rank symmetry)
                local, lin = _in_proj_local_ok(dec, x)
            except Exception as exc:  # noqa: BLE001
                if _CUDA_ERR.search(str(exc)):
                    raise
                _log.warning("tf_fp8_w8a8: FP8 all-gather local check raised %r: this layer votes no", exc)
                local, lin = False, None
            agreed = _agree_min(local)
            AG.decision[id(dec)] = ent = (dec, agreed, lin)
            _log.info("tf_fp8_w8a8: FP8 all-gather for %s: local %s, agreed %s", type(dec).__name__ +
                      f"#{getattr(dec, 'layer_idx', '?')}", local, agreed)
        if not ent[1]:
            return orig(x)
    except Exception as exc:  # noqa: BLE001 - the decision raised before any collective: bf16 gather
        if _CUDA_ERR.search(str(exc)):
            raise
        _log_once("ag-err", logging.WARNING, "tf_fp8_w8a8: FP8 all-gather decision raised %r: bf16 gather", exc)
        return orig(x)
    if AG.pending is not None:
        _materialize("a new gather began")
    return fp8ag_gather(x, ent[2])


def fp8ag_install() -> bool:
    """Wrap the glm5next model module's sp_all_gather (once; called at weight loading, when the module exists)."""
    AG.tried = True
    mod = _sys.modules.get(MODEL_MOD)
    if mod is None or not callable(getattr(mod, "sp_all_gather", None)):
        AG.reason = f"{MODEL_MOD} not loaded / has no sp_all_gather"
        _log.warning("tf_fp8_w8a8: %s=1 but %s: FP8 all-gather off", ENV_FP8AG, AG.reason)
        return False
    if getattr(mod.sp_all_gather, "_tf_fp8ag", False):
        AG.installed = True
        return True
    AG.mod, AG.orig = mod, mod.sp_all_gather
    _sp_all_gather_fp8._tf_fp8ag = True
    mod.sp_all_gather = _sp_all_gather_fp8
    AG.installed, AG.reason = True, None
    _log.info("tf_fp8_w8a8: FP8 all-gather installed (%s=1): the sequence-parallel attention gather of a KDA layer "
              "whose in_proj_qkvbfg_a is served carries per-token fp8 + scales (half the bytes); in_proj output "
              "unchanged bit for bit", ENV_FP8AG)
    return True


def fp8ag_uninstall() -> None:
    if AG.mod is not None and getattr(AG.mod.sp_all_gather, "_tf_fp8ag", False):
        AG.mod.sp_all_gather = AG.orig
    AG.__init__()


def summary() -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(COUNTERS.items())) or "no calls"


def plugin_register() -> None:
    """Called from integrate.plugin_register (vllm.general_plugins, every vLLM process); never raises."""
    try:
        on = env_enabled()
        _log.info("tf_fp8_w8a8 plugin loaded (pid %d): %s=%r -> %s", os.getpid(), ENV, os.environ.get(ENV),
                  "installing" if on else "off, production's dense FP8 path unchanged")
        if on:
            install()
    except Exception as exc:  # noqa: BLE001
        _log.warning("tf_fp8_w8a8 plugin install failed (dense FP8 path unchanged): %r", exc)
