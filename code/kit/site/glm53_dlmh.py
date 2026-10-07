"""GLM53_DEC_DLMH: the DFlash2 drafter's candidate head reads a 4-bit coarse copy of the lm_head instead of the full
FP8 lm_head, then recomputes the EXACT FP8 logits of the few rows it needs (docs/DEC_DLMH.md).

Production, every decode step, per TP rank: ``DFlash2Qwen3ForCausalLM.compute_candidates`` runs the shared target
lm_head (FP8 e4m3 per-channel, Marlin layout, ``glm53_runtime.convert_lm_head_fp8``; fp8_gemv kernel) over the 7 query
rows of each request: 77440 x 4096 B = 317 MB read (~1.34 ms in the R15 trace), an all-gather of the bf16 logits
(~86 us), then ``torch.topk(logits, 16)``. Only the 16 candidates (ids + bf16 values) are used afterwards.

With GLM53_DEC_DLMH=1 (default off = production unchanged):
  1. coarse: a Triton GEMV over a per-rank int4 copy (symmetric, per-row groups of GLM53_DEC_DLMH_GROUP = 128 along K,
     fp16 group scales; 158.6 + 5.0 MB) gives approximate logits; per row the coarse top-C (GLM53_DEC_DLMH_C = 128)
     local vocabulary rows are kept;
  2. exact: those T x C rows of the Marlin FP8 weight (+ scales) are gathered into a small Marlin-layout buffer and the
     SAME fp8_gemv kernel with the SAME configuration production uses at this row count computes their logits: each
     output column of fp8_gemv depends only on its own weight column, the scale and the configuration (tensor-core
     mma per 16-k tile, fixed-order split-K reduction, scale, bf16 rounding), so the values are bit-identical to the
     full head's at those vocabulary ids (tests/dlmh/test_dlmh.py: gathered == full, rtol = atol = 0);
  3. the (id, value) pairs of every rank are all-gathered (T x 2C int32 instead of T x 77440 bf16), scattered into a
     -inf full-vocabulary bf16 tensor and production's own ``torch.topk`` runs on it.
torch.topk's result depends only on the values and indices of the elements >= its k-th value (radix select, index
order among ties, then a sort of those k elements; tested with tie-heavy bf16 inputs), so the candidates and unary
logits are byte-identical to production's whenever every element >= the 16th exact value is in the coarse top-C of
its rank. Measured on the real lm_head with the real drafter's hidden states (tests/dlmh/recall_probe.py,
4032 rows, near-uniform to confident regimes and random directions): the worst row needed C = 61 (int4 g128), so
C = 128 has a 2.1x margin. A miss can only change the drafter's proposal (the target's distribution, acceptance
rule and outputs are untouched: speculative decoding stays lossless); GLM53_DEC_DLMH=verify measures the miss rate
in production while serving production's own candidates.

Served row counts: those for which production's lm_head call itself runs fp8_gemv (fp8_gemv.select_config for the
77440 x 4096 shape) and T <= 16 (1-2 requests; the coarse kernel's BLOCK_M), T a multiple of 7; any other call, any call first seen while capturing, a TP layout
other than vocab-parallel all-gather, a logits soft cap or scale -> production's compute_candidates unchanged.
Memory per rank: coarse copy 163.6 MB (g = 128) + gather buffers 11 MB (C = 128).
"""
from __future__ import annotations

import logging
import os
import threading

import torch

_log = logging.getLogger("vllm.glm53_dlmh")
ENV = "GLM53_DEC_DLMH"
ENV_C = "GLM53_DEC_DLMH_C"
ENV_GROUP = "GLM53_DEC_DLMH_GROUP"
ENV_LOG = "GLM53_DEC_DLMH_LOG"
_MODES = {"1": "on", "on": "on", "true": "on", "yes": "on", "verify": "verify"}
VERSION = 1
SNAP_LEAD = 32      # the stats line prints the device counters snapshotted this many drafter steps earlier
# [review] rows per chunk of the coarse build. The build runs inside vLLM's profile run (no empty_cache follows it):
# at 4096 rows it peaked at +477 MiB and left +634-654 MiB reserved in the caching allocator per rank (nodeC,
# tests/dlmh/review_adv.py R1) on hosts with 3-6 GB of unified memory available; 1024 rows cut the transient ~4x.
# Rows are independent, so the coarse bytes do not depend on the chunk size.
BUILD_CHUNK_ROWS = 1024
_LOCK = threading.Lock()


class _Cfg:
    def __init__(self) -> None:
        self.mode = "off"
        self.C = 128         # column octets per row (8 C columns rescored per row)
        self.group = 128
        self.log_every = 2000
        self.strict = False        # tests: raise instead of falling back
        self.rescore_warps = 4     # same KW (= per-column arithmetic) as production's config, 1 n-tile per block


CFG = _Cfg()
COUNTERS = {"served": 0, "prod_rows": 0, "prod_capture_first": 0, "verify_calls": 0, "setup": 0,
            "eager": 0, "captured": 0}   # [review] two-stage executions outside / inside a graph capture (host-side)
TUNE_CANDIDATES = [(64, 32, 4, 4), (32, 32, 4, 3), (64, 64, 4, 3), (32, 64, 4, 3), (64, 64, 4, 4), (32, 128, 4, 3),
                   (64, 128, 4, 3), (64, 128, 8, 3), (16, 128, 4, 3), (32, 128, 4, 4), (128, 64, 8, 3)]
COARSE_CFG = {"block_n": 32, "block_w": 128, "num_warps": 4, "num_stages": 3}   # tests/dlmh/bench_coarse.py (g128)


def parse_env(environ=None) -> None:
    e = os.environ if environ is None else environ
    raw = (e.get(ENV) or "").strip().lower()
    CFG.mode = _MODES.get(raw, "off") if raw not in ("", "0", "off", "false", "no") else "off"
    if raw and raw not in ("0", "off", "false", "no") and raw not in _MODES:
        raise ValueError(f"{ENV} must be 0/1/verify (got {raw!r})")
    c = int(e.get(ENV_C) or 128)
    g = int(e.get(ENV_GROUP) or 128)
    if c < 16 or c % 8 or c > 256:
        raise ValueError(f"{ENV_C} (column octets per row) must be a multiple of 8 in 16..256 (got {c})")
    if g not in (32, 64, 128):
        raise ValueError(f"{ENV_GROUP} must be 32, 64 or 128 (got {g})")
    CFG.C, CFG.group = c, g
    CFG.log_every = int(e.get(ENV_LOG) or 2000)


# ---------------------------------------------------------------------------------------------------------------
# kernels (built lazily: importing this module must not need triton / vllm)
_K: dict = {}


def _kernels():
    if _K:
        return _K
    try:
        from vllm.triton_utils import tl, triton
    except Exception:  # noqa: BLE001  (tests outside vllm)
        import triton
        import triton.language as tl

    @triton.jit
    def _coarse_kernel(x_ptr, w_ptr, s_ptr, y_ptr, M, N, stride_xm, stride_ym,
                       K8: tl.constexpr, G: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                       BLOCK_W: tl.constexpr):
        # w: int32 [N, K8]; word j of row n holds the 4-bit codes (value + 8) of k = s*K8 + j at bits 4s..4s+3
        # s: fp16 [N, 8*K8/G] group scales; y: fp32 [M, N] = x @ dequant(w)^T
        pid = tl.program_id(0)
        rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        rm = tl.arange(0, BLOCK_M)
        rw = tl.arange(0, BLOCK_W)
        nmask = rn < N
        mmask = rm < M
        NG: tl.constexpr = (8 * K8) // G
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for j0 in range(0, K8, BLOCK_W):
            w = tl.load(w_ptr + rn[:, None] * K8 + j0 + rw[None, :], mask=nmask[:, None], other=0)
            for s in tl.static_range(8):
                q = (((w >> (4 * s)) & 15) - 8).to(tl.bfloat16)
                xs = tl.load(x_ptr + rm[:, None] * stride_xm + s * K8 + j0 + rw[None, :], mask=mmask[:, None],
                             other=0.0)
                part = tl.dot(xs, tl.trans(q))
                sc = tl.load(s_ptr + rn * NG + (s * K8 + j0) // G, mask=nmask, other=0.0).to(tl.float32)
                acc += part * sc[None, :]
        tl.store(y_ptr + rm[:, None] * stride_ym + rn[None, :], acc, mask=mmask[:, None] & nmask[None, :])

    @triton.jit
    def _gather_kernel(src_ptr, dst_ptr, ids_ptr, U, src_ld, dst_ld, KT,
                       BLOCK_C: tl.constexpr, BLOCK_K: tl.constexpr):
        # Marlin FP8 layout (kernels/fp8_gemv.cu): int32 [K/16, 4*Npad]; column col of k-tile kt lives in the 4 words
        # (col//64)*256 + 32*(col%8 within its 16-group) + 8*q + 2*((col%64)//16) + (col%16)//8, q = 0..3
        pc = tl.program_id(0)
        pk = tl.program_id(1)
        c = pc * BLOCK_C + tl.arange(0, BLOCK_C)
        cm = c < U
        n = tl.load(ids_ptr + c, mask=cm, other=0)
        q = tl.arange(0, 4)
        so = (n // 64) * 256 + 32 * ((n % 16) % 8) + 2 * ((n % 64) // 16) + (n % 16) // 8
        do = (c // 64) * 256 + 32 * ((c % 16) % 8) + 2 * ((c % 64) // 16) + (c % 16) // 8
        kt = pk * BLOCK_K + tl.arange(0, BLOCK_K)
        km = kt < KT
        sp = src_ptr + kt[:, None, None].to(tl.int64) * src_ld + so[None, :, None] + 8 * q[None, None, :]
        dp = dst_ptr + kt[:, None, None].to(tl.int64) * dst_ld + do[None, :, None] + 8 * q[None, None, :]
        mk = km[:, None, None] & cm[None, :, None]
        v = tl.load(sp, mask=mk, other=0)
        tl.store(dp, v, mask=mk)

    @triton.jit
    def _coarse_oct_kernel(x_ptr, w_ptr, s_ptr, o_ptr, M, N, stride_xm, stride_om,
                           K8: tl.constexpr, G: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_W: tl.constexpr):
        # as _coarse_kernel over one 64-column n-tile, but stores per row the max over each column octet
        # {64 * tile + r + 8 i : i = 0..7} at o[m, 8 * tile + r] (the octet score)
        pid = tl.program_id(0)
        rn = pid * 64 + tl.arange(0, 64)
        rm = tl.arange(0, BLOCK_M)
        rw = tl.arange(0, BLOCK_W)
        nmask = rn < N
        mmask = rm < M
        NG: tl.constexpr = (8 * K8) // G
        acc = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
        for j0 in range(0, K8, BLOCK_W):
            w = tl.load(w_ptr + rn[:, None] * K8 + j0 + rw[None, :], mask=nmask[:, None], other=0)
            for s in tl.static_range(8):
                q = (((w >> (4 * s)) & 15) - 8).to(tl.bfloat16)
                xs = tl.load(x_ptr + rm[:, None] * stride_xm + s * K8 + j0 + rw[None, :], mask=mmask[:, None],
                             other=0.0)
                part = tl.dot(xs, tl.trans(q))
                sc = tl.load(s_ptr + rn * NG + (s * K8 + j0) // G, mask=nmask, other=0.0).to(tl.float32)
                acc += part * sc[None, :]
        om = tl.max(tl.reshape(acc, (BLOCK_M, 8, 8)), axis=1)          # [m, r] = max_i acc[m, 8 i + r]
        ro = tl.arange(0, 8)
        tl.store(o_ptr + rm[:, None] * stride_om + pid * 8 + ro[None, :], om, mask=mmask[:, None])

    @triton.jit
    def _post_kernel(y_ptr, octs_ptr, pack_ptr, stride_y, stride_p, base, C8: tl.constexpr, BLOCK: tl.constexpr):
        # row r, its local virtual column u (global vc = r * C8 + u): source column of slot 8 (vc // 64) + vc % 8,
        # i = (vc % 64) // 8 -> pack[r, u] = global id, pack[r, C8 + u] = the bf16 bits of y[r, vc] (sign-extended)
        r = tl.program_id(0)
        u = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        um = u < C8
        vc = r * C8 + u
        o = tl.load(octs_ptr + 8 * (vc // 64) + vc % 8, mask=um, other=0)
        gid = (o >> 3) * 64 + (o & 7) + 8 * ((vc % 64) // 8) + base
        v = tl.load(y_ptr + r * stride_y + vc, mask=um, other=0.0)
        bits = v.to(tl.int16, bitcast=True).to(tl.int32)
        tl.store(pack_ptr + r * stride_p + u, gid.to(tl.int32), mask=um)
        tl.store(pack_ptr + r * stride_p + C8 + u, bits, mask=um)

    _K.update(coarse=_coarse_kernel, gather=_gather_kernel, coarse_oct=_coarse_oct_kernel, post=_post_kernel,
              triton=triton)
    return _K


def coarse_gemv(x, wq, s, y, group, block_n=None, block_w=None, num_warps=None, num_stages=None):
    """y[M, N] (fp32) = x[M, K] (bf16) @ int4-dequant(wq, s)^T."""
    K = _kernels()
    M, Kd = x.shape
    N, K8 = wq.shape
    assert Kd == 8 * K8 and y.shape == (M, N) and M <= 16
    bn = block_n or COARSE_CFG["block_n"]
    bw = block_w or COARSE_CFG["block_w"]
    assert group % bw == 0 and K8 % bw == 0
    grid = (K["triton"].cdiv(N, bn),)
    K["coarse"][grid](x, wq, s, y, M, N, x.stride(0), y.stride(0), K8=K8, G=group, BLOCK_M=16, BLOCK_N=bn,
                      BLOCK_W=bw, num_warps=num_warps or COARSE_CFG["num_warps"],
                      num_stages=num_stages or COARSE_CFG["num_stages"])
    return y


OCT_CFG = {"block_w": 128, "num_warps": 4, "num_stages": 2}   # tests/dlmh/bench_octmax.py


def coarse_octmax(x, wq, s, osc, group, block_w=None, num_warps=None, num_stages=None):
    """osc[M, N/8] (fp32) = per column octet {64 t + r + 8 i} the max of x @ int4-dequant(wq, s)^T."""
    K = _kernels()
    M, Kd = x.shape
    N, K8 = wq.shape
    bw = block_w or OCT_CFG["block_w"]
    assert Kd == 8 * K8 and N % 64 == 0 and osc.shape == (M, N // 8) and M <= 16 and group % bw == 0
    K["coarse_oct"][(N // 64,)](x, wq, s, osc, M, N, x.stride(0), osc.stride(0), K8=K8, G=group, BLOCK_M=16,
                                BLOCK_W=bw, num_warps=num_warps or OCT_CFG["num_warps"],
                                num_stages=num_stages or OCT_CFG["num_stages"])
    return osc


def perm_col(n: torch.Tensor) -> torch.Tensor:
    """stored position of logical column n in a Marlin-permuted per-channel vector (scale_perm_single)."""
    c = n & 31
    return (n & ~31) + 8 * ((c & 7) >> 1) + (c & 1) + 2 * (c >> 3)


def gather_marlin(src_w, src_s, ids, dst_w, dst_s):
    """dst columns c = src columns ids[c] of a Marlin FP8 weight (int32 [K/16, 4*Npad]) and its permuted scales."""
    K = _kernels()
    KT = src_w.shape[0]
    U = ids.numel()
    assert dst_w.shape == (KT, 4 * U) and dst_s.numel() == U and U % 64 == 0
    grid = (K["triton"].cdiv(U, 64), K["triton"].cdiv(KT, 32))
    K["gather"][grid](src_w, dst_w, ids, U, src_w.shape[1], dst_w.shape[1], KT, BLOCK_C=64, BLOCK_K=32, num_warps=4)
    c = torch.arange(U, device=ids.device, dtype=torch.int64)
    dst_s.view(-1)[perm_col(c)] = src_s.view(-1)[perm_col(ids.to(torch.int64))]


def _unpack_perm(device) -> torch.Tensor:
    """index into a tile's 1024 bytes for logical (col 0..63, k 0..15) of the Marlin FP8 tile."""
    perm = torch.empty(64 * 16, dtype=torch.int64)
    for t in range(32):
        for w in range(4):
            for h in range(2):
                j = 8 * t + 2 * w + h
                col = 16 * w + t // 4 + 8 * h
                for b, dk in enumerate((0, 8, 1, 9)):
                    k = 2 * (t % 4) + dk
                    perm[col * 16 + k] = 4 * j + b
    return perm.to(device)


def unpack_marlin_fp8(wq, ws, n, k, row0, row1):
    """logical fp32 weight rows [row0, row1) (e4m3 value x per-channel scale) of a Marlin FP8 layer."""
    assert row0 % 64 == 0 and row1 % 64 == 0 or row1 == wq.shape[1] // 4
    KT = k // 16
    npad = wq.shape[1] // 4
    nt0, nt1 = row0 // 64, (row1 + 63) // 64
    b = wq.view(torch.uint8).view(KT, npad // 64, 1024)[:, nt0:nt1]
    t = b.index_select(2, _unpack_perm(wq.device)).view(KT, nt1 - nt0, 64, 16)
    w8 = t.permute(1, 2, 0, 3).reshape((nt1 - nt0) * 64, k).contiguous().view(torch.float8_e4m3fn)
    rows = torch.arange(row0, nt1 * 64, device=wq.device)
    sc = ws.view(-1)[perm_col(rows)].float() * (2.0 ** -120)
    return (w8.float() * sc[:, None])[: row1 - row0]


def build_coarse_from_marlin(wq, ws, n, k, group=32, chunk=4096):
    """int4 coarse copy: wq int32 [Npad, K/8] (word j: codes of k = s*K/8 + j, s = 0..7), s fp16 [Npad, K/group]."""
    npad = wq.shape[1] // 4
    K8 = k // 8
    out_w = torch.empty((npad, K8), dtype=torch.int32, device=wq.device)
    out_s = torch.empty((npad, k // group), dtype=torch.float16, device=wq.device)
    for r0 in range(0, npad, chunk):
        r1 = min(npad, r0 + chunk)
        w = unpack_marlin_fp8(wq, ws, n, k, r0, r1)
        wg = w.view(r1 - r0, k // group, group)
        s = (wg.abs().amax(-1) / 7.0).clamp(min=1e-30).to(torch.float16).float()
        s = torch.where(s > 0, s, torch.full_like(s, 1e-30))
        qv = torch.clamp(torch.round(wg / s[..., None]), -8, 7).to(torch.int64) + 8
        qv = qv.view(r1 - r0, 8, K8)
        word = torch.zeros((r1 - r0, K8), dtype=torch.int64, device=wq.device)
        for sh in range(8):
            word |= qv[:, sh, :] << (4 * sh)
        word = torch.where(word >= 2 ** 31, word - 2 ** 32, word)
        out_w[r0:r1] = word.to(torch.int32)
        out_s[r0:r1] = s.to(torch.float16)
        del w, wg, qv, word
    return out_w, out_s


def coarse_dequant(wq, s, group):
    """fp32 [N, K] of the coarse copy (tests)."""
    N, K8 = wq.shape
    out = torch.empty((N, 8, K8), dtype=torch.float32, device=wq.device)
    for sh in range(8):
        out[:, sh] = ((wq >> (4 * sh)) & 15).float() - 8.0
    out = out.view(N, 8 * K8)
    return (out.view(N, -1, group) * s.float()[..., None]).view(N, 8 * K8)


# ---------------------------------------------------------------------------------------------------------------
class _Head:
    """per-process state of one candidate head (the drafter's shared lm_head)."""

    def __init__(self) -> None:
        self.ready = False
        self.failed = None
        self.holder = None
        self.coarse_w = self.coarse_s = None
        self.bufs: dict = {}
        self.tp = 1
        self.rank = 0
        self.vloc = 0
        self.org = 0
        self.counters = None       # device int64 [4]: calls, rows, mismatched rows (verify), served calls
        self.pinned = None
        self.log_n = 0
        self.snap_n = 0            # drafter step at which the pinned counter snapshot was enqueued
        self.snap_eager = 0        # COUNTERS["eager"] at that moment
        self.serving_logged = False
        self.dbg = None
        self.dbg_seen = 0


HEAD = _Head()


def _tp_info(lm):
    tp = int(getattr(lm, "tp_size", 1) or 1)
    rank = 0
    if tp > 1:
        from vllm.distributed import get_tensor_model_parallel_rank
        rank = int(get_tensor_model_parallel_rank())
    return tp, rank


def _config_for(m: int, head=None):
    import fp8_gemv as G
    hd = HEAD if head is None else head
    if not G.STATE.enabled:
        return None
    return G.select_config(hd.holder.weight.shape[1] // 4, int(hd.holder.glm53_fp8_k), m)


def _all_gather(t):
    if HEAD.tp == 1:
        return t
    from vllm.distributed import tensor_model_parallel_all_gather
    return tensor_model_parallel_all_gather(t, dim=-1)


_EXT: dict = {}


def _dlmh_ext():
    if "ext" not in _EXT:
        try:
            import tf_dlmh_ext as m
        except ImportError:
            import fp8_gemv as G
            from torch.utils.cpp_extension import load
            here = os.path.dirname(os.path.abspath(__file__))
            m = load("tf_dlmh_ext", [os.path.join(here, "kernels", "dlmh_gemv.cu")], extra_cflags=["-O3"],
                     extra_cuda_cflags=["-O3", *G._include_shim()], verbose=False)
        _EXT["ext"] = m
    return _EXT["ext"]


def _vmaps(T, C, device):
    """static maps of the virtual columns for T rows x C octets: slot (= octet position in the flat list) and i."""
    nv = 8 * T * C
    vc = torch.arange(nv, device=device, dtype=torch.int64)
    slot = 8 * (vc // 64) + (vc % 8)
    i = (vc % 64) // 8
    return slot, i, perm_col(vc)


def local_pack(x, head=None):
    """this rank's part: coarse top-C column octets per row, the exact FP8 logits of their 8*C columns
    -> int32 [T, 16C] (global ids | bf16 bits)."""
    hd = HEAD if head is None else head
    T = x.shape[0]
    C = CFG.C
    h = hd.holder
    k = int(h.glm53_fp8_k)
    cfg = _config_for(T, hd)
    osc = torch.empty((T, hd.coarse_w.shape[0] // 8), dtype=torch.float32, device=x.device)
    coarse_octmax(x, hd.coarse_w, hd.coarse_s, osc, CFG.group)
    octs = torch.topk(osc, C, dim=-1, sorted=False).indices.view(-1)            # [T*C] int64, row-major
    nv = 8 * T * C
    y = torch.empty((T, nv), dtype=torch.bfloat16, device=x.device)
    _dlmh_ext().dlmh_gemv_out(y, x, h.weight, h.weight_scale.view(-1), octs, nv, k, CFG.rescore_warps, cfg[1],
                              bool(cfg[3]))
    pack = torch.empty((T, 16 * C), dtype=torch.int32, device=x.device)
    K = _kernels()
    K["post"][(T, K["triton"].cdiv(8 * C, 256))](y, octs, pack, y.stride(0), pack.stride(0), hd.rank * hd.vloc,
                                                C8=8 * C, BLOCK=256, num_warps=4)
    return pack


def merge(allp, T, top_k, tp, vloc, org):
    """every rank's packs (all-gathered along the last dim, rank-major) -> production's torch.topk on a -inf
    full-vocabulary tensor holding only the candidates' exact values."""
    W = 8 * CFG.C
    allp = allp.view(T, tp, 2, W)
    ids = allp[:, :, 0].reshape(T, tp * W).to(torch.int64)
    v = allp[:, :, 1].reshape(T, tp * W).to(torch.int16).view(torch.bfloat16)
    full = torch.full((T, tp * vloc), float("-inf"), dtype=torch.bfloat16, device=allp.device)
    full.scatter_(1, ids, v)
    logits = full[..., :org]
    unary, cand = torch.topk(logits, top_k, dim=-1)
    return cand, unary


def two_stage(x, top_k: int):
    """(candidate_ids int64 [T, top_k], unary bf16 [T, top_k]) from hidden rows x [T, H] bf16."""
    pack = local_pack(x)
    return merge(_all_gather(pack), x.shape[0], top_k, HEAD.tp, HEAD.vloc, HEAD.org)


def _setup(model) -> None:
    """build the coarse copy + buffers; eager only, once per process, at the drafter's first eager call."""
    import fp8_gemv as G
    lm = model.lm_head
    holder = getattr(lm, "glm53_fp8_head", None)
    proc = model.candidate_logits_processor
    why = None
    if holder is None:
        why = "lm_head is not the glm53 FP8 head (GLM53_LMHEAD_FP8 off?)"
    elif not G.STATE.enabled or G.STATE.ext is None:
        why = "fp8_gemv is not enabled (GLM53_FP8_GEMV)"
    elif getattr(proc, "soft_cap", None) is not None or float(getattr(proc, "scale", 1.0)) != 1.0:
        why = "candidate logits soft cap / scale set"
    elif getattr(proc, "logits_as_input", False):
        why = "logits_as_input"
    elif int(getattr(lm, "tp_size", 1) or 1) > 1 and not getattr(proc, "use_all_gather", True):
        why = "vocab-parallel gather (not all-gather) logits"
    if why is not None:
        HEAD.failed = why
        _log.info("glm53_dlmh: not wired (%s); production's candidate head stays", why)
        return
    HEAD.holder = holder
    HEAD.tp, HEAD.rank = _tp_info(lm)
    HEAD.vloc = int(holder.glm53_fp8_n)
    HEAD.org = int(proc.org_vocab_size)
    npad = holder.weight.shape[1] // 4
    k = int(holder.glm53_fp8_k)
    if HEAD.vloc != npad or HEAD.tp * HEAD.vloc < HEAD.org:
        HEAD.failed = f"unexpected vocab layout (local {HEAD.vloc}, Npad {npad}, tp {HEAD.tp}, org {HEAD.org})"
        _log.warning("glm53_dlmh: %s; production's candidate head stays", HEAD.failed)
        return
    HEAD.coarse_w, HEAD.coarse_s = build_coarse_from_marlin(holder.weight, holder.weight_scale, HEAD.vloc, k,
                                                            CFG.group, chunk=BUILD_CHUNK_ROWS)
    for T in range(1, 17):      # the coarse kernel's BLOCK_M = 16; production runs fp8_gemv there (MB 1, 2)
        if T % 7 == 0 and _config_for(T) is not None:
            HEAD.bufs[T] = ()
    HEAD.counters = torch.zeros(4, dtype=torch.int64, device=holder.weight.device)
    if CFG.mode == "verify":   # the last mismatching row (hidden state, production's and the two-stage's candidates)
        dv = holder.weight.device
        HEAD.dbg = (torch.zeros((1, k), dtype=torch.bfloat16, device=dv), torch.zeros((1, 16), dtype=torch.int64, device=dv),
                    torch.zeros((1, 16), dtype=torch.bfloat16, device=dv), torch.zeros((1, 16), dtype=torch.int64, device=dv),
                    torch.zeros((1, 16), dtype=torch.bfloat16, device=dv))
    HEAD.pinned = torch.zeros(4, dtype=torch.int64, pin_memory=True)
    HEAD.ready = True
    COUNTERS["setup"] += 1
    _dlmh_ext()
    mib = (HEAD.coarse_w.numel() * 4 + HEAD.coarse_s.numel() * 2 +
           sum(t.numel() * 8 for v in HEAD.bufs.values() for t in v)) / 2**20
    _log.info("glm53_dlmh: rank %d/%d coarse candidate head built (int4 g%d, C=%d, vocab rows %d, rows served %s, "
              "+%.1f MiB)", HEAD.rank, HEAD.tp, CFG.group, CFG.C, HEAD.vloc, sorted(HEAD.bufs), mib)


def _selftest(model, orig) -> None:
    """two-stage == production on peaked synthetic rows (every element >= the 16th surely in the coarse top-C):
    proves the gather / exact-GEMV / all-gather / topk path byte for byte. Both ranks run it at the same call."""
    h = HEAD.holder
    k = int(h.glm53_fp8_k)
    g = torch.Generator(device=h.weight.device).manual_seed(7)
    bad = 0
    for T in sorted(HEAD.bufs)[:2]:
        # rows = a few lm_head rows of this rank mixed (peaked logits) + noise, at the drafter's hidden scale
        rows = torch.randint(0, HEAD.vloc, (T, 3), generator=g, device=h.weight.device)
        w = torch.stack([unpack_marlin_fp8(h.weight, h.weight_scale, HEAD.vloc, k, int(r) // 64 * 64,
                                           int(r) // 64 * 64 + 64)[int(r) % 64] for r in rows.view(-1).tolist()])
        x = w.view(T, 3, k).sum(1)
        x = x / x.norm(dim=-1, keepdim=True) * 110.0
        x = (x + torch.randn(x.shape, generator=g, device=x.device) * 0.3).to(torch.bfloat16)
        c0, u0 = orig(model, x)
        c1, u1 = two_stage(x, model.model.candidate_selector.top_k)
        ok = torch.equal(c0, c1) and torch.equal(u0.view(torch.int16), u1.view(torch.int16))
        bad += 0 if ok else 1
    if bad:
        HEAD.ready = False
        HEAD.failed = "self-test mismatch"
        _log.warning("glm53_dlmh: rank %d self-test: two-stage candidates differ from production's; production's "
                     "candidate head stays", HEAD.rank)
    else:
        _log.info("glm53_dlmh: rank %d self-test: candidates and unary logits byte-equal to production's (T in %s) "
                  "PROOF mode=%s C=%d g=%d", HEAD.rank, sorted(HEAD.bufs)[:2], CFG.mode, CFG.C, CFG.group)


def _agree(ok: bool, tp: int) -> bool:
    """True iff every TP rank reports ok (eager all-reduce of a 0/1 flag; tp == 1: ok)."""
    if tp <= 1:
        return bool(ok)
    from vllm.distributed import tensor_model_parallel_all_reduce
    t = torch.tensor([1 if ok else 0], dtype=torch.int32, device=torch.device("cuda", torch.cuda.current_device()))
    return int(tensor_model_parallel_all_reduce(t).item()) == tp


def _install_model() -> None:
    from vllm.model_executor.models import qwen3_dflash2 as Q
    cls = Q.DFlash2Qwen3ForCausalLM
    if getattr(cls.compute_candidates, "_glm53_dlmh", False):
        return
    orig = cls.compute_candidates

    def compute_candidates(self, hidden_states):
        if CFG.mode == "off" or HEAD.failed is not None:
            return orig(self, hidden_states)
        capturing = torch.cuda.is_current_stream_capturing()
        if not HEAD.ready:
            if capturing:
                COUNTERS["prod_capture_first"] += 1
                return orig(self, hidden_states)
            with _LOCK:
                if not HEAD.ready and HEAD.failed is None:
                    # TP: the two-stage head all-gathers a different tensor than production's head, so BOTH ranks
                    # must take the same decision: setup and self-test results are agreed on (all-reduce) before
                    # anything is served; one rank failing turns the head off on every rank
                    tp = int(getattr(self.lm_head, "tp_size", 1) or 1)
                    for stage in ("setup", "self-test"):
                        try:
                            if stage == "setup":
                                _setup(self)
                            elif HEAD.ready:
                                _selftest(self, orig)
                        except Exception as exc:  # noqa: BLE001
                            HEAD.ready = False
                            HEAD.failed = f"{stage} failed: {exc!r}"
                            _log.warning("glm53_dlmh: setup failed (%s); production's candidate head stays",
                                         HEAD.failed)
                            if CFG.strict:
                                raise
                        if not _agree(HEAD.ready, tp):
                            if HEAD.ready:
                                _log.warning("glm53_dlmh: setup failed on another TP rank (%s); production's "
                                             "candidate head stays on every rank", stage)
                            HEAD.ready = False
                            HEAD.failed = HEAD.failed or f"another TP rank failed its {stage}"
                            break
            if not HEAD.ready:
                return orig(self, hidden_states)
        x = hidden_states
        T = x.shape[0]
        if (T not in HEAD.bufs or x.dtype != torch.bfloat16 or x.dim() != 2 or not x.is_contiguous()):
            COUNTERS["prod_rows"] += 1
            return orig(self, hidden_states)
        top_k = self.model.candidate_selector.top_k
        if CFG.mode == "verify":
            c0, u0 = orig(self, hidden_states)
            c1, u1 = two_stage(x, top_k)
            bad = ((c0 != c1) | (u0.view(torch.int16) != u1.view(torch.int16))).any(-1)
            HEAD.counters[0] += 1
            HEAD.counters[1] += T
            HEAD.counters[2] += bad.sum()
            if HEAD.dbg is not None and top_k == 16:
                anyb = bad.any()
                i = torch.argmax(bad.to(torch.int32)).view(1)
                for buf, src in zip(HEAD.dbg, (x, c0, u0, c1, u1)):
                    buf.copy_(torch.where(anyb, src.index_select(0, i), buf))
            COUNTERS["verify_calls"] += 1
            COUNTERS["captured" if capturing else "eager"] += 1
            return c0, u0
        HEAD.counters[3] += 1
        COUNTERS["served"] += 1
        COUNTERS["captured" if capturing else "eager"] += 1
        return two_stage(x, top_k)

    compute_candidates._glm53_dlmh = True
    compute_candidates._glm53_orig = orig
    cls.compute_candidates = compute_candidates


def _diagnose() -> str:
    """eager, this rank only: why did the last mismatching row differ (coarse recall vs tie order vs arithmetic)?"""
    import fp8_gemv as G
    x, c0, u0, c1, u1 = (t.clone() for t in HEAD.dbg)
    h = HEAD.holder
    lg = G.linear(x, h.weight, h.weight_scale, h.workspace, None, int(h.glm53_fp8_n), int(h.glm53_fp8_k)).float()[0]
    kth = u0.float()[0, -1]
    lo, hi = HEAD.rank * HEAD.vloc, (HEAD.rank + 1) * HEAD.vloc
    need = (lg >= kth).nonzero().view(-1)                                       # this rank's elements >= the 16th
    osc = torch.empty((1, HEAD.coarse_w.shape[0] // 8), dtype=torch.float32, device=x.device)
    coarse_octmax(x, HEAD.coarse_w, HEAD.coarse_s, osc, CFG.group)
    order = torch.argsort(osc[0], descending=True)
    rank_of = torch.empty_like(order)
    rank_of[order] = torch.arange(order.numel(), device=x.device)
    octs = (need // 64) * 8 + need % 8
    ranks = rank_of[octs].tolist()
    same_set = set(c0[0].tolist()) == set(c1[0].tolist())
    return (f"16th value {kth.item():.4f}; elements >= it on this rank {need.numel()} (ties at it "
            f"{int((lg == kth).sum())}), their octet ranks {sorted(ranks)[-6:]} (C {CFG.C}); same candidate set "
            f"{same_set}; prod ids {c0[0].tolist()} vals {[round(v, 4) for v in u0.float()[0].tolist()]}; two-stage "
            f"ids {c1[0].tolist()} vals {[round(v, 4) for v in u1.float()[0].tolist()]}; rank range [{lo}, {hi})")


def _install_logger() -> None:
    from vllm.v1.worker.gpu.spec_decode.dflash2 import speculator as S
    cls = S.DFlash2Speculator
    if getattr(cls.propose, "_glm53_dlmh", False):
        return
    orig = cls.propose

    def propose(self, *a, **kw):
        r = orig(self, *a, **kw)
        if HEAD.ready and CFG.log_every > 0:
            HEAD.log_n += 1
            n = HEAD.log_n
            if n % CFG.log_every == 0 or n == 64:
                # [review] the snapshot enqueued SNAP_LEAD steps before this log point (complete by now: every step
                # syncs the host on the sampled ids). Before the review fix the copy was enqueued only AFTER each
                # print, so the first line (step 64) always read zeros and every later line one period stale.
                p = HEAD.pinned.tolist()
                _log.info("[glm53-dlmh] rank %d mode %s C %d g %d: steps %d, served graph-calls %d; verify calls %d "
                          "rows %d mismatched rows %d (counters as of step %d)", HEAD.rank, CFG.mode, CFG.C,
                          CFG.group, n, p[3], p[0], p[1], p[2], HEAD.snap_n)
                # device-counted calls minus the eager ones (host-counted up to the snapshot) = graph replays
                replays = (p[3] if CFG.mode == "on" else p[0]) - HEAD.snap_eager
                if not HEAD.serving_logged and COUNTERS["captured"] > 0 and replays > 0:
                    # one fixed line per rank once CUDA-graph replays have demonstrably run the two-stage head: the
                    # A/B's PROOF string (the boot self-test line alone does not show that anything was served, and
                    # graphs captured before the setup would keep production's head)
                    HEAD.serving_logged = True
                    _log.info("[glm53-dlmh] rank %d serving confirmed (mode %s): graph replays %d (captured graphs %d, "
                              "eager calls %d) by step %d", HEAD.rank, CFG.mode, replays, COUNTERS["captured"],
                              HEAD.snap_eager, HEAD.snap_n)
                if HEAD.dbg is not None and p[2] > HEAD.dbg_seen:
                    HEAD.dbg_seen = p[2]
                    try:
                        _log.info("[glm53-dlmh] rank %d mismatch diagnosis: %s", HEAD.rank, _diagnose())
                    except Exception as exc:  # noqa: BLE001
                        _log.info("[glm53-dlmh] rank %d mismatch diagnosis failed: %r", HEAD.rank, exc)
            if n == 64 - SNAP_LEAD or (n + SNAP_LEAD) % CFG.log_every == 0:
                HEAD.pinned.copy_(HEAD.counters, non_blocking=True)
                HEAD.snap_n = n
                HEAD.snap_eager = COUNTERS["eager"]
        return r

    propose._glm53_dlmh = True
    propose._glm53_orig = orig
    cls.propose = propose


def _hook_loader() -> None:
    import vllm.model_executor.model_loader.base_loader as BL
    if getattr(BL.process_weights_after_loading, "_glm53_dlmh_orig", None) is not None:
        return
    orig = BL.process_weights_after_loading

    def process_weights_after_loading(model, model_config, target_device):
        orig(model, model_config, target_device)
        if type(model).__name__ == "DFlash2Qwen3ForCausalLM":
            try:
                _install_model()
                _install_logger()
                _log.info("glm53_dlmh: hooked DFlash2 compute_candidates (mode %s, C %d, g %d)", CFG.mode, CFG.C,
                          CFG.group)
            except Exception as exc:  # noqa: BLE001
                HEAD.failed = f"hook failed: {exc!r}"
                _log.warning("glm53_dlmh: %s; production's candidate head stays", HEAD.failed)

    process_weights_after_loading._glm53_dlmh_orig = orig
    BL.process_weights_after_loading = process_weights_after_loading


def plugin_install() -> None:
    """Called from integrate.plugin_register in every vLLM process. Inert unless GLM53_DEC_DLMH is set."""
    try:
        parse_env()
    except ValueError as exc:
        _log.warning("glm53_dlmh: bad configuration (%s): off, production's candidate head stays", exc)
        CFG.mode = "off"
        return
    _log.info("glm53_dlmh plugin loaded (pid %d): %s=%r -> %s", os.getpid(), ENV, os.environ.get(ENV),
              f"installing (mode {CFG.mode}, C {CFG.C}, g {CFG.group})" if CFG.mode != "off"
              else "off, production's candidate head unchanged")
    if CFG.mode == "off":
        return
    try:
        _hook_loader()
    except Exception as exc:  # noqa: BLE001
        _log.warning("glm53_dlmh not installed (production unchanged): %r", exc)


def summary() -> dict:
    return {"mode": CFG.mode, "C": CFG.C, "group": CFG.group, "ready": HEAD.ready, "failed": HEAD.failed,
            "counters": dict(COUNTERS)}
