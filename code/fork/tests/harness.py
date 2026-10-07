"""Shared test harness (docs/DESIGN.md §E.0).

Production modules used for real (no stubs): vllm/model_executor/layers/quantization/exl3.py as the container has
it — the image's own file (byte-identical to docs/prod_exl3_reference.py) or, with
GPU_RUN_BIND="docs/ref/prod_live/overlay_exl3.py=<that path>", the launcher overlay production installs over it
(docs/ref/prod_live/overlay_exl3.py). load_prod() prints which one (sha256 and label, integrate.KNOWN_PROD_MODULES)
so every log says what it ran against. Tests call its map_topk_to_local / apply_exl3_fused_moe / _exl3_moe_launch /
build_exl3_fused_state / Exl3MoEMethod.process_weights_after_loading directly. exllamav3_ext is imported as the top-level module
(`import exllamav3_ext`; only `import exllamav3` needs flash_attn).

Weights are built exactly like production create_weights (stacked, gate/up interleaved), and the layer is
finished by production's own process_weights_after_loading (LinearEXL3 inners on slices of the stacked
Parameters -> build_exl3_fused_state pointer tables + shared temps).
"""
from __future__ import annotations

import math
import os
import sys
import time
import traceback
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for p in (str(REPO), str(REPO / "kernels"), str(REPO / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

GiB = 2 ** 30
MCG_MARKER = -877912083


# ---------------------------------------------------------------------------------------------------------
# failure / exit discipline

class Checks:
    def __init__(self) -> None:
        self.n = 0
        self.failed: list[str] = []

    def __call__(self, cond: bool, msg: str) -> bool:
        self.n += 1
        if not cond:
            self.failed.append(msg)
            print(f"  CHECK FAILED: {msg}", flush=True)
        return bool(cond)

    def summary(self) -> None:
        print(f"checks: {self.n - len(self.failed)}/{self.n} passed", flush=True)
        if self.failed:
            raise AssertionError(f"{len(self.failed)} check(s) failed; first: {self.failed[0]}")


def run_main(fn) -> None:
    t0 = time.time()
    try:
        fn()
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        print(f"RESULT: FAIL ({time.time() - t0:.1f}s)", flush=True)
        sys.exit(1)
    print(f"RESULT: PASS ({time.time() - t0:.1f}s)", flush=True)
    sys.exit(0)


# ---------------------------------------------------------------------------------------------------------
# memory guard (nodeC hosts other services)

def gpu_guard(budget_gib: float = 8.0) -> None:
    """Cap this process's CUDA caching allocator at `budget_gib` (<= 8 GiB) and refuse to start without 2x
    headroom. GB10 is unified memory: cudaMemGetInfo's 'free' excludes reclaimable page cache (it reported
    4.9 GiB free while MemAvailable was 85 GiB), so headroom = max(cuda free, /proc/meminfo MemAvailable)."""
    assert budget_gib <= 8.0, "per-test GPU budget must be <= 8 GiB"
    free, total = torch.cuda.mem_get_info()
    avail = 0
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                avail = int(line.split()[1]) * 1024
    except OSError:
        pass
    headroom = max(free, avail)
    print(f"gpu_guard: budget {budget_gib:.1f} GiB, cuda free {free / GiB:.1f} GiB, MemAvailable {avail / GiB:.1f} GiB,"
          f" total {total / GiB:.1f} GiB", flush=True)
    if headroom < 2 * budget_gib * GiB:
        raise SystemExit(f"gpu_guard: headroom {headroom / GiB:.1f} GiB < 2 x budget; refusing to run")
    torch.cuda.set_per_process_memory_fraction(budget_gib * GiB / total)


def report_peak(budget_gib: float = 8.0) -> None:
    peak = torch.cuda.max_memory_allocated()
    print(f"peak CUDA allocation {peak / GiB:.2f} GiB (budget {budget_gib:.1f} GiB)", flush=True)
    assert peak <= budget_gib * GiB


# ---------------------------------------------------------------------------------------------------------
# modules

def load_xl():
    try:
        import exllamav3_ext as xl  # noqa: F401
    except ImportError:
        import importlib.util

        so = "/usr/local/lib/python3.12/dist-packages/exllamav3_ext.cpython-312-aarch64-linux-gnu.so"
        spec = importlib.util.spec_from_file_location("exllamav3_ext", so)
        xl = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(xl)
        sys.modules["exllamav3_ext"] = xl
    return xl


def load_prod():
    import vllm.model_executor.layers.quantization.exl3 as q  # the real production module

    import integrate

    ident = integrate.prod_identity(q)
    print(f"production module: {ident['known'] or 'UNKNOWN'} (sha256 {ident['sha256']}); MX path "
          f"{'present' if hasattr(q, 'mx_enabled') else 'absent'}, GLM53_EXL3_MOE_FAST path "
          f"{'present' if hasattr(q, 'exl3_moe_fast_requested') else 'absent'}", flush=True)
    return q


def load_tf(require_fresh: bool = True, require_aot: bool = True):
    """tf_exl3_moe with its compiled extension. Asserts the active extension is the AOT module this repo
    ships (tests/run_all.sh builds it in place first; a silent fallback to the JIT build would leave the
    shipped artifact untested) and that it is newer than the kernel sources."""
    import tf_exl3_moe as tf

    ext = tf.load_ext()
    src = max((REPO / "kernels" / f).stat().st_mtime for f in ("exl3.cu", "exl3.cpp"))
    so = Path(ext.__file__)
    print(f"tf extension: {tf.EXT_SOURCE} parity={ext.parity()}", flush=True)
    if require_aot:
        assert tf.EXT_SOURCE.startswith("aot:") and so.resolve().parent == REPO.resolve(), (
            f"active extension is {tf.EXT_SOURCE}, expected the in-place AOT build under {REPO}")
    if require_fresh:
        assert so.stat().st_mtime >= src, f"stale extension {so} (older than kernels/); rebuild it"
    assert ext.parity() == 1, "tests expect the TF_PARITY=1 build"
    return tf


def prod_orig(xl):
    fn = xl.exl3_moe
    return getattr(fn, "_tf_exl3_orig", fn)


# ---------------------------------------------------------------------------------------------------------
# E.0 weights: production create_weights layout

def _signs(shape, g, dev):
    return (torch.randint(0, 2, shape, generator=g, device=dev) * 2 - 1).to(torch.float32)


def _scales(shape, scale, g, dev):
    mag = 1.0 + 0.25 * (torch.rand(shape, generator=g, device=dev) * 2 - 1)
    return (_signs(shape, g, dev) * mag * scale).to(torch.float16)


_CB_STD = None


def codebook_std() -> float:
    """std of the mcg codebook over all 16-bit states (random trellis words ~ uniform states)."""
    global _CB_STD
    if _CB_STD is None:
        import exl3_format_ref as R

        _CB_STD = float(R.mcg_values().astype("float64").std())
    return _CB_STD


class Weights:
    """One MoE layer's routed experts on this rank, production layout (docs/prod_exl3_reference.py:2070-2112)."""

    def __init__(self, n: int, K: int, N: int, dev, seed: int, g_std: float = 5.0, svh_d: float | None = None,
                 shared_suh: bool = False):
        g = torch.Generator(device=dev)
        g.manual_seed(seed)
        self.n, self.K, self.N = n, K, N
        s_suh = 1.0
        # std(g) = sqrt(K) * cb_std * s_suh * E|mag| * s_svh  (Hadamard orthonormal, x ~ N(0,1)) -> s_svh
        mag_rms = math.sqrt(1 + 0.25 ** 2 / 3)
        s_svh = g_std / (math.sqrt(K) * codebook_std() * s_suh * mag_rms * mag_rms)
        self.w13_trellis = torch.randint(-32768, 32767, (n, 2, K // 16, N // 16, 64), dtype=torch.int16,
                                         generator=g, device=dev)
        self.w13_suh = _scales((n, 2, K), s_suh, g, dev)
        if shared_suh:   # gate and up share one input rotation (production: layer._exl3_shared_w13_suh)
            self.w13_suh[:, 1].copy_(self.w13_suh[:, 0])
        self.w13_svh = _scales((n, 2, N), s_svh, g, dev)
        self.w13_mcg = torch.full((n, 2, 1), MCG_MARKER, dtype=torch.int32, device=dev)
        self.w2_trellis = torch.randint(-32768, 32767, (n, N // 16, K // 16, 64), dtype=torch.int16, generator=g,
                                        device=dev)
        # |act| ~ 5..10 after the limit; d0 = H(act * suh_d) @ Wq_d: keep suh_d ~ 1/sqrt(N) scale so fp16 d0 ~ O(10)
        self.w2_suh = _scales((n, N), 1.0 / math.sqrt(N) * 8.0, g, dev)
        self.w2_svh = _scales((n, K), 1.0 if svh_d is None else svh_d, g, dev)
        self.w2_mcg = torch.full((n, 1), MCG_MARKER, dtype=torch.int32, device=dev)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.w13_trellis, self.w13_suh, self.w13_svh,
                                                          self.w2_trellis, self.w2_suh, self.w2_svh))

    def rescale_svh_d(self, factor: float) -> None:
        self.w2_svh.copy_((self.w2_svh.float() * factor).to(torch.float16))


def make_layer(prod, W: Weights, *, method: str = "process_weights", require_ptrs: bool = True) -> torch.nn.Module:
    """A torch.nn.Module holding the stacked Parameters and every attribute production's
    create_weights sets, finished by production's own process_weights_after_loading."""
    from torch.nn import Parameter

    layer = torch.nn.Module()
    for name in ("w13_trellis", "w13_suh", "w13_svh", "w13_mcg", "w2_trellis", "w2_suh", "w2_svh", "w2_mcg"):
        layer.register_parameter(name, Parameter(getattr(W, name), requires_grad=False))
    layer._exl3_hidden_size = W.K
    layer._exl3_intermediate_local = W.N
    layer._exl3_k_words = 64
    layer._exl3_bits = 4
    layer.expert_map = None
    if method == "process_weights":
        m = prod.Exl3MoEMethod.__new__(prod.Exl3MoEMethod)   # bypass FusedMoEMethodBase(moe) config plumbing
        m.bits = 4
        m._logged = True
        m.process_weights_after_loading(layer)
    else:
        inners = [{"gate": types.SimpleNamespace(trellis=W.w13_trellis[e, 0], suh=W.w13_suh[e, 0], svh=W.w13_svh[e, 0]),
                   "up": types.SimpleNamespace(trellis=W.w13_trellis[e, 1], suh=W.w13_suh[e, 1], svh=W.w13_svh[e, 1]),
                   "down": types.SimpleNamespace(trellis=W.w2_trellis[e], suh=W.w2_suh[e], svh=W.w2_svh[e])}
                  for e in range(W.n)]
        layer._exl3_inners = inners
        prod.build_exl3_fused_state(layer, inners)
    if require_ptrs:
        assert getattr(layer, "_exl3_ptrs", None), "production did not build pointer tables"
    return layer


class SeparateExperts:
    """Second variant: one tensor per expert matrix (tests/prod_baseline.py make_experts style) -> proves the
    pointer tables are general, not tied to the stacked layout."""

    def __init__(self, n: int, K: int, N: int, dev, seed: int, like: Weights | None = None):
        g = torch.Generator(device=dev)
        g.manual_seed(seed)
        self.n, self.K, self.N = n, K, N
        self.ex = []
        for e in range(n):
            if like is not None:   # same values as the stacked layer, different memory
                d = {"gate": {"trellis": like.w13_trellis[e, 0].clone(), "suh": like.w13_suh[e, 0].clone(),
                              "svh": like.w13_svh[e, 0].clone()},
                     "up": {"trellis": like.w13_trellis[e, 1].clone(), "suh": like.w13_suh[e, 1].clone(),
                            "svh": like.w13_svh[e, 1].clone()},
                     "down": {"trellis": like.w2_trellis[e].clone(), "suh": like.w2_suh[e].clone(),
                              "svh": like.w2_svh[e].clone()}}
            self.ex.append(d)
        self.ptrs = {f"{w}_{a}": torch.tensor([int(d[w][a].data_ptr()) for d in self.ex], dtype=torch.int64, device=dev)
                     for w in ("gate", "up", "down") for a in ("trellis", "suh", "svh")}

    def regions(self) -> dict:
        """For tf.register(): each pointer must fall inside one allocation. Separate tensors are not one
        region, so give the union's bounds per kind with matrix size 1 (alignment is still checked)."""
        out = {}
        for w in ("gate", "up", "down"):
            for a in ("trellis", "suh", "svh"):
                ts = [d[w][a] for d in self.ex]
                lo = min(int(t.data_ptr()) for t in ts)
                mb = ts[0].numel() * ts[0].element_size()
                hi = max(int(t.data_ptr()) for t in ts) + mb
                out[f"{w}_{a}"] = (lo, hi - lo, 16)
        # (region check is "base <= p <= base + nbytes - mb and (p - base) % mb == 0" with mb = 16 here)
        return out


# ---------------------------------------------------------------------------------------------------------
# routing and argument capture through the real production code

def kernel_names(fn, *args) -> list[str]:
    """Names of the CUDA kernels fn(*args) launches (torch.profiler / CUPTI, demangled). Used to show which native
    kernel production's exl3_moe dispatched to: the thin-decode build's is glm53_exl3_moe_fast_kernel<4, 256,
    shared_input>, the stock one exl3_moe_kernel<...> (docs/ref/launcher_0924/patch_exl3_decode_pipeline.py)."""
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn(*args)
        torch.cuda.synchronize()
    return sorted({e.name for e in prof.events() if str(getattr(e, "device_type", "")).endswith("CUDA")})


def capture_args(prod, xl, x2d, ids, weights, layer, limit, expert_map=None):
    """Run production apply_exl3_fused_moe with a recorder in place of exllamav3_ext.exl3_moe (same __doc__, so
    production's num_active detector answers as for the real one) and return the exact positional args
    production passes. The recorder computes nothing."""
    rec = []
    real = xl.exl3_moe

    def recorder(*args):
        rec.append(args)

    recorder.__doc__ = real.__doc__
    xl.exl3_moe = recorder
    try:
        if expert_map is not None:
            layer.expert_map = expert_map
        emap = prod.pin_exl3_expert_map(layer, x2d.device)
        prod.apply_exl3_fused_moe(x2d, ids.to(torch.long), weights, layer, layer._exl3_inners, emap, float(limit))
    finally:
        xl.exl3_moe = real
        layer.expert_map = None
    assert len(rec) == 1, f"production made {len(rec)} exl3_moe calls (expected 1: tokens <= cap)"
    return rec[0]


def with_out(args: tuple, out: torch.Tensor) -> tuple:
    return args[:1] + (out,) + args[2:]


def with_temps(args: tuple, temps: tuple) -> tuple:
    return args[:5] + tuple(temps) + args[9:]


def c1_temps(layer) -> tuple:
    """Concurrency-1 temps (deterministic exl3_moe: one kernel group processes every expert in order)."""
    t = layer._exl3_fused_temps
    return tuple(torch.empty((1,) + tuple(x.shape[1:]), dtype=x.dtype, device=x.device) for x in t)


def random_ids(T: int, n: int, topk: int, g: torch.Generator, dev) -> torch.Tensor:
    return torch.stack([torch.randperm(n, generator=g)[:topk] for _ in range(T)]).to(dev)


def random_weights(T: int, topk: int, g: torch.Generator, dev) -> torch.Tensor:
    # production passes router weights incl. routed_scaling_factor (2.5 for GLM); softmax * 2.5 here
    return (torch.softmax(torch.randn(T, topk, generator=g), -1) * 2.5).to(dev)


def xin(T: int, K: int, g: torch.Generator, dev) -> torch.Tensor:
    return torch.randn(T, K, generator=g).to(torch.bfloat16).to(dev)   # production hidden dtype; exl3.py casts .half()


def seq_lengths(T: int, g: torch.Generator) -> list:
    """Split T decode rows into sequences the way DFlash batches them: each sequence contributes k + 1 rows,
    k in {4, 5, 7} (adaptive), so lengths are drawn from {5, 6, 8}; a remainder < 5 becomes one short sequence
    (T = 1, 2, 4 are a single sequence). Deterministic for a given generator state."""
    out, left = [], T
    choices = (5, 6, 8)
    while left >= 5:
        fits = [c for c in choices if c == left or left - c >= 5] or [c for c in choices if c <= left]
        L = fits[int(torch.randint(0, len(fits), (1,), generator=g))]
        out.append(L)
        left -= L
    if left:
        out.append(left)
    return out


def correlated_ids(T: int, n: int, topk: int, pool: int, g: torch.Generator, dev) -> torch.Tensor:
    """Speculative-decoding-like routing: rows are grouped into sequences (seq_lengths); every row of a sequence
    draws its topk distinct experts from a pool of `pool` experts local to that sequence (pools of different
    sequences are independent random subsets of the n experts). pool = n gives random routing; smaller pools
    give fewer distinct experts per row (about 3..8 for pool 24..n at topk 8)."""
    rows = []
    for L in seq_lengths(T, g):
        local = torch.randperm(n, generator=g)[:max(pool, topk)]
        for _ in range(L):
            rows.append(local[torch.randperm(local.numel(), generator=g)[:topk]])
    return torch.stack(rows).to(dev)


def routing_ids(kind: str, T: int, n: int, topk: int, g: torch.Generator, dev) -> torch.Tensor:
    """kind: "rand" (independent top-k per row) or "corr<pool>" (correlated_ids with that pool, e.g. corr40)."""
    if kind == "rand":
        return random_ids(T, n, topk, g, dev)
    assert kind.startswith("corr"), kind
    return correlated_ids(T, n, topk, int(kind[4:]), g, dev)
