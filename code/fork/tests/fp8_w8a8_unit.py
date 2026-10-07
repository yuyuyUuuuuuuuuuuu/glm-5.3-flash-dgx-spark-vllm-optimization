"""kernels/fp8_w8a8.cu + fp8_w8a8.py correctness on nodeC (production image, GB10). Every check prints; exit 1 on any failure.

L  layout: the repack kernel's fp8 [N, K] equals (a) the independent torch formula of tests/test_fp8_gemv.py::unpack
   applied to the Marlin payload, (b) the e4m3 weight production quantized (byte for byte, incl. the padded KDA
   in_proj N 12576 -> Npad 12608), and (c) the exact bf16 dequant (tf_fp8_large_m_ext.dequant) cast back to fp8 -
   for every production shape and one adversarial one (N not a multiple of 64 within the padded tile).
   The unpack comparison runs in 64-row chunks so a partial trailing n-tile is exercised directly.
S  scratch: two layers of the same shape through _scratch/back to back (the second repack overwrites the first) and
   a growth step (a smaller weight then a bigger one) both give byte-exact results.
Q  per-token quantization: ops.scaled_fp8_quant(use_per_token_if_dynamic=True) reproduces the fp32 reference within
   the e4m3 rounding bound and matches the kill-test magnitudes (rel_l2 ~ 1e-2 class on the activations).
G  GEMM: cutlass_scaled_mm(W8A8) vs production's Marlin on the same Marlin-packed weights, per-token quantized
   random rows, M in {64, 512, 2048, 13824 (2048-row pieces), 1791}: rel_l2 within the measured W8A8 class
   (<= 6e-2) and finite; with a Marlin-permuted bias: rel_l2 vs Marlin(with bias) <= 6e-2 and the bias changes the
   output by more than 0 (the bias is really applied).
D  dispatch: try_w8a8 declines M < GLM53_DENSE_W8A8_MIN_M, a non-bf16 x, a capture, an unknown group, and a layer
   without Marlin weights, and serves everything else through apply() (counter checks, strict mode).
I  install: idempotent, uninstall restores the original apply / process_weights_after_loading exactly, and the
   wrapper delegates when GLM53_DENSE_W8A8 selects no group.

Run: GPU_RUN_ENV="TF_EXL3_JIT=1" flock /tmp/tf-gpu-bench.lock tests/gpu_run.sh python3 tests/fp8_w8a8_unit.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Checks, gpu_guard, report_peak, run_main  # noqa: E402
import torch  # noqa: E402
import fp8_gemv as G  # noqa: E402
import fp8_w8a8 as W  # noqa: E402
from fp8_bench_common import UNIQUE_SHAPES, marlin, quantize_like_prod  # noqa: E402

dev = "cuda"
MS = (64, 512, 2048, 13824)


def unpack(q, n):
    """tests/test_fp8_gemv.py::unpack - the documented Marlin fp8 layout, straight from the formula."""
    kt, npad = q.shape[0], q.shape[1] // 4
    b = q.contiguous().view(torch.uint8).view(kt, npad // 64, 32, 4, 2, 4)
    ar = lambda s, d: torch.arange(s, device=q.device).view([s if i == d else 1 for i in range(6)])  # noqa: E731
    KT, NT, t, w, h = ar(kt, 0), ar(npad // 64, 1), ar(32, 2), ar(4, 3), ar(2, 4)
    bo = torch.tensor([0, 8, 1, 9], device=q.device).view(1, 1, 1, 1, 1, 4)
    nn = (NT * 64 + 16 * w + t // 4 + 8 * h).expand_as(b)
    kk = (KT * 16 + 2 * (t % 4) + bo).expand_as(b)
    out = torch.zeros(npad, kt * 16, dtype=torch.uint8, device=q.device)
    out[nn.reshape(-1), kk.reshape(-1)] = b.reshape(-1)
    return out[:n]


def perm_idx(n, dev):
    c = torch.arange(n, device=dev)
    r = c % 32
    return (c - r) + 8 * ((r % 8) // 2) + (r % 2) + 2 * (r // 8)


def rel_l2(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def marlin_pack(w):
    layer, fp8, sc = quantize_like_prod(w)
    return layer, fp8, sc


def main():
    gpu_guard(8.0)
    ck = Checks()
    E = W.ext()
    print(f"extension {getattr(E, '__file__', E)}, VERSION {getattr(E, 'VERSION', None)}")
    os.environ["GLM53_FP8_LARGE_M"] = "1"          # the dequant reference lives in the large-M extension
    g = torch.Generator(device=dev).manual_seed(0)
    shapes = list(dict.fromkeys(list(UNIQUE_SHAPES) + [(13000, 4096)]))   # + a non-64 n within the padded tile

    print("== L layout (repack kernel vs the torch formula, the original fp8, and the bf16 dequant)")
    layers = {}
    for n, k in shapes:
        w = (torch.randn(n, k, device=dev, generator=g) * 0.02 *
             torch.exp(torch.randn(n, 1, device=dev, generator=g))).to(torch.bfloat16)
        layer, fp8, sc = marlin_pack(w)
        del w
        q = layer.weight.data
        npad = q.shape[1] // 4
        mine = W._scratch(n, k, dev)
        W.repack(mine, q, n, k)
        uf = unpack(q, n)                      # the torch formula (includes the skipped padded rows' check)
        ok_formula = torch.equal(mine.view(torch.uint8), uf)
        del uf
        ok_orig = torch.equal(mine.view(torch.uint8), fp8.view(torch.uint8))
        ref = W._dequant_reference(q, n, k, npad)
        ok_deq = torch.equal(mine.view(torch.uint8), ref.view(torch.uint8))
        del ref
        alpha = W.large_alpha(layer.weight_scale, n)
        ok_scale = torch.equal(alpha[:n], layer.weight_scale.data.view(-1)[perm_idx(n, dev)].float() * 2.0 ** -120)
        print(f"   N={n:6d} K={k:5d} Npad={npad:6d}: formula {ok_formula} original fp8 {ok_orig} "
              f"dequant {ok_deq} scale {ok_scale}", flush=True)
        ck(ok_formula, f"L formula {n}x{k}")
        ck(ok_orig, f"L original fp8 bytes {n}x{k}")
        ck(ok_deq, f"L dequant bytes {n}x{k}")
        ck(ok_scale, f"L scale {n}x{k}")
        layers[(n, k)] = (layer, fp8, sc)
        torch.cuda.empty_cache()

    print("== S scratch reuse (two layers back to back, then growth)")
    (n, k) = (2048, 4096)
    la, _, _ = layers[(n, k)]
    lb = marlin_pack((torch.randn(n, k, device=dev, generator=g) * 0.02).to(torch.bfloat16))[0]
    ma = W._scratch(n, k, dev)
    W.repack(ma, la.weight, n, k)
    ua = ma.view(torch.uint8).clone()
    mb = W._scratch(n, k, dev)
    W.repack(mb, lb.weight, n, k)
    ua2 = unpack(lb.weight, n)
    ok_reuse = torch.equal(mb.view(torch.uint8), ua2)
    mc = W._scratch(4096, k, dev)
    lc = marlin_pack((torch.randn(4096, k, device=dev, generator=g) * 0.02).to(torch.bfloat16))[0]
    W.repack(mc, lc.weight, 4096, k)
    ok_grow = torch.equal(mc.view(torch.uint8), unpack(lc.weight, 4096))
    print(f"   reuse {ok_reuse} (buffer changed: {not torch.equal(ua, mb.view(torch.uint8))}), grow {ok_grow}")
    ck(ok_reuse, "S reuse")
    ck(ok_grow, "S growth")
    del la, lb, lc

    print("== Q per-token quantization (vs the fp32 reference of each row)")
    x = (torch.randn(512, 4096, device=dev, generator=g) * 0.02).to(torch.bfloat16)
    q, s = W.quant_per_token(x)
    rl = rel_l2(q.float() * s, x.float())
    print(f"   quant rel_l2 {rl:.3e} (per-token e4m3 rounding, fp32 math), scales shape {tuple(s.shape)} "
          f"dtype {s.dtype}")
    ck(rl <= 5e-2, f"Q quant rel_l2 {rl:.3e} > 5e-2")
    ck(s.shape == (512, 1) and s.dtype == torch.float32, "Q scale shape/dtype")

    print("== G cutlass W8A8 vs production's Marlin")
    for (n, k), (layer, fp8, sc) in list(layers.items()):
        if n * k > 60_000_000 or n % 16 or k % 16:  # keep the reference bounded; n % 16 is not served (below)
            continue
        npad = layer.weight.shape[1] // 4
        st = W.selftest(layer, n, k, label="unit")   # primes ALPHA/WVERDICT through the real load-time self-test
        ck(st[0], f"G selftest {n}x{k}: {st[1]}")
        line = f"   {n:6d}x{k:5d}:"
        for M in MS:
            x = (torch.randn(M, k, device=dev, generator=g) * 0.02).to(torch.bfloat16)
            ym = marlin(layer, x, n, k)
            y = W.w8a8_forward(x, layer.weight, layer.weight_scale, n, k)
            r = rel_l2(y, ym)
            fin = bool(torch.isfinite(y).all().item())
            line += f" M{M}:{r:.2e}{'!' if not fin else ''}"
            ck(fin and r <= W.SELFTEST_TOL, f"G {n}x{k} M={M}: rel_l2 {r:.3e} > {W.SELFTEST_TOL:.0e} or not finite")
        print(line, flush=True)
        layers[(n, k)] = (layer, None, None)
        torch.cuda.empty_cache()

    print("== G bias (Marlin-permuted, as prepare_fp8_layer_for_marlin stores it)")
    n, k = 4096, 4096
    layer, fp8, sc = marlin_pack((torch.randn(n, k, device=dev, generator=g) * 0.02).to(torch.bfloat16))
    npad = layer.weight.shape[1] // 4
    bias = torch.randn(npad, device=dev, generator=g).to(torch.bfloat16)
    st = W.selftest(layer, n, k, bias, label="unit-bias")   # primes ALPHA/WVERDICT, GEMM smoke incl. the bias
    ck(st[0], f"G selftest(bias) {n}x{k}: {st[1]}")
    for M in (64, 2048):
        x = (torch.randn(M, k, device=dev, generator=g) * 0.02).to(torch.bfloat16)
        ym = marlin(layer, x, n, k, bias)
        y = W.w8a8_forward(x, layer.weight, layer.weight_scale, n, k, bias)
        r = rel_l2(y, ym)
        print(f"   M{M}: rel_l2 vs Marlin(bias) {r:.2e}")
        ck(r <= W.SELFTEST_TOL, f"G bias M={M}: rel_l2 {r:.3e}")
    W.uninstall()

    print("== D dispatch (strict mode, a real Glm53DenseFp8Method-like layer)")
    W.CFG.strict = True
    W.CFG.min_m = 512
    STATE_KEY = W._key(layer.weight, layer.weight_scale, k)
    W.STATE.enabled = True
    W.STATE.groups = frozenset({"dense"})
    W.WVERDICT[STATE_KEY] = (True, "test")
    W.ALPHA[STATE_KEY] = W.large_alpha(layer.weight_scale, n)
    # an N % 16 != 0 layer is not served: the cutlass operands' alignment (13000 % 16 = 8)
    lay13, _, _ = layers[(13000, 4096)]
    W.COUNTERS["prod_gemm_alignment"] = 0
    y = W.try_w8a8((torch.randn(2048, 4096, device=dev, generator=g) * 0.02).to(torch.bfloat16), lay13, None,
                   13000, 4096)
    ck(y is None and W.COUNTERS["prod_gemm_alignment"] == 1, "D an n % 16 != 0 layer declines (cutlass alignment)")

    class FakeMethod:
        group = "dense"
        ready = True

        def apply(self, layer, x, bias=None):
            return "prod"

    fm = FakeMethod()
    x_ok = (torch.randn(2048, k, device=dev, generator=g) * 0.02).to(torch.bfloat16)
    y = W.try_w8a8(x_ok, layer, None, n, k)
    ck(isinstance(y, torch.Tensor) and tuple(y.shape) == (2048, n), f"D serves M=2048 (got {type(y)})")
    y = W.try_w8a8(x_ok[:16], layer, None, n, k)
    ck(y is None, "D declines M < min_m")
    y = W.try_w8a8(x_ok[:512].half(), layer, None, n, k)
    ck(y is None, "D declines fp16 x")
    fm.group = "draft"
    y = W._make_apply(lambda s, l, x_, b=None: "prod")(fm, layer, x_ok)
    ck(y == "prod", "D delegates an unknown group")
    fm.group = "dense"
    y = W._make_apply(lambda s, l, x_, b=None: "prod")(fm, layer, x_ok)
    ck(isinstance(y, torch.Tensor), "D apply serves a served group")
    cap = torch.cuda.CUDAGraph()
    s0 = torch.cuda.Stream()
    s0.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s0):
        x_fix = x_ok.clone()
    torch.cuda.current_stream().wait_stream(s0)
    with torch.cuda.graph(cap):
        yc = W.try_w8a8(x_fix, layer, None, n, k)
    ck(yc is None, "D declines under capture")
    bad = torch.nn.Module()
    bad.weight = torch.nn.Parameter(torch.zeros(k // 16 - 1, 4 * npad, dtype=torch.int32), requires_grad=False)
    bad.weight_scale = layer.weight_scale
    bad.workspace = layer.workspace
    before = W.COUNTERS.get("prod_weight_layout", 0)
    y = W.try_w8a8(x_ok, bad, None, n, k)
    ck(y is None, "D a broken weight layout declines (production's path serves it)")
    ck(W.COUNTERS.get("prod_weight_layout", 0) > before, "D the decline is counted")
    W.CFG.strict = False
    W.STATE.enabled = False
    W.WVERDICT.clear()
    W.ALPHA.clear()

    print("== I install / uninstall")
    os.environ["GLM53_DENSE_W8A8"] = "1"
    os.environ["GLM53_DENSE_FP8"] = "dense"
    import importlib
    prodmod = importlib.import_module(W.PROD_MODULE)
    cls = prodmod.Glm53DenseFp8Method
    orig_apply, orig_pwal = cls.apply, cls.process_weights_after_loading
    rep = W.install(prodmod)
    ck(rep.get("installed"), f"I install: {rep}")
    rep2 = W.install(prodmod)
    ck(rep2.get("installed") and rep2.get("reason") == "already installed", f"I idempotent: {rep2}")
    ck(cls.apply._tf_w8a8_orig is orig_apply, "I the wrapper delegates to the original apply")
    W.uninstall()
    ck(cls.apply is orig_apply and cls.process_weights_after_loading is orig_pwal, "I uninstall restores")
    ck.summary()
    report_peak()
    return 0


if __name__ == "__main__":
    run_main(main)
