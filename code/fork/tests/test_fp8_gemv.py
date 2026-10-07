"""kernels/fp8_gemv.cu correctness on nodeC (production image, GB10). Every check prints; exit 1 on any failure.

L  layout: the Marlin FP8 repack (production's prepare_fp8_layer_for_marlin) unpacked with the formula in the kernel
   header equals the e4m3 weight byte for byte, and the stored scale un-permuted x 2^-120 equals the per-channel scale,
   for every production shape (incl. the padded KDA in_proj, N 12576 -> Npad 12608).
N  numerics vs float64: y vs x.double() @ (fp8 * scale).double().T for every production shape and
   M in {1,2,4,5,6,8,12,16,24,32,48,64} with the shipped config (the table's, else a default one): every element within
   0.5 bf16 ulp + an fp32-accumulation bound (2^-20 * sum|x*w|), i.e. the only freedom is the fp32 summation order;
   fraction correctly rounded and agreement with production's Marlin are reported and bounded.
C  every instantiated configuration (W8 KW 1/2/4/8 at MB 1..8, W16 KW 4/8/16 at MB 1..2, W4 KW 1/2/4 at MB 3..8) on
   the padded shape, output
   written through a view of a NaN-poisoned larger buffer: no write outside [M, N].
B  bias (Marlin-permuted, as prepare_fp8_layer_for_marlin stores it): y(bias) == bf16(y(no bias) + bias) bitwise
   (Marlin's epilogue order) and agrees with Marlin(bias).
S  strided activations (a column slice of a wider tensor, as KDA's f_a / g_a are) == contiguous, bitwise.
E  edge numerics: e4m3 subnormals / zeros / +-448, activations over 2^-60..2^10: the shipped kernel stays within the
   bound; the Marlin-style conversion without the 2^120 rescale (test-only instance) is reported for comparison.
G  CUDA graph: capture at M = 5 / 8 / 16 / 64, refill x, replay == eager, bitwise; repeated calls bitwise identical.
X  input checks raise (fp16 x, M = 0 / 65, misaligned x, wrong weight shape, unknown config) instead of launching.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Checks, run_main, gpu_guard, report_peak  # noqa: E402
import torch  # noqa: E402
import fp8_gemv as G  # noqa: E402
from fp8_bench_common import UNIQUE_SHAPES, quantize_like_prod, marlin  # noqa: E402
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_permute_bias, marlin_pad_dim  # noqa: E402
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_layer_for_marlin  # noqa: E402

dev = "cuda"
DEFAULT_CFG = (8, 8, 2, True)
MS = (1, 2, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64)


def unpack(q, n):
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


def bf16_ulp(a):
    """ulp of bf16 at |a| (float64 tensor); subnormal floor 2^-133."""
    e = torch.floor(torch.log2(a.abs().clamp(min=2.0 ** -126)))
    return torch.pow(2.0, e - 7)


def check_vs_f64(y, x, fp8, sc, bias=None, rows=8192):
    """(max err / allowed, fraction correctly rounded) of bf16 y vs the float64 reference, chunked over N."""
    n = fp8.shape[0]
    worst, exact, tot = 0.0, 0, 0
    xd = x.double()
    for a in range(0, n, rows):
        deq = fp8[a:a + rows].double() * sc[a:a + rows].double()[:, None]
        ref = xd @ deq.t()
        if bias is not None:
            ref = ref + bias[a:a + rows].double()
        S = xd.abs() @ deq.abs().t()
        acc = S * 2.0 ** -20
        yy = y[:, a:a + rows].double()
        err = (yy - ref).abs()
        allowed = 0.5 * bf16_ulp(ref.abs() + acc) + acc + 1e-300
        worst = max(worst, (err / allowed).max().item())
        exact += (err <= 0.5 * bf16_ulp(ref) * (1 + 1e-9)).sum().item()
        tot += err.numel()
    return worst, exact / tot


def main():
    gpu_guard(8.0)
    ck = Checks()
    E = G.ext()
    print(f"extension {E.__file__ if hasattr(E, '__file__') else E}, VERSION {getattr(E, 'VERSION', None)}")
    g = torch.Generator(device=dev).manual_seed(0)

    print("== L layout")
    layers = {}
    for n, k in UNIQUE_SHAPES:
        w = (torch.randn(n, k, device=dev, generator=g) * 0.02 *
             torch.exp(torch.randn(n, 1, device=dev, generator=g))).to(torch.bfloat16)
        layer, fp8, sc = quantize_like_prod(w)
        del w
        q = layer.weight.data
        npad = q.shape[1] // 4
        ok_w = torch.equal(unpack(q, n), fp8.view(torch.uint8))
        ok_s = torch.equal(layer.weight_scale.data.view(-1)[perm_idx(n, dev)].float() * 2.0 ** -120, sc.float())
        pad_zero = npad == n or (layer.weight_scale.data.view(-1)[perm_idx(npad, dev)[n:]] == 0).all().item()
        print(f"   N={n:6d} K={k:5d} Npad={npad:6d}: bytes {ok_w} scale {ok_s} padded scales zero {pad_zero}")
        ck(ok_w and ok_s and pad_zero, f"layout {n}x{k}")
        layers[(n, k)] = (layer, fp8, sc)

    print("== N numerics vs float64 and vs Marlin (shipped config per M; '*' = table config, else default)")
    for (n, k), (layer, fp8, sc) in layers.items():
        npad = layer.weight.shape[1] // 4
        line = f"   {n:6d}x{k:5d}:"
        for M in MS:
            cfg = G.select_config(npad, k, M)
            tag = "*" if cfg else ""
            cfg = cfg or DEFAULT_CFG
            x = torch.randn(M, k, device=dev, generator=g).to(torch.bfloat16)
            y = E.fp8_gemv(x, layer.weight, layer.weight_scale.view(-1), None, n, k, *cfg)
            ym = marlin(layer, x, n, k)
            r, fr = check_vs_f64(y, x, fp8, sc)
            rm, frm = check_vs_f64(ym, x, fp8, sc)
            same = (y == ym).float().mean().item()
            ck(r <= 1.0, f"N {n}x{k} M={M} cfg={cfg}: err/allowed {r:.3f} > 1")
            ck(fr >= 0.999, f"N {n}x{k} M={M}: correctly rounded {fr:.5f} < 0.999")
            ck(same >= 0.99, f"N {n}x{k} M={M}: identical to Marlin {same:.4f} < 0.99")
            line += f" M{M}{tag}:{r:.2f}/{fr*100:.2f}%/{rm:.2f}/{frm*100:.2f}%/{same*100:.1f}%"
        print(line)
    print("   (per M: new err/allowed / new correctly-rounded % / Marlin err/allowed / Marlin correctly-rounded % / "
          "new == Marlin %)")

    print("== C every instantiated configuration, padded shape, poisoned output buffer")
    layer, fp8, sc = layers[(12576, 4096)]
    n, k = 12576, 4096
    cfgs = [(8, kw, 2, pol) for kw in (1, 2, 4, 8) for pol in (True, False)] + [(16, kw, 2, True) for kw in (4, 8, 16)] \
        + [(4, kw, 2, True) for kw in (1, 2, 4)]
    valid = lambda c, M: not (c[0] == 16 and M > 16) and not (c[0] == 4 and M <= 16)  # noqa: E731
    for M in (5, 12, 20, 32, 40, 48, 56, 64):
        x = torch.randn(M, k, device=dev, generator=g).to(torch.bfloat16)
        ref = None
        for cfg in cfgs:
            if not valid(cfg, M):
                continue
            big = torch.full((M + 8, n + 72), float("nan"), dtype=torch.bfloat16, device=dev)
            out = big[:M, :n]
            E.fp8_gemv_out(out, x, layer.weight, layer.weight_scale.view(-1), None, n, k, cfg[0], cfg[1], cfg[2],
                           cfg[3], True)
            clean = torch.isnan(big[M:]).all().item() and torch.isnan(big[:M, n:]).all().item()
            if ref is None:
                r, _ = check_vs_f64(out, x, fp8, sc)
                ck(r <= 1.0, f"C M={M} cfg={cfg}: err/allowed {r:.3f}")
                ref = out.clone()
            same = torch.equal(out, ref)
            ck(clean, f"C M={M} cfg={cfg}: wrote outside [M, N]")
            ck(torch.isfinite(out).all().item(), f"C M={M} cfg={cfg}: non-finite output")
            # different KW = different fp32 summation order: allow 1-ulp flips, but equal within the bound
            if not same:
                r, _ = check_vs_f64(out, x, fp8, sc)
                ck(r <= 1.0, f"C M={M} cfg={cfg}: err/allowed {r:.3f}")
        print(f"   M={M}: {len([c for c in cfgs if valid(c, M)])} configs ok")

    print("== B bias")
    for (n, k) in ((4096, 4096), (12576, 4096)):
        layer, fp8, sc = layers[(n, k)]
        npad = layer.weight.shape[1] // 4
        bias = (torch.randn(n, device=dev, generator=g) * 0.1).to(torch.bfloat16)
        bperm = marlin_permute_bias(marlin_pad_dim(bias, n, npad))
        for M in (1, 8, 16):
            x = torch.randn(M, k, device=dev, generator=g).to(torch.bfloat16)
            y = E.fp8_gemv(x, layer.weight, layer.weight_scale.view(-1), bperm, n, k, *DEFAULT_CFG)
            y0 = E.fp8_gemv(x, layer.weight, layer.weight_scale.view(-1), None, n, k, *DEFAULT_CFG)
            ym = marlin(layer, x, n, k, bias=bperm)
            order = torch.equal(y, (y0.float() + bias.float()).to(torch.bfloat16))
            same = (y == ym).float().mean().item()
            print(f"   {n}x{k} M={M}: y(bias) == bf16(y + b) {order}, == Marlin(bias) {same*100:.2f}%")
            ck(order and same >= 0.99, f"B {n}x{k} M={M}")

    print("== S strided activations")
    layer, fp8, sc = layers[(4096, 128)]
    for M in (1, 5, 8, 16):
        wide = torch.randn(M, 12576, device=dev, generator=g).to(torch.bfloat16)
        xs = wide[:, 12320:12448]                      # f_a's position in the KDA in_proj output
        y1 = E.fp8_gemv(xs, layer.weight, layer.weight_scale.view(-1), None, 4096, 128, *DEFAULT_CFG)
        y2 = E.fp8_gemv(xs.contiguous(), layer.weight, layer.weight_scale.view(-1), None, 4096, 128, *DEFAULT_CFG)
        ck(torch.equal(y1, y2), f"S M={M}: strided != contiguous")
    print("   strided == contiguous (bitwise) at M = 1, 5, 8, 16")

    print("== E edge numerics (e4m3 subnormals / zeros / +-448, scales 2^-40..2^1, activation rows at 2^-60..2^10)")
    n, k = 4096, 4096
    raw = torch.randint(0, 256, (n, k), device=dev, generator=g, dtype=torch.int32).to(torch.uint8)
    raw[(raw & 0x7F) == 0x7F] = 0x3F                  # no NaN codes
    sub = torch.rand(n, k, device=dev, generator=g) < 0.3
    raw[sub] = raw[sub] & 0x87                        # 30 % subnormals (exponent field 0) incl. zeros
    raw[:, :8] = torch.tensor([0x7E, 0xFE, 0x00, 0x80, 0x01, 0x81, 0x07, 0x87], dtype=torch.uint8, device=dev)
    allsub = torch.arange(n, device=dev) % 16 == 3    # some rows made of subnormals only
    raw[allsub] = raw[allsub] & 0x87
    fp8 = raw.view(torch.float8_e4m3fn)
    # production scale = amax/448 of a BF16 row: 2^-40..2^1 covers every realistic row (x 2^120 stays finite in bf16)
    scale = torch.pow(2.0, torch.rand(n, device=dev, generator=g) * 41 - 40).to(torch.bfloat16)
    lay = torch.nn.Module()
    lay.output_size_per_partition, lay.input_size_per_partition = n, k
    lay.orig_dtype = torch.bfloat16
    lay.weight = torch.nn.Parameter(fp8.clone(), requires_grad=False)
    lay.weight_scale = torch.nn.Parameter(scale.clone(), requires_grad=False)
    lay.weight_block_size = None
    prepare_fp8_layer_for_marlin(lay, size_k_first=False)
    ck(torch.isfinite(lay.weight_scale).all().item(), "E: stored Marlin scales finite")
    for M in (1, 8):
        rowmag = torch.pow(2.0, torch.linspace(-60, 10, M, device=dev)).view(M, 1)
        x = (torch.randn(M, k, device=dev, generator=g) * rowmag).to(torch.bfloat16)
        y = E.fp8_gemv(x, lay.weight, lay.weight_scale.view(-1), None, n, k, *DEFAULT_CFG)
        y0 = torch.empty_like(y)
        E.fp8_gemv_out(y0, x, lay.weight, lay.weight_scale.view(-1), None, n, k, 8, 8, 2, True, False)
        ym = marlin(lay, x, n, k)
        r, fr = check_vs_f64(y, x, fp8, scale)
        r0, fr0 = check_vs_f64(y0, x, fp8, scale)
        rm, frm = check_vs_f64(ym, x, fp8, scale)
        print(f"   M={M}: shipped err/allowed {r:.3f} rounded {fr*100:.3f}% | without 2^120 rescale {r0:.3g} "
              f"{fr0*100:.3f}% | Marlin {rm:.3g} {frm*100:.3f}%")
        ck(r <= 1.0 and fr >= 0.999, f"E M={M}: shipped kernel err/allowed {r:.3f}, rounded {fr:.5f}")

    print("== G CUDA graphs and determinism")
    layer, fp8, sc = layers[(4096, 4096)]
    n, k = 4096, 4096
    for M in (5, 8, 16, 64):
        cfg = G.select_config(4096, 4096, M) or DEFAULT_CFG
        x = torch.randn(M, k, device=dev, generator=g).to(torch.bfloat16)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            E.fp8_gemv(x, layer.weight, layer.weight_scale.view(-1), None, n, k, *cfg)
        torch.cuda.current_stream().wait_stream(s)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            yg = E.fp8_gemv(x, layer.weight, layer.weight_scale.view(-1), None, n, k, *cfg)
        ok = True
        for _ in range(3):
            x.copy_(torch.randn(M, k, device=dev, generator=g).to(torch.bfloat16))
            gr.replay()
            ye = E.fp8_gemv(x, layer.weight, layer.weight_scale.view(-1), None, n, k, *cfg)
            ye2 = E.fp8_gemv(x, layer.weight, layer.weight_scale.view(-1), None, n, k, *cfg)
            torch.cuda.synchronize()
            ok &= torch.equal(yg, ye) and torch.equal(ye, ye2)
        ck(ok, f"G M={M}: replay != eager or not deterministic")
        print(f"   M={M} cfg={cfg}: replay == eager == repeat (bitwise) {ok}")

    print("== X input checks raise")
    x = torch.randn(8, k, device=dev).to(torch.bfloat16)
    W, Sc = layer.weight, layer.weight_scale.view(-1)
    bad = {
        "fp16 x": lambda: E.fp8_gemv(x.half(), W, Sc, None, n, k, *DEFAULT_CFG),
        "M=65": lambda: E.fp8_gemv(torch.randn(65, k, device=dev).to(torch.bfloat16), W, Sc, None, n, k, *DEFAULT_CFG),
        "M=0": lambda: E.fp8_gemv(x[:0], W, Sc, None, n, k, *DEFAULT_CFG),
        "misaligned x": lambda: E.fp8_gemv(torch.randn(8 * k + 1, device=dev).to(torch.bfloat16)[1:].view(8, k),
                                           W, Sc, None, n, k, *DEFAULT_CFG),
        "K mismatch": lambda: E.fp8_gemv(x[:, :2048], W, Sc, None, n, 2048, *DEFAULT_CFG),
        "N > Npad": lambda: E.fp8_gemv(x, W, Sc, None, n + 64, k, *DEFAULT_CFG),
        "unknown cfg": lambda: E.fp8_gemv(x, W, Sc, None, n, k, 8, 16, 2, True),
        "U=4": lambda: E.fp8_gemv(x, W, Sc, None, n, k, 8, 8, 4, True),
        "W16 MB8": lambda: E.fp8_gemv(torch.randn(64, k, device=dev).to(torch.bfloat16), W, Sc, None, n, k, 16, 8, 2, True),
        "W4 MB1": lambda: E.fp8_gemv(x, W, Sc, None, n, k, 4, 4, 2, True),
        "fp32 bias": lambda: E.fp8_gemv(x, W, Sc, torch.zeros(n, device=dev), n, k, *DEFAULT_CFG),
    }
    for name, fn in bad.items():
        try:
            fn()
            torch.cuda.synchronize()
            ck(False, f"X {name}: did not raise")
        except RuntimeError as exc:
            print(f"   {name}: raised ({str(exc).splitlines()[0][:90]})")
    torch.cuda.synchronize()
    report_peak(8.0)
    ck.summary()


run_main(main)
