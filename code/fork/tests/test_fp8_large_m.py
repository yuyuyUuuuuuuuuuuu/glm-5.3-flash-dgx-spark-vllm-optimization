"""kernels/fp8_large_m.cu + fp8_gemv.large_forward correctness on nodeC (production image, GB10), both GEMM backends
(tilelang: the fused-scale TileLang kernel; cublas: torch.mm fp32-out + scale_cast). Exit 1 on any failure.

D  dequant: rows n0..n0+nc of the Marlin repack -> bf16 equals fp8.to(bfloat16) bitwise, for every large-M production
   shape (incl. the padded KDA in_proj N 12576 -> Npad 12608), unaligned n0 / nc, and an e4m3 edge-value weight
   (all 256 byte patterns except NaN: subnormals, zeros, +-448); nothing written outside [nc, K].
S  scale_cast == torch's (y32 * alpha).to(bf16) bitwise; with bias == (that + bias).to(bf16) (Marlin's order);
   writes only its column slice (NaN-poisoned neighbours).
A  alpha: the un-permuted fp32 scale equals the stored per-channel scale (bitwise), zero padding after N.
F  the fp32 intermediate: large_forward (both backends) == bf16(torch.mm(x, dequant(W)^T, fp32) * scale) bitwise, and
   |y32 - float64 dot| <= (K/16 + 2) * 2^-23 * sum|x*w| (fp32 accumulation over K/16 tensor-core k-steps,
   round-toward-zero allowed): the only freedom is the fp32 summation.
N  numerics of large_forward vs float64 and vs production's Marlin, synthetic weights (per-row spread) and REAL
   weights (GLM-5.3-Flash BF16 samples: KDA in_proj rank-0 shards of layers 1/20/42, dense gate_up of layer 1, the
   DFlash2 drafter fc), activations with outlier channels, M in {512, 1791, 2202, 4608, 13824} (float64 on a row
   subset incl. every chunk boundary). Checked: every element within 0.5 bf16 ulp + the fp32 bound above (scaled);
   >= 99 % correctly rounded and within 0.5 points of Marlin's own fraction; >= 99 % bitwise equal to Marlin and
   <= 0.1 % more than 1 ulp apart. Reported side by side for new and Marlin: err / (0.5 ulp + 2^-20 sum|x*w|) (a
   tight "typical fp32" yardstick, not a bound), fraction correctly rounded; |new - Marlin| histogram, rel_l2.
C  chunk plans: large_plan() within the budget for both backends; results for several (nc, mc) plans and both backends
   agree with the default (reported bitwise %, all within the float64 bound); repeated calls bitwise identical.
T  the TileLang kernel on odd shapes: M in {1, 7, 129, 777}, column chunks that are not multiples of the 256-wide
   tile, an output that is a column slice of a wider tensor (NaN-poisoned neighbours untouched): bitwise the same
   rows / columns of one full M = 777 call (a row's result does not depend on M or on the chunking) and within the
   float64 bound; a strided x.
X  input checks raise instead of launching (wrong dtype / shape / alignment).
"""
import json
import os
import struct
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Checks, run_main, gpu_guard  # noqa: E402
import torch  # noqa: E402
import fp8_gemv as G  # noqa: E402
from fp8_bench_common import quantize_like_prod, marlin  # noqa: E402

dev = "cuda"
SAMPLES = os.environ.get("BF16_SAMPLES", os.path.join(os.environ.get("TF_EXL3_MODELS") or os.path.expanduser("~/models"), "GLM-5.3-Flash-EXL3-TR3-4bpw-partial/bf16_samples"))
DRAFTER = os.environ.get("DFLASH2", os.path.join(os.environ.get("TF_EXL3_MODELS") or os.path.expanduser("~/models"), "GLM-5.3-Flash-DFlash2-dc77ff1c/model.safetensors"))


def bf16_ulp(a):
    e = torch.floor(torch.log2(a.abs().clamp(min=2.0 ** -126)))
    return torch.pow(2.0, e - 7)


def vs_f64(y, x, fp8, sc, rows, bias=None):
    """(max err / (0.5 ulp + 2^-20 sum|xw|), max err / (0.5 ulp + fp32 bound), fraction correctly rounded) of bf16
    y[rows] vs float64, chunked over N. fp32 bound = (K/16 + 2) * 2^-23 * sum|x*w| (+ the scale product's rounding)."""
    n, k = fp8.shape
    gam = (k / 16 + 2) * 2.0 ** -23
    xd = x[rows].double()
    worst, worst_b, exact, tot = 0.0, 0.0, 0, 0
    for a in range(0, n, 4096):
        deq = fp8[a:a + 4096].double() * sc[a:a + 4096].double()[:, None]
        ref = xd @ deq.t()
        if bias is not None:
            ref = ref + bias[a:a + 4096].double()
        S = xd.abs() @ deq.abs().t()
        acc = S * 2.0 ** -20
        err = (y[rows, a:a + 4096].double() - ref).abs()
        allowed = 0.5 * bf16_ulp(ref.abs() + acc) + acc + 1e-300
        fb = S * gam + ref.abs() * 2.0 ** -24
        allowed_b = 0.5 * bf16_ulp(ref.abs() + fb) + fb + 1e-300
        worst = max(worst, (err / allowed).max().item())
        worst_b = max(worst_b, (err / allowed_b).max().item())
        exact += (err <= 0.5 * bf16_ulp(ref) * (1 + 1e-9)).sum().item()
        tot += err.numel()
    return worst, worst_b, exact / tot


def ulp_hist(y, ym):
    d = (y.float() - ym.float()).abs() / bf16_ulp(ym.float()).float()
    return [(d == 0).float().mean().item(), ((d > 0) & (d <= 1)).float().mean().item(), (d > 1).float().mean().item()]


def acts(M, k, g):
    x = torch.randn(M, k, device=dev, generator=g)
    idx = torch.randperm(k, device=dev, generator=g)[: max(1, k // 256)]
    x[:, idx] *= 30.0
    return x.to(torch.bfloat16)


def load_sample(man, name):
    e = next(m for m in man if m["name"] == name)
    raw = open(os.path.join(SAMPLES, e["file"]), "rb").read()
    return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).view(*e["shape"]).to(dev)


def load_safetensor(path, name):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
        e = h[name]
        a, b = e["data_offsets"]
        fh.seek(8 + n + a)
        raw = fh.read(b - a)
    assert e["dtype"] == "BF16"
    return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).view(*e["shape"]).to(dev)


def real_weights():
    out = []
    if not os.path.isdir(SAMPLES):
        print(f"   (real samples not mounted: {SAMPLES})")
        return out
    man = json.load(open(os.path.join(SAMPLES, "manifest.json")))
    P = "model.language_model.layers.{}.self_attn.{}_proj.weight"
    for l in (1, 20, 42):
        q, k, v = (load_sample(man, P.format(l, s)) for s in ("q", "k", "v"))
        b, fa, ga = (load_sample(man, P.format(l, s)) for s in ("b", "f_a", "g_a"))
        h = q.shape[0] // 2
        w = torch.cat([q[:h], k[:h], v[:h], b[: b.shape[0] // 2], fa, ga])
        del q, k, v, b, fa, ga
        out.append((f"real kda in_proj L{l}", w))
    ga = load_sample(man, "model.language_model.layers.1.mlp.gate_proj.weight")
    up = load_sample(man, "model.language_model.layers.1.mlp.up_proj.weight")
    h = ga.shape[0] // 2
    out.append(("real dense gate_up L1", torch.cat([ga[:h], up[:h]])))
    del ga, up
    if os.path.isfile(DRAFTER):
        out.append(("real drafter fc", load_safetensor(DRAFTER, "fc.weight")))
    return out


def main():
    gpu_guard(8.0)
    ck = Checks()
    G.STATE.large = True
    L = G.ext_large()
    G.STATE.large_gemm = "tilelang"
    print(f"extension {getattr(L, '__file__', L)}, VERSION {getattr(L, 'VERSION', None)}")
    g = torch.Generator(device=dev).manual_seed(0)

    print("== D dequant exactness")
    shapes = [(12576, 4096), (12288, 4096), (4096, 20480)]
    syn = {}
    for n, k in shapes:
        w = (torch.randn(n, k, device=dev, generator=g) * 0.02 *
             torch.exp(torch.randn(n, 1, device=dev, generator=g))).to(torch.bfloat16)
        layer, fp8, sc = quantize_like_prod(w)
        del w
        syn[(n, k)] = (layer, fp8, sc)
        ref = fp8.to(torch.bfloat16)
        for n0, nc in [(0, n), (0, 64), (64, 1000), (1000, 1000), (n - 136, 136), (n - 8, 8)]:
            big = torch.full((nc + 2, k + 16), float("nan"), dtype=torch.bfloat16, device=dev)
            out = big[1:nc + 1, :k]
            L.dequant(out, layer.weight, n0, nc, k)
            ok = torch.equal(out, ref[n0:n0 + nc])
            clean = torch.isnan(big[0]).all().item() and torch.isnan(big[nc + 1]).all().item() and \
                torch.isnan(big[:, k:]).all().item()
            ck(ok and clean, f"D {n}x{k} n0={n0} nc={nc}: exact {ok} clean {clean}")
        print(f"   {n}x{k} (Npad {layer.weight.shape[1] // 4}): 6 row ranges bitwise == fp8.to(bf16), no stray writes")
    # every e4m3 byte pattern (NaN excluded) in the weight
    n, k = 256, 4096
    pats = torch.arange(256, dtype=torch.int32, device=dev)
    pats = pats[(pats & 0x7F) != 0x7F]
    idx = torch.randint(0, pats.numel(), (n, k), device=dev, generator=g)
    fp8e = pats[idx].to(torch.uint8).view(torch.float8_e4m3fn)
    lay = torch.nn.Module()
    lay.output_size_per_partition, lay.input_size_per_partition = n, k
    lay.orig_dtype = torch.bfloat16
    lay.weight = torch.nn.Parameter(fp8e.clone(), requires_grad=False)
    lay.weight_scale = torch.nn.Parameter(torch.full((n,), 2.0 ** -9, dtype=torch.bfloat16, device=dev),
                                          requires_grad=False)
    lay.weight_block_size = None
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_layer_for_marlin
    prepare_fp8_layer_for_marlin(lay, size_k_first=False)
    out = torch.empty(n, k, dtype=torch.bfloat16, device=dev)
    L.dequant(out, lay.weight, 0, n, k)
    ok = torch.equal(out, fp8e.to(torch.bfloat16))
    ck(ok, "D edge values")
    print(f"   all 254 non-NaN e4m3 byte patterns (subnormals, +-0, +-448): exact {ok}")

    print("== A alpha (un-permuted fp32 scale)")
    for (n, k), (layer, fp8, sc) in syn.items():
        a = G.large_alpha(layer.weight_scale, n)
        ok = torch.equal(a[:n], sc.float()) and a.numel() == n + G.LARGE_ALPHA_PAD and not a[n:].any().item()
        ck(ok, f"A {n}x{k}")
        print(f"   {n}x{k}: alpha[:N] == stored per-channel scale (fp32), {G.LARGE_ALPHA_PAD} zeros after: {ok}")

    print("== S scale_cast")
    for M, nc in [(7, 8), (300, 4096), (2048, 6336)]:
        y32 = torch.randn(M, nc, device=dev, generator=g) * 10
        alpha = torch.rand(nc, device=dev, generator=g) * 1e-3
        bias = (torch.randn(nc, device=dev, generator=g)).to(torch.bfloat16)
        big = torch.full((M, nc + 16), float("nan"), dtype=torch.bfloat16, device=dev)
        o = big[:, 8:8 + nc]
        L.scale_cast(o, y32, alpha, None)
        ok1 = torch.equal(o, (y32 * alpha).to(torch.bfloat16))
        L.scale_cast(o, y32, alpha, bias)
        ok2 = torch.equal(o, ((y32 * alpha).to(torch.bfloat16).float() + bias.float()).to(torch.bfloat16))
        clean = torch.isnan(big[:, :8]).all().item() and torch.isnan(big[:, 8 + nc:]).all().item()
        ck(ok1 and ok2 and clean, f"S M={M} nc={nc}: {ok1} {ok2} {clean}")
        print(f"   M={M} nc={nc}: == torch reference (no bias {ok1}, bias {ok2}), neighbours untouched {clean}")

    print("== F the fp32 intermediate (one cuBLAS call per plan chunk)")
    for (n, k), (layer, fp8, sc) in syn.items():
        alpha = G.large_alpha(layer.weight_scale, n)
        for M in (1791, 4608):
            x = acts(M, k, g)
            y = G.large_forward(x, layer.weight, alpha, None, n, k, nc=-(-n // 64) * 64, mc=M, backend="cublas")
            yt = G.large_forward(x, layer.weight, alpha, None, n, k, backend="tilelang")
            w = torch.empty(n, k, dtype=torch.bfloat16, device=dev)
            L.dequant(w, layer.weight, 0, n, k)
            y32 = torch.mm(x, w.t(), out_dtype=torch.float32)
            want = (y32 * alpha[:n]).to(torch.bfloat16)
            same = torch.equal(y, want) and torch.equal(yt, want)
            rows = torch.arange(0, M, max(1, M // 32), device=dev)
            xd = x[rows].double()
            worst = 0.0
            for a in range(0, n, 4096):
                f = fp8[a:a + 4096].double()
                dot = xd @ f.t()
                bound = (k / 16 + 2) * 2.0 ** -23 * (xd.abs() @ f.abs().t()) + 1e-300
                worst = max(worst, ((y32[rows, a:a + 4096].double() - dot).abs() / bound).max().item())
            ck(same and worst <= 1.0, f"F {n}x{k} M={M}: bitwise {same}, fp32 error / bound {worst:.3f}")
            print(f"   {n}x{k} M={M}: large_forward (cublas and tilelang) == bf16(mm_fp32 * scale) bitwise {same}; max |y32 - dot| / "
                  f"((K/16+2) 2^-23 sum|xw|) = {worst:.4f}", flush=True)
            del x, y, yt, w, y32

    print("== N numerics vs float64 and vs Marlin (default backend: tilelang; '!=C' = elements differing from the cublas"
          " backend)")
    cases = [(f"synthetic {n}x{k}", syn[(n, k)]) for n, k in shapes]
    for name, w in real_weights():
        layer, fp8, sc = quantize_like_prod(w)
        del w
        cases.append((name, (layer, fp8, sc)))
    for name, (layer, fp8, sc) in cases:
        n, k = fp8.shape
        alpha = G.large_alpha(layer.weight_scale, n)
        line = f"   {name} [{n}x{k}]:"
        for M in (512, 1791, 2202, 4608, 13824):
            x = acts(M, k, g)
            y = G.large_forward(x, layer.weight, alpha, None, n, k)
            yc = G.large_forward(x, layer.weight, alpha, None, n, k, backend="cublas")
            ndiff = int((y != yc).sum().item())
            ck(ndiff <= 1e-5 * y.numel(), f"N {name} M={M}: tilelang vs cublas backend differ in {ndiff} elements")
            del yc
            ym = marlin(layer, x, n, k)
            nc, mc = G.large_plan(n, k, M, backend="cublas")
            bnd = sorted({0, M - 1, *range(0, M, max(1, M // 24)), *[b for m0 in range(mc, M, mc) for b in (m0 - 1, m0)]})
            rows = torch.tensor(bnd[:48], device=dev)
            r, rb, fr = vs_f64(y, x, fp8, sc, rows)
            rm, rbm, frm = vs_f64(ym, x, fp8, sc, rows)
            h = ulp_hist(y, ym)
            rl2 = ((y.float() - ym.float()).norm() / ym.float().norm()).item()
            ck(rb <= 1.0, f"N {name} M={M}: outside the fp32 bound ({rb:.3f})")
            ck(fr >= 0.99 and fr >= frm - 0.005, f"N {name} M={M}: correctly rounded {fr:.5f} (Marlin {frm:.5f})")
            ck(h[0] >= 0.99 and h[2] <= 1e-3, f"N {name} M={M}: vs Marlin histogram {h}")
            ck(rl2 <= 1e-3, f"N {name} M={M}: rel_l2 vs Marlin {rl2:.1e}")
            line += (f" M{M}: {r:.2f}/{rb:.3f}/{fr*100:.2f}% (Marlin {rm:.2f}/{rbm:.3f}/{frm*100:.2f}%) =M {h[0]*100:.2f}% "
                     f"1ulp {h[1]*100:.3f}% >1ulp {h[2]*100:.4f}% rl2 {rl2:.1e} !=C {ndiff}/{y.numel()};")
            del x, y, ym
        print(line, flush=True)
        torch.cuda.empty_cache()
    print("   (per M: new err/(0.5ulp+2^-20 sum|xw|) / err/(0.5ulp+fp32 bound) / correctly rounded (Marlin: same three) /"
          " bitwise == Marlin / |diff| <= 1 ulp / > 1 ulp / rel_l2 vs Marlin)")
    del cases
    torch.cuda.empty_cache()

    print("== C chunk plans and determinism")
    for (n, k), (layer, fp8, sc) in syn.items():
        alpha = G.large_alpha(layer.weight_scale, n)
        for M in (1791, 13824):
            nct, _ = G.large_plan(n, k, M, backend="tilelang")
            ck(nct * k * 2 / 2 ** 20 <= G.CFG.large_temp_mib * 1.05, f"C tilelang plan {n}x{k}: {nct}")
            nc, mc = G.large_plan(n, k, M, backend="cublas")
            tmp_mib = (nc * k * 2 + mc * nc * 4) / 2 ** 20
            ck(tmp_mib <= G.CFG.large_temp_mib * 1.05, f"C plan {n}x{k} M={M}: {tmp_mib:.1f} MiB")
            x = acts(M, k, g)
            y0 = G.large_forward(x, layer.weight, alpha, None, n, k)
            y1 = G.large_forward(x, layer.weight, alpha, None, n, k)
            ck(torch.equal(y0, y1), f"C {n}x{k} M={M}: not deterministic")
            rows = torch.arange(0, M, max(1, M // 32), device=dev)
            line = (f"   {n}x{k} M={M}: tilelang plan nc={nct} ({nct * k * 2 / 2**20:.0f} MiB), cublas plan nc={nc} mc={mc} "
                    f"({tmp_mib:.0f} MiB), repeat bitwise {torch.equal(y0, y1)};")
            for pnc, pmc, be in [(n, M, "cublas"), (64 * ((n // 3 + 63) // 64), 1024, "cublas"), (1024, 4096, "cublas"),
                                 (64 * ((n // 3 + 63) // 64), 1000, "tilelang"), (None, None, "cublas")]:
                y2 = G.large_forward(x, layer.weight, alpha, None, n, k, nc=pnc and min(pnc, -(-n // 64) * 64), mc=pmc,
                                     backend=be)
                _, rb, _ = vs_f64(y2, x, fp8, sc, rows)
                ck(rb <= 1.0, f"C {n}x{k} M={M} plan ({pnc},{pmc}): outside the fp32 bound {rb:.3f}")
                line += f" {be}({pnc},{pmc}): =default {(y2 == y0).float().mean().item()*100:.2f}% bound {rb:.3f};"
            print(line, flush=True)
            del x, y0, y1, y2

    print("== T TileLang kernel on odd shapes")
    kern = G._tl_kernel(4096)
    layer, fp8, sc = syn[(12576, 4096)]
    wfull = torch.empty(12576, 4096, dtype=torch.bfloat16, device=dev)
    L.dequant(wfull, layer.weight, 0, 12576, 4096)
    alpha = G.large_alpha(layer.weight_scale, 12576)
    xall = acts(777, 4096, g)
    yall = torch.empty(777, 12576, dtype=torch.bfloat16, device=dev)
    kern(xall, wfull, alpha[:12576], yall)
    for M, n0, c in [(1, 0, 256), (7, 64, 200), (129, 1000, 1000), (777, 12576 - 136, 136), (777, 0, 12576)]:
        x = xall[:M]
        big = torch.full((M, 12576 + 32), float("nan"), dtype=torch.bfloat16, device=dev)
        out = big[:, 16 + n0:16 + n0 + c]
        kern(x, wfull[n0:n0 + c], alpha[n0:n0 + c], out)
        same = torch.equal(out, yall[:M, n0:n0 + c])        # a row's result does not depend on M or the chunk
        _, rb, _ = vs_f64(big[:, 16:16 + 12576].nan_to_num(0.0) if c == 12576 else yall[:M], x, fp8, sc,
                          torch.arange(M, device=dev))
        rest = torch.cat([big[:, :16 + n0].reshape(-1), big[:, 16 + n0 + c:].reshape(-1)])
        clean = bool(torch.isnan(rest).all().item())
        ck(same and clean and rb <= 1.0, f"T M={M} n0={n0} c={c}: same {same} clean {clean} bound {rb:.3f}")
        print(f"   M={M} columns {n0}..{n0 + c}: == the same rows/columns of an M=777 full call {same}, fp32 bound "
              f"ratio {rb:.3f}, nothing written outside {clean}")
    wide = acts(777, 4096 + 256, g)
    xs = wide[:, 128:128 + 4096]
    ys = G.large_forward(xs, layer.weight, alpha, None, 12576, 4096, backend="tilelang")
    ok = torch.equal(ys, G.large_forward(xs.contiguous(), layer.weight, alpha, None, 12576, 4096, backend="tilelang"))
    ck(ok, "T strided x")
    print(f"   strided x (column slice of a wider tensor) == contiguous: {ok}")

    print("== X input checks")
    layer, fp8, sc = syn[(12288, 4096)]
    w16 = torch.empty(64, 4096, dtype=torch.bfloat16, device=dev)
    bad = [
        ("dequant fp16 out", lambda: L.dequant(torch.empty(64, 4096, dtype=torch.float16, device=dev), layer.weight, 0, 64, 4096)),
        ("dequant rows out of range", lambda: L.dequant(w16, layer.weight, 12288 - 32, 64, 4096)),
        ("dequant wrong K", lambda: L.dequant(torch.empty(64, 2048, dtype=torch.bfloat16, device=dev), layer.weight, 0, 64, 2048)),
        ("dequant misaligned out", lambda: L.dequant(torch.empty(64 * 4096 + 1, dtype=torch.bfloat16, device=dev)[1:].view(64, 4096), layer.weight, 0, 64, 4096)),
        ("scale_cast nc % 8", lambda: L.scale_cast(torch.empty(4, 12, dtype=torch.bfloat16, device=dev), torch.empty(4, 12, device=dev), torch.empty(12, device=dev), None)),
        ("scale_cast alpha size", lambda: L.scale_cast(torch.empty(4, 16, dtype=torch.bfloat16, device=dev), torch.empty(4, 16, device=dev), torch.empty(8, device=dev), None)),
        ("scale_cast fp16 y32", lambda: L.scale_cast(torch.empty(4, 16, dtype=torch.bfloat16, device=dev), torch.empty(4, 16, dtype=torch.float16, device=dev), torch.empty(16, device=dev), None)),
    ]
    for nm, fn in bad:
        try:
            fn()
            torch.cuda.synchronize()
            ck(False, f"X {nm}: did not raise")
            print(f"   {nm}: NOT rejected")
        except RuntimeError as exc:
            print(f"   {nm}: rejected ({str(exc).splitlines()[0][:80]})")
    ck.summary()


run_main(main)
