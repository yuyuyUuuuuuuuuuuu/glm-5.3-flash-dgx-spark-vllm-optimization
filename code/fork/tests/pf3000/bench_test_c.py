"""PF3000 TEST C - W8A8 dense GEMMs (prefill3000 plan step 3 kill test).

For every production dense FP8 shape (per rank, TP=2, GLM53_DENSE_FP8=dense,kda,mla,shared):
  - production's CURRENT path through Glm53DenseFp8Method.apply (GLM53_FP8_LARGE_M=1, as in
    production env_nonsecret.txt: the TileLang large-M path for (12608,4096)/(12288,4096)/(4096,20480)
    from their M thresholds, Marlin for everything else) and production's own apply (Marlin) separately;
  - cutlass_scaled_mm (the image's vLLM _C, sm_120/sm_121 kernels) at M = 13824 in 2048-row pieces,
    INCLUDING the per-token e4m3 activation quantization;
  - per-GEMM relative error of the W8A8 output against an fp32 reference (and against production's path);
  - the weight-layout cost of moving the FP8 weights out of the Marlin repack:
    (i) a resident standard-layout copy (exact GiB/rank, allocated for real in resident_copy.py),
    (ii) a per-call re-layout (production's own Marlin->BF16 dequant kernel + bf16->fp8 cast, timed),
        and the fp8-direct floor 2*N*K bytes at the measured 230 GB/s DRAM bandwidth.

Run: GPU_RUN_ENV="TF_EXL3_JIT=1" GPU_RUN_RO="$TF_EXL3_MODELS/GLM-5.3-Flash-EXL3-TR3-4bpw-partial" \
     flock /tmp/tf-gpu-bench.lock tests/gpu_run.sh python3 tests/pf3000/bench_test_c.py
"""
import inspect
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import run_main, gpu_guard, load_prod  # noqa: E402
import torch  # noqa: E402

dev = "cuda"
M = 13824
PIECE = 2048
SAMPLES = os.path.join(os.environ.get("TF_EXL3_MODELS") or os.path.expanduser("~/models"), "GLM-5.3-Flash-EXL3-TR3-4bpw-partial/bf16_samples")
DRAM_GBPS = 230.0  # nodeC measured 227-234 GB/s (E4_FAT_MOE.md ceilings.log / l2bw.cu)

# name: (N, K, group, prefix, calls per 13,824-token chunk per rank, real bf16_samples source)
#   real source: list of (file, row-slice, col-slice) concatenated along rows, or None = synthetic
SHAPES = {
    # real source: list of (file, file shape, row-slice, col-slice) concatenated along rows
    "kda.in_proj":    (12576, 4096, "kda", "model.layers.0.self_attn.in_proj_qkvbfg_a", 34,
                       [("layers.1.self_attn.q_proj.weight.bin", (8192, 4096), slice(0, 4096), slice(None)),
                        ("layers.1.self_attn.k_proj.weight.bin", (8192, 4096), slice(0, 4096), slice(None)),
                        ("layers.1.self_attn.v_proj.weight.bin", (8192, 4096), slice(0, 4096), slice(None)),
                        ("layers.1.self_attn.b_proj.weight.bin", (64, 4096), slice(0, 32), slice(None)),
                        ("layers.1.self_attn.f_a_proj.weight.bin", (128, 4096), slice(None), slice(None)),
                        ("layers.1.self_attn.g_a_proj.weight.bin", (128, 4096), slice(None), slice(None))]),
    "kda.o_proj":     (4096, 4096, "kda", "model.layers.0.self_attn.o_proj", 34,
                       [("layers.1.self_attn.o_proj.weight.bin", (4096, 8192), slice(None), slice(0, 4096))]),
    "mla.qkv_a":      (2048, 4096, "mla", "model.layers.3.self_attn.fused_qkv_a_proj", 11, None),
    "mla.q_b":        (8192, 1536, "mla", "model.layers.3.self_attn.q_b_proj", 11, None),
    "mla.o_proj":     (4096, 8192, "mla", "model.layers.3.self_attn.o_proj", 11, None),
    "shared.gate_up": (2048, 4096, "shared", "model.layers.3.mlp.shared_experts.gate_up_proj", 42,
                       [("layers.10.mlp.shared_experts.gate_proj.weight.bin", (2048, 4096), slice(None), slice(None))]),
    "shared.down":    (4096, 1024, "shared", "model.layers.3.mlp.shared_experts.down_proj", 42,
                       [("layers.10.mlp.shared_experts.down_proj.weight.bin", (4096, 2048), slice(None), slice(0, 1024))]),
    "dense.gate_up":  (12288, 4096, "dense", "model.layers.1.mlp.gate_up_proj", 3,
                       [("layers.1.mlp.gate_proj.weight.bin", (12288, 4096), slice(None), slice(None))]),
    "dense.down":     (4096, 6144, "dense", "model.layers.1.mlp.down_proj", 3,
                       [("layers.1.mlp.down_proj.weight.bin", (4096, 12288), slice(None), slice(0, 6144))]),
    "draft.fc":       (4096, 20480, "draft", "model.fc", 1, None),
}


def load_real(src):
    """bf16_samples .bin files are BF16 stored as uint16 (lm_head/manifest.json: dtype BF16)."""
    import numpy as np
    rows = []
    for fn, fshape, rs, cs in src:
        a = np.fromfile(f"{SAMPLES}/{fn}", dtype=np.uint16)
        assert a.size == fshape[0] * fshape[1], (fn, a.size, fshape)
        a = (a.astype(np.uint32) << 16).view(np.float32)
        t = torch.from_numpy(a).to(torch.bfloat16).reshape(fshape)[rs, cs]
        rows.append(t.to(dev).contiguous())
    return torch.cat(rows, dim=0)


def per_token_quant(x):
    """Per-token e4m3 quantization through the image's own CUDA kernel (dynamic_per_token_scaled_fp8_quant,
    fp32 math in one pass: absmax/448 per row, cast) — production's own primitive, not a python re-write."""
    import vllm._custom_ops as ops
    q, s = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
    return q, (s if s.dim() == 2 else s.unsqueeze(1)).float().contiguous()


def main():
    gpu_guard(8.0)
    for v in ("GLM53_FP8_GEMV", "GLM53_FP8_GEMV_MAX_M", "TF_EXL3_MOE", "GLM53_KDA_BF16_LARGE_M",
              "GLM53_DEC_FP8ROOF"):
        os.environ.pop(v, None)
    os.environ["GLM53_FP8_LARGE_M"] = "1"          # production's value (env_nonsecret.txt:52)
    from test_fp8_integrate import single_rank_tp, L
    single_rank_tp()
    prod = load_prod()
    import fp8_gemv as F
    cls = prod.Glm53DenseFp8Method
    orig_apply = cls.apply
    rep = F.install(prod)
    assert rep["installed"] and F.STATE.large, rep
    print("install:", rep)
    takes_prefix = "prefix" in inspect.signature(cls.__init__).parameters
    p = torch.cuda.get_device_properties(0)
    print(f"torch {torch.__version__}, {p.name} cc {p.major}.{p.minor}, M={M} in {PIECE}-row pieces, "
          f"rounds interleaved median of 5")

    import vllm._custom_ops as ops
    tot = {"prod": 0.0, "marlin": 0.0, "w8a8": 0.0, "w8a8_nopiece": 0.0}
    tot_layout = {"relayout": 0.0, "floor": 0.0}
    resident_bytes = 0
    marlin_bytes = 0
    rounds = int(os.environ.get("BENCH_ROUNDS", "5"))
    g = torch.Generator(device=dev).manual_seed(0)

    for name, (n, k, grp, pre, calls, src) in SHAPES.items():
        w = load_real(src) if src else (torch.randn(n, k, device=dev, generator=g) * 0.02).to(torch.bfloat16)
        assert w.shape == (n, k), (name, w.shape)
        real = src is not None
        lay = L(w)
        m = cls(grp, pre) if takes_prefix else cls(grp)
        # production's own per-output-channel e4m3 quantization (overlay_exl3.py:1978-1980), for the
        # standard-layout cutlass operand; process_weights_after_loading then Marlin-packs it
        wf = w.float()
        scales = wf.abs().amax(dim=1).clamp(min=1e-12) / 448.0
        wb = (wf / scales[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
        del wf
        sb = scales.to(torch.bfloat16).float().unsqueeze(1).contiguous()   # stored scale is bf16, as production
        m.process_weights_after_loading(lay)         # quantizes to e4m3 per-channel, Marlin repacks
        del w
        npad = lay.weight.shape[1] // 4
        min_m = F.LARGE_TABLE.get((npad, k))
        cur_is_large = min_m is not None and M >= min_m
        # activations: gaussian and the x30-outlier-channel recipe of FP8_LARGE_M
        x = (torch.randn(M, k, device=dev, generator=g) * 0.02).to(torch.bfloat16)
        xo = (torch.randn(M, k, device=dev, generator=g) * 0.02).to(torch.bfloat16)
        xo[:, ::30] *= 30.0
        alpha = F.large_alpha(lay.weight_scale, n)   # un-permuted fp32 scale from production's packed scales
        # ---- cutlass W8A8 operands (standard layout) ----
        # (wb/sb built above from production's quantization formula; alpha is the same scale Marlin multiplies by)
        bias = None

        def prod_cur():
            return m.apply(lay, x)

        def prod_marlin():
            return orig_apply(m, lay, x)

        def w8a8(xin=x):
            out = torch.empty(xin.shape[0], n, dtype=torch.bfloat16, device=dev)
            for r0 in range(0, xin.shape[0], PIECE):
                r = min(PIECE, xin.shape[0] - r0)
                q, sa = per_token_quant(xin[r0:r0 + r])
                torch.ops._C.cutlass_scaled_mm(out[r0:r0 + r], q, wb.t(), sa, sb, None)
            return out

        fns = {"prod": prod_cur, "marlin": prod_marlin, "w8a8": w8a8}
        for f in fns.values():
            f()
        torch.cuda.synchronize()
        ts = {a: [] for a in fns}
        reps = 3
        for r in range(rounds):
            order = ("prod", "marlin", "w8a8")
            for a in order[r % 3:] + order[:r % 3]:
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record()
                for _ in range(reps):
                    fns[a]()
                e1.record()
                torch.cuda.synchronize()
                ts[a].append(e0.elapsed_time(e1) / reps)
        med = {a: statistics.median(v) for a, v in ts.items()}
        spread = {a: (max(v) - min(v)) / med[a] * 100 for a, v in ts.items()}
        # single-call reference (no pieces)
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        out1 = torch.empty(M, n, dtype=torch.bfloat16, device=dev)
        q1, sa1 = per_token_quant(x)

        def nopiece():
            torch.ops._C.cutlass_scaled_mm(out1, q1, wb.t(), sa1, sb, None)
        nopiece()
        torch.cuda.synchronize()
        nv = []
        for _ in range(5):
            e0.record()
            for _ in range(reps):
                nopiece()
            e1.record()
            torch.cuda.synchronize()
            nv.append(e0.elapsed_time(e1) / reps)
        med_np = statistics.median(nv)

        # breakdown of the W8A8 path: per-token quantization alone vs the piece GEMMs alone
        outp = torch.empty(M, n, dtype=torch.bfloat16, device=dev)
        qs = []
        for r0 in range(0, M, PIECE):
            r = min(PIECE, M - r0)
            qs.append(per_token_quant(x[r0:r0 + r]))

        def quant_only():
            for r0 in range(0, M, PIECE):
                per_token_quant(x[r0:r0 + min(PIECE, M - r0)])

        def gemm_only():
            for i, r0 in enumerate(range(0, M, PIECE)):
                r = min(PIECE, M - r0)
                torch.ops._C.cutlass_scaled_mm(outp[r0:r0 + r], qs[i][0], wb.t(), qs[i][1], sb, None)
        bk = {}
        for label, fn in (("quant", quant_only), ("gemm", gemm_only)):
            fn()
            torch.cuda.synchronize()
            vv = []
            for _ in range(5):
                e0.record()
                for _ in range(reps):
                    fn()
                e1.record()
                torch.cuda.synchronize()
                vv.append(e0.elapsed_time(e1) / reps)
            bk[label] = statistics.median(vv)

        # ---- numerics (both activation distributions) ----
        lines = []
        for tag, xin in (("gauss", x), ("outlier", xo)):
            yref = torch.mm(xin.float(), (wb.float() * sb).t())
            yp = m.apply(lay, xin)
            yw = w8a8(xin)
            rl_ref = ((yw.float() - yref).norm() / yref.norm()).item()
            rl_prod = ((yp.float() - yref).norm() / yref.norm()).item()
            rl_vs_prod = ((yw.float() - yp.float()).norm() / yp.float().norm()).item()
            lines.append(f"{tag}: W8A8 rel_l2 vs fp32 ref {rl_ref:.3e} (production path {rl_prod:.3e}, "
                         f"W8A8 vs production {rl_vs_prod:.3e})")

        # ---- weight layout cost ----
        # (ii) per-call re-layout: production's own Marlin->bf16 dequant kernel + bf16->fp8 cast
        Lx = F.ext_large()
        wtmp = torch.empty(n, k, dtype=torch.bfloat16, device=dev)

        def relayout():
            for n0 in range(0, n, 2048):
                c = min(2048, n - n0)
                Lx.dequant(wtmp[n0:n0 + c], lay.weight, n0, c, k)
            return wtmp.to(torch.float8_e4m3fn)
        wcheck = relayout()
        agree = (wcheck.view(torch.uint8) == wb.view(torch.uint8)).float().mean().item()
        torch.cuda.synchronize()
        rv = []
        for _ in range(5):
            e0.record()
            for _ in range(reps):
                relayout()
            e1.record()
            torch.cuda.synchronize()
            rv.append(e0.elapsed_time(e1) / reps)
        rel_ms = statistics.median(rv)
        floor_ms = 2.0 * n * k / (DRAM_GBPS * 1e9) * 1e3
        resident_bytes += calls * n * k
        marlin_bytes += calls * lay.weight.numel() * 4
        # production current path: TileLang large-M (wraps Marlin) only for LARGE_TABLE shapes
        cur = med["prod"] if cur_is_large else med["marlin"]
        tot["prod"] += calls * med["prod"]
        tot["marlin"] += calls * med["marlin"]
        tot["w8a8"] += calls * med["w8a8"]
        tot["w8a8_nopiece"] += calls * med_np
        tot_layout["relayout"] += calls * rel_ms
        tot_layout["floor"] += calls * floor_ms
        fl = 2.0 * M * n * k
        if name == "kda.in_proj":
            med_w8a8_inproj = med["w8a8"]
        print(f"{name:15s} [{n}x{k}] real={'Y' if real else 'N'} Npad={npad} large-M thr={min_m}: "
              f"production(current) {cur:7.3f} ms ({fl / cur / 1e9:5.1f} TFLOPS, spread {spread['prod' if cur_is_large else 'marlin']:.1f}%) | "
              f"Marlin-only {med['marlin']:7.3f} ms | W8A8 pieces {med['w8a8']:7.3f} ms "
              f"({fl / med['w8a8'] / 1e9:5.1f} TFLOPS, spread {spread['w8a8']:.1f}%) | "
              f"W8A8 single {med_np:7.3f} ms | speedup vs current {cur / med['w8a8']:.2f}x", flush=True)
        for ln in lines:
            print(f"      numerics {ln}")
        print(f"      W8A8 breakdown: per-token quant {bk['quant']:.3f} ms + piece GEMMs {bk['gemm']:.3f} ms "
              f"({fl / bk['gemm'] / 1e9:.1f} TFLOPS GEMM only)", flush=True)
        print(f"      layout: per-call re-layout {rel_ms:.3f} ms (Marlin->bf16 dequant + fp8 cast), "
              f"fp8-direct floor {floor_ms:.3f} ms at {DRAM_GBPS:.0f} GB/s; resident fp8 copy "
              f"{calls * n * k / 2**30:.3f} GiB/rank ({calls} calls/chunk); "
              f"my un-permuted fp8 vs production's Marlin payload: bitwise match {agree * 100:.2f}%",
              flush=True)
        del x, xo, yref, lay, m, wb, wtmp
        torch.cuda.empty_cache()

    print("\n==== per 13,824-token chunk per rank (calls per chunk from DEC_FP8ROOF/FP8_LARGE_M) ====")
    print(f"weights resident fp8 standard-layout copy: {resident_bytes / 2**30:.2f} GiB/rank "
          f"(Marlin packed holds {marlin_bytes / 2**30:.2f} GiB/rank)")
    print(f"production current (TileLang large-M + Marlin): {tot['prod']:.1f} ms/chunk")
    print(f"Marlin only:                                    {tot['marlin']:.1f} ms/chunk")
    print(f"W8A8 cutlass pieces (incl. per-token quant):    {tot['w8a8']:.1f} ms/chunk "
          f"(-> {tot['prod'] - tot['w8a8']:+.1f} ms vs current)")
    print(f"W8A8 cutlass single-call (no pieces):           {tot['w8a8_nopiece']:.1f} ms/chunk")
    print(f"per-call re-layout total: {tot_layout['relayout']:.1f} ms/chunk "
          f"(fp8-direct floor {tot_layout['floor']:.1f} ms/chunk at {DRAM_GBPS:.0f} GB/s)")
    print(f"resident copy total: {resident_bytes / 2**30:.2f} GiB/rank vs 0.5 GiB/rank budget")
    ok_speed = med_w8a8_inproj <= 11.0
    ok_layout = (resident_bytes / 2**30 <= 0.5) or (tot_layout["relayout"] <= 10.0)
    print(f"VERDICT step 3: KDA in_proj W8A8 {med_w8a8_inproj:.3f} ms / 13,824 rows "
          f"({'PASS' if ok_speed else 'KILL'}: threshold 11 ms) | layout: resident {resident_bytes / 2**30:.2f} GiB/rank "
          f"({'fits' if resident_bytes / 2**30 <= 0.5 else 'over'}, 0.5 budget) vs re-layout {tot_layout['relayout']:.1f} "
          f"ms/chunk ({'fits' if tot_layout['relayout'] <= 10.0 else 'over'}, 10 ms/chunk budget; fp8-direct floor "
          f"{tot_layout['floor']:.1f} ms/chunk) -> {'PASS' if ok_layout else 'KILL'}")
    # REVIEW FIX: the 0.5 GiB / 10 ms budgets are not in PLAN.md (the critique only estimates ~3.2 GiB and
    # ~30 ms/chunk). The decision criterion is the NET saving: W8A8 + per-call re-layout vs production current
    # (production's TileLang large-M path already pays its own per-call Marlin->bf16 dequant inside tot['prod']).
    net_relayout = tot["prod"] - (tot["w8a8"] + tot_layout["relayout"])
    net_floor = tot["prod"] - (tot["w8a8"] + tot_layout["floor"])
    print(f"NET per chunk/rank with per-call re-layout: {net_relayout:+.1f} ms saved "
          f"(with an fp8->fp8 repack at the DRAM floor: {net_floor:+.1f} ms) -> "
          f"{'PASS (layout option ii, 0 GiB resident)' if net_relayout > 0 and ok_speed else 'KILL'}")


if __name__ == "__main__":
    run_main(main)
