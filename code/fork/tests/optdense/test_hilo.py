"""opt-dense: the hi+lo W8A8 fast path (GLM53_DENSE_W8A8_HILO + _SEL=first) on nodeC, JIT extension (REPACK_LD).
 R  repack into a row-strided [N, K + C] view == the contiguous repack, byte for byte (and the pad columns untouched)
 Q  hilo_quant's q(x) half == the image's per-token quant (ops.scaled_fp8_quant) bytes and scales; its residual half
    == the prototype's e4m3((x_S - q_S s) / s)
 F  fast path == prototype (same frozen channel set) BITWISE through the hooked apply, per production shape
 E  error: ||Y - Y_marlin(bf16 x)||^2 of hi+lo / of plain W8A8 on rows with a few outlier channels
 T  wall per call at M 13824 / 4289: plain W8A8 vs hi+lo (production shapes and the channel budgets of the KL arm)
Usage: source tests/w8a82/env.sh; tests/gpu_run.sh python3 tests/optdense/test_hilo.py"""
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
import torch  # noqa: E402

dev = "cuda"
FAILS = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAILS.append(msg)


def tmed(fn, n=9):
    for _ in range(2): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return statistics.median(ts)


SHAPES = (("kda.o_proj", 4096, 4096, "kda", "model.layers.0.self_attn.o_proj", 256),
          ("mla.o_proj", 4096, 8192, "mla", "model.layers.3.self_attn.o_proj", 512),
          ("mla.q_b_proj", 8192, 1536, "mla", "model.layers.3.self_attn.q_b_proj", 256),
          ("dense.down_proj", 4096, 6144, "dense", "model.layers.0.mlp.down_proj", 512),
          ("shared.down_proj", 4096, 1024, "shared", "model.layers.3.mlp.shared_experts.down_proj", 128))


def main():
    from harness import gpu_guard, load_prod
    gpu_guard(8.0)
    for v in ("GLM53_FP8_GEMV", "GLM53_FP8_GEMV_MAX_M", "TF_EXL3_MOE", "GLM53_KDA_BF16_LARGE_M", "GLM53_DEC_FP8ROOF"):
        os.environ.pop(v, None)
    os.environ.update({"GLM53_FP8_LARGE_M": "1", "GLM53_DENSE_FP8": "dense,kda,mla,shared", "GLM53_DENSE_W8A8": "1",
                       "GLM53_DENSE_W8A8_HILO": ",".join(f"{s[0]}:{s[5]}" for s in SHAPES),
                       "GLM53_DENSE_W8A8_HILO_SEL": "first"})
    from test_fp8_integrate import single_rank_tp, L
    single_rank_tp()
    prod = load_prod()
    import fp8_gemv as F
    import fp8_w8a8 as W
    F.install(prod)
    rep = W.install(prod)
    print("install:", rep, "hilo", W.CFG.hilo, W.CFG.hilo_sel, flush=True)
    check(W.hilo_fast_ok(), "extension has REPACK_LD and the custom GEMM (fast path available)")
    cls = prod.Glm53DenseFp8Method
    import vllm._custom_ops as ops
    g = torch.Generator(device=dev).manual_seed(4)
    for name, n, k, grp, pre, c in SHAPES:
        w = (torch.randn(n, k, device=dev, generator=g) * 0.02).to(torch.bfloat16)
        lay = L(w)
        m = cls(grp, pre)
        m.process_weights_after_loading(lay)
        key = W._key(lay.weight, lay.weight_scale, k)
        # R
        ref8 = torch.empty(n, k, dtype=torch.float8_e4m3fn, device=dev)
        W.repack(ref8, lay.weight, n, k)
        big = torch.full((n, k + c), 0x7E, dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
        W.repack(big[:, :k], lay.weight, n, k)
        bu = big.view(torch.uint8)
        check(torch.equal(bu[:, :k], ref8.view(torch.uint8)) and bool((bu[:, k:] == 0x7E).all()),
              f"R {name}: strided repack == contiguous repack, pad columns untouched")
        # inputs with outlier channels
        x = (torch.randn(13824, k, device=dev, generator=g) * 0.5)
        oc = torch.randperm(k, generator=torch.Generator().manual_seed(k))[:16].to(dev)
        x[:, oc] *= 25.0
        x = x.to(torch.bfloat16)
        # Q
        cls.apply(m, lay, x[:600])                                      # self-test -> ALPHA (and a first freeze)
        S = W.hilo_select(x[:4096], ref8, W.ALPHA[key], n, k, c, key)
        a2, sa = W.hilo_quant(x, S)
        qv, sv = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
        sv = sv.float().reshape(-1, 1)
        okq = torch.equal(a2[:, :k].contiguous().view(torch.uint8), qv.view(torch.uint8)) and torch.equal(sa, sv)
        nd = (a2[:, :k].contiguous().view(torch.uint8) != qv.view(torch.uint8)).sum().item()
        check(okq, f"Q {name}: q(x) half == ops.scaled_fp8_quant per-token bytes + scales ({nd} bytes differ)")
        r = x.float() - qv.float() * sv
        qr = (r[:, S.long()] / sv).clamp(-448, 448).to(torch.float8_e4m3fn)
        check(torch.equal(a2[:, k:].contiguous().view(torch.uint8), qr.view(torch.uint8)),
              f"Q {name}: residual half == e4m3((x_S - q_S s)/s) (prototype formula)")
        # F: fast vs prototype with the same frozen set
        W.HILO_SEL[key] = S
        y_fast = cls.apply(m, lay, x)
        cf = W.COUNTERS.get("hilo_fast_calls", 0)
        # prototype with the frozen set: emulate its arithmetic exactly
        w8 = ref8
        a2p = torch.cat([qv, qr], 1).contiguous()
        w2p = torch.cat([w8, w8.view(torch.uint8)[:, S.long()].view(torch.float8_e4m3fn)], 1).contiguous()
        y_proto = torch.empty(x.shape[0], n, dtype=torch.bfloat16, device=dev)
        torch.ops._C.cutlass_scaled_mm(y_proto, a2p, w2p.t(), sv, W.ALPHA[key][:n].view(n, 1), None)
        check(cf >= 1 and torch.equal(y_fast, y_proto),
              f"F {name}: fast path (served {cf}) == prototype arithmetic bitwise; custom GEMM check "
              f"{W.HILO_GEMMCHECK.get((n, k + c))}")
        # E
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import apply_fp8_marlin_linear
        xm = x[:4096]
        y_m = apply_fp8_marlin_linear(input=xm, weight=lay.weight, weight_scale=lay.weight_scale,
                                      workspace=lay.workspace, size_n=n, size_k=k, bias=None).float()
        y_w = W.w8a8_forward(xm, lay.weight, lay.weight_scale, n, k, None, layer_key=key).float()
        e_w = (y_w - y_m).pow(2).sum().item()
        e_h = (y_fast[:4096].float() - y_m).pow(2).sum().item()
        print(f"E {name} c={c}: error^2 hi+lo / W8A8 = {e_h / e_w:.3f} (16 outlier channels x25)", flush=True)
        # T
        for M in (13824, 4289):
            xx = x[:M].contiguous()
            W.CFG.hilo = {}
            tw = tmed(lambda: cls.apply(m, lay, xx))
            W.CFG.hilo = {name: c}
            th = tmed(lambda: cls.apply(m, lay, xx))
            tq = tmed(lambda: W.hilo_quant(xx, S))
            tv = tmed(lambda: ops.scaled_fp8_quant(xx, use_per_token_if_dynamic=True))
            print(f"T {name} [{n}x{k}] c={c} M={M}: W8A8 {tw:.3f} ms -> hi+lo {th:.3f} ms ({th - tw:+.3f}); "
                  f"quant: image {tv:.3f} vs hilo_quant {tq:.3f}", flush=True)
        W.CFG.hilo = {s[0]: s[5] for s in SHAPES}
        del x, a2, qv, r, qr, y_fast, y_proto, w2p, a2p, lay, big
        torch.cuda.empty_cache()
    print("counters:", W.summary())
    print("RESULT:", "FAIL" if FAILS else "PASS", f"({len(FAILS)} failures)")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
