"""opt-dense: W8A8 for the drafter fc (GLM53_DENSE_W8A8_ONLY=draft.fc), 4096 x 20480, group "draft", prefix model.fc.
Through the real hooked apply on production's overlay exl3.py (+ fp8_gemv large-M path as production installs it):
production wall (TileLang large-M from M 8192, Marlin below) vs W8A8 wall, the GEMM cfg sweep for this shape, and the
output error of W8A8 vs production (rel_l2) on activations shaped like the aux hidden states (5 layer slices with
different scales). Also checks the opt-in: with ONLY unset ("all") the draft group is NOT served.
Usage: source tests/w8a82/env.sh; tests/gpu_run.sh python3 tests/optdense/bench_draft_fc.py"""
import os, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pf3000"))
from harness import run_main, gpu_guard, load_prod  # noqa
import torch
dev = "cuda"


def tmed(fn, n=7):
    for _ in range(2): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def main():
    gpu_guard(8.0)
    for v in ("GLM53_FP8_GEMV", "GLM53_FP8_GEMV_MAX_M", "TF_EXL3_MOE", "GLM53_KDA_BF16_LARGE_M", "GLM53_DEC_FP8ROOF"):
        os.environ.pop(v, None)
    os.environ["GLM53_FP8_LARGE_M"] = "1"
    os.environ["GLM53_DENSE_FP8"] = "dense,kda,mla,shared"
    from test_fp8_integrate import single_rank_tp, L
    single_rank_tp()
    prod = load_prod()
    import fp8_gemv as F
    import fp8_w8a8 as W
    cls = prod.Glm53DenseFp8Method
    F.install(prod)
    prod_apply = cls.apply
    os.environ["GLM53_DENSE_W8A8"] = "1"
    os.environ.pop("GLM53_DENSE_W8A8_ONLY", None)
    rep = W.install(prod)
    print("install (ONLY unset):", rep, "groups", sorted(W.STATE.groups), flush=True)
    g = torch.Generator(device=dev).manual_seed(3)
    n, k = 4096, 20480
    w = (torch.randn(n, k, device=dev, generator=g) * 0.01).to(torch.bfloat16)
    lay = L(w)
    takes_prefix = "prefix" in cls.__init__.__code__.co_varnames
    m = cls("draft", "model.fc") if takes_prefix else cls("draft")
    m.process_weights_after_loading(lay)
    x0 = torch.randn(13824, k, device=dev, generator=g)
    for i, s in enumerate((0.3, 1.0, 3.0, 8.0, 20.0)):          # aux hidden states of 5 layers: different scales
        x0[:, i * 4096:(i + 1) * 4096] *= s
    x0 = x0.to(torch.bfloat16)
    c0 = W.COUNTERS.get("w8a8_eager", 0)
    cls.apply(m, lay, x0[:4289])
    print("ONLY unset: draft.fc served?", W.COUNTERS.get("w8a8_eager", 0) != c0, "(expected False)", flush=True)
    W.uninstall()
    os.environ["GLM53_DENSE_W8A8_ONLY"] = "draft.fc"
    rep = W.install(prod)
    print("install (ONLY=draft.fc):", rep, "groups", sorted(W.STATE.groups), flush=True)
    m.process_weights_after_loading(lay) if False else None
    for M in (13824, 8192, 4289, 1791):
        x = x0[:M].contiguous()
        c0 = W.COUNTERS.get("w8a8_eager", 0)
        y = cls.apply(m, lay, x)
        served = W.COUNTERS.get("w8a8_eager", 0) - c0
        ref = prod_apply._tf_w8a8_orig(m, lay, x) if hasattr(prod_apply, "_tf_w8a8_orig") else None
        W.STATE.enabled = False
        yp = cls.apply(m, lay, x)
        tp = tmed(lambda: cls.apply(m, lay, x))
        W.STATE.enabled = True
        tw = tmed(lambda: cls.apply(m, lay, x))
        rel = ((y.float() - yp.float()).norm() / yp.float().norm()).item()
        print(f"draft.fc M={M}: served {served}, production {tp:.3f} ms -> W8A8 {tw:.3f} ms ({tw - tp:+.3f}); "
              f"rel_l2 W8A8 vs production {rel:.2e}; cfg {W.gemm_choice(n, k, M)}", flush=True)
    # GEMM cfg sweep for (4096, 20480)
    key = W._key(lay.weight, lay.weight_scale, k)
    alpha = W.ALPHA[key]
    w8 = W._scratch(n, k, x0.device)
    W.repack(w8, lay.weight, n, k)
    for M in (13824, 4289, 1791):
        q, sa = W.quant_per_token(x0[:M].contiguous())
        out = torch.empty(M, n, dtype=torch.bfloat16, device=dev)
        res = []
        for cfg in range(4):
            for sw in (1, 2, 4, 8):
                for ro in (0, 1, 2):
                    try:
                        t = tmed(lambda: W.ext().fp8_w8a8_gemm(out, q, w8, sa.view(-1), alpha, cfg, sw, ro, W._ws(dev)), 5)
                        res.append((t, cfg, sw, ro))
                    except Exception as e:  # noqa
                        pass
        res.sort()
        fl = 2 * M * n * k
        print(f"GEMM sweep M={M}: best " + ", ".join(f"{t:.3f} ms ({fl / t / 1e9:.0f} TF) cfg{c} s{s} r{r}"
                                                  for t, c, s, r in res[:4]), flush=True)
    print("counters:", W.summary())


if __name__ == "__main__":
    run_main(main)
