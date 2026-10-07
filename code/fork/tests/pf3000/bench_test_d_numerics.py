"""PF3000 TEST D, part 2: exit-B numerics emulation on real GLM-5.3-Flash MoE-FFN weights.

The real EXL3 routed-expert weights (trellis/mcg/suh/svh, MoE layers 3..45) are listed in the local partial
checkpoint's model.safetensors.index.json but the 120 safetensors shards are NOT on this node's disk (only
bf16_samples/ + lm_head/ are present, 3.0 GB total). This script proves that and then substitutes the closest
real thing: the real BF16 weights of GLM-5.3-Flash MoE-FFN linears from bf16_samples (shared_experts of MoE
layer 10, dense MLP layer 1), raw and Hadamard-rotated exactly as the routed path rotates its experts (128-
point Sylvester H / sqrt(128) per 128-input/128-output block). The EXL3 4-bpw weight error itself is a
rate-distortion model (the moe-kernels reader's dq_sim: 6.25 % RD bound - 6.70 % QTIP-class relative RMS),
applied as noise on the rotated weights; the report says so.

Emulated per (weights, rotation, activation distribution):
  W_exl3 = fp16(rot(W) + noise(rel RMS rho))     # trellis decode output is fp16 codebook values
  reference y_ref = x_f64 @ W_original_f64 (the untouched real BF16 weights)
  today:         y = fp32acc(x_fp16, W_exl3_fp16)               -> e_native
  exit B (a):    y = fp32acc(e4m3(x, per-row), e4m3(W, direct cvt satfinite))   -> e_f8a
  exit B (b):    y = fp32acc(e4m3(x, per-row), e4m3(W, per-128-col-block))      -> e_f8b
  KILL exit B if e_f8 > 1.2 * e_native

Run: GPU_RUN_RO="$TF_EXL3_MODELS/GLM-5.3-Flash-EXL3-TR3-4bpw-partial" \
     flock /tmp/tf-gpu-bench.lock tests/gpu_run.sh python3 tests/pf3000/bench_test_d_numerics.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]        # .../tests
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "kernels"))
from harness import run_main, gpu_guard  # noqa: E402
import torch  # noqa: E402

dev = "cuda"
SAMPLES = os.path.join(os.environ.get("TF_EXL3_MODELS") or os.path.expanduser("~/models"), "GLM-5.3-Flash-EXL3-TR3-4bpw-partial")
RHOS = (0.0625, 0.0670)          # EXL3 4bpw relative RMS: RD bound / QTIP-class (moe-kernels.md 6)
ROWS = 2048                       # activation rows per emulated call (one prefill piece)
# routed-expert GEMM shapes per rank (TP=2): gate/up [2048, 4096], down [4096, 1024]
# (file, HF file shape, row-slice, col-slice, per-rank emulated shape)
SHAPES = {
    "gate/up [2048x4096]": ("layers.10.mlp.shared_experts.gate_proj.weight.bin", (2048, 4096),
                            slice(None), slice(None), (2048, 4096)),
    "down [4096x1024]":    ("layers.10.mlp.shared_experts.down_proj.weight.bin", (4096, 2048),
                            slice(None), slice(0, 1024), (4096, 1024)),
    "dense mlp l1 [12288x4096]": ("layers.1.mlp.gate_proj.weight.bin", (12288, 4096),
                                  slice(None), slice(None), (12288, 4096)),
}


def load_real(fn, fshape, rs, cs, shape):
    import numpy as np
    a = np.fromfile(f"{SAMPLES}/bf16_samples/{fn}", dtype=np.uint16)
    assert a.size == fshape[0] * fshape[1], (fn, a.size, fshape)
    w = (a.astype(np.uint32) << 16).view(np.float32)
    w = torch.from_numpy(w).to(torch.bfloat16).reshape(fshape)[rs, cs]
    return w.reshape(shape).to(dev)


def had128():
    """The repo's reference 128-point Sylvester Hadamard (kernels/exl3_format_ref.py); fallback: recursive."""
    try:
        from exl3_format_ref import hadamard
        return torch.from_numpy(hadamard(128)).to(dev).float()
    except Exception as e:
        print("fallback hadamard (recursive):", repr(e))
        h = torch.tensor([[1.0]], device=dev)
        while h.shape[0] < 128:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        return h / (128 ** 0.5)


def rotate_128(w, H):
    """block-wise H_K @ W @ H_N (suh/svh = 1), preserving the fp16-ish class by returning bf16."""
    n, k = w.shape
    wn = w.float().reshape(n // 128, 128, k // 128, 128)          # a,k-block-internal,l-block,j
    wn = torch.einsum("ailj,mi->amlj", wn, H)      # H on the 128 rows of each output block
    wn = torch.einsum("amlj,jn->amln", wn, H)      # H on the 128 inputs of each k block
    return wn.reshape(n, k).to(torch.bfloat16)


def rotate_x(x, H):
    t = x.float().reshape(x.shape[0], x.shape[1] // 128, 128)
    t = torch.einsum("blk,kj->blj", t, H)                          # H applied on the 128-input block
    return t.reshape(x.shape).to(torch.float16)


def add_exl3_noise(w, rho, g):
    noise = (torch.randn(w.shape, device=dev, generator=g) * rho * w.float().std()).to(torch.bfloat16)
    return (w.float() + noise.float()).to(torch.float16)


def rowwise_e4m3(x):
    amax = x.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    s = amax / 448.0
    return ((x.float() / s).clamp(-448, 448)).to(torch.float8_e4m3fn), s


def main():
    gpu_guard(6.0)
    import json
    idx = json.load(open(f"{SAMPLES}/model.safetensors.index.json"))
    shard = sorted({v for v in idx["weight_map"].values()})[0]
    print(f"routed-expert shard {shard}: {'PRESENT' if os.path.exists(f'{SAMPLES}/{shard}') else 'ABSENT on this node'} "
          f"(index lists {len(idx['weight_map'])} tensors; local files:")
    print("   ", sorted(os.listdir(SAMPLES)))
    H = had128()
    g = torch.Generator(device=dev).manual_seed(0)
    for name, (fn, fshape, rs, cs, shape) in SHAPES.items():
        w = load_real(fn, fshape, rs, cs, shape)
        n, k = w.shape
        for rot in ("raw", "hadamard-rotated"):
            wr = rotate_128(w, H) if rot == "hadamard-rotated" else w
            x = (torch.randn(ROWS, k, device=dev, generator=g) * 0.02).to(torch.bfloat16)
            xo = x.clone()
            xo[:, ::30] *= 30.0
            for act, xin in (("gauss", x), ("outlier x30", xo)):
                xin16 = xin.to(torch.float16)
                if rot == "hadamard-rotated":
                    xin16 = rotate_x(xin16, H)          # the routed gather's rotation, then fp16 rows
                for rho in RHOS:
                    wq = add_exl3_noise(wr, rho, g)     # the EXL3-reconstructed weights, fp16
                    # reference: the ORIGINAL bf16 weights in fp64; production's path (fp16 operands,
                    # fp32 accumulate, fp16 epilogue) is "EXL3-native"
                    y_ref = torch.mm(xin16.double(), wr.double().t())
                    e_nat = ((torch.mm(xin16.float(), wq.float().t()).to(torch.float16).float()
                              - y_ref).norm() / y_ref.norm()).item()
                    out = {}
                    # (a) direct satfinite cvt, no weight scale (what raw in-register conversion gives)
                    xqa, sa = rowwise_e4m3(xin16)
                    wqa = wq.float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
                    ya = (torch.mm(xqa.float(), wqa.float().t()) * sa).to(torch.float16).float()
                    out["direct cvt"] = ((ya - y_ref).norm() / y_ref.norm()).item()
                    # (b) per-128-col-block e4m3 weights with per-block scale (favourable variant)
                    wbl = wq.float().reshape(n, k // 128, 128)
                    amax = wbl.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
                    ws_ = amax / 448.0
                    wqb = ((wbl / ws_).clamp(-448, 448)).to(torch.float8_e4m3fn)
                    xb = xqa.float().reshape(ROWS, k // 128, 128)
                    y_acc = torch.zeros(ROWS, n, device=dev)
                    for b in range(k // 128):
                        sc = sa * ws_[:, b, 0][None, :]            # per-row x per-block, [ROWS, n]
                        y_acc += torch.mm(xb[:, b, :], wqb[:, b, :].t().float()) * sc
                    yb = y_acc.to(torch.float16).float()
                    out["per-128-block"] = ((yb - y_ref).norm() / y_ref.norm()).item()
                    for lbl, e in out.items():
                        print(f"{name} {rot:16s} act={act:12s} rho={rho:.4f}: EXL3-native rel_l2 {e_nat:.6f} | "
                              f"exitB {lbl:13s} rel_l2 {e:.6f} | ratio x{e / e_nat:.3f} "
                              f"{'KILL' if e > 1.2 * e_nat else 'pass'}", flush=True)
        del w
        torch.cuda.empty_cache()


run_main(main)
