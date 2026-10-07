"""Router top-8 agreement: production's router GEMM (F.linear(x, W).to(float32), cuBLAS) vs kernels/gemv_bf16.cu
(out_mode 2, the same bf16 rounding of the logits) on real GLM-5.3-Flash router weights (docs/BF16_GEMV.md).

Weights: mlp.gate.weight [288, 4096] bf16, mlp.gate.e_score_correction_bias [288] fp32 and
post_attention_layernorm.weight of layers 10..13, read from the local GLM-5.3-Flash checkpoint shards
(GLM53_CKPT, default $TF_EXL3_MODELS/GLM-5.3-Flash-Uncensored-NVFP4; its router / norm tensors are bf16 / fp32,
not quantized). Inputs: the router's input on this model is the RMS-normalized residual times the norm weight
(mhc_pre_big_fuse_with_norm), so x = rmsnorm(randn) * norm_weight, bf16.
Top-8: production's own routing kernel (vllm._custom_ops.grouped_topk: sigmoid, + e_score_correction_bias,
n_group 1, renormalize, routed_scaling_factor 2.5).
Reported per M: tokens whose logits differ at all, max |d logit|, tokens whose top-8 id SET differs, max |d weight|.
Baselines for scale: the same tokens through cuBLAS at another M (cuBLAS picks another kernel -> another
summation order) and against the float64 logits rounded to bf16.

Run: GPU_RUN_RO=$TF_EXL3_MODELS/GLM-5.3-Flash-Uncensored-NVFP4 tests/gpu_run.sh python3 tests/router_topk_agreement.py
"""
from __future__ import annotations

import glob
import json
import os
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import glm53_bf16_gemv as G  # noqa: E402

CKPT = os.environ.get("GLM53_CKPT", os.path.join(os.environ.get("TF_EXL3_MODELS") or os.path.expanduser("~/models"), "GLM-5.3-Flash-Uncensored-NVFP4"))
LAYERS = (10, 11, 12, 13)
TOKENS = int(os.environ.get("ROUTER_TOKENS", str(1 << 16)))   # per layer and M
MS = (5, 8, 16, 32, 48)


def tensor_index() -> dict[str, str]:
    idx = {}
    for f in sorted(glob.glob(os.path.join(CKPT, "model-*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            for k in json.loads(fh.read(n)):
                if k != "__metadata__":
                    idx[k] = f
    return idx


def load(idx: dict[str, str], name: str) -> torch.Tensor:
    from safetensors import safe_open
    with safe_open(idx[name], framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def topk(logits: torch.Tensor, bias: torch.Tensor):
    from vllm import _custom_ops as ops
    w, ids = ops.grouped_topk(logits.contiguous(), 1, 1, 8, True, 2.5, bias, 1)
    return w, ids.long()


def main() -> None:
    dev = torch.device("cuda")
    G.load_ext()
    print("ext", G.EXT_SOURCE, "| checkpoint", CKPT, "| tokens per (layer, M):", TOKENS)
    idx = tensor_index()
    gen = torch.Generator(device=dev).manual_seed(1234)
    tot = {}
    for L in LAYERS:
        pre = f"model.language_model.layers.{L}."
        W = load(idx, pre + "mlp.gate.weight").to(dev)
        b = load(idx, pre + "mlp.gate.e_score_correction_bias").to(dev).float()
        nw = load(idx, pre + "post_attention_layernorm.weight").to(dev).float()
        assert W.dtype == torch.bfloat16 and tuple(W.shape) == (288, 4096)
        ctx = G.GemmCtx(288, 4096, dev)
        # the same tokens are evaluated at every M (cuBLAS-at-another-M baseline)
        X = torch.randn(TOKENS, 4096, device=dev, generator=gen)
        X = (X * torch.rsqrt(X.pow(2).mean(-1, keepdim=True) + 1e-6) * nw).to(torch.bfloat16)
        W64 = W.double()
        ref_bf = torch.cat([(X[i:i + 8192].double() @ W64.t()).to(torch.bfloat16).float()
                            for i in range(0, TOKENS, 8192)])
        _, ids_ref = topk(ref_bf, b)
        per_m = {}
        for M in MS:
            base = torch.empty(TOKENS, 288, device=dev)
            new = torch.empty(TOKENS, 288, device=dev)
            for i in range(0, TOKENS - M + 1, M):
                x = X[i:i + M]
                base[i:i + M] = F.linear(x, W).to(torch.float32)
                new[i:i + M] = G.gemm(x, W, ctx, out_mode=2)
            n = (TOKENS // M) * M
            base, new = base[:n], new[:n]
            wb, ib = topk(base, b)
            wn, inn = topk(new, b)
            sb, sn = ib.sort(-1).values, inn.sort(-1).values
            flips = int((sb != sn).any(-1).sum())
            same_ids = (sb == sn).all(-1)
            # weights compared where the id sets agree, in id order
            wb_s = wb.gather(-1, ib.argsort(-1))
            wn_s = wn.gather(-1, inn.argsort(-1))
            dw = float((wb_s - wn_s)[same_ids].abs().max()) if bool(same_ids.any()) else float("nan")
            diff_tok = int((base != new).any(-1).sum())
            dl = float((base - new).abs().max())
            flips_ref = int((sb != ids_ref[:n].sort(-1).values).any(-1).sum())
            flips_new_ref = int((sn != ids_ref[:n].sort(-1).values).any(-1).sum())
            per_m[M] = (base, ib)
            r = dict(tokens=n, logit_diff_tokens=diff_tok, max_dlogit=dl, top8_set_flips=flips, max_dweight=dw,
                     cublas_vs_f64_flips=flips_ref, new_vs_f64_flips=flips_new_ref)
            for k, v in r.items():
                if isinstance(v, int):
                    tot.setdefault(k, 0)
                    tot[k] += v
            print(f"layer {L} M={M:2d}: tokens {n}, logits differ in {diff_tok} ({diff_tok / n:.4%}), max |dlogit| "
                  f"{dl:.3g}; top-8 set flips new vs cuBLAS {flips} ({flips / n:.5%}), max |dweight| {dw:.3g} | "
                  f"cuBLAS vs f64->bf16 flips {flips_ref} ({flips_ref / n:.5%}), new vs f64->bf16 flips {flips_new_ref} "
                  f"({flips_new_ref / n:.5%})", flush=True)
        # cuBLAS against itself at another M (other kernel / split): the inherent flip rate of production
        b5, i5 = per_m[5]
        for M in MS[1:]:
            bm, im = per_m[M]
            n = min(b5.shape[0], bm.shape[0])
            f_ = int((i5[:n].sort(-1).values != im[:n].sort(-1).values).any(-1).sum())
            d_ = int((b5[:n] != bm[:n]).any(-1).sum())
            tot.setdefault("cublas_m5_vs_m_flips", 0)
            tot["cublas_m5_vs_m_flips"] += f_
            tot.setdefault("cublas_m5_vs_m_tokens", 0)
            tot["cublas_m5_vs_m_tokens"] += n
            print(f"layer {L} cuBLAS M=5 vs cuBLAS M={M}: logits differ in {d_} tokens, top-8 set flips {f_} "
                  f"({f_ / n:.5%})", flush=True)
    print("TOTAL", json.dumps(tot))
    print(f"TOTAL new vs cuBLAS top-8 set flips: {tot['top8_set_flips']} / {tot['tokens']} "
          f"= {tot['top8_set_flips'] / tot['tokens']:.5%}; cuBLAS M=5 vs other M: {tot['cublas_m5_vs_m_flips']} / "
          f"{tot['cublas_m5_vs_m_tokens']} = {tot['cublas_m5_vs_m_flips'] / tot['cublas_m5_vs_m_tokens']:.5%}; "
          f"cuBLAS vs f64: {tot['cublas_vs_f64_flips'] / tot['tokens']:.5%}; new vs f64: "
          f"{tot['new_vs_f64_flips'] / tot['tokens']:.5%}")


if __name__ == "__main__":
    main()
