"""Which top-p does the verifier apply? The image routes Sampler.apply_top_k_top_p -> vllm/v1/sample/ops/
topk_topp_sampler.apply_top_k_top_p: Triton (Qrita pivot search) when the batch has >= 8 rows, the sort-based
PyTorch path below 8. A DFlash2 verify batch has (n + 1) rows per request, so one request at adaptive K n = 4 or 5
(5-6 rows) takes the PyTorch path and n = 7 (8 rows) or two requests take Triton. This probe compares the two masks
on LLM-like rows at the production vocab (V = 154880, top_p 0.95) and on pathological rows (a -1e4 floor, which
breaks the Triton pivot's first-tile statistics). Informational; exit 0 unless the realistic rows disagree.
"""
import sys

import torch

from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton

dev = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(min(1.0, 8 * 2**30 / torch.cuda.get_device_properties(0).total_memory))
V = 154880
g = torch.Generator(device=dev).manual_seed(5)
bad_real = 0


def compare(name, logits, p=0.95):
    global bad_real
    pp = torch.full((logits.shape[0],), p, device=dev)
    a = apply_top_k_top_p_triton(logits.clone(), None, pp)
    b = apply_top_k_top_p_pytorch(logits.clone(), None, pp)
    ma, mb = torch.isinf(a), torch.isinf(b)
    rows_diff = int((ma != mb).any(dim=1).sum())
    probs = torch.softmax(logits.double(), dim=-1)
    kept_a = (probs * (~ma)).sum(1)
    kept_b = (probs * (~mb)).sum(1)
    print(f"{name}: rows {logits.shape[0]}, rows whose mask differs {rows_diff}, kept mass triton "
          f"[{float(kept_a.min()):.4f}, {float(kept_a.max()):.4f}] pytorch [{float(kept_b.min()):.4f}, {float(kept_b.max()):.4f}], "
          f"max |kept diff| {float((kept_a - kept_b).abs().max()):.2e}, tokens kept triton/pytorch "
          f"{int((~ma).sum())}/{int((~mb).sum())}")
    return rows_diff


R = 512
# LLM-like: Gaussian bulk (std 2.5) + a head of 1..64 tokens 8..20 above the mean with Zipf-like spacing
for head, lo, hi in ((1, 14, 20), (4, 10, 16), (16, 8, 14), (64, 6, 12)):
    x = 2.5 * torch.randn(R, V, device=dev, generator=g)
    idx = torch.randint(0, V, (R, head), device=dev, generator=g)
    vals = lo + (hi - lo) * torch.rand(R, head, device=dev, generator=g)
    x.scatter_(1, idx, vals)
    bad_real += compare(f"LLM-like head={head} ({lo}..{hi} above the bulk)", x)
# a real-ish decode row set: temperature-scaled Gumbel-ish logits
x = torch.distributions.Gumbel(0.0, 3.0).sample((R, V)).to(dev)
bad_real += compare("Gumbel(0,3) rows", x)
# pathological: 4 active tokens over a -1e4 floor (the first harness filler)
x = torch.full((R, V), -1e4, device=dev)
x[:, [11, 40000, 100003, 154870]] = torch.randn(R, 4, device=dev, generator=g) * 2
compare("pathological -1e4 floor (4 finite tokens)", x)
print("REALISTIC ROWS AGREE" if bad_real == 0 else f"REALISTIC ROWS DISAGREE: {bad_real}")
sys.exit(0 if bad_real == 0 else 1)
