"""opt-dense-rev (refuter): glm53_mla_prefill.fused_index vs production's chain on ADVERSARIAL index inputs that the
opt-dense test (one request, req_id all 0, logical top-k from topk_logical) never builds:
  - several requests (req_id ragged runs, block_table rows of different live widths, padded with garbage),
  - -1 holes in the MIDDLE of a row (not only a -1 tail), other negative values (-5, INT_MIN+1),
  - token indices whose block id is >= max_num_blocks_per_req (out of the table) and huge values,
  - a non-contiguous req_id view, a kv_indices buffer bigger than T*W (bytes past T*W must stay untouched).
Reference: triton_convert_req_index_to_global_index(..., return_valid_counts=True) then clamp_(min=0).to(int32)
copied into kv_indices (production forward_mqa's bytes). Byte equality of the WHOLE buffer + valid counts.
Run: source tests/mla_env.sh; tests/gpu_run.sh python3 tests/optdense_rev/adv_fused_index.py"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import glm53_mla_prefill as M  # noqa: E402
from vllm.v1.attention.backends.mla.sparse_utils import triton_convert_req_index_to_global_index  # noqa: E402

FAIL = []


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg, flush=True)
    if not ok:
        FAIL.append(msg)


def case(seed, T, W, nreq, nblk, bs, hole_p, oob_p, neg_p, noncontig):
    g = torch.Generator(device="cpu").manual_seed(seed)
    cuts = sorted(torch.randperm(T - 1, generator=g)[: nreq - 1].add(1).tolist())
    rid = torch.zeros(T, dtype=torch.int32)
    for i, c in enumerate(cuts):
        rid[c:] = i + 1
    bt = torch.randint(0, 1 << 20, (nreq + 2, nblk), generator=g, dtype=torch.int32)
    tok = torch.randint(0, nblk * bs, (T, W), generator=g, dtype=torch.int32)
    u = torch.rand(T, W, generator=g)
    tok[u < hole_p] = -1
    tok[(u >= hole_p) & (u < hole_p + oob_p)] = nblk * bs + 7          # block id >= max_num_blocks_per_req
    tok[(u >= hole_p + oob_p) & (u < hole_p + oob_p + neg_p)] = -5
    tok[0, : W // 2] = -1                                              # leading holes
    tok[1, :] = -1                                                     # an all-invalid row
    tok[2, -1] = 2 ** 31 - 1
    rid, bt, tok = rid.cuda(), bt.cuda(), tok.cuda()
    if noncontig:
        big = torch.zeros(2 * T, dtype=torch.int32, device="cuda")
        big[::2] = rid
        rid = big[::2]
    ref_s, ref_v = triton_convert_req_index_to_global_index(rid, bt, tok, BLOCK_SIZE=bs, NUM_TOPK_TOKENS=W,
                                                            return_valid_counts=True)
    ki_ref = torch.full(((T + 9) * W,), -7, dtype=torch.int32, device="cuda")
    ki_ref[: T * W].copy_(ref_s.reshape(-1).clamp(min=0).to(torch.int32))
    ki = torch.full_like(ki_ref, -7)
    r = M.fused_index(rid, bt, tok, bs, ki)
    if r is None:
        check(False, f"seed {seed}: fused_index declined (T {T} W {W})")
        return
    s, v = r
    torch.cuda.synchronize()
    nd = int((ki != ki_ref).sum())
    check(nd == 0, f"seed {seed} T={T} W={W} nreq={nreq} bs={bs} holes={hole_p} oob={oob_p} neg={neg_p} "
                   f"noncontig={noncontig}: whole kv_indices == production bytes ({nd} differ)")
    check(torch.equal(v, ref_v.to(torch.int32)), f"seed {seed}: valid counts == triton_convert's")
    check(s.data_ptr() == ki.data_ptr(), f"seed {seed}: slots is the kv_indices view")


def main():
    i = 0
    for T in (13824, 4289, 1791, 300):
        for (nreq, hole, oob, neg, nc) in ((1, 0.0, 0.0, 0.0, False), (5, 0.3, 0.05, 0.02, False),
                                           (17, 0.6, 0.1, 0.05, True), (3, 0.95, 0.0, 0.0, True)):
            case(100 + i, T, 2048, min(nreq, T // 4), 600, 64, hole, oob, neg, nc)
            i += 1
    case(999, 2048, 1024, 4, 300, 64, 0.2, 0.05, 0.0, False)     # W 1024
    # declines
    check(M.fused_index(torch.zeros(8, dtype=torch.int32, device="cuda"), torch.zeros(1, 4, dtype=torch.int32,
          device="cuda"), torch.zeros(8, 2000, dtype=torch.int32, device="cuda"), 64,
          torch.zeros(8 * 2000, dtype=torch.int32, device="cuda")) is None, "W=2000 (not a power of 2) declined")
    print("ALL PASSED" if not FAIL else f"FAILED {len(FAIL)}: {FAIL[:5]}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
