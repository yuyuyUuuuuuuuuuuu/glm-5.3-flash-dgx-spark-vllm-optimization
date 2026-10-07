"""DEC_DLMH U8: two TP ranks (two processes on nodeC's one GB10; the TP collectives emulated over gloo: NCCL refuses
two ranks on one device). Real lm_head shards (77440 rows each), the installed DFlash2Qwen3ForCausalLM.compute_candidates
on a model stand-in whose stock LogitsProcessor all-gathers the bf16 logits like production's.

  A  both ranks set up, agree, self-test, serve: hooked == stock on both ranks (peaked, random and noisy rows, T = 7,
     14; T = 21 -> production path), served counters equal on both ranks
  B  rank 1's coarse build raises: the agreement turns the head off on BOTH ranks (no rank serves the two-stage head
     alone, no collective mismatch / hang), every call == stock

  GPU_RUN_RO=<common.RO> tests/gpu_run.sh python3 tests/dlmh/test_dlmh_tp2.py
"""
from __future__ import annotations

import os
import sys
import types

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, "/w")
sys.path.insert(0, "/w/tests/dlmh")


def gloo_ag(t, dim=-1):
    parts = [torch.empty_like(t, device="cpu") for _ in range(2)]
    dist.all_gather(parts, t.cpu())
    return torch.cat(parts, dim=dim).to(t.device)


def gloo_ar(t):
    c = t.cpu()
    dist.all_reduce(c)
    return c.to(t.device)


def worker(rank: int, out_q) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29531")
    dist.init_process_group("gloo", rank=rank, world_size=2)
    torch.cuda.set_per_process_memory_fraction(0.12)
    import common as Cm
    import glm53_dlmh as D
    import vllm.distributed as VD
    G = Cm.enable_fp8_gemv()
    VD.tensor_model_parallel_all_reduce = gloo_ar
    VD.get_tensor_model_parallel_rank = lambda: rank
    D._all_gather = gloo_ag
    VL = Cm.V // 2
    lm = Cm.load_lm_bf16()
    holder, fp8, _ = Cm.fp8_holder(lm[rank * VL:(rank + 1) * VL].contiguous())
    del fp8

    class _QM:
        def apply(self, layer, x, bias=None):
            return Cm.prod_logits(G, holder, x)

    class _Proc:
        soft_cap, scale, logits_as_input, use_all_gather = None, 1.0, False, True
        org_vocab_size = Cm.V

        def __call__(self, lm_head, hs):
            return gloo_ag(lm_head.quant_method.apply(lm_head, hs))[..., : self.org_vocab_size]

    fake = types.SimpleNamespace()
    fake.lm_head = types.SimpleNamespace(tp_size=2, glm53_fp8_head=holder, quant_method=_QM())
    fake.candidate_logits_processor = _Proc()
    fake.model = types.SimpleNamespace(candidate_selector=types.SimpleNamespace(top_k=16))
    from vllm.model_executor.models import qwen3_dflash2 as Q
    stock = Q.DFlash2Qwen3ForCausalLM.compute_candidates
    D._install_model()
    hook = Q.DFlash2Qwen3ForCausalLM.compute_candidates
    # identical rows on both ranks (production replicates the drafter's hidden states): peaked mixes of full lm_head
    # rows, pure random directions and mixtures, at the drafter's hidden scale
    g = torch.Generator(device="cuda").manual_seed(5)
    xs = []
    for kind in ("peaked", "random", "mix"):
        for T in (7, 14, 21):
            r = torch.randint(0, Cm.V, (T, 3), generator=g, device="cuda")
            pk = lm[r].float().sum(1)
            rn = torch.randn(T, Cm.H, generator=g, device="cuda")
            x = {"peaked": pk, "random": rn, "mix": pk / pk.norm(dim=-1, keepdim=True) + 0.7 * rn / rn.norm(dim=-1, keepdim=True)}[kind]
            xs.append((kind, (x / x.norm(dim=-1, keepdim=True) * 110.0).to(torch.bfloat16)))
    res = {}
    for case in ("A", "B"):
        D.HEAD.__init__()
        D.parse_env({"GLM53_DEC_DLMH": "1"})
        orig_build = D.build_coarse_from_marlin
        if case == "B" and rank == 1:
            def boom(*a, **k):
                raise RuntimeError("injected coarse-build failure on rank 1")
            D.build_coarse_from_marlin = boom
        bad = 0
        for kind, x in xs:
            c0, u0 = stock(fake, x)
            c1, u1 = hook(fake, x)
            bad += int(not (torch.equal(c0, c1) and torch.equal(u0.view(torch.int16), u1.view(torch.int16))))
        D.build_coarse_from_marlin = orig_build
        torch.cuda.synchronize()
        res[case] = dict(bad=bad, ready=D.HEAD.ready, failed=D.HEAD.failed, served=D.COUNTERS["served"],
                         prod_rows=D.COUNTERS["prod_rows"])
        D.COUNTERS.update(served=0, prod_rows=0)
    out_q.put((rank, res))
    dist.destroy_process_group()


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=worker, args=(r, q)) for r in range(2)]
    for p in ps:
        p.start()
    got = {}
    for _ in range(2):
        r, res = q.get(timeout=1200)
        got[r] = res
    for p in ps:
        p.join(timeout=120)
    for r in (0, 1):
        print(f"[dlmh-tp2] rank {r}: {got[r]}", flush=True)
    A0, A1, B0, B1 = got[0]["A"], got[1]["A"], got[0]["B"], got[1]["B"]
    okA = A0["bad"] == 0 and A1["bad"] == 0 and A0["ready"] and A1["ready"] and A0["served"] == A1["served"] == 6 \
        and A0["prod_rows"] == A1["prod_rows"] == 3
    okB = B0["bad"] == 0 and B1["bad"] == 0 and not B0["ready"] and not B1["ready"] and B0["served"] == B1["served"] == 0
    print(f"[dlmh-tp2] A both ranks serve, hooked == stock: {'OK' if okA else 'FAIL'}", flush=True)
    print(f"[dlmh-tp2] B rank 1 setup failure -> off on both ranks, == stock, no hang: {'OK' if okB else 'FAIL'}",
          flush=True)
    print(f"[dlmh-tp2] {'ALL OK' if okA and okB else 'FAILED'}", flush=True)
    sys.exit(0 if okA and okB else 1)
