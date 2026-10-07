"""r16z8 C1-C4: GLM53_DEC_DLMH=1 and GLM53_DEC_AR1SHOT=1 TOGETHER on two vLLM TP ranks (nodeC: two processes on the one
GB10, NCCL's net transport over loopback sockets with NCCL_HOSTID per rank, P2P/SHM/IB off - as tests/decode6; unlike
decode5's U8, which emulated the collectives over gloo, every collective here is production's: vLLM's TP group,
pynccl all-gather / all-reduce through CudaCommunicator).

  C1 production's ORDER: both plugins installed (integrate.py's order: ar1shot, then dlmh) BEFORE the process group
     exists; the AR1SHOT readiness agreement happens at CudaCommunicator construction on both ranks (TP + world group);
     the DLMH module stays inert until its first eager call
  C2 the drafter's first eager compute_candidates on the real lm_head shards (77440 rows per rank): DLMH setup +
     self-test + its TP agreement (an int32 all-reduce: NOT one-shot-eligible, production's ring) on both ranks; then
     peaked / random / mixed drafter rows at T = 7, 14 (served) and 21 (production path): hooked == stock bitwise
     (candidate ids and unary logits; the stock head all-gathers its bf16 logits through the real NCCL all-gather)
  C3 a CUDA graph captured under vLLM's GroupCoordinator.graph_capture like the drafter graph: a decode-size
     all-reduce (served one-shot) -> the DLMH two-stage head (its int32 pack all-gather inside the graph) -> another
     all-reduce consuming the first; 6 replays with new rank-specific partials and new replicated rows: every output
     bit-identical to production (AR1SHOT off = NCCL's ring all-reduce, eager; the stock candidate head, eager)
  C4 the boot / serving log lines both modules print in this run (the strings boot_checks and the A/B PROOF read):
     the AR1SHOT agreement + serving-confirmed lines and the DLMH built / self-test lines, on both ranks; no
     'not ready' / 'differs' / 'setup failed' line
Usage: GPU_RUN_SHM=2g GPU_RUN_RO=<tests/dlmh/common.py RO> tests/gpu_run.sh python3 tests/r16z8/test_combo_tp2.py
"""
from __future__ import annotations

import io
import logging
import os
import sys
import types

import torch

sys.path.insert(0, "/w")
sys.path.insert(0, "/w/tests/dlmh")
sys.path.insert(0, "/w/tests/decode6")


def worker(rank: int, port: int, q) -> None:
    os.environ.update({"NCCL_HOSTID": f"z8-host{rank}", "NCCL_IB_DISABLE": "1", "NCCL_SOCKET_IFNAME": "lo",
                       "NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "1", "NCCL_NVLS_ENABLE": "0", "NCCL_DEBUG": "WARN",
                       "GLM53_DEC_AR1SHOT": "1", "GLM53_DEC_DLMH": "1"})
    out, logbuf = [], io.StringIO()
    try:
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(0.12)
        import vllm.logger  # noqa: F401  (vLLM's logging config first: it would drop a handler attached before it)
        h = logging.StreamHandler(logbuf)
        h.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        logging.getLogger("vllm").addHandler(h)    # the modules' loggers are vllm.glm53_ar1shot / vllm.glm53_dlmh
        import glm53_ar1shot as A
        import glm53_dlmh as D
        A.plugin_install()                       # integrate.plugin_register's order, before the process group exists
        D.plugin_install()
        ok1 = A.ST.mode == "on" and D.CFG.mode == "on"
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
        from vllm.distributed.parallel_state import set_custom_all_reduce
        set_custom_all_reduce(False)
        with set_current_vllm_config(VllmConfig()):
            init_distributed_environment(world_size=2, rank=rank, local_rank=0,
                                         distributed_init_method=f"tcp://127.0.0.1:{port}", backend="nccl")
            ensure_model_parallel_initialized(2, 1)
        from vllm.distributed import get_tp_group, tensor_model_parallel_all_gather, tensor_model_parallel_all_reduce
        tp = get_tp_group()
        ok1 &= A.ST.agreed.get(id(tp.device_communicator)) is True and A.COUNTERS.get("agree_at_init", 0) >= 1
        ok1 &= not D.HEAD.ready and D.HEAD.failed is None
        out.append(("C1", ok1, f"rank {rank}: modes ar1shot={A.ST.mode} dlmh={D.CFG.mode}; ar1shot agreed at "
                               f"construction (tp) {A.ST.agreed.get(id(tp.device_communicator))}, agree_at_init "
                               f"{A.COUNTERS.get('agree_at_init', 0)}; dlmh inert before its first call"))

        # ---- C2: the real lm_head shards, production's FP8 head, the stock all-gathering candidate head
        import common as Cm
        G = Cm.enable_fp8_gemv()
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
                return tensor_model_parallel_all_gather(lm_head.quant_method.apply(lm_head, hs), dim=-1)[
                    ..., : self.org_vocab_size]

        fake = types.SimpleNamespace()
        fake.lm_head = types.SimpleNamespace(tp_size=2, glm53_fp8_head=holder, quant_method=_QM())
        fake.candidate_logits_processor = _Proc()
        fake.model = types.SimpleNamespace(candidate_selector=types.SimpleNamespace(top_k=16))
        from vllm.model_executor.models import qwen3_dflash2 as Q
        stock = Q.DFlash2Qwen3ForCausalLM.compute_candidates
        D._install_model()                       # what the loader hook does after the drafter's weights load
        hook = Q.DFlash2Qwen3ForCausalLM.compute_candidates
        g = torch.Generator(device="cuda").manual_seed(11)     # replicated rows (same seed on both ranks)

        def rows(kind, T):
            r = torch.randint(0, Cm.V, (T, 3), generator=g, device="cuda")
            pk = lm[r].float().sum(1)
            rn = torch.randn(T, Cm.H, generator=g, device="cuda")
            x = {"peaked": pk, "random": rn,
                 "mix": pk / pk.norm(dim=-1, keepdim=True) + 0.7 * rn / rn.norm(dim=-1, keepdim=True)}[kind]
            return (x / x.norm(dim=-1, keepdim=True) * 110.0).to(torch.bfloat16)

        bad = 0
        for kind in ("peaked", "random", "mix"):
            for T in (7, 14, 21):
                x = rows(kind, T)
                c1, u1 = hook(fake, x)           # the first call runs setup + self-test + the TP agreement
                c0, u0 = stock(fake, x)
                bad += int(not (torch.equal(c0, c1) and torch.equal(u0.view(torch.int16), u1.view(torch.int16))))
        torch.cuda.synchronize()
        ok2 = bad == 0 and D.HEAD.ready and D.HEAD.failed is None and D.COUNTERS["served"] == 6 \
            and D.COUNTERS["prod_rows"] == 3
        out.append(("C2", ok2, f"rank {rank}: dlmh ready={D.HEAD.ready} failed={D.HEAD.failed}; 9 eager calls hooked "
                               f"== stock bitwise: {bad == 0} ({bad} differ); served {D.COUNTERS['served']} (T 7/14), "
                               f"production path {D.COUNTERS['prod_rows']} (T 21); ar1shot counters {A.summary()}"))

        # ---- C3: drafter-like graph: AR (one-shot) -> DLMH two-stage head -> AR, replayed
        from test_ar1shot import H, bits, mk_pair
        M = 7
        xa = torch.zeros(M, H, dtype=torch.bfloat16, device="cuda")
        hs = torch.zeros(M, H, dtype=torch.bfloat16, device="cuda")
        cap0 = A.COUNTERS["captured"]
        gph = torch.cuda.CUDAGraph()
        with tp.graph_capture() as gc:
            with torch.cuda.graph(gph, stream=gc.stream):
                y1 = tensor_model_parallel_all_reduce(xa * 1)
                cg, ug = hook(fake, hs)
                y2 = tensor_model_parallel_all_reduce(y1 * 1)
        ncap = A.COUNTERS["captured"] - cap0
        ok3 = ncap == 2 and A.COUNTERS["fallback"] == 0 and D.COUNTERS["captured"] >= 1
        same = 0
        for it in range(6):
            a, b = mk_pair(M * H, 500 + it)
            xa.copy_((a if rank == 0 else b).view(M, H))
            hs.copy_(rows(("peaked", "random", "mix")[it % 3], M))
            gph.replay()
            torch.cuda.synchronize()
            A.ST.mode = "off"                    # production's all-reduce (NCCL ring), eager
            r1 = tensor_model_parallel_all_reduce(xa * 1)
            r2 = tensor_model_parallel_all_reduce(r1 * 1)
            A.ST.mode = "on"
            c0, u0 = stock(fake, hs.clone())
            torch.cuda.synchronize()
            same += int(torch.equal(bits(r1), bits(y1)) and torch.equal(bits(r2), bits(y2)) and torch.equal(c0, cg)
                        and torch.equal(u0.view(torch.int16), ug.view(torch.int16)))
        # an eager eligible all-reduce after the capture -> AR1SHOT's serving-confirmed line (production: the eager
        # embedding all-reduce of the first step after the graphs were captured)
        tensor_model_parallel_all_reduce(torch.ones(1, H, dtype=torch.bfloat16, device="cuda"))
        torch.cuda.synchronize()
        ok3 &= same == 6
        out.append(("C3", ok3, f"rank {rank}: graph [AR one-shot -> DLMH two-stage -> AR]: {ncap} one-shot all-reduces "
                               f"captured (fallback {A.COUNTERS['fallback']}), dlmh captured {D.COUNTERS['captured']}; "
                               f"6 replays bit-identical to production (ring AR + stock head): {same}/6"))

        # ---- C4: the log lines
        txt = logbuf.getvalue()
        want = ["glm53_ar1shot plugin loaded", "-> installing (mode on)",
                "glm53_ar1shot: hooked CudaCommunicator.all_reduce (mode on",
                "agreement: every rank ready -> one-shot all-reduce armed (mode on",
                f"[glm53-ar1shot] rank {rank} serving confirmed (mode on): ",
                "glm53_dlmh plugin loaded", "-> installing (mode on, C 128, g 128)",
                f"glm53_dlmh: rank {rank}/2 coarse candidate head built (int4 g128, C=128, vocab rows 77440",
                f"glm53_dlmh: rank {rank} self-test: candidates and unary logits byte-equal to production's"]
        never = ["agreement: a rank is not ready", "differs from production", "glm53_dlmh: setup failed",
                 "not installed", "self-test: two-stage candidates differ", "glm53_dlmh: setup failed on another TP rank"]
        miss = [w for w in want if w not in txt]
        hit = [n for n in never if n in txt]
        out.append(("C4", not miss and not hit, f"rank {rank}: {len(want) - len(miss)}/{len(want)} expected lines, "
                                                f"missing {miss}, forbidden present {hit}"))
        q.put((rank, out, None, txt))
    except Exception:  # noqa: BLE001
        import traceback
        q.put((rank, out, traceback.format_exc(), logbuf.getvalue()))


def main():
    import socket
    import torch.multiprocessing as mp
    with socket.socket() as s_:
        s_.bind(("127.0.0.1", 0))
        port = s_.getsockname()[1]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=worker, args=(r, port, q)) for r in range(2)]
    for p in ps:
        p.start()
    res = [q.get(timeout=2400) for _ in ps]
    for p in ps:
        p.join(120)
    allok = True
    for rank, out, err, txt in sorted(res, key=lambda t: t[0]):
        print(f"---- rank {rank} module log lines:")
        for line in txt.splitlines():
            if "glm53" in line:
                print(f"  [r{rank}] {line}")
        for item in out:
            allok &= bool(item[1])
            print(f"{item[0]} {'OK ' if item[1] else 'FAIL'} {item[2]}")
        if err:
            allok = False
            print(f"rank {rank} raised:\n{err}")
        if len(out) < 4:
            allok = False
    print("ALL OK" if allok else "FAILED")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
