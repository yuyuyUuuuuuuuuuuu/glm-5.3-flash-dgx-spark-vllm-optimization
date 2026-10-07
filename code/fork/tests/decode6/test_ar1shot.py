"""GLM53_DEC_AR1SHOT on nodeC: two vLLM TP ranks = two processes on the one GB10, NCCL over loopback sockets
(NCCL_HOSTID per rank, P2P/SHM/IB off: NCCL's net transport with its proxy thread, as on the RoCE pair, but over TCP).

A  NCCL arithmetic: production's pynccl all_reduce == one_shot (all_gather + bf16 add) bitwise, and both == the
   torch reference bf16(float(a) + float(b)) computed from both partials (every rank builds both from seeds).
   Data: per element a random mix of normal values over 2^-40..2^40, exact ties at the bf16 rounding midpoint
   (b = +-half an ulp of a, so round-to-even decides), cancellations, exponent gaps of 8..30, subnormals, +-0,
   overflow to +-inf. M x 4096 for M in 1..64 (decode) and fp16 / fp32.
B  the installed wrapper: tensor_model_parallel_all_reduce (mode on) == production's result, eager and inside a CUDA
   graph replayed with new inputs; a 2 MiB tensor (> MAX_KB) takes production's path; counters.
C  verify mode: production's result served, mismatch counter 0.
D  latency (loopback sockets, NOT production's RoCE): AR vs one-shot per call, eager and in a CUDA graph of 50
   collectives, M = 1, 3, 5, 6, 8, 16, 64 rows of 4096 bf16.
Usage: GPU_RUN_SHM=2g tests/gpu_run.sh python3 tests/decode6/test_ar1shot.py [bench]
"""
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

H = 4096


def mk_pair(n, seed, dtype=torch.bfloat16):
    """both ranks' partials (a, b), adversarial mix, on cuda"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    kind = torch.randint(0, 9, (n,), generator=g)
    e = torch.randint(-40, 41, (n,), generator=g).float()
    a = torch.randn(n, generator=g) * torch.exp2(e)
    b = torch.randn(n, generator=g) * torch.exp2(e)
    a = a.to(dtype).float()
    fin = torch.finfo(dtype)
    # ulp of a in this dtype
    mant = {torch.bfloat16: 7, torch.float16: 10, torch.float32: 23}[dtype]
    ea = torch.floor(torch.log2(a.abs().clamp_min(fin.tiny)))
    half_ulp = torch.exp2(ea - mant - 1)
    sgn = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0)
    tie = sgn * half_ulp                                            # exact midpoint sums
    canc = -a * torch.where(torch.rand(n, generator=g) < 0.5, 1.0, 1.0 + 2.0 ** -mant)
    gap = a * torch.exp2(-torch.randint(8, 31, (n,), generator=g).float()) * sgn
    sub = torch.randn(n, generator=g) * fin.tiny * 0.25            # subnormal range
    zero = torch.where(torch.rand(n, generator=g) < 0.5, -0.0, 0.0) * torch.ones(n)
    big = torch.full((n,), float(fin.max)) * torch.where(torch.rand(n, generator=g) < 0.5, 0.75, -0.75)
    b = torch.where(kind == 1, tie, b)
    b = torch.where(kind == 2, canc, b)
    b = torch.where(kind == 3, gap, b)
    b = torch.where(kind == 4, sub, b)
    a = torch.where(kind == 4, torch.randn(n, generator=g) * fin.tiny * 0.25, a)
    b = torch.where(kind == 5, zero, b)
    a = torch.where(kind == 5, torch.where(torch.rand(n, generator=g) < 0.5, -0.0, 0.0) * torch.ones(n), a)
    a = torch.where(kind == 6, big, a)
    b = torch.where(kind == 6, big * torch.where(torch.rand(n, generator=g) < 0.7, 1.0, -0.9), b)
    return a.to(dtype).cuda(), b.to(dtype).cuda()


def bits(t):
    return t.view(-1).view({2: torch.int16, 4: torch.int32}[t.element_size()])


def worker(rank, port, q, bench):
    os.environ.update({"NCCL_HOSTID": f"ar1-host{rank}", "NCCL_IB_DISABLE": "1", "NCCL_SOCKET_IFNAME": "lo",
                       "NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "1", "NCCL_NVLS_ENABLE": "0",
                       "NCCL_DEBUG": "WARN"})
    os.environ.pop("GLM53_DEC_AR1SHOT", None)
    out = []
    try:
        torch.cuda.set_device(0)
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
        from vllm.distributed.parallel_state import set_custom_all_reduce
        set_custom_all_reduce(False)
        with set_current_vllm_config(VllmConfig()):
            init_distributed_environment(world_size=2, rank=rank, local_rank=0,
                                         distributed_init_method=f"tcp://127.0.0.1:{port}", backend="nccl")
            ensure_model_parallel_initialized(2, 1)
        from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
        import glm53_ar1shot as A
        tp = get_tp_group()
        dc = tp.device_communicator
        pc = dc.pynccl_comm
        out.append(("info", f"rank {rank}: pynccl {pc is not None and not pc.disabled}, nccl "
                            f"{pc.nccl.ncclGetVersion() if pc is not None else '-'}"))

        # ---- A: NCCL arithmetic
        bad = 0
        tot = 0
        cases = 0
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            Ms = (1, 2, 3, 5, 6, 7, 8, 14, 16, 21, 32, 64) if dtype == torch.bfloat16 else (1, 5, 8, 64)
            for M in Ms:
                for s in range(6 if dtype == torch.bfloat16 else 2):
                    a, b = mk_pair(M * H, 1000 * M + 17 * s + (0 if dtype == torch.bfloat16 else 7),
                                   dtype)
                    x = (a if rank == 0 else b).view(M, H).contiguous()
                    ar = pc.all_reduce(x)
                    os_ = A.one_shot(pc, x)
                    ref = (a.float() + b.float()).to(dtype).view(M, H)
                    torch.cuda.synchronize()
                    nan = torch.isnan(ref)
                    d1 = (bits(ar) != bits(os_)) & ~nan.view(-1)
                    d2 = (bits(ar) != bits(ref)) & ~nan.view(-1)
                    bad += int(d1.sum()) + int(d2.sum())
                    tot += x.numel()
                    cases += 1
        out.append(("A", bad == 0, f"rank {rank}: {cases} tensors, {tot} elements: NCCL all_reduce vs one_shot vs "
                                   f"bf16(a+b) reference: {bad} differing (NaN positions excluded)"))

        # ---- B: the installed wrapper
        os.environ["GLM53_DEC_AR1SHOT"] = "1"
        rep = A.install_now()
        ok_b = rep["mode"] == "on"
        msgs = [f"install {rep}"]
        for M in (1, 5, 8, 64):
            a, b = mk_pair(M * H, 77 + M)
            x = (a if rank == 0 else b).view(M, H).contiguous()
            y = tensor_model_parallel_all_reduce(x)
            A.ST.mode = "off"
            yp = tensor_model_parallel_all_reduce(x)
            A.ST.mode = "on"
            e = torch.equal(bits(y), bits(yp))
            ok_b &= e
            msgs.append(f"eager M={M} equal={e}")
        big = torch.randn(256, H, device="cuda").to(torch.bfloat16)  # 2 MiB > 512 KiB: production path
        c0 = dict(A.COUNTERS)
        tensor_model_parallel_all_reduce(big)
        took_prod = A.COUNTERS["eager"] == c0["eager"]
        ok_b &= took_prod
        msgs.append(f"2 MiB -> production path {took_prod}")
        # graph capture / replay
        M = 5
        xs = torch.zeros(M, H, dtype=torch.bfloat16, device="cuda")
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                ys = tensor_model_parallel_all_reduce(xs * 1)
        torch.cuda.current_stream().wait_stream(s)
        gph = torch.cuda.CUDAGraph()
        cap0 = A.COUNTERS["captured"]
        with torch.cuda.graph(gph):
            ys = tensor_model_parallel_all_reduce(xs * 1)
            ys2 = tensor_model_parallel_all_reduce(ys * 1)
        ncap = A.COUNTERS["captured"] - cap0
        ok_g = ncap == 2
        for it in range(8):
            a, b = mk_pair(M * H, 5000 + it)
            xs.copy_((a if rank == 0 else b).view(M, H))
            gph.replay()
            torch.cuda.synchronize()
            A.ST.mode = "off"
            r1 = tensor_model_parallel_all_reduce(xs * 1)
            r2 = tensor_model_parallel_all_reduce(r1 * 1)
            A.ST.mode = "on"
            ok_g &= torch.equal(bits(ys), bits(r1)) and torch.equal(bits(ys2), bits(r2))
        ok_b &= ok_g
        msgs.append(f"graph: {ncap} captured one-shot calls, 8 replays with new inputs equal={ok_g}")
        out.append(("B", ok_b, f"rank {rank}: " + "; ".join(msgs) + f"; counters {A.summary()}"))

        # ---- C: verify mode
        A.ST.mode = "verify"
        A.ST.mism = None
        for M in (1, 5, 8, 64):
            for it in range(4):
                a, b = mk_pair(M * H, 9000 + 10 * M + it)
                x = (a if rank == 0 else b).view(M, H).contiguous()
                y = tensor_model_parallel_all_reduce(x)
                A.ST.mode = "off"
                yp = tensor_model_parallel_all_reduce(x)
                A.ST.mode = "verify"
                ok_v = torch.equal(bits(y), bits(yp))
        m = A.ST.mism.tolist()
        out.append(("C", m[0] == 0 and m[1] > 0 and ok_v,
                    f"rank {rank}: verify mode: {m[0]} differing of {m[1]} compared elements, served == production"))
        A.ST.mode = "on"

        # ---- D: latency
        if bench:
            res = []
            for M in (1, 3, 5, 6, 8, 16, 64):
                x = torch.randn(M, H, device="cuda").to(torch.bfloat16)
                rounds = {"ar": [], "os": [], "ar_g": [], "os_g": []}
                graphs = {}
                for kind in ("ar", "os"):
                    xs = x.clone()
                    s = torch.cuda.Stream()
                    s.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(s):
                        for _ in range(3):
                            y = pc.all_reduce(xs) if kind == "ar" else A.one_shot(pc, xs)
                    torch.cuda.current_stream().wait_stream(s)
                    gph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(gph):
                        y = xs
                        for _ in range(50):
                            y = pc.all_reduce(y) if kind == "ar" else A.one_shot(pc, y)
                            y = y * 0.5
                    graphs[kind] = gph
                for r in range(9):
                    for kind in (("ar", "os") if r % 2 == 0 else ("os", "ar")):
                        tp.barrier()
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        y = x
                        for _ in range(100):
                            y = pc.all_reduce(x) if kind == "ar" else A.one_shot(pc, x)
                        torch.cuda.synchronize()
                        rounds[kind].append((time.perf_counter() - t0) / 100 * 1e6)
                        tp.barrier()
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        graphs[kind].replay()
                        torch.cuda.synchronize()
                        rounds[kind + "_g"].append((time.perf_counter() - t0) / 50 * 1e6)
                med = {k: statistics.median(v) for k, v in rounds.items()}
                res.append((M, med))
            out.append(("D", True, res))
        q.put((rank, out, None))
    except Exception:  # noqa: BLE001
        import traceback
        q.put((rank, out, traceback.format_exc()))


def main():
    import socket
    import torch.multiprocessing as mp
    bench = len(sys.argv) > 1 and sys.argv[1] == "bench"
    with socket.socket() as s_:
        s_.bind(("127.0.0.1", 0))
        port = s_.getsockname()[1]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=worker, args=(r, port, q, bench)) for r in range(2)]
    for p in ps:
        p.start()
    res = [q.get(timeout=1800) for _ in ps]
    for p in ps:
        p.join(60)
    allok = True
    for rank, out, err in sorted(res, key=lambda t: t[0]):
        for item in out:
            if item[0] == "info":
                print(item[1])
            elif item[0] == "D":
                print(f"D rank {rank} latency, loopback sockets (NOT RoCE), us per collective (median of 9 rounds):")
                print("   M  bytes   AR eager  1shot eager   AR graph  1shot graph(+mul)")
                for M, med in item[2]:
                    print(f"  {M:2d} {M * H * 2:6d} {med['ar']:9.1f} {med['os']:11.1f} {med['ar_g']:10.1f} "
                          f"{med['os_g']:11.1f}")
            else:
                allok &= bool(item[1])
                print(f"{item[0]} {'OK ' if item[1] else 'FAIL'} {item[2]}")
        if err:
            allok = False
            print(f"rank {rank} raised:\n{err}")
    print("ALL OK" if allok else "FAILED")


if __name__ == "__main__":
    main()
