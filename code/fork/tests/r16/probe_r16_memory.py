"""deploy-r16: extra GPU memory per rank of the R16 features that no design doc sized from device-level data.

GB10 is unified memory: torch.cuda.mem_get_info()'s "free" is the system's and moves with other processes and page
cache, so every quantity is a DELTA inside this process, repeated (median of REPS), read two ways: (a) mem_get_info
used = total - free, (b) NVML's per-process used GPU memory of THIS pid (what nvidia-smi --query-compute-apps shows;
None when NVML is unavailable). A pure-PyTorch allocation of known size is the ruler (it must read back as its size).

  M.1 CUDA-graph memory of the decode-graph additions. GLM53_DEC_FP8ROOF adds per decode graph ~145 prefetch groups
      (side stream waits the main stream, tf_fp8_roof_ext.l2_prefetch, main stream waits the side stream; doc §6) and
      GLM53_DEC_MOEGLUE_WARM 42 groups (tf_exl3_moe_ext.l2_warm on its side stream). Measured: one graph of G main-stream
      ops with vs without G fork/kernel/join groups (G = 2000), KG = 10 graphs kept alive (as vLLM keeps its captured
      graphs), each shape in REPS fresh processes (the driver keeps a grown graph pool, so only a fresh context shows
      the growth) -> per-group cost; then 28 production-shaped graphs (14 capture sizes of production's start.sh x
      FULL + PIECEWISE, the upper bound; 187 main ops + 145 roof + 42 warm groups vs 187 main ops) -> total.
  M.2 kernel images: used memory after one launch of each new R16 kernel (glm53_mla_prefill_ext v4, tf_fp8_roof_ext,
      the moeglue l2_warm of the new tf_exl3_moe_ext, glm53_smallops_ext dconv) minus before (CUDA lazy loading: a
      module image is loaded at its first launch).
  M.3 persistent PyTorch allocations the features make (from the code; printed for the record): moeglue warm sink
      16 B per device; TF Scratch `inv` int32[P_cap = 1024] = 4 KiB (allocated whenever TF is on, flags off too);
      fp8roof / MLA / quickwins / hostloop / smallops dconv: none (hostloop's snapshot is a pinned HOST vector).
      [sidestream] R16_MEM_GRAPHS / R16_MEM_ROOF_GROUPS / R16_MEM_WARM_GROUPS set the production shape: production runs
      MAX_NUM_SEQS=8 = 23 capture sizes (the first R16 boot: "Capturing CUDA graphs (PIECEWISE): 0/23") -> 24 FULL decode
      graphs with 145 + 42 groups and 23 breakable PIECEWISE graphs with 100 + 42 (t1 34 / t4 11 are not forked inside
      a breakable segment); run once per shape.
Run: tests/r16/gpu.sh python3 -u tests/r16/probe_r16_memory.py
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import torch  # noqa: E402

MiB = 2 ** 20
REPS = int(os.environ.get("R16_MEM_REPS", "5"))
try:
    import pynvml
    pynvml.nvmlInit()
    _H = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception as exc:  # noqa: BLE001
    print(f"NVML unavailable ({type(exc).__name__}: {exc}); per-process column = None", flush=True)
    _H = None


def used() -> tuple[int, int | None]:
    torch.cuda.synchronize()
    f, t = torch.cuda.mem_get_info()
    mine = None
    if _H is not None:
        try:
            for p in pynvml.nvmlDeviceGetComputeRunningProcesses(_H):
                if p.pid == os.getpid() and p.usedGpuMemory is not None:
                    mine = int(p.usedGpuMemory)
        except Exception:  # noqa: BLE001
            mine = None
    return t - f, mine


def delta(a, b):
    return (b[0] - a[0], None if a[1] is None or b[1] is None else b[1] - a[1])


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def fmt(v, unit=MiB, nd=2):
    return "None" if v is None else f"{v / unit:.{nd}f}"


def main() -> int:
    dev = torch.device("cuda", 0)
    torch.ones(1, device=dev)
    child = len(sys.argv) > 1 and sys.argv[1] == "--child"
    r = []
    for _ in range(0 if child else REPS):
        torch.cuda.empty_cache()
        u0 = used()
        x = torch.empty(256 * MiB, dtype=torch.uint8, device=dev)
        x.fill_(1)
        r.append(delta(u0, used()))
        del x
    if not child:
        print(f"M.0 ruler: a 256 MiB tensor reads as {fmt(med([a for a, _ in r]))} MiB (mem_get_info) / "
              f"{fmt(med([b for _, b in r]))} MiB (NVML this pid); runs {[(fmt(a, nd=1), fmt(b, nd=1)) for a, b in r]}",
              flush=True)

    import tf_fp8_roof_ext as RX
    import tf_exl3_moe as tf
    E = tf.load_ext()
    w = torch.randint(-2 ** 31, 2 ** 31 - 1, (4 * MiB,), dtype=torch.int32, device=dev)   # 16 MiB "weight"
    regs = [w[: MiB // 4], w[MiB // 4: MiB]]
    sink = torch.zeros(4, dtype=torch.int32, device=dev)
    side = torch.cuda.Stream(device=dev)
    y = torch.zeros(1024, device=dev)

    def body(n_main: int, n_roof: int, n_warm: int) -> None:
        cur = torch.cuda.current_stream()
        for i in range(n_main):
            y.add_(1.0)                                    # stands for the layer's GEMV on the main stream
            for kind, n in (("roof", n_roof), ("warm", n_warm)):
                if i < n:
                    side.wait_stream(cur)                  # fork
                    with torch.cuda.stream(side):
                        if kind == "roof":
                            RX.l2_prefetch(w, (i % 64) * 65536, 65536, 8, 0)
                        else:
                            E.l2_warm(regs, 16, 8, sink)
                    cur.wait_stream(side)                  # join

    def warmup():
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            body(2, 2, 2)                                  # first launches outside capture (module images)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        # one fresh process = one fresh CUDA context and graph pool: capture n graphs of one shape, all kept alive
        # (as vLLM keeps its captured graphs), print the used-memory growth
        n, shape = int(sys.argv[2]), tuple(int(v) for v in sys.argv[3].split(","))
        warmup()
        u0 = used()
        keep = []
        for _ in range(n):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                body(*shape)
            g.replay()
            keep.append(g)
        d = delta(u0, used())
        print("CHILD", json.dumps({"n": n, "shape": shape, "get_info": d[0], "nvml": d[1]}), flush=True)
        return 0

    def children(n, shape):
        res = []
        for _ in range(REPS):
            p = subprocess.run([sys.executable, "-u", __file__, "--child", str(n), ",".join(map(str, shape))],
                               capture_output=True, text=True, timeout=900)
            ln = [x for x in p.stdout.splitlines() if x.startswith("CHILD ")]
            if not ln:
                print(f"child {n} {shape} failed rc={p.returncode}: {p.stderr[-800:]}", flush=True)
                continue
            r = json.loads(ln[0][6:])
            res.append((r["get_info"], r["nvml"]))
        return res

    G = int(os.environ.get("R16_MEM_GROUPS", "2000"))
    KG = int(os.environ.get("R16_MEM_KG", "10"))
    base = children(KG, (G, 0, 0))
    print(f"M.1 {KG} graphs of {G} main ops (fresh process each): {[(fmt(a), fmt(b)) for a, b in base]} MiB", flush=True)
    for kind, nr, nw in (("roof", G, 0), ("warm", 0, G)):
        on = children(KG, (G, nr, nw))
        pa = (med([x for x, _ in on]) - med([x for x, _ in base])) / (KG * G)
        nb = [y for _, y in on if y is not None]
        pb = (med(nb) - med([y for _, y in base])) / (KG * G) if nb and med([y for _, y in base]) is not None else None
        print(f"M.1 {kind}: {KG} graphs of {G} main ops + {G} {kind} groups: {[(fmt(a), fmt(b)) for a, b in on]} MiB -> "
              f"per fork/kernel/join group {fmt(pa, 1024, 3)} KiB (mem_get_info) / {fmt(pb, 1024, 3)} KiB (NVML)",
              flush=True)
    NG = int(os.environ.get("R16_MEM_GRAPHS", "28"))
    NR = int(os.environ.get("R16_MEM_ROOF_GROUPS", "145"))
    NW = int(os.environ.get("R16_MEM_WARM_GROUPS", "42"))
    pb0 = children(NG, (187, 0, 0))
    pon = children(NG, (187, NR, NW))
    da = med([x for x, _ in pon]) - med([x for x, _ in pb0])
    yb, yo = med([y for _, y in pb0]), med([y for _, y in pon])
    db = None if yb is None or yo is None else yo - yb
    print(f"M.1 production shape: {NG} decode graphs of 187 main ops: "
          f"{[(fmt(a), fmt(b)) for a, b in pb0]} MiB; with {NR} roof + {NW} warm groups each: "
          f"{[(fmt(a), fmt(b)) for a, b in pon]} MiB -> extra {fmt(da)} MiB (mem_get_info) / {fmt(db)} MiB (NVML)",
          flush=True)
    del w, regs, y
    torch.cuda.empty_cache()

    # ---- M.2 kernel images (first launch of each new kernel in a process that has not launched it yet)
    import glm53_mla_prefill as MP
    import glm53_smallops as SO
    import mla_prefill_common as C
    T = 256
    case = C.Case(T, 3000, "indep", seed=1)
    out = torch.empty(T, C.HEADS, C.D, dtype=torch.bfloat16, device=dev)
    H, gs, taps = 4096, 16, 2
    xb = (torch.randn(8, H, device=dev) * 0.5).bfloat16()
    coeff = (torch.randn(8, taps, H // gs, device=dev) * 0.5).bfloat16()
    base = (torch.randn(taps, H, device=dev) * 0.3).bfloat16()
    ext_mla = MP.load_ext()
    SO.load_ext()
    u0 = used()
    MP.run(ext_mla, case.q, case.cache, case.slots, case.valid, out, C.SM_SCALE, case.k_scale, variant=4)
    SO.dconv(xb, coeff, base, gs, 8)
    u1 = used()
    d = delta(u0, u1)
    print(f"M.2 kernel images of the MLA prefill v4 + smallops dconv first launches: {fmt(d[0])} MiB (mem_get_info) / "
          f"{fmt(d[1])} MiB (NVML); tf_fp8_roof l2_prefetch and tf_exl3_moe l2_warm were loaded in M.1 "
          f"(CUDA_MODULE_LOADING={os.environ.get('CUDA_MODULE_LOADING', 'default = LAZY')})", flush=True)
    so = {n: os.path.getsize(ROOT / f"{n}.cpython-312-aarch64-linux-gnu.so") for n in
          ("glm53_mla_prefill_ext", "glm53_smallops_ext", "tf_fp8_roof_ext", "tf_exl3_moe_ext")}
    print(f"M.2 AOT file sizes (upper bound of their device images): "
          f"{ {k: f'{v / MiB:.2f} MiB' for k, v in so.items()} }", flush=True)
    has_inv = "self.inv = torch.zeros((P_cap,), dtype=i32" in (ROOT / "tf_exl3_moe.py").read_text()
    print(f"M.3 persistent allocations: moeglue warm sink 16 B/device; TF Scratch 'inv' int32[1024] = 4 KiB "
          f"(present: {has_inv}); fp8roof / MLA / quickwins / hostloop / smallops dconv: 0", flush=True)
    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
