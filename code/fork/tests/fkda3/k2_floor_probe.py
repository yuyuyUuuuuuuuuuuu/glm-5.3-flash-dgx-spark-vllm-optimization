#!/usr/bin/env python3
"""FKDA3 probe: K2's per-tile time with its inputs hot in L2 (small T, repeated) vs cold (T=13,824), to bound what a
K1/K2 L2-resident pipeline could reach. K2-only build _fk3b_C on a workspace the K1-only build _fk3a_C filled."""
import statistics, sys, torch
sys.path.insert(0, "/fkda/builds/k1only"); sys.path.insert(0, "/fkda/builds/k2only")
import _fk3a_C, _fk3b_C  # noqa
props = torch.cuda.get_device_properties(0)
print("L2 bytes", getattr(props, "L2_cache_size", "?"), "sms", props.multi_processor_count)
D, H = 128, 32
for T in (256, 512, 1024, 2048, 4608, 13824):
    g = torch.Generator(device="cpu").manual_seed(0)
    rn = lambda *s, sc=1.0: (torch.randn(*s, generator=g) * sc).cuda().to(torch.bfloat16)
    p = dict(q=rn(1, T, H, D), k=rn(1, T, H, D), v=rn(1, T, H, D), g=rn(1, T, H, D, sc=0.5), b=rn(1, T, H),
             s0=(torch.randn(1, H, D, D, generator=g) * 0.3).cuda(), A=(torch.randn(H, generator=g) * .2).cuda(),
             dtb=(torch.rand(H, D, generator=g) * 8 - 10).cuda(), cu=torch.tensor([0, T], dtype=torch.int32, device="cuda"))
    p["ws"] = torch.empty(int(torch.ops._fk3a_C.get_workspace_size(T, H, 1)), dtype=torch.uint8, device="cuda")
    p["out"] = torch.empty(1, T, H, D, dtype=torch.bfloat16, device="cuda"); p["fs"] = torch.empty(1, H, D, D, device="cuda")
    run = lambda ns: getattr(torch.ops, ns).fwd(p["q"], p["k"], p["v"], p["g"], p["b"], D ** -.5, p["out"], p["ws"], p["A"],
                                                 p["dtb"], -5.0, p["s0"], p["fs"], p["cu"], None, None)
    run("_fk3a_C"); torch.cuda.synchronize()
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    def t(fn, cold):
        ts = []
        for _ in range(15):
            if cold: flush.zero_()
            else: fn()
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
        return statistics.median(ts)
    k2h, k2c = t(lambda: run("_fk3b_C"), False), t(lambda: run("_fk3b_C"), True)
    k1h, k1c = t(lambda: run("_fk3a_C"), False), t(lambda: run("_fk3a_C"), True)
    tiles = T // 16
    print(f"T={T:6d} ws {p['ws'].numel()/2**20:7.1f} MiB | K2 hot {k2h:.3f} ms ({k2h*1e3/tiles:.2f} us/tile) cold {k2c:.3f} ms "
          f"({k2c*1e3/tiles:.2f} us/tile) | K1 hot {k1h:.3f} cold {k1c:.3f} ms", flush=True)
