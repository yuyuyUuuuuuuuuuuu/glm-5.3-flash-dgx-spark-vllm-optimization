"""fused octet-max coarse kernel configs (cold: 2 copies), CUDA graphs, median us."""
import statistics as st, sys, torch
sys.path.insert(0, "/w")
import glm53_dlmh as D
D.parse_env({"GLM53_DEC_DLMH": "1"})
dev = "cuda"; N, K = 77440, 4096
cw = [torch.randint(-2**31, 2**31 - 1, (N, K // 8), device=dev, dtype=torch.int32) for _ in range(2)]
cs = [(torch.rand(N, K // 128, device=dev) * 0.01).half() for _ in range(2)]
for T in (7, 14):
    x = torch.randn(T, K, device=dev).to(torch.bfloat16)
    osc = torch.empty(T, N // 8, device=dev); yc = torch.empty(T, N, device=dev)
    res = {}
    cands = [(128, 8, 2), (128, 4, 2), (64, 8, 3), (64, 4, 3), (64, 8, 4), (32, 4, 4), (32, 8, 4)]
    for bw, nw, ns in cands:
        f = lambda i, bw=bw, nw=nw, ns=ns: D.coarse_octmax(x, cw[i], cs[i], osc, 128, bw, nw, ns)
        try:
            f(0); f(1); torch.cuda.synchronize()
        except Exception as e:
            print("skip", bw, nw, ns, str(e)[:80]); continue
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for i in range(4): f(i % 2)
        res[f"oct bw{bw} w{nw} s{ns}"] = g
    f = lambda i: D.coarse_gemv(x, cw[i], cs[i], yc, 128)
    f(0); f(1)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(4): f(i % 2)
    res["plain coarse (bn32 bw128)"] = g
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t = {k: [] for k in res}
    for r in range(9):
        for k, g in (list(res.items()) if r % 2 == 0 else list(res.items())[::-1]):
            g.replay(); torch.cuda.synchronize(); e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
            t[k].append(e0.elapsed_time(e1) * 250)
    for k, v in sorted(t.items(), key=lambda kv: st.median(kv[1])):
        print(f"T={T} {k:28s} {st.median(v):7.1f} us", flush=True)
    # topk options on the octet scores
    osc.normal_()
    tk = {}
    for name, fn in (("topk [T,9680] f32", lambda: torch.topk(osc, 128, dim=-1, sorted=False)),
                     ("topk [T,9680] f16", lambda: torch.topk(osc.half(), 128, dim=-1, sorted=False)),
                     ("topk pad [T,19360]", lambda: torch.topk(torch.nn.functional.pad(osc, (0, 9680), value=float("-inf")), 128, dim=-1, sorted=False)),
                     ("topk [T,77440] f32", lambda: torch.topk(yc, 128, dim=-1, sorted=False))):
        fn(); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for i in range(4): fn()
        tk[name] = g
    for k, g in tk.items():
        v = []
        for r in range(9):
            g.replay(); torch.cuda.synchronize(); e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
            v.append(e0.elapsed_time(e1) * 250)
        print(f"T={T} {k:28s} {st.median(v):7.1f} us", flush=True)
