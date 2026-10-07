"""Per-kernel floor inside a CUDA graph on this GPU: N back-to-back tiny kernels (torch.cuda._sleep(1)), and device
properties relevant to the L2-prefetch idea (docs/DEC_SMALLOPS.md)."""
import torch
p = torch.cuda.get_device_properties(0)
print(p.name, "SMs", p.multi_processor_count, "L2", getattr(p, "L2_cache_size", None), "persist max",
      getattr(p, "persisting_l2_cache_max_size", None))
N = 89
g = torch.cuda.CUDAGraph()
x = torch.zeros(1, device="cuda")
with torch.cuda.graph(g):
    for _ in range(N):
        torch.cuda._sleep(1)
for _ in range(3):
    g.replay()
torch.cuda.synchronize()
ts = []
for r in range(11):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(20):
        g.replay()
    e.record(); torch.cuda.synchronize()
    ts.append(s.elapsed_time(e) * 1000 / (20 * N))
ts.sort()
print(f"graph node floor (sleep kernel): median {ts[5]:.2f} us/kernel, min {ts[0]:.2f} max {ts[-1]:.2f}")
