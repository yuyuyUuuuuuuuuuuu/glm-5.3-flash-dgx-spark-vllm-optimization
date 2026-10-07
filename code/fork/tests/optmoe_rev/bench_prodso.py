"""opt-moe-rev: production's path (fp32 accumulator, per-pair gather, .to(bf16)) timed with whichever extension
OPTMOE_EXT points at (unset = overlay/ = this branch's build); saves the bf16 outputs to OUT_PT for a cross-build
comparison. Guards against a slowdown/drift of the DEFAULT path from the template changes."""
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "moee4m3"))
sys.path.insert(0, os.path.join(HERE, "..", "optmoe"))
import torch  # noqa: E402

import prefill_cap_common as C  # noqa: E402
import harness as H  # noqa: E402
from real_layer import make_real_layer  # noqa: E402

H.gpu_guard(8.0)
prod = H.load_prod()
H.load_xl()
import _ext as OX  # noqa: E402
OX.preload()
import glm53_moe_e4m3 as M  # noqa: E402
import glm53_prefill_cap as PC  # noqa: E402

dev = torch.device("cuda", 0)
L = make_real_layer(prod, dev)
emap = prod.pin_exl3_expert_map(L, dev)
assert PC.install(prodmod=prod, n=1)["installed"]
outs = {}
for T in (13824, 4289):
    g = torch.Generator().manual_seed(1000 + T)
    x = torch.randn(T, 4096, generator=g).to(torch.bfloat16).to(dev)
    ids = C.routing("real", T, 1000 + T, dev)
    w = C.weights_for(T, 1000 + T, dev).float()
    f = lambda: M.run(prod, x, ids, w, L, C.LIMIT, expert_map=emap, sched={"acc": "f32", "tg": False}).to(torch.bfloat16)  # noqa: E731
    outs[T] = f().cpu()
    for v in (16,):
        outs[(T, v)] = M.run(prod, x, ids, w, L, C.LIMIT, expert_map=emap, sched={"acc": "f32", "tg": False, "variant": v}).to(torch.bfloat16).cpu()
    f()
    ts = []
    for _ in range(21):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); f(); e.record(); e.synchronize()
        ts.append(s.elapsed_time(e))
    print(f"[T={T}] prod path median {statistics.median(ts):.2f} ms (min {min(ts):.2f})", flush=True)
torch.save(outs, os.environ.get("OUT_PT", "/w/tests/optmoe_rev/prodso.pt"))
print("saved", flush=True)
