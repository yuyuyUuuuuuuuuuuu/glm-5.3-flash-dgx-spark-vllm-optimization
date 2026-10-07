"""opt-moe-rev: one small TG + bf16-accumulator fused call per variant, for compute-sanitizer racecheck/memcheck."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "moee4m3"))
import torch  # noqa: E402

import prefill_cap_common as C  # noqa: E402
import harness as H  # noqa: E402
from real_layer import make_real_layer  # noqa: E402

prod = H.load_prod()
H.load_xl()
import glm53_moe_e4m3 as M  # noqa: E402
import glm53_prefill_cap as PC  # noqa: E402

dev = torch.device("cuda", 0)
L = make_real_layer(prod, dev)
emap = prod.pin_exl3_expert_map(L, dev)
assert PC.install(prodmod=prod, n=1)["installed"]
T = int(os.environ.get("RACE_T", "300"))
x = torch.randn(T, 4096).to(torch.bfloat16).to(dev)
ids = C.routing("collapsed", T, 5, dev)
w = C.weights_for(T, 5, dev).float()
for sch in ({"acc": "f32", "tg": True}, {"acc": "bf16", "tg": True}, {"acc": "bf16", "tg": True, "variant": 16}):
    S = torch.zeros(T, 4096, dtype=torch.bfloat16, device=dev)
    o = M.run(prod, x, ids, w, L, C.LIMIT, expert_map=emap, sched=sch, fold_into=S if sch["acc"] == "bf16" else None)
    torch.cuda.synchronize()
    print("done", sch, float(o.float().norm()), flush=True)
