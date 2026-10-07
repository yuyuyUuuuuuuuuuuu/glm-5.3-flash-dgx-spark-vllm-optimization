"""Review: one small launch per variant for compute-sanitizer (memcheck / racecheck / synccheck)."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mla_prefill_common as C  # noqa: E402
import glm53_mla_prefill as M  # noqa: E402
ext = M.load_ext()
case = C.Case(60, 5000, "indep", seed=2)
valid = case.valid.clone()
valid[:8] = torch.tensor([0, 1, 31, 32, 33, 65, 0, 2047], dtype=torch.int32, device="cuda").clamp(max=case.valid[:8])
slots = case.slots.clone()
slots[torch.arange(C.TOPK, device="cuda")[None] >= valid[:, None]] = -1
out = torch.empty(60, 32, 512, dtype=torch.bfloat16, device="cuda")
for var in [int(v) for v in (sys.argv[1:] or ["4", "3", "2"])]:
    M.run(ext, case.q, case.cache, slots, valid, out, C.SM_SCALE, 1.0, variant=var)
    M.run(ext, case.q, case.cache, slots, valid, out, C.SM_SCALE, 0.0371, variant=var)
    torch.cuda.synchronize()
    print("variant", var, "ok", float(out.float().abs().mean()), flush=True)
