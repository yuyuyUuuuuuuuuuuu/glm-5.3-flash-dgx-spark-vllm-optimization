"""DEC_DLMH review test L1: the stats / PROOF logger (the wrapped DFlash2Speculator.propose).

Before the review fix the counters were snapshotted only AFTER each print, so the first stats line (step 64) always
read zeros and no line showed that graph replays had served the two-stage head (the A/B PROOF was a boot line).
  L1a the step-64 line shows the counters as of step 32 (non-zero once served)
  L1b 'serving confirmed (mode on)' appears once, only when graph replays (device count - eager count) > 0 and a graph
      was captured with the two-stage head; never for eager-only service
  L1c the line after the period shows counters at most SNAP_LEAD steps old

  tests/gpu_run.sh python3 tests/dlmh/test_dlmh_logger.py
"""
from __future__ import annotations

import logging
import sys
import types

import torch

sys.path.insert(0, "/w")
import glm53_dlmh as D  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"[dlmh-logger] {name}: {'OK' if ok else 'FAIL'} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


class _Cap(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, rec):
        self.lines.append(rec.getMessage())


cap = _Cap()
D._log.addHandler(cap)
D._log.setLevel(logging.INFO)


def fresh_module():
    """a stub vllm speculator module whose propose 'serves' one call per step (device counter += 1)."""
    S = types.ModuleType("speculator")

    class DFlash2Speculator:
        def propose(self, served=True):
            if served:
                D.HEAD.counters[3] += 1
            return None

    S.DFlash2Speculator = DFlash2Speculator
    for name in ("vllm", "vllm.v1", "vllm.v1.worker", "vllm.v1.worker.gpu", "vllm.v1.worker.gpu.spec_decode",
                 "vllm.v1.worker.gpu.spec_decode.dflash2"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["vllm.v1.worker.gpu.spec_decode.dflash2.speculator"] = S
    sys.modules["vllm.v1.worker.gpu.spec_decode.dflash2"].speculator = S
    return S


def reset(mode="on", log_every=2000, captured=1, eager=0):
    D.HEAD.__init__()
    D.HEAD.ready = True
    D.HEAD.counters = torch.zeros(4, dtype=torch.int64, device="cuda")
    D.HEAD.pinned = torch.zeros(4, dtype=torch.int64, pin_memory=True)
    D.CFG.mode, D.CFG.log_every = mode, log_every
    D.COUNTERS.update(captured=captured, eager=eager)
    cap.lines.clear()


# graph service: every step one replay (device +1, no host count)
S = fresh_module()
D._install_logger()
sp = S.DFlash2Speculator()
reset()
for _ in range(2100):
    sp.propose()
    torch.cuda.synchronize()
stats = [ln for ln in cap.lines if "served graph-calls" in ln]
conf = [ln for ln in cap.lines if "serving confirmed (mode on)" in ln]
check("L1a step-64 line shows the step-32 counters", len(stats) >= 1 and "steps 64, served graph-calls 32;" in stats[0]
      and "as of step 32" in stats[0], f"{stats[:1]}")
check("L1b serving confirmed once with graph replays", len(conf) == 1 and "graph replays 32" in conf[0], f"{conf}")
check("L1c the step-2000 line is at most 32 steps old", len(stats) >= 2 and "steps 2000, served graph-calls 1968;" in
      stats[1], f"{stats[1:2]}")

# eager-only service (device +1 per step AND host eager +1): never 'serving confirmed'
reset(captured=0)


def eager_propose(self, served=True):
    D.HEAD.counters[3] += 1
    D.COUNTERS["eager"] += 1


# a fresh stub class (install_logger skips an already wrapped propose)
S2 = fresh_module()
S2.DFlash2Speculator.propose = eager_propose
D._install_logger()
sp2 = S2.DFlash2Speculator()
for _ in range(200):
    sp2.propose()
    torch.cuda.synchronize()
conf = [ln for ln in cap.lines if "serving confirmed" in ln]
check("L1b eager-only service never confirms", conf == [], f"{conf}")
# captured graphs exist but every call ran eagerly: still no confirmation
reset(captured=3)
for _ in range(200):
    sp2.propose()
    torch.cuda.synchronize()
conf = [ln for ln in cap.lines if "serving confirmed" in ln]
check("L1b captured but only eager calls never confirms", conf == [], f"{conf}")

print(f"[dlmh-logger] {'ALL OK' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}", flush=True)
sys.exit(1 if FAILS else 0)
