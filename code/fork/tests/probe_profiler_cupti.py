"""[dec-hostloop] does torch.profiler (kineto/CUPTI) record CUDA kernels in the situations a vLLM worker is in?
  S1 one session, eager kernels + CUDA-graph replays;  S2 a second session in the same process;
  S3 a side thread blocked in cudaEventSynchronize / stream sync during the session (MRv2 async-output thread);
  S4 session started from a SIGUSR2-armed hook (glm53_runtime's own _prof_step, fake Worker);
  S5 glm53_runtime twice (re-arm)."""
import glob
import gzip
import json
import os
import signal
import sys
import tempfile
import threading
import time
import types

sys.path.insert(0, "/w")
import torch
from torch.profiler import ProfilerActivity, profile

print("torch", torch.__version__, "cuda", torch.version.cuda, "kineto", torch.autograd.kineto_available(),
      "activities", torch.profiler.supported_activities())
x = torch.randn(1024, 1024, device="cuda")
g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    y = x @ x
torch.cuda.current_stream().wait_stream(s)
with torch.cuda.graph(g):
    y = x @ x
    y = y.relu()


def work(n=5):
    for _ in range(n):
        g.replay()
        z = (x * 2).sum()
    torch.cuda.synchronize()


def count(prof):
    ev = prof.events()
    k = sum(1 for e in ev if getattr(e, "device_type", None) == torch.autograd.DeviceType.CUDA)
    return k, len(ev)


def session(tag):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        work()
    k, n = count(p)
    print(f"{tag}: cuda events {k} / all {n}")
    return k


r = {}
r["S1"] = session("S1 first session")
r["S2"] = session("S2 second session")
stop = [False]


def side():
    ev = torch.cuda.Event()
    while not stop[0]:
        ev.record()
        ev.synchronize()
        time.sleep(0.0005)


t = threading.Thread(target=side, daemon=True)
t.start()
r["S3"] = session("S3 with a side thread syncing")
stop[0] = True
t.join()

# S4/S5: glm53_runtime's own SIGUSR2 path on a fake Worker
d = tempfile.mkdtemp()
os.environ["GLM53_TF_PROFILE"] = d
os.environ["GLM53_TF_PROFILE_STEPS"] = "5"
import glm53_runtime as GR
mod = types.ModuleType("fake_gpu_worker")


class Worker:
    def load_model(self):
        pass

    def compile_or_warm_up_model(self):
        pass

    def execute_model(self, *a):
        work(1)


mod.Worker = Worker
GR._patch(mod)
w = Worker()
w.compile_or_warm_up_model()
for rnd in ("S4", "S5"):
    before = set(glob.glob(d + "/*.json.gz"))
    os.kill(os.getpid(), signal.SIGUSR2)
    for _ in range(8):
        w.execute_model()
    new = sorted(set(glob.glob(d + "/*.json.gz")) - before)
    k = 0
    if new:
        tr = json.load(gzip.open(new[-1]))
        k = sum(1 for e in tr["traceEvents"] if e.get("cat") == "kernel")
    print(f"{rnd} glm53_runtime SIGUSR2 trace {new[-1:] and os.path.basename(new[-1])}: kernel events {k}")
    r[rnd] = k
print("RESULT", r)
