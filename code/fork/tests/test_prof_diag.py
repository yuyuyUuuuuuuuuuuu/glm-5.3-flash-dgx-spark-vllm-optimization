"""[dec-hostloop] GLM53_DEC_PROF_DIAG on nodeC (production image), glm53_runtime's own SIGUSR2 path on a fake Worker:
  D1 diag on: boot preflight logs 'CUDA activity recorded'; a SIGUSR2 session writes rank<r>-<pid>-<time>-s<k> with
     kernel events and logs the device-event count; torch.distributed rank is used when initialized (gloo, world 1);
  D2 a session that records no CUDA activity (simulated: CUDA activity removed from the request) -> WARNING with the
     context, ONE automatic re-armed session, then no further re-arm;
  D3 diag off: the original behaviour (no preflight, 'rankx'-style name without -s<k>).
Run: tests/hostloop_gpu.sh python3 tests/test_prof_diag.py
"""
import glob
import gzip
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile

CHILD = r'''
import glob, gzip, json, logging, os, signal, sys, types, tempfile
sys.path.insert(0, "/w")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
import torch
mode = os.environ["CHILD_MODE"]
if mode != "off":
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29533", rank=0, world_size=1)
import glm53_runtime as GR
if mode == "nocuda":
    import torch.profiler as TP
    real = TP.profile
    def fake(*a, activities=None, **k):
        return real(*a, activities=[x for x in (activities or []) if x != TP.ProfilerActivity.CUDA], **k)
    TP.profile = fake
mod = types.ModuleType("fake_gpu_worker")
x = torch.randn(512, 512, device="cuda")
class Worker:
    def load_model(self): pass
    def compile_or_warm_up_model(self): pass
    def execute_model(self, *a):
        (x @ x).relu_()
mod.Worker = Worker
GR._patch(mod)
w = Worker()
w.compile_or_warm_up_model()
os.kill(os.getpid(), signal.SIGUSR2)
for _ in range(40):
    w.execute_model()
torch.cuda.synchronize()
out = []
for f in sorted(glob.glob(os.environ["GLM53_TF_PROFILE"] + "/*.json.gz")):
    tr = json.load(gzip.open(f))
    out.append((os.path.basename(f), sum(1 for e in tr["traceEvents"] if e.get("cat") == "kernel")))
print("RESULT", out, GR._DIAG["sessions"], GR._DIAG["retried"])
'''


def run(mode, diag):
    d = tempfile.mkdtemp()
    env = {k: v for k, v in os.environ.items() if not k.startswith("GLM53_")}
    env.update(CHILD_MODE=mode, GLM53_TF_PROFILE=d, GLM53_TF_PROFILE_STEPS="5")
    if diag:
        env["GLM53_DEC_PROF_DIAG"] = "1"
    p = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True, timeout=900)
    res = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT")]
    logs = [ln for ln in (p.stdout + p.stderr).splitlines() if "[glm53-prof]" in ln]
    if p.returncode or not res:
        print(p.stdout[-2000:], p.stderr[-4000:])
        raise SystemExit(f"child {mode} failed rc={p.returncode}")
    files, sessions, retried = eval(res[-1][len("RESULT "):].rsplit(" ", 2)[0]), None, None
    tail = res[-1].rsplit(" ", 2)
    return files, int(tail[1]), tail[2] == "True", logs


fails = []
files, sessions, retried, logs = run("ok", True)
print("D1 files", files, "sessions", sessions)
for ln in logs:
    print("   ", ln[:260])
if not (any("preflight: CUDA activity recorded" in ln for ln in logs) and len(files) == 1
        and files[0][0].startswith("rank0-") and "-s2" in files[0][0] and files[0][1] > 0
        and any("device events recorded" in ln for ln in logs)):
    fails.append("D1")
files, sessions, retried, logs = run("nocuda", True)
print("D2 files", files, "sessions", sessions, "retried", retried)
for ln in logs:
    print("   ", ln[:260])
if not (len(files) == 2 and all(k == 0 for _, k in files) and retried
        and sum("recorded NO CUDA activity" in ln for ln in logs) == 2
        and sum("re-arming one more session" in ln for ln in logs) == 1
        and any("preflight: NO CUDA activity" in ln for ln in logs)):
    fails.append("D2")
files, sessions, retried, logs = run("off", False)
print("D3 files", files, "sessions", sessions)
if not (len(files) == 1 and files[0][0].startswith("rankx-") and "-s" not in files[0][0] and files[0][1] > 0
        and not any("preflight" in ln for ln in logs) and sessions == 0):
    fails.append("D3")
print("RESULT:", "PASS" if not fails else f"FAIL {fails}")
sys.exit(1 if fails else 0)
