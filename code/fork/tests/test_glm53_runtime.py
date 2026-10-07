"""glm53_runtime: import-hook patching, memory hygiene and the SIGUSR2-armed profiler (with CUDA-graph replays).
Part 1 uses a fake vllm.v1.worker.gpu_worker (PYTHONPATH shadow) so the Worker methods can be exercised; part 2 checks
that the real image module gets patched by the import hook. Run: tests/gpu_run.sh python3 tests/test_glm53_runtime.py"""
import os, sys, subprocess, tempfile, textwrap, glob
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAKE = textwrap.dedent('''
    import torch
    class Worker:
        def load_model(self):
            keep = []
            for _ in range(64):                                  # 64 x 8 MiB pinned blocks, freed -> host cache
                t = torch.empty(8 << 20, dtype=torch.uint8, pin_memory=True); t.fill_(1); keep.append(t)
            del keep
            b = [bytearray(1 << 20) for _ in range(512)]          # 512 MiB of small heap objects, freed
            del b
        def compile_or_warm_up_model(self):
            self.x = torch.randn(512, 512, device="cuda"); self.w = torch.randn(512, 512, device="cuda")
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3): self.y = self.x @ self.w
            torch.cuda.current_stream().wait_stream(s)
            self.g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.g): self.y = self.x @ self.w
        def execute_model(self):
            self.g.replay(); return self.y
''')
CHILD = textwrap.dedent('''
    import os, sys, signal, logging, glob
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s", stream=sys.stdout)
    import glm53_runtime as R
    R.install()
    import vllm.v1.worker.gpu_worker as gw
    assert R._STATE["patched"], "not patched"
    w = gw.Worker(); w.load_model(); w.compile_or_warm_up_model()
    for _ in range(3): w.execute_model()
    os.kill(os.getpid(), signal.SIGUSR2)
    for _ in range(14): w.execute_model()
    import torch; torch.cuda.synchronize()
    outs = glob.glob(os.environ["GLM53_TF_PROFILE"] + "/*")
    print("OUTPUTS", sorted(os.path.basename(o).split(".", 1)[1] for o in outs))
    txt = [o for o in outs if o.endswith(".txt")]
    body = open(txt[0]).read() if txt else ""
    print("GEMM_IN_TABLE", any(k in body.lower() for k in ("gemm", "sgemm", "cutlass", "ampere", "kernel")))
''')
def run(env, code, pypath):
    e = dict(os.environ, **env, PYTHONPATH=pypath)
    return subprocess.run([sys.executable, "-c", code], env=e, capture_output=True, text=True, timeout=600)
ok = True
with tempfile.TemporaryDirectory() as d:
    pk = os.path.join(d, "vllm", "v1", "worker"); os.makedirs(pk)
    for p in (os.path.join(d, "vllm"), os.path.join(d, "vllm", "v1"), pk): open(os.path.join(p, "__init__.py"), "w").close()
    open(os.path.join(pk, "gpu_worker.py"), "w").write(FAKE)
    prof = os.path.join(d, "prof")
    r = run({"GLM53_MEM_HYGIENE": "1", "GLM53_TF_PROFILE": prof, "GLM53_TF_PROFILE_STEPS": "10"}, CHILD, f"{d}:{REPO}")
    out = r.stdout + r.stderr
    print("\n".join(l for l in out.splitlines() if "glm53" in l or l.startswith(("OUTPUTS", "GEMM", "Traceback", "AssertionError")) or "Error" in l)[-4000:])
    ok &= r.returncode == 0 and "OUTPUTS ['json.gz', 'txt']" in out and "GEMM_IN_TABLE True" in out and "hygiene after load_model" in out
# part 2: the real module is patched by the import hook; unset env -> nothing patched
REAL = 'import logging,sys; logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s"); import glm53_runtime as R; R.install(); import vllm.v1.worker.gpu_worker as gw; print("REAL_PATCHED", R._STATE["patched"], gw.Worker.load_model.__name__, gw.Worker.execute_model.__qualname__)'
r2 = run({"GLM53_MEM_HYGIENE": "1", "GLM53_TF_PROFILE": "/tmp/x"}, REAL, REPO)
print([l for l in (r2.stdout + r2.stderr).splitlines() if "REAL_PATCHED" in l or "Traceback" in l][-3:])
ok &= "REAL_PATCHED True" in r2.stdout and "_patch.<locals>.execute_model" in r2.stdout
r3 = run({"GLM53_MEM_HYGIENE": "", "GLM53_TF_PROFILE": ""}, REAL, REPO)
print([l for l in r3.stdout.splitlines() if "REAL_PATCHED" in l][-1:])
ok &= "REAL_PATCHED False" in r3.stdout and "Worker.execute_model" in r3.stdout
print("ALL PASSED" if ok else "FAILED"); sys.exit(0 if ok else 1)
