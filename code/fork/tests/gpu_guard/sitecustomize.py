# GPU guard for the test containers (tests/gpu_run.sh, tests/handoff/run.sh, tests/fkda/gpu_run.sh mount this directory at
# /opt/gpuguard and put it on PYTHONPATH). GB10 memory is unified with the host: a test that over-allocates can hang the
# whole machine instead of failing. This caps every process's PyTorch CUDA allocations at GPU_MEM_CAP_GB (default 40;
# 0 disables) so an oversized test fails with a normal CUDA OOM.
# The cap is set when the process first initializes CUDA, not when it imports torch: setting it needs a CUDA context,
# and creating one at import would break the tests that assert a plugin load creates none (tests/test_hostloop_plugin.py).
import os, sys, importlib.abc, importlib.util
_CAP = float(os.environ.get("GPU_MEM_CAP_GB", "40") or 0)
def _set_cap(torch):
    try:
        tot = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, _CAP * 2**30 / tot), 0)
        if os.environ.get("GPU_GUARD_VERBOSE"): print(f"[gpu-guard] torch CUDA cap {_CAP:.0f} GiB", file=sys.stderr)
    except Exception as e:  # never break the test because of the guard
        print(f"[gpu-guard] could not set cap: {e!r}", file=sys.stderr)
def _apply(torch):
    # torch's C++ side initializes CUDA through the module attribute torch.cuda._lazy_init (looked up at every call), so
    # wrapping that attribute sees the first initialization by a tensor, a stream or a graph; the cap is set right after
    try:
        tc = torch.cuda
        if not tc.is_available(): return
        orig = tc._lazy_init
        done = []
        def _lazy_init_capped():
            orig()
            if not done:
                done.append(1); _set_cap(torch)
        tc._lazy_init = _lazy_init_capped
    except Exception as e:
        print(f"[gpu-guard] could not install the cap: {e!r}", file=sys.stderr)
class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != "torch": return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec("torch")
        if spec and spec.loader:
            orig = spec.loader.exec_module
            def exec_module(m, _o=orig):
                _o(m); _apply(m)
            spec.loader.exec_module = exec_module
        return spec
if _CAP > 0: sys.meta_path.insert(0, _Finder())
# chain the image's own sitecustomize (Ubuntu's /usr/lib/python3.12/sitecustomize.py), which PYTHONPATH shadows
_sys_sc = "/usr/lib/python3.12/sitecustomize.py"
if os.path.exists(_sys_sc):
    try: exec(compile(open(_sys_sc).read(), _sys_sc, "exec"), {"__name__": "_sys_sitecustomize"})
    except Exception: pass
