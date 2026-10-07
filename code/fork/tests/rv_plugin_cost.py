"""What integrate.plugin_register() costs in a process that only loads vLLM's general plugins (API server, engine core,
registry subprocess): CUDA primary context created?, host RSS, MemAvailable delta, wall time. Mode from argv[1]:
none | gemv | gemv+large | large."""
import ctypes
import os
import sys
import time

mode = sys.argv[1]
for v in ("GLM53_FP8_GEMV", "GLM53_FP8_LARGE_M", "TF_EXL3_MOE", "GLM53_BF16_GEMV"):
    os.environ.pop(v, None)
if "gemv" in mode or "prod" in mode:
    os.environ["GLM53_FP8_GEMV"] = "1"
if "prod" in mode:
    os.environ["TF_EXL3_MOE"] = "1"
if "large" in mode:
    os.environ["GLM53_FP8_LARGE_M"] = "1"
sys.path.insert(0, "/w")


def rss():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS"):
            return int(line.split()[1]) // 1024
    return -1


def avail():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) // 1024
    return -1


def ctx_active():
    cu = ctypes.CDLL("libcuda.so.1")
    cu.cuInit(0)
    dev = ctypes.c_int()
    cu.cuDeviceGet(ctypes.byref(dev), 0)
    flags, active = ctypes.c_uint(), ctypes.c_int()
    cu.cuDevicePrimaryCtxGetState(dev, ctypes.byref(flags), ctypes.byref(active))
    return bool(active.value)


import torch  # noqa: E402
import vllm  # noqa: E402,F401
a0, r0 = avail(), rss()
t0 = time.time()
import integrate  # noqa: E402
integrate.plugin_register()
dt = time.time() - t0
time.sleep(1.0)
a1, r1 = avail(), rss()
print(f"RESULT mode={mode} ctx_active={ctx_active()} torch.cuda.is_initialized={torch.cuda.is_initialized()} "
      f"tilelang_imported={'tilelang' in sys.modules} rss +{r1 - r0} MiB (now {r1}) MemAvailable delta {a0 - a1} MiB "
      f"plugin_register {dt:.2f} s", flush=True)
