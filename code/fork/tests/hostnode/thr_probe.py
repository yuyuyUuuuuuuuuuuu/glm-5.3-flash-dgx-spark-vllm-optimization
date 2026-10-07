import ctypes, os, torch
torch.cuda.init(); torch.zeros(1, device="cuda")
def lib(name):
    for line in open("/proc/self/maps"):
        p = line.split()[-1]
        if name in p: return p
cudart = ctypes.CDLL(lib("libcudart.so"))
cb = ctypes.CDLL(_cb_lib())
cudart.cudaLaunchHostFunc.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
slot = (ctypes.c_long * 4)()
s = torch.cuda.Stream()
before = {t: open(f"/proc/self/task/{t}/comm").read().strip() for t in os.listdir("/proc/self/task")}
print("threads before host func:", before)
cudart.cudaLaunchHostFunc(ctypes.c_void_p(s.cuda_stream), ctypes.cast(cb.glm53_cb, ctypes.c_void_p), ctypes.cast(slot, ctypes.c_void_p))
s.synchronize()
after = {t: open(f"/proc/self/task/{t}/comm").read().strip() for t in os.listdir("/proc/self/task")}
print("threads after:", after)
print("callback ran on tid", slot[2], "comm", after.get(str(slot[2])), "cpu", slot[3])
for irq in ("487", "489"):
    try: print(irq, open(f"/proc/irq/{irq}/effective_affinity_list").read().strip())
    except Exception as e: print(irq, repr(e))
print([l[:60] for l in open("/proc/interrupts") if "nvidia" in l][:3])
