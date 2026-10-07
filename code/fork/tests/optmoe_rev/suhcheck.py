# read-only: for every MoE layer, are all 288 experts' gate_proj.suh and up_proj.suh identical? (TOKGATHER qualification)
import json, os, struct, hashlib, collections
D = os.path.join(os.environ.get("TF_EXL3_MODELS") or os.path.expanduser("~/models"), "GLM-5.3-Flash-EXL3-TR3-4bpw/")
m = json.load(open(D + "model.safetensors.index.json"))["weight_map"]
hdr = {}
def rd(name):
    f = m[name]
    if f not in hdr:
        with open(D + f, "rb") as b:
            n = struct.unpack("<Q", b.read(8))[0]; hdr[f] = (json.loads(b.read(n)), 8 + n)
    h, off = hdr[f]; a, z = h[name]["data_offsets"]
    fd = os.open(D + f, os.O_RDONLY)
    try:
        data = os.pread(fd, z - a, off + a)
        os.posix_fadvise(fd, off + a, z - a, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return data, h[name]["dtype"], h[name]["shape"]
layers = sorted({int(k.split(".layers.")[1].split(".")[0]) for k in m if ".experts.0.gate_proj.suh" in k})
bad = []
for L in layers:
    p = f"model.language_model.layers.{L}.mlp.experts."
    g0, dt, sh = rd(p + "0.gate_proj.suh")
    neq_g = neq_u = 0
    for e in range(288):
        g = rd(p + f"{e}.gate_proj.suh")[0]; u = rd(p + f"{e}.up_proj.suh")[0]
        neq_g += g != g0; neq_u += u != g0
    print(L, dt, sh, "gate!=e0:", neq_g, "up!=e0gate:", neq_u, hashlib.md5(g0).hexdigest()[:8], flush=True)
    if neq_g or neq_u: bad.append(L)
print("layers", len(layers), "non-qualifying", bad)
