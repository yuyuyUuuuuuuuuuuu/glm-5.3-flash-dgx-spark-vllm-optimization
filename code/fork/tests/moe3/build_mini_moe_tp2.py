#!/usr/bin/env python3
"""Build the "handoff MoE mini, TP=2 rank-0 shapes" checkpoint (host side, numpy only, no GPU) from the handoff MoE mini
(branch moefq tests/moee4m3/build_mini_moe.py: 10 real GLM-5.3-Flash layers, layers 3..9 = 32 REAL layer-10 EXL3
experts at TP=1 shapes, local intermediate 2048). Production runs TP=2 (local intermediate 1024), which is the only
shape the MoE prefill kernels (GLM53_MOE_FUSED16 / GLM53_MOE_E4M3) serve, so this variant cuts every routed expert and
the shared expert to the rank-0 shard exactly like production's weight loader (overlay exl3.py shard_exl3_col /
shard_exl3_row, as tests/moee4m3/extract_layer.py): gate/up trellis dim 1 + svh column-parallel, down trellis dim 0 +
suh row-parallel; shared gate/up rows and shared down columns; config moe_intermediate_size 1024. At TP=1 the routed +
shared outputs are then rank 0's partial sums (no all-reduce): numerically not the real model, but a real engine on
production's shapes and code path - the wiring / A-B test bed for the MoE prefill kernels (tests/moe3/kl_driver.py).
Usage: build_mini_moe_tp2.py [src, default $TF_EXL3_MODELS/GLM-5.3-Flash-handoff-moe-mini]
                             [out, default <src>-tp2r0]
Deterministic: same source -> byte-identical output (MANIFEST.sha256 next to it; the -dflash2 dir is copied).
"""
import hashlib
import json
import os
import re
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("TF_EXL3_MODELS") or os.path.expanduser("~/models"), "GLM-5.3-Flash-handoff-moe-mini"))
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else str(SRC) + "-tp2r0")
DT = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32, "I16": np.int16, "I32": np.int32}
RX = re.compile(r"^model\.layers\.(\d+)\.mlp\.(experts\.\d+\.(gate_proj|up_proj|down_proj)\.(trellis|suh|svh|mcg)|"
                r"shared_experts\.(gate_proj|up_proj|down_proj)\.weight)$")


def half(a, dim):
    c = a.shape[dim] // 2
    return np.ascontiguousarray(np.take(a, np.arange(c), axis=dim))


def main():
    f = SRC / "model.safetensors"
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    base = 8 + n
    meta = hdr.pop("__metadata__", None)
    out = {}
    cut = 0
    for name in sorted(hdr):
        m = hdr[name]
        a0, a1 = m["data_offsets"]
        arr = np.fromfile(f, dtype=DT[m["dtype"]], count=(a1 - a0) // np.dtype(DT[m["dtype"]]).itemsize,
                          offset=base + a0).reshape(m["shape"])
        mm = RX.match(name)
        if mm:
            if mm.group(5):                                   # shared expert (BF16 dense)
                arr = half(arr, 1 if mm.group(5) == "down_proj" else 0)
            else:
                proj, suf = mm.group(3), mm.group(4)
                if proj in ("gate_proj", "up_proj"):
                    if suf == "trellis":
                        arr = half(arr, 1)                    # [K/16, N/16, 64] -> N/2
                    elif suf == "svh":
                        arr = half(arr, 0)
                else:
                    if suf == "trellis":
                        arr = half(arr, 0)                    # [N/16 (intermediate), K/16, 64] -> rows/2
                    elif suf == "suh":
                        arr = half(arr, 0)
            cut += 1
        out[name] = (m["dtype"], arr)
    OUT.mkdir(parents=True, exist_ok=True)
    h, off = {}, 0
    for name in sorted(out):
        dt, arr = out[name]
        nb = arr.nbytes
        h[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + nb]}
        off += nb
    if meta is not None:
        h["__metadata__"] = meta
    hb = json.dumps(h, separators=(",", ":"), sort_keys=True).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(OUT / "model.safetensors", "wb") as fo:
        fo.write(struct.pack("<Q", len(hb)))
        fo.write(hb)
        for name in sorted(out):
            fo.write(out[name][1].tobytes())
    c = json.load(open(SRC / "config.json"))
    assert int(c["moe_intermediate_size"]) == 2048
    c["moe_intermediate_size"] = 1024
    c["_handoff_moe_mini"]["tp2_rank0_shard"] = True
    c["_handoff_moe_mini"]["shared_expert_intermediate"] = 1024
    json.dump(c, open(OUT / "config.json", "w"), indent=1, sort_keys=True)
    for g in ("generation_config.json",):
        if (SRC / g).exists():
            shutil.copy2(SRC / g, OUT / g)
    D0, D1 = Path(str(SRC) + "-dflash2"), Path(str(OUT) + "-dflash2")
    D1.mkdir(exist_ok=True)
    for p in D0.iterdir():
        q = D1 / p.name
        if q.is_symlink() or q.exists():
            q.unlink()
        if p.is_symlink():
            q.symlink_to(os.readlink(p))
        elif p.name != "MANIFEST.sha256":
            shutil.copy2(p, q)
    for d in (OUT, D1):
        rows = []
        for fn in sorted(os.listdir(d)):
            if fn == "MANIFEST.sha256":
                continue
            hh = hashlib.sha256()
            p = d / fn
            with open(p.resolve() if p.is_symlink() else p, "rb") as fh:
                for blk in iter(lambda: fh.read(1 << 24), b""):
                    hh.update(blk)
            rows.append(f"{hh.hexdigest()}  {fn}")
        (d / "MANIFEST.sha256").write_text("\n".join(rows) + "\n")
    print(f"wrote {OUT} ({off / 2**30:.2f} GiB, {cut} expert / shared tensors cut to the TP=2 rank-0 shard)")


if __name__ == "__main__":
    main()
