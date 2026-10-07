"""Host-side (numpy, no GPU): one MoE layer's 288 routed experts of the real GLM-5.3-Flash EXL3-TR3-4bpw checkpoint,
TP-sharded exactly like production's weight loader (overlay exl3.py shard_exl3_col / shard_exl3_row: gate/up trellis
dim 1 + svh column-parallel, down trellis dim 0 + suh row-parallel), in production's stacked create_weights layout:

  w13_trellis int16 [288, 2, K/16, N_loc/16, 64]  w13_suh fp16 [288, 2, K]  w13_svh fp16 [288, 2, N_loc]
  w2_trellis  int16 [288, N_loc/16, K/16, 64]     w2_suh  fp16 [288, N_loc] w2_svh  fp16 [288, K]

Reads the read-only shards in $TF_EXL3_ASSETS/moee4m3-shards/ (layer 10 is complete there), writes .npy
files to $TF_EXL3_ASSETS/moee4m3/layer{L}_tp{size}r{rank}/ (outside the repo, ~1.8 GB per rank).
Usage: python3 tests/moee4m3/extract_layer.py [layer=10] [rank=0] [size=2]
"""
import json
import os
import struct
import sys

import numpy as np

D = os.path.join(os.environ.get("TF_EXL3_ASSETS") or os.path.expanduser("~/tf-exl3-assets"), "moee4m3-shards/")
IDX = json.load(open(os.path.join(os.environ.get("TF_EXL3_MODELS") or os.path.expanduser("~/models"), "GLM-5.3-Flash-EXL3-TR3-4bpw-partial/model.safetensors.index.json")))["weight_map"]
L = int(sys.argv[1]) if len(sys.argv) > 1 else 10
RANK = int(sys.argv[2]) if len(sys.argv) > 2 else 0
SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else 2
OUT = os.path.join(os.environ.get("TF_EXL3_ASSETS") or os.path.expanduser("~/tf-exl3-assets"), f"moee4m3/layer{L}_tp{SIZE}r{RANK}/")
_hdr = {}


def tensor(name):
    f = IDX[name]
    if f not in _hdr:
        with open(D + f, "rb") as b:
            n = struct.unpack("<Q", b.read(8))[0]
            _hdr[f] = (json.loads(b.read(n)), 8 + n)
    h, off = _hdr[f]
    m = h[name]
    a, z = m["data_offsets"]
    dt = {"I16": np.int16, "F16": np.float16, "I32": np.int32}[m["dtype"]]
    return np.fromfile(D + f, dtype=dt, count=(z - a) // np.dtype(dt).itemsize, offset=off + a).reshape(m["shape"])


def narrow(t, dim):
    c = t.shape[dim] // SIZE
    return np.take(t, np.arange(c * RANK, c * (RANK + 1)), axis=dim)


def main():
    os.makedirs(OUT, exist_ok=True)
    P = lambda e, k, t: tensor(f"model.language_model.layers.{L}.mlp.experts.{e}.{k}.{t}")  # noqa: E731
    n = 288
    g0 = narrow(P(0, "gate_proj", "trellis"), 1)
    d0 = narrow(P(0, "down_proj", "trellis"), 0)
    K = g0.shape[0] * 16
    NL = g0.shape[1] * 16
    w13t = np.empty((n, 2) + g0.shape, np.int16)
    w13s = np.empty((n, 2, K), np.float16)
    w13v = np.empty((n, 2, NL), np.float16)
    w2t = np.empty((n,) + d0.shape, np.int16)
    w2s = np.empty((n, NL), np.float16)
    w2v = np.empty((n, K), np.float16)
    for e in range(n):
        for j, k in enumerate(("gate_proj", "up_proj")):
            w13t[e, j] = narrow(P(e, k, "trellis"), 1)
            w13s[e, j] = P(e, k, "suh")
            w13v[e, j] = narrow(P(e, k, "svh"), 0)
            assert P(e, k, "mcg").size == 1
        w2t[e] = narrow(P(e, "down_proj", "trellis"), 0)
        w2s[e] = narrow(P(e, "down_proj", "suh"), 0)
        w2v[e] = P(e, "down_proj", "svh")
    for name, arr in (("w13_trellis", w13t), ("w13_suh", w13s), ("w13_svh", w13v), ("w2_trellis", w2t),
                      ("w2_suh", w2s), ("w2_svh", w2v)):
        np.save(OUT + name + ".npy", arr)
    shared = bool(np.array_equal(w13s[:, 0], w13s[:, 1]))
    print(f"layer {L} tp{SIZE} rank {RANK}: K {K} N_loc {NL}, w13_trellis {w13t.shape}, w2_trellis {w2t.shape}, "
          f"gate/up suh shared across all experts: {shared} -> {OUT}")


if __name__ == "__main__":
    main()
