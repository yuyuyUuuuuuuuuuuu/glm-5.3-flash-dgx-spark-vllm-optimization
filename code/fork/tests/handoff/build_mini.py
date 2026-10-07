#!/usr/bin/env python3
"""Build the 10-layer GLM-5.3-Flash "handoff mini" checkpoint used by tests/handoff/ (host side, numpy only, no GPU).

Every tensor is a REAL checkpoint tensor (the partial GLM-5.3-Flash-Uncensored-NVFP4 download on nodeC: layers 0, 1, 10-13,
45 + embed / lm_head; attention, indexer, mHC and dense MLP weights are BF16/F32 there), cut to the per-rank shapes
production runs at TP=2 (rank 0's head / column slice), so every kernel sees production's local shapes at TP=1:
  KDA:  32 of 64 heads -> q/k/v [4096, 4096], conv [4096, 1, 4] x 3, b [32], f_b/g_b [4096, 128], A_log [32],
        dt_bias [4096], o_proj [4096, 4096]; f_a / g_a / o_norm replicated
  MLA:  32 of 64 heads -> q_b [8192, 1536], kv_b [16384, 512] (head-major nope|v blocks), o_proj [4096, 8192];
        q_a / kv_a / norms and the whole sparse indexer (32 index heads, kpool 4, top-k 2048) replicated as at TP=2
  MLP:  dense, intermediate 6144 (rank 0's half of the 12288 dense MLP)
Layer plan (mini <- real attention): KDA 0, KDA 1, KDA 10, DSA 11, KDA 12, DSA 45 (the MTP layer's attention +
indexer), KDA 13, DSA 11, DSA 45, DSA 11 (LAYERS below; mHC / norms from real 10-13). Ten layers because production's
DFlash2 drafter KV layout (patch_glm5_drafter_group.py) co-locates drafter layer i with MLA layer i and needs at least
as many MLA layers as the drafter's 5. MLP of mini layer i <- real layer (i % 2) dense MLP. Final norm <- 45's
shared_head.norm. Config: glm5_next_text / Glm5NextForCausalLM, no quantization, no MTP.

Also writes <out>-dflash2/: the DFlash2 drafter (real weights, symlinked) with target_layer_ids [0..4] (the mini model's
layers; each aux layer is followed by another mHC layer, so the mHC aux path runs at every aux layer).
Usage: build_mini.py [out dir, default $TF_EXL3_MODELS/GLM-5.3-Flash-handoff-mini]
Deterministic: same sources -> byte-identical output (MANIFEST.sha256 written next to it).
"""
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

SRC = Path(os.path.join(os.environ.get("TF_EXL3_MODELS") or os.path.expanduser("~/models"), "GLM-5.3-Flash-Uncensored-NVFP4"))
DRAFT = Path(os.path.join(os.environ.get("TF_EXL3_MODELS") or os.path.expanduser("~/models"), "GLM-5.3-Flash-DFlash2-dc77ff1c"))
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("TF_EXL3_MODELS") or os.path.expanduser("~/models"), "GLM-5.3-Flash-handoff-mini"))
DT = {"BF16": (np.uint16, 2), "F32": (np.float32, 4), "F16": (np.float16, 2)}
P = "model.language_model."

KDA_HEADS, HD = 32, 128            # rank-0 local KDA heads (64 / TP 2)
MLA_HEADS, NOPE, VD = 32, 256, 256  # rank-0 local MLA heads
INTER = 6144                        # rank-0 half of the dense MLP
LAYERS = [  # (type, attention source layer, mhc/norm source layer, mlp source layer)
    ("linear_attention", 0, 0, 0),
    ("linear_attention", 1, 1, 1),
    ("linear_attention", 10, 10, 0),
    ("deepseek_sparse_attention", 11, 11, 1),
    ("linear_attention", 12, 12, 0),
    ("deepseek_sparse_attention", 45, 13, 1),
    ("linear_attention", 13, 13, 0),
    ("deepseek_sparse_attention", 11, 10, 1),
    ("deepseek_sparse_attention", 45, 12, 0),
    ("deepseek_sparse_attention", 11, 11, 1),
]


def index():
    idx = {}
    for f in sorted(SRC.glob("model-*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        for k, v in h.items():
            if k != "__metadata__":
                idx[k] = (f, 8 + n + v["data_offsets"][0], v["data_offsets"][1] - v["data_offsets"][0], v["dtype"],
                          v["shape"])
    return idx


IDX = index()


def load(name):
    f, off, nbytes, dt, shape = IDX[name]
    npdt, _ = DT[dt]
    with open(f, "rb") as fh:
        fh.seek(off)
        a = np.frombuffer(fh.read(nbytes), dtype=npdt).reshape(shape)
    return dt, a


def build():
    out = {}   # name -> (dtype str, array)

    def put(name, dt, a):
        out[name] = (dt, np.ascontiguousarray(a))

    put("model.embed_tokens.weight", *load(P + "embed_tokens.weight"))
    put("lm_head.weight", *load("lm_head.weight"))
    put("model.norm.weight", *load(P + "layers.45.shared_head.norm.weight"))
    for i, (typ, a_src, h_src, m_src) in enumerate(LAYERS):
        d = f"model.layers.{i}."
        for nm in ("hc_attn_base", "hc_attn_fn", "hc_attn_scale", "hc_ffn_base", "hc_ffn_fn", "hc_ffn_scale",
                   "input_layernorm.weight", "post_attention_layernorm.weight"):
            put(d + nm, *load(P + f"layers.{h_src}.{nm}"))
        s = P + f"layers.{a_src}.self_attn."
        a = d + "self_attn."
        if typ == "linear_attention":
            n = KDA_HEADS * HD
            for nm in ("q_proj", "k_proj", "v_proj"):
                dt, w = load(s + nm + ".weight"); put(a + nm + ".weight", dt, w[:n])
                dt, w = load(s + nm[0] + "_conv1d.weight"); put(a + nm[0] + "_conv1d.weight", dt, w[:n])
            dt, w = load(s + "b_proj.weight"); put(a + "b_proj.weight", dt, w[:KDA_HEADS])
            for nm in ("f_a_proj", "g_a_proj"):
                put(a + nm + ".weight", *load(s + nm + ".weight"))
            for nm in ("f_b_proj", "g_b_proj"):
                dt, w = load(s + nm + ".weight"); put(a + nm + ".weight", dt, w[:n])
            dt, w = load(s + "A_log"); put(a + "A_log", dt, w[:KDA_HEADS])
            dt, w = load(s + "dt_bias"); put(a + "dt_bias", dt, w[:n])
            put(a + "o_norm.weight", *load(s + "o_norm.weight"))
            dt, w = load(s + "o_proj.weight"); put(a + "o_proj.weight", dt, w[:, :n])
        else:
            for nm in ("q_a_proj.weight", "q_a_layernorm.weight", "kv_a_proj_with_mqa.weight", "kv_a_layernorm.weight"):
                put(a + nm, *load(s + nm))
            dt, w = load(s + "q_b_proj.weight"); put(a + "q_b_proj.weight", dt, w[:MLA_HEADS * NOPE])
            dt, w = load(s + "kv_b_proj.weight"); put(a + "kv_b_proj.weight", dt, w[:MLA_HEADS * (NOPE + VD)])
            dt, w = load(s + "o_proj.weight"); put(a + "o_proj.weight", dt, w[:, :MLA_HEADS * VD])
            for nm in ("index_kpool_compress_ape", "index_kpool_compress_gate", "k_norm.bias", "k_norm.weight",
                       "weights_proj.weight", "wk.weight", "wq_b.weight"):
                put(a + "indexer." + nm, *load(s + "indexer." + nm))
        m = P + f"layers.{m_src}.mlp."
        dt, w = load(m + "gate_proj.weight"); put(d + "mlp.gate_proj.weight", dt, w[:INTER])
        dt, w = load(m + "up_proj.weight"); put(d + "mlp.up_proj.weight", dt, w[:INTER])
        dt, w = load(m + "down_proj.weight"); put(d + "mlp.down_proj.weight", dt, w[:, :INTER])
    return out


def write_st(path, tensors):
    header, off = {}, 0
    for k in sorted(tensors):
        dt, a = tensors[k]
        header[k] = {"dtype": dt, "shape": list(a.shape), "data_offsets": [off, off + a.nbytes]}
        off += a.nbytes
    header["__metadata__"] = {"format": "pt"}
    hb = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(hb)))
        fh.write(hb)
        for k in sorted(tensors):
            fh.write(tensors[k][1].tobytes())


def config():
    c = json.load(open(SRC / "config.json"))["text_config"]
    L = len(LAYERS)
    c.update(
        architectures=["Glm5NextForCausalLM"], model_type="glm5_next_text", num_hidden_layers=L,
        layer_types=[t for t, *_ in LAYERS], mlp_layer_types=["dense"] * L, first_k_dense_replace=L,
        indexer_types=c["indexer_types"][:L], num_attention_heads=MLA_HEADS, num_key_value_heads=MLA_HEADS,
        intermediate_size=INTER, num_nextn_predict_layers=0, n_routed_experts=None, torch_dtype="bfloat16",
    )
    la = dict(c["linear_attn_config"])
    la.update(num_heads=KDA_HEADS, full_attn_layers=[i for i, (t, *_) in enumerate(LAYERS) if t != "linear_attention"],
              kda_layers=[i for i, (t, *_) in enumerate(LAYERS) if t == "linear_attention"])
    c["linear_attn_config"] = la
    c.pop("quantization_config", None)
    c["_handoff_mini"] = {"source": str(SRC), "layers": [list(x) for x in LAYERS], "kda_heads": KDA_HEADS,
                          "mla_heads": MLA_HEADS, "intermediate": INTER}
    return c


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = build()
    write_st(OUT / "model.safetensors", t)
    json.dump(config(), open(OUT / "config.json", "w"), indent=1, sort_keys=True)
    gen = SRC / "generation_config.json"
    if gen.exists():
        (OUT / "generation_config.json").write_bytes(gen.read_bytes())
    D = Path(str(OUT) + "-dflash2")
    D.mkdir(exist_ok=True)
    dc = json.load(open(DRAFT / "config.json"))
    dc["dflash_config"]["target_layer_ids"] = [0, 1, 2, 3, 4]
    dc["num_target_layers"] = len(LAYERS)
    json.dump(dc, open(D / "config.json", "w"), indent=1, sort_keys=True)
    lk = D / "model.safetensors"
    if lk.is_symlink() or lk.exists():
        lk.unlink()
    lk.symlink_to(DRAFT / "model.safetensors")
    for d in (OUT, D):
        rows = []
        for f in sorted(os.listdir(d)):
            if f == "MANIFEST.sha256":
                continue
            h = hashlib.sha256()
            with open(d / f, "rb") as fh:
                for blk in iter(lambda: fh.read(1 << 24), b""):
                    h.update(blk)
            rows.append(f"{h.hexdigest()}  {f}")
        (d / "MANIFEST.sha256").write_text("\n".join(rows) + "\n")
    n = sum(a.nbytes for _, a in t.values())
    print(f"wrote {OUT} ({len(t)} tensors, {n / 2**30:.2f} GiB) and {D}")


if __name__ == "__main__":
    main()
