import json, struct, urllib.request, re, collections, concurrent.futures as cf
R = "https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw/resolve/main/"
def hdr(i):
    f = f"model-{i:05d}-of-00120.safetensors"
    req = urllib.request.Request(R + f, headers={"Range": "bytes=0-7"})
    n = struct.unpack("<Q", urllib.request.urlopen(req, timeout=60).read())[0]
    req = urllib.request.Request(R + f, headers={"Range": f"bytes=8-{7+n}"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())
with cf.ThreadPoolExecutor(8) as ex:
    heads = list(ex.map(hdr, range(1, 121)))
cat = collections.Counter(); cnt = collections.Counter()
per_layer_routed = collections.Counter()
for h in heads:
    for k, v in h.items():
        if k == "__metadata__": continue
        nb = v["data_offsets"][1] - v["data_offsets"][0]
        if ".mlp.experts." in k and "shared" not in k:
            c = "routed_experts"; m = re.search(r"layers\.(\d+)\.", k); per_layer_routed[int(m.group(1))] += nb
        elif "shared_expert" in k: c = "shared_experts"
        elif "embed_tokens" in k: c = "embed_tokens"
        elif "lm_head" in k: c = "lm_head"
        elif re.search(r"layers\.\d+\.mlp\.", k): c = "dense_mlp(+gate)"
        elif "self_attn" in k or "attn" in k: c = "attention"
        elif "visual" in k or "vision" in k: c = "vision"
        elif re.search(r"layers\.\d+\..*norm", k) or k.endswith("norm.weight"): c = "norms"
        else: c = "other"
        cat[c] += nb; cnt[c] += 1
tot = sum(cat.values())
print(f"total {tot/2**30:.2f} GiB in {sum(cnt.values())} tensors")
for c, b in cat.most_common(): print(f"  {c:18s} {b/2**30:8.2f} GiB  ({cnt[c]} tensors)")
L = sorted(per_layer_routed); print("routed layers:", len(L), L[:3], "...", L[-3:], f" per layer {per_layer_routed[L[5]]/2**20:.0f} MiB")
json.dump({k: v for k, v in cat.items()}, open("budget.json", "w"))
# sample names in 'other' and 'attention' to sanity-check the bins
for c in ("other", "dense_mlp(+gate)"):
    ex = [k for h in heads for k in h if k != "__metadata__" and c in ("other",) and not any(s in k for s in (".mlp.experts.","shared_expert","embed_tokens","lm_head","self_attn","attn","visual","vision","norm"))][:6]
    if ex: print("other sample:", ex)
