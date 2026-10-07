"""② GLM53_DENSE_FP8 に mla,shared を足す前の nodeC 検証(実重み・本番と同じ量子化手順・本番の Marlin カーネル)。

実重み: HF Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw の BF16 テンソル(tests 外で Range 取得したもの)。
量子化: 本番 Glm53DenseFp8Method.process_weights_after_loading と同じ(出力チャネルごと amax/448 → e4m3 → Marlin 詰め替え)。
比較: 本番で既に FP8 の group(kda, dense)と、追加候補(mla, shared)の誤差を同じ物差しで並べる。
TP=2 の rank0 の持ち分に切り出す(column-parallel は行、row-parallel は列を半分、MLA fused_qkv_a は複製)。
"""
import json, os, statistics, sys, time
import torch
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear, prepare_fp8_layer_for_marlin)

D = os.environ.get("BF16_SAMPLES", "/models/bf16_samples")
man = json.load(open(os.path.join(D, "manifest.json")))
dev = "cuda"
torch.manual_seed(0)
# GB10 は統合メモリ: cudaMemGetInfo の free はページキャッシュ(回収可能)を数えない。空きはホストの MemAvailable で見る
avail = int(next(l for l in open('/proc/meminfo') if l.startswith('MemAvailable')).split()[1]) * 1024
assert avail > 40 * 2**30, f'host MemAvailable {avail/2**30:.1f} GiB < 40'

def load(name):
    x = next(m for m in man if m["name"] == name)
    raw = open(os.path.join(D, x["file"]), "rb").read()
    return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).view(*x["shape"]).to(dev)

P = "model.language_model.layers.{}."
def rank0(group, l):
    """rank0 の持ち分の行列(vLLM のマージ済み射影の形)を返す: [(label, W[N,K])]"""
    g = lambda s: load(P.format(l) + s)
    if group == "mla":
        qa, kva = g("self_attn.q_a_proj.weight"), g("self_attn.kv_a_proj_with_mqa.weight")
        qb, o = g("self_attn.q_b_proj.weight"), g("self_attn.o_proj.weight")
        return [("fused_qkv_a", torch.cat([qa, kva])), ("q_b", qb[: qb.shape[0] // 2]), ("o_proj", o[:, : o.shape[1] // 2].contiguous())]
    if group == "shared":
        ga, up, dn = (g(f"mlp.shared_experts.{s}_proj.weight") for s in ("gate", "up", "down"))
        h = ga.shape[0] // 2
        return [("gate_up", torch.cat([ga[:h], up[:h]])), ("down", dn[:, :h].contiguous())]
    if group == "kda":
        q, k, v, o = (g(f"self_attn.{s}_proj.weight") for s in ("q", "k", "v", "o"))
        h = q.shape[0] // 2
        return [("in_proj_qkv", torch.cat([q[:h], k[:h], v[:h]])), ("o_proj", o[:, :h].contiguous())]
    if group == "dense":
        ga, up, dn = (g(f"mlp.{s}_proj.weight") for s in ("gate", "up", "down"))
        h = ga.shape[0] // 2
        return [("gate_up", torch.cat([ga[:h], up[:h]])), ("down", dn[:, :h].contiguous())]

def quantize(w):
    """本番 Glm53DenseFp8Method と同一の手順。戻り値: Marlin 用の層と、比較用の逆量子化 BF16 行列。"""
    n, k = w.shape
    wf = w.float()
    scales = wf.abs().amax(dim=1).clamp(min=1e-12) / 448.0
    fp8 = (wf / scales[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    deq = (fp8.float() * scales[:, None])
    layer = torch.nn.Module()
    layer.output_size_per_partition, layer.input_size_per_partition = n, k
    layer.orig_dtype = w.dtype
    layer.weight = torch.nn.Parameter(fp8, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scales.to(w.dtype), requires_grad=False)
    layer.weight_block_size = None
    prepare_fp8_layer_for_marlin(layer, size_k_first=False)
    return layer, deq

def marlin(layer, x, n, k):
    return apply_fp8_marlin_linear(input=x, weight=layer.weight, weight_scale=layer.weight_scale,
                                   workspace=layer.workspace, size_n=n, size_k=k, bias=None)

def acts(T, k, outliers):
    x = torch.randn(T, k, device=dev)
    if outliers:            # LLM 活性化によくある「一部チャネルだけ桁違いに大きい」状態
        idx = torch.randperm(k, device=dev)[: max(1, k // 256)]
        x[:, idx] *= 30.0
    return x.to(torch.bfloat16)

def rel(a, b): return float((a.float() - b.float()).norm() / b.float().norm())

rows = []
layers = {"mla": (3, 23, 43), "shared": (3, 10, 23, 43), "kda": (1, 20, 42), "dense": (1,)}
fail = []
for grp, ls in layers.items():
    for l in ls:
        for lab, w in rank0(grp, l):
            n, k = w.shape
            layer, deq = quantize(w)
            r = {"group": grp, "layer": l, "mat": lab, "shape": (n, k), "w_rel": rel(deq, w.float())}
            for outl in (False, True):
                x = acts(8, k, outl)
                ref = x.float() @ w.float().t()                  # BF16 重み(現状)の出力
                y_fp8_ref = x.float() @ deq.t()                  # FP8 化した重みの理想出力
                y_marlin = marlin(layer, x, n, k)                # 本番の Marlin カーネル
                r[f"out_rel{'_outl' if outl else ''}"] = rel(y_marlin, ref)
                r[f"kernel_vs_deq{'_outl' if outl else ''}"] = rel(y_marlin, y_fp8_ref)
            if r["kernel_vs_deq"] > 1e-2 or not torch.isfinite(y_marlin).all():
                fail.append((grp, l, lab, r["kernel_vs_deq"]))
            rows.append(r)
            del layer, deq, w
torch.cuda.empty_cache()

print(f"{'group':7s} {'layer':>5s} {'matrix':12s} {'N x K':>13s} | {'w_rel':>8s} {'out_rel':>8s} {'out_rel(outl)':>13s} | {'marlin vs deq':>13s}")
for r in rows:
    print(f"{r['group']:7s} {r['layer']:5d} {r['mat']:12s} {str(r['shape']):>13s} | {r['w_rel']:.2e} {r['out_rel']:.2e} {r['out_rel_outl']:13.2e} | {r['kernel_vs_deq']:13.2e}")
print("\n== group summary (median / max over matrices) — kda, dense = 本番で既に FP8 ; mla, shared = 追加候補")
summ = {}
for grp in layers:
    v = [r for r in rows if r["group"] == grp]
    s = {m: (statistics.median(r[m] for r in v), max(r[m] for r in v)) for m in ("w_rel", "out_rel", "out_rel_outl")}
    summ[grp] = s
    print(f"  {grp:7s} w_rel {s['w_rel'][0]:.2e}/{s['w_rel'][1]:.2e}  out_rel {s['out_rel'][0]:.2e}/{s['out_rel'][1]:.2e}  out_rel(outliers) {s['out_rel_outl'][0]:.2e}/{s['out_rel_outl'][1]:.2e}")
base_max = max(summ["kda"]["out_rel"][1], summ["dense"]["out_rel"][1])
cand_max = max(summ["mla"]["out_rel"][1], summ["shared"]["out_rel"][1])
print(f"\ncandidate max out_rel {cand_max:.2e} vs production-accepted max {base_max:.2e}  -> {'<= (同等以下)' if cand_max <= base_max * 1.1 else '> (悪化)'}")
print("Marlin kernel failures:", fail or "none")

# ---- 速度: BF16 (F.linear) vs FP8 Marlin, rank0 形状, 冷えた重み(8 コピーを巡回 > L2 24 MiB)
print("\n== speed per matrix (us, median of 5 rounds, cold weights), T = tokens per call")
shapes = {("mla", "fused_qkv_a"): (2048, 4096, 11), ("mla", "q_b"): (8192, 1536, 11), ("mla", "o_proj"): (4096, 8192, 11),
          ("shared", "gate_up"): (2048, 4096, 42), ("shared", "down"): (4096, 1024, 42)}
def timeit(fns, it=40):
    for f in fns: f()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record()
    for i in range(it): fns[i % len(fns)]()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / it * 1000
saved = {1: 0.0, 8: 0.0, 64: 0.0}
for (grp, lab), (n, k, nl) in shapes.items():
    copies = max(8, int(64 * 2**20 // (n * k * 2)) + 1)
    ws = [torch.randn(n, k, device=dev).to(torch.bfloat16) * 0.02 for _ in range(copies)]
    qs = [quantize(w)[0] for w in ws]
    line = f"  {grp:6s} {lab:12s} {n}x{k} x{nl} layers:"
    for T in (1, 8, 64):
        x = torch.randn(T, k, device=dev).to(torch.bfloat16)
        b, f8 = [], []
        for _ in range(5):
            b.append(timeit([lambda w=w: torch.nn.functional.linear(x, w) for w in ws]))
            f8.append(timeit([lambda q=q: marlin(q, x, n, k) for q in qs]))
        bm, fm = statistics.median(b), statistics.median(f8)
        saved[T] += (bm - fm) * nl
        line += f"  T={T}: bf16 {bm:6.1f} fp8 {fm:6.1f} ({bm/fm:.2f}x)"
    print(line)
    del ws, qs; torch.cuda.empty_cache()
print("\n== time saved per forward per rank if mla+shared go FP8 (sum over layers): " +
      ", ".join(f"T={T}: {v/1000:.2f} ms" for T, v in saved.items()))
sys.exit(1 if fail else 0)
