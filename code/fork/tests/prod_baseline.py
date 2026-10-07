"""本番 exllamav3_ext.exl3_moe を合成重みで単独実行する基準線ハーネス。

本番 vLLM(exl3.py build_exl3_fused_state / apply_exl3_fused_moe)と同じ引数の作り方を再現する:
  - expert ごとの trellis/suh/svh の data_ptr を int64 テーブルにする
  - temps: (concurrency, rows, hidden)x2, (concurrency, rows, intermediate)x2 fp16
  - routing: token_sorted = トークン番号を expert 順に並べたもの、expert_count は末尾に番兵バケツ
  - MOE_ACT_SILU = 0, K=4, mcg=True, mul1=False
"""
from __future__ import annotations
import importlib.util
import torch

MOE_ACT_SILU = 0
SO = "/usr/local/lib/python3.12/dist-packages/exllamav3_ext.cpython-312-aarch64-linux-gnu.so"


def load_xl():
    spec = importlib.util.spec_from_file_location("exllamav3_ext", SO)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def make_experts(E, D, NI, dev, seed=0, scale=0.05):
    """expert ごとに別テンソル(本番の inners と同じく個別 data_ptr を持つ)。"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    def tr(k, n):
        return torch.randint(-2**15, 2**15 - 1, (k // 16, n // 16, 64), dtype=torch.int16, generator=g).to(dev)
    def sc(n):
        # suh/svh は ±1 に近い符号付きスケール(EXL3 の実物は符号×スケール)
        return ((torch.randint(0, 2, (n,), generator=g) * 2 - 1).to(torch.float16) * scale).to(dev)
    ex = []
    for _ in range(E):
        ex.append({
            "gate": {"trellis": tr(D, NI), "suh": sc(D), "svh": sc(NI)},
            "up":   {"trellis": tr(D, NI), "suh": sc(D), "svh": sc(NI)},
            "down": {"trellis": tr(NI, D), "suh": sc(NI), "svh": sc(D)},
        })
    return ex


def ptr_tables(ex, dev):
    def p(w, a):
        return torch.tensor([int(e[w][a].data_ptr()) for e in ex], dtype=torch.int64, device=dev)
    return {f"{w}_{a}": p(w, a) for w in ("gate", "up", "down") for a in ("trellis", "suh", "svh")}


def routing(ids: torch.Tensor, weights: torch.Tensor, n_exp: int):
    """本番 apply_exl3_fused_moe と同じ作り方(map_topk_to_local 済みの ids を受ける)。"""
    tokens, topk = ids.shape
    dev = ids.device
    local = ids.reshape(-1)
    flat_token = torch.arange(tokens, device=dev, dtype=torch.long).repeat_interleave(topk)
    flat_weight = weights.reshape(-1).to(torch.float16)
    order = local.argsort()
    token_sorted = flat_token[order]
    weight_sorted = flat_weight[order]
    expert_count = torch.zeros(n_exp + 1, dtype=torch.long, device=dev)
    expert_count.scatter_add_(0, local.long(), torch.ones(local.shape, dtype=torch.long, device=dev))
    return expert_count, token_sorted, weight_sorted


def run_prod(xl, x_fp16, ids, w, ex, ptrs, D, NI, limit=7.0, rows_cap=128, num_active=-1):
    dev = x_fp16.device
    E = len(ex)
    conc = max(1, int(xl.exl3_moe_max_concurrency(dev.index or 0)))
    temps = (
        torch.empty((conc, rows_cap, D), dtype=torch.float16, device=dev),
        torch.empty((conc, rows_cap, D), dtype=torch.float16, device=dev),
        torch.empty((conc, rows_cap, NI), dtype=torch.float16, device=dev),
        torch.empty((conc, rows_cap, NI), dtype=torch.float16, device=dev),
    )
    expert_count, token_sorted, weight_sorted = routing(ids, w, E)
    out = torch.zeros(x_fp16.shape[0], D, dtype=torch.float32, device=dev)
    args = (x_fp16, out, expert_count, token_sorted, weight_sorted, *temps, MOE_ACT_SILU, 4, 4, 4,
            ptrs["gate_trellis"], ptrs["gate_suh"], ptrs["gate_svh"],
            ptrs["up_trellis"], ptrs["up_suh"], ptrs["up_svh"],
            ptrs["down_trellis"], ptrs["down_suh"], ptrs["down_svh"],
            True, False, True, False, True, False, float(limit))
    # 本番ビルドの exl3_moe は num_active を受け付けない(29引数)。
    # 例外経由の再試行は pybind11 がテンソルを文字列化するため ~16ms かかり、計測を壊す。
    # よって一度だけ判定してキャッシュする(本番 _exl3_moe_accepts_num_active と同じ判定)。
    if _accepts_num_active(xl.exl3_moe):
        xl.exl3_moe(*args, num_active)
    else:
        xl.exl3_moe(*args)
    return out


_NA_CACHE: dict = {}


def _accepts_num_active(fn) -> bool:
    k = id(fn)
    if k not in _NA_CACHE:
        doc = getattr(fn, "__doc__", None) or ""
        _NA_CACHE[k] = "num_active" in doc or "arg29" in doc
    return _NA_CACHE[k]


if __name__ == "__main__":
    dev = "cuda"
    xl = load_xl()
    print("exl3_moe doc:", (xl.exl3_moe.__doc__ or "")[:300].replace("\n", " | "))
    E, D, NI, T, TOPK = 16, 4096, 768, 4, 8
    ex = make_experts(E, D, NI, dev)
    ptrs = ptr_tables(ex, dev)
    x = (torch.randn(T, D, device=dev) * 0.5).to(torch.float16)
    ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(T)])
    w = torch.softmax(torch.randn(T, TOPK, device=dev), -1)
    out = run_prod(xl, x, ids, w, ex, ptrs, D, NI)
    torch.cuda.synchronize()
    print("prod out finite:", bool(torch.isfinite(out).all()), "absmax:", float(out.abs().max()), "mean|.|:", float(out.abs().mean()))
    out2 = run_prod(xl, x, ids, w, ex, ptrs, D, NI)
    print("prod deterministic:", bool(torch.equal(out, out2)), " maxdiff:", float((out - out2).abs().max()))
