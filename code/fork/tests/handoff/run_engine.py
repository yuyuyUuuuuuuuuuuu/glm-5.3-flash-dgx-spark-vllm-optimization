"""State-handoff harness, in-container driver (one feature configuration per process). Run by tests/handoff/run.sh
inside a production-composed container (the launcher's GLM53_OVERLAY_ORDER chain incl. patch_tf_bundle.py with this
worktree's bundle, production's SM90_KV mounts and production env); never on its own.

What it runs: the REAL vLLM engine of the production image (V2 model runner, breakable CUDA graphs, prefix caching in
mamba "align" mode, fp8 KV cache, FLASHINFER_MLA_SPARSE_SM90, DFlash2 speculative decoding with adaptive K {4,5,7},
the tf-exl3 plugin loaded through its vllm.general_plugins entry point) on the 10-layer handoff mini model
(tests/handoff/build_mini.py: real GLM-5.3-Flash KDA / DSA+indexer / mHC / dense-MLP weights at TP=2 per-rank shapes)
and the real DFlash2 drafter. Requests A_i (--prompts, e.g. 6000,3001), one after the other, each its own cache_salt:
  prompt of n real-text tokens (6000 spans the prefill chunk boundary: chunk 1 without, chunk 2 with initial state;
  3001 is one chunk, like production's 5-8k probe prompts at MNBT 13824, with a kpool tail of 1), then --gen greedy
  tokens through spec-verify steps (M = K+1, K in {7, 5, 4}).

SHADOW (--shadow 1, the bitwise handoff check; one process, so no cross-process nondeterminism):
  every eager target-model forward with >= --shadow-min-t tokens (the prefill chunks) runs up to three times from the
  SAME starting bytes of every KV allocation (KDA conv_state / recurrent state slots, MLA KV pages, indexer K cache
  pages with their scales, kpool tail pages, drafter KV) and of the persistent side buffers (SM90 wrapper kv_indices /
  kv_len_arr, the sparse top-k indices buffer):
    P  every feature forced onto production's statements (quickwins MIN_T = inf and idx_gate off, MLA prefill
       MIN_TOKENS = inf): exactly production's computation in this process
    C  P again (control: proves the forward is deterministic in-process; a feature diff is only meaningful if C == P)
    F  the configured features (what production ran with them on); the engine continues from F's state
  and compares bitwise: the forward's outputs (hidden states + DFlash2 aux hidden states), every decoder layer's /
  self_attn's / mlp's output (forward hooks: the first module where F leaves P), and every byte of every state buffer,
  reported per layer view, row (block / slot, marked when the row belongs to the request) and element.
  Decode steps then run from F's state through production's decode path (CUDA-graph replays with eager-break ops);
  the fast-path counters are logged per step (a fast path must never run in decode or capture).
Per-step hashes of every layer view / raw block are also recorded (records.pt) for cross-run comparisons (compare.py).
Outputs in --out: result.json (per request: prompt ids, generated ids, top-5 logprobs), shadow.json, records.pt,
harness.log. Optional --consistency N: B_j = fresh prefill of prompt + A's first j tokens (own cache_salt).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
import time


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.path.join(os.environ.get("TF_EXL3_MODELS") or os.path.expanduser("~/models"), "GLM-5.3-Flash-handoff-mini"))
    ap.add_argument("--prompts", default="6000,3001", help="comma list of prompt lengths (one request each)")
    ap.add_argument("--gen", type=int, default=24)
    ap.add_argument("--shadow", type=int, default=1)
    ap.add_argument("--shadow-control", type=int, default=1)
    ap.add_argument("--shadow-min-t", type=int, default=256)
    ap.add_argument("--dump-decode", type=int, default=0, help="save full view contents at prefill + N decode steps")
    ap.add_argument("--consistency", type=int, default=0, help="number of B_j prefills (0 = none)")
    ap.add_argument("--mnbt", type=int, default=16384)
    ap.add_argument("--batch-a", type=int, default=0,
                    help="submit the first N A-requests in ONE llm.generate call: their prefills land in the same "
                         "engine step (multi-sequence cu_seqlens, N>1 chunk); 0 = one request per call (as before)")
    ap.add_argument("--kv-bytes", type=int, default=1 << 30)
    ap.add_argument("--gpu-util", type=float, default=0.125, help="also nodeC's GPU budget (< 16 GiB of 121.6)")
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--topk", default="sort", choices=("prod", "sort", "full"),
                    help="prefill top-k: prod = production's op as is (not reproducible in-process), sort = the op's "
                         "selection in a canonical order (score desc, index asc), full = stable-sort top-k")
    ap.add_argument("--topk-probe", type=int, default=1, help="run production's op twice per call and log set/order drift")
    ap.add_argument("--mla-probe", type=int, default=1,
                    help="on the feature pass: FA2 / installed path / fp32 references (valid count, planned kv_len) "
                         "per sparse-MLA prefill call")
    ap.add_argument("--mla-kv-indices", type=int, default=1,
                    help="0: glm53_mla_prefill does not write the FA2 wrapper's kv_indices (the deploy-r16 defect, for "
                         "the regression check); 1: the fixed module (default)")
    ap.add_argument("--label", default="")
    return ap.parse_args()


ARGS = parse()
os.makedirs(ARGS.out, exist_ok=True)
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")     # engine core + worker in this process
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")

import torch  # noqa: E402

LOG = open(os.path.join(ARGS.out, "harness.log"), "a")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


# ------------------------------------------------------------------------------------------------ prompt (real text)
_TEXT = {}


def prompt_ids(n, i):
    from tokenizers import Tokenizer
    if "ids" not in _TEXT:
        tok = Tokenizer.from_file(os.path.join(os.environ.get("TF_EXL3_MODELS") or os.path.expanduser("~/models"), "GLM-OCR/tokenizer.json"))
        files = sorted(glob.glob("/usr/share/common-licenses/*")) + sorted(glob.glob("/usr/lib/python3.12/*.py"))
        t = "".join("\n\n### %s\n%s" % (os.path.basename(f), open(f, errors="ignore").read()) for f in files)
        _TEXT["ids"] = [int(x) for x in tok.encode(t).ids]
    ids = _TEXT["ids"]
    s = random.Random("handoff-%d" % i).randrange(0, len(ids) - n - 1)
    return ids[s:s + n]


def prompt_ids_ext(n, i, total):
    """prefixhit: the same start as prompt_ids(n, i), but `total` tokens (prompt_ids(n, i) == the first n)."""
    prompt_ids(n, i)
    ids = _TEXT["ids"]
    s = random.Random("handoff-%d" % i).randrange(0, len(ids) - n - 1)
    assert s + total <= len(ids), (s, total, len(ids))
    return ids[s:s + total]


# ------------------------------------------------------------------------------------------------ engine hooks
import vllm.v1.worker.gpu.attn_utils as AU  # noqa: E402
import vllm.v1.worker.gpu.model_runner as MR  # noqa: E402

H = {"raw": {}, "kv": None, "runner": None, "active": False, "step": 0, "cur": None, "records": [], "coef": {},
     "dumped": [], "decode_steps": 0, "blocks": {}}
SH = {"steps": [], "trace": None, "hooked": False, "wrapped": False, "phase": None, "mla": []}

_orig_alloc = AU._allocate_kv_cache


def _alloc(*a, **k):
    d = _orig_alloc(*a, **k)
    for name, t in d.items():
        H["raw"][name] = t
    return d


AU._allocate_kv_cache = _alloc
_orig_init_kv = MR.init_kv_cache


def _init_kv(*a, **k):
    d = _orig_init_kv(*a, **k)
    H["kv"] = d
    return d


MR.init_kv_cache = _init_kv


def _flat(x, out, pre):
    if isinstance(x, torch.Tensor):
        out.append((pre, x))
    elif isinstance(x, (list, tuple)):
        for i, y in enumerate(x):
            _flat(y, out, f"{pre}[{i}]")


def _sfc():
    return H["runner"].compilation_config.static_forward_context


def views():
    """[(name, tensor)] every per-layer KV view (sorted), dim 0 = block / slot rows."""
    out = []
    for ln in sorted(H["kv"] or {}):
        mod = _sfc().get(ln) if H["runner"] is not None else None
        kv = getattr(mod, "kv_cache", None) if mod is not None else None
        if kv is None:
            kv = H["kv"][ln]
        _flat(kv, out, ln)
    return out


def uniq_raw():
    seen, out = set(), []
    for ln in sorted(H["raw"]):
        t = H["raw"][ln]
        if t.data_ptr() not in seen:
            seen.add(t.data_ptr())
            out.append((ln, t))
    return out


def _coef(n, key):
    c = H["coef"].get((key, n))
    if c is None:
        g = torch.Generator(device="cpu").manual_seed(1234567 + n)
        c = (torch.randint(-(2 ** 62), 2 ** 62, (n,), generator=g, dtype=torch.int64) * 2 + 1).cuda()
        H["coef"][(key, n)] = c
    return c


def row_hash(t2d_u8):
    """[rows, bytes] uint8 (contiguous) -> [rows] int64 hash (sum of int64 words x odd random coefficients, wrapping)."""
    rows, nb = t2d_u8.shape
    pad = (-nb) % 8
    if pad:
        t2d_u8 = torch.nn.functional.pad(t2d_u8, (0, pad))
    w = t2d_u8.view(torch.int64)
    c = _coef(w.shape[1], "row")
    return (w * c).sum(dim=1)


def tensor_hash(t):
    return int(row_hash(t.contiguous().view(torch.uint8).reshape(1, -1))[0])


def view_row_hashes(t, chunk_bytes=256 << 20):
    rows = t.shape[0]
    per = max(1, t[0].numel() * t.element_size())
    step = max(1, chunk_bytes // per)
    hs = []
    for r0 in range(0, rows, step):
        x = t[r0:r0 + step].contiguous().view(torch.uint8).reshape(min(step, rows - r0), -1)
        hs.append(row_hash(x))
    return torch.cat(hs)


def raw_block_hashes(raw, nblocks, chunk_bytes=256 << 20):
    n = raw.numel() * raw.element_size()
    if nblocks and n % nblocks == 0:
        page = n // nblocks
    else:
        page = 1 << 20
        nblocks = (n + page - 1) // page
    v = raw.view(torch.uint8).reshape(-1)
    hs = []
    step = max(1, chunk_bytes // page)
    for b0 in range(0, nblocks, step):
        b1 = min(nblocks, b0 + step)
        x = v[b0 * page:min(n, b1 * page)]
        if x.numel() < (b1 - b0) * page:
            x = torch.nn.functional.pad(x, (0, (b1 - b0) * page - x.numel()))
        hs.append(row_hash(x.reshape(b1 - b0, page)))
    return torch.cat(hs)


def side_tensors():
    """[(name, tensor)] persistent buffers outside the KV allocations that a forward writes and a later step may read."""
    out = []
    try:
        sm = sys.modules.get("vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90")
        st = getattr(sm, "_SM90_STATE", None) if sm is not None else None
        if st is not None:
            out += [("sm90.kv_indices", st.kv_indices), ("sm90.kv_len_arr", st.kv_len_arr)]
    except Exception as e:  # noqa: BLE001
        log("side_tensors sm90:", repr(e))
    seen = {t.data_ptr() for _, t in out}
    for ln, mod in sorted(_sfc().items()):
        for holder in (mod, getattr(mod, "impl", None), getattr(mod, "indexer", None)):
            tb = getattr(holder, "topk_indices_buffer", None) if holder is not None else None
            if isinstance(tb, torch.Tensor) and tb.data_ptr() not in seen:
                seen.add(tb.data_ptr())
                out.append((f"topk_indices_buffer({ln})", tb))
    return out


def _kv_rows(ntok):
    """Rows of the SM90 kv_indices a step leaves for later FA2 calls: all ntok, or glm53_mla_prefill's bounded
    write-back (opt-kdamhc, STATE.kv_rows: every row an FA2 call can read beyond its own) when it is installed."""
    M = sys.modules.get("glm53_mla_prefill")
    R = getattr(getattr(M, "STATE", None), "kv_rows", None) if M is not None and M.STATE.installed else None
    return ntok if R is None else min(ntok, R)


def side_buffers(ntok):
    out = {}
    for name, t in side_tensors():
        if name == "sm90.kv_indices":
            try:
                W = sys.modules["vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90"]._SM90_STATE.topk_width
                out[name + "[readable rows of step]"] = tensor_hash(t[: _kv_rows(ntok) * W])
            except Exception as e:  # noqa: BLE001
                out[name] = repr(e)
        elif name.startswith("topk_indices_buffer"):
            out[name + "[rows of step]"] = tensor_hash(t[:ntok])
        else:
            out[name] = tensor_hash(t)
    return out


def fast_path_counters():
    c = {}
    try:
        import glm53_prefill_quickwins as Q
        c.update({k: v for k, v in Q.STATS.items() if v})
    except Exception:  # noqa: BLE001
        pass
    try:
        import glm53_mla_prefill as M
        c["mla_prefill_calls"] = M.STATE.calls
        if getattr(M.STATE, "kv_indices_writes", 0):
            c["mla_prefill_kv_indices_writes"] = M.STATE.kv_indices_writes
        c.update({f"mla_prefill_fallback_{k}": v for k, v in M.STATE.fallbacks.items()})
    except Exception:  # noqa: BLE001
        pass
    return c


def _delta(a, b):
    return {k: b.get(k, 0) - a.get(k, 0) for k in sorted(set(a) | set(b)) if b.get(k, 0) != a.get(k, 0)}


def _layer_idx(name):
    try:
        return int(name.split("layers.", 1)[1].split(".", 1)[0])
    except (IndexError, ValueError):
        return -1


# ------------------------------------------------------------------------------------------------ top-k determinizer
# The sparse indexer's prefill top-k (torch.ops._C.top_k_per_row_prefill, kpool pools, select 512 of up to
# context/4) is not reproducible in-process: two calls on the same logits return a different order and/or set, and the
# kpool path keeps only the first select_k - 1 pools (expand_pools_and_append_tail), so which pool is dropped moves too.
# Production has that nondeterminism in every configuration; the harness removes it the same way in P, C and F so that a
# feature's effect is not buried under it (--topk sort keeps the op's selection, only the order is canonical).
TOPK = {"calls": 0, "set_drift": 0, "order_drift": 0, "rows": 0, "logged": 0}



def _install_tailptr_probe():
    """prefixhit-adv (HANDOFF_TAILPTR=1): log KpoolTailMetadataBuilder.build's output slot_mapping address vs the
    persistent input buffer address, per call (capture-time calls happen inside LLM() init; runtime FULL replays read
    the capture-time address, so a runtime address that differs from it is never read by the graph)."""
    import vllm.v1.attention.backends.mla.indexer as IX
    orig = IX.KpoolTailMetadataBuilder.build
    seen = {"n": 0}

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        out = orig(self, common_prefix_len, common_attn_metadata, fast_build)
        seen["n"] += 1
        if seen["n"] <= 60 or (H.get("active") and seen["n"] % 50 == 0):
            cam = common_attn_metadata
            log(f"tailptr call {seen['n']} step {H['step']} active {H.get('active')} tokens {cam.num_actual_tokens} "
                f"reqs {cam.num_reqs} in_ptr {cam.slot_mapping.data_ptr():#x} out_ptr {out.slot_mapping.data_ptr():#x} "
                f"same {cam.slot_mapping.data_ptr() == out.slot_mapping.data_ptr()} pos {'set' if cam.positions is not None else 'None'}")
        return out
    IX.KpoolTailMetadataBuilder.build = build


def _install_topk_determinizer():
    if ARGS.topk == "prod":
        return
    ns = torch.ops._C
    orig = ns.top_k_per_row_prefill

    def canon_rows(logits, ks, out, n, k):
        idx = out[:n, :k].to(torch.int64)
        valid = idx >= 0
        cols = logits.shape[1]
        col = (idx + ks[:n, None].to(torch.int64)).clamp(0, cols - 1)
        sc = torch.gather(logits[:n].float(), 1, col)
        sc = torch.where(valid, sc, torch.full_like(sc, float("-inf")))
        big = torch.iinfo(torch.int64).max
        o1 = torch.sort(torch.where(valid, idx, torch.full_like(idx, big)), dim=1, stable=True).indices
        idx, sc, valid = idx.gather(1, o1), sc.gather(1, o1), valid.gather(1, o1)
        o2 = torch.sort(-sc, dim=1, stable=True).indices
        idx, valid = idx.gather(1, o2), valid.gather(1, o2)
        out[:n, :k] = torch.where(valid, idx, torch.full_like(idx, -1)).to(out.dtype)

    def full_topk(logits, ks, ke, out, n, k):
        cols = logits.shape[1]
        ar = torch.arange(cols, device=logits.device)
        valid = (ar[None, :] >= ks[:n, None]) & (ar[None, :] < ke[:n, None])
        v = torch.where(valid, logits[:n].float(), torch.full((1, 1), float("-inf"), device=logits.device))
        kk = min(k, cols)
        o = torch.sort(-v, dim=1, stable=True).indices[:, :kk]
        sv = valid.gather(1, o)
        rel = o - ks[:n, None].to(torch.int64)
        out[:n, :kk] = torch.where(sv, rel, torch.full_like(rel, -1)).to(out.dtype)
        if kk < k:
            out[:n, kk:k] = -1

    def top_k_per_row_prefill(logits, ks, ke, out, num_rows, s0, s1, k):
        n = int(num_rows)
        if ARGS.topk_probe and not torch.cuda.is_current_stream_capturing():
            orig(logits, ks, ke, out, num_rows, s0, s1, k)
            a = out[:n, :k].clone()
            orig(logits, ks, ke, out, num_rows, s0, s1, k)
            b = out[:n, :k]
            od = int((a != b).any(dim=1).sum())
            sd = int((a.sort(dim=1).values != b.sort(dim=1).values).any(dim=1).sum())
            TOPK["calls"] += 1
            TOPK["rows"] += n
            TOPK["order_drift"] += od
            TOPK["set_drift"] += sd
            if (od or sd) and TOPK["logged"] < 4:
                TOPK["logged"] += 1
                log(f"top-k probe: production's op twice on the same logits ({n} rows, k {k}): order differs in {od} "
                    f"rows, selected set differs in {sd} rows")
        else:
            orig(logits, ks, ke, out, num_rows, s0, s1, k)
        if ARGS.topk == "sort":
            canon_rows(logits, ks, out, n, int(k))
        else:
            full_topk(logits, ks, ke, out, n, int(k))

    ns.top_k_per_row_prefill = top_k_per_row_prefill
    log(f"top-k determinizer: torch.ops._C.top_k_per_row_prefill -> {ARGS.topk} (probe {ARGS.topk_probe})")


# ------------------------------------------------------------------------------------------------ sparse-MLA probe
M_SM90 = "vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90"


def _mla_reference(qn, kv_flat, slots, lens, sm_scale, k_scale, rows_per=96):
    """fp32 sparse MQA: out[t] = softmax(q[t] . K[s]^T * sm_scale) @ K[s] over s = slots[t, :lens[t]] (K = V = the
    fp8 latent row x k_scale). slots entries past the valid prefix are production's clamped -1 (slot 0)."""
    T, Hh, D = qn.shape
    W = slots.shape[1]
    out = torch.empty((T, Hh, D), dtype=torch.float32, device=qn.device)
    ar = torch.arange(W, device=qn.device)
    for r0 in range(0, T, rows_per):
        r1 = min(T, r0 + rows_per)
        sl = slots[r0:r1].long().clamp(min=0)
        K = kv_flat[sl.reshape(-1)].view(torch.float8_e4m3fn).float().reshape(r1 - r0, W, D) * k_scale
        q = qn[r0:r1].float()
        sc = torch.einsum("thd,twd->thw", q, K) * sm_scale
        m = (ar[None, :] < lens[r0:r1, None])[:, None, :]
        sc = sc.masked_fill(~m, float("-inf"))
        p = torch.softmax(sc, dim=-1)
        out[r0:r1] = torch.einsum("thw,twd->thd", p, K)
        del K, sc, p
    return out


def _rel_rows(a, ref):
    d = (a.float() - ref).flatten(1).norm(dim=1)
    n = ref.flatten(1).norm(dim=1).clamp(min=1e-30)
    return d / n


def _install_mla_probe():
    sm = sys.modules.get(M_SM90)
    if sm is None or not ARGS.mla_probe:
        return
    cls = sm.FlashInferMLASparseSM90Impl
    cur = cls.forward_mqa
    if getattr(cur, "_handoff_probe", False):
        return

    def forward_mqa(self, q, kv_cache, md, layer):
        num_tokens = q[1].shape[0] if isinstance(q, tuple) else q.shape[0]
        if (SH["phase"] != "F" or num_tokens < ARGS.shadow_min_t or torch.cuda.is_current_stream_capturing()
                or not isinstance(q, tuple)):
            return cur(self, q, kv_cache, md, layer)
        st = sm._SM90_STATE
        ki0 = st.kv_indices.clone()
        with ProdMode():
            o_fa2, _ = cur(self, q, kv_cache, md, layer)
        o_fa2 = o_fa2.clone()
        ki_prod = st.kv_indices.clone()
        st.kv_indices.copy_(ki0)
        out = cur(self, q, kv_cache, md, layer)
        o_inst = out[0]
        from vllm.v1.attention.backends.mla.sparse_utils import triton_convert_req_index_to_global_index
        ti = self.topk_indices_buffer[:num_tokens]
        slots, valid = triton_convert_req_index_to_global_index(
            md.req_id_per_token[:num_tokens], md.block_table, ti, BLOCK_SIZE=md.block_size,
            NUM_TOPK_TOKENS=ti.shape[1], return_valid_counts=True)
        valid = valid.to(torch.int64)
        planned = torch.as_tensor(st._lens_cpu[:num_tokens]).to(valid.device, torch.int64)
        qn = q[0]
        kv_flat = kv_cache.view(torch.uint8).reshape(-1, kv_cache.shape[-1])
        ks = float(getattr(layer, "_k_scale_float", 1.0) or 1.0)
        W = slots.shape[1]
        lens_p = planned.clamp(max=W)
        ref_v = _mla_reference(qn, kv_flat, slots, valid, float(self.scale), ks)
        ref_p = _mla_reference(qn, kv_flat, slots, lens_p, float(self.scale), ks)
        # FA2's own addressing: kv_indptr = row * W, kv_len = planned -> row t reads ki[t*W : t*W + planned[t]], i.e.
        # past its row into row t+1 (and, for the last row, past the rows this call wrote) whenever planned > W
        Wm = int(planned.max())
        pos_ = (torch.arange(num_tokens, device=valid.device)[:, None] * W
                + torch.arange(Wm, device=valid.device)[None, :]).clamp(max=ki_prod.numel() - 1)
        slots_fa2 = ki_prod[pos_]
        ref_o = _mla_reference(qn, kv_flat, slots_fa2, planned, float(self.scale), ks)
        over = (planned - W).clamp(min=0)
        pos = md.req_id_per_token.new_zeros(0)
        ctx = planned.clone()   # planned == ctx for ctx <= topk; bucket by planned > topk
        topk = int(ti.shape[1])
        long_ = planned > topk
        res = {"layer": getattr(layer, "layer_name", "?"), "tokens": int(num_tokens), "width": int(W),
               "planned_minus_valid": {str(int(k)): int(v) for k, v in zip(*torch.unique(planned - valid,
                                                                                        return_counts=True))},
               "slot0_content_absmax": float(kv_flat[0].view(torch.float8_e4m3fn).float().abs().max().item()),
               "kv_indices_written_by_installed": bool(not torch.equal(st.kv_indices, ki0)),
               # opt-kdamhc: with the bounded write-back (glm53_mla_prefill STATE.kv_rows) only the rows an FA2 call can
               # read beyond its own rows are written; compare those
               "kv_indices_equal_prod": bool(torch.equal(st.kv_indices[: _kv_rows(num_tokens) * W],
                                                         ki_prod[: _kv_rows(num_tokens) * W]))}
        res["rows_reading_past_their_row"] = int((over > 0).sum())
        res["last_row_reads_past_call"] = int(over[-1])
        for nm, o in (("fa2", o_fa2), ("installed", o_inst)):
            for rn, ref in (("ref_valid", ref_v), ("ref_planned", ref_p), ("ref_fa2_addressing", ref_o)):
                e = _rel_rows(o, ref)
                for bn, m in (("ctx<=topk", ~long_), ("ctx>topk", long_)):
                    if int(m.sum()):
                        res[f"{nm}_vs_{rn}[{bn}]"] = (float(e[m].mean()), float(e[m].max()), int(m.sum()))
        e = _rel_rows(ref_p, ref_v)
        if int(long_.sum()):
            res["ref_planned_vs_ref_valid[ctx>topk]"] = (float(e[long_].mean()), float(e[long_].max()), int(long_.sum()))
        res["installed_equals_fa2"] = bool(torch.equal(o_inst.contiguous().view(torch.uint8),
                                                       o_fa2.contiguous().view(torch.uint8)))
        SH["mla"].append(res)
        if len(SH["mla"]) <= 12:
            log("mla probe " + json.dumps(res))
        del ref_v, ref_p, ref_o
        return out

    forward_mqa._handoff_probe = True
    cls.forward_mqa = forward_mqa
    log(f"mla probe: wrapping {cls.__name__}.forward_mqa (installed: {getattr(cur, '__qualname__', cur)})")


# ------------------------------------------------------------------------------------------------ shadow
class ProdMode:
    """Force every installed fast path onto production's own statements (the gates each item already has)."""

    def __enter__(self):
        self.saved = []
        Q = sys.modules.get("glm53_prefill_quickwins")
        if Q is not None:
            self.saved.append((Q._STATE, "min_t", Q._STATE["min_t"]))
            self.saved.append((Q._HG, "off", Q._HG["off"]))
            Q._STATE["min_t"] = 1 << 60
            Q._HG["off"] = True
        M = sys.modules.get("glm53_mla_prefill")
        if M is not None:
            self.saved.append((M.STATE, None, M.STATE.min_tokens))
            M.STATE.min_tokens = 1 << 60
        # [glm53-kda-flashkda] the FlashKDA chunked-prefill wrapper (installed by
        # overlay/patch_flashkda.py; absent in a stock or switch-unset tree)
        F = sys.modules.get("glm53_flashkda")
        if F is not None:
            self.saved.append((F.STATE, "off", F.STATE["off"]))
            F.STATE["off"] = True
        return self

    def __exit__(self, *exc):
        for obj, key, val in reversed(self.saved):
            if key is None:
                obj.min_tokens = val
            else:
                obj[key] = val
        return False


def _out_list(o):
    out = []
    _flat(o, out, "out")
    return out


def _bytes_equal(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def _elem_report(p, f, dtype_hint=None):
    """element-level diff of two same-shaped tensors: (n differing, numel, max |diff| as floats, first index)."""
    pb, fb = p.contiguous().view(torch.uint8), f.contiguous().view(torch.uint8)
    if p.element_size() > 1:
        wide = {2: torch.int16, 4: torch.int32, 8: torch.int64}[p.element_size()]
        ne = p.contiguous().view(wide) != f.contiguous().view(wide)
    else:
        ne = pb != fb
    n = int(ne.sum().item())
    first = tuple(int(x) for x in torch.nonzero(ne)[0].tolist()) if n else None
    mad = float("nan")
    try:
        if p.is_floating_point():
            mad = float((p.float() - f.float()).abs().max().item())
        elif dtype_hint is not None:
            mad = float((p.view(dtype_hint).float() - f.view(dtype_hint).float()).abs().max().item())
    except Exception:  # noqa: BLE001
        pass
    return n, int(ne.numel()), mad, first


def _req_blocks():
    """{layer name: set(logical block ids held by the active requests in that layer's KV group)}"""
    out = {}
    try:
        kvc = H["runner"].kv_cache_config
        for gi, g in enumerate(kvc.kv_cache_groups):
            rows = set()
            for blocks in H["blocks"].values():
                if gi < len(blocks):
                    rows.update(blocks[gi])
            for ln in g.layer_names:
                out[ln] = rows
    except Exception as e:  # noqa: BLE001
        log("req blocks:", repr(e))
    return out


def _owned_rows(vname, nrows):
    """view rows (kernel blocks) that belong to the active requests: row r is logical block r // (rows / num_blocks)"""
    nb = H["runner"].kv_cache_config.num_blocks
    blocks = _req_blocks().get(vname.split("[")[0], set())
    f = nrows // nb if nb and nrows % nb == 0 else 1
    return {b * f + i for b in blocks for i in range(f)}


def _cmp_state(st, P_state, max_rows=6):
    """Compare the live state tensors st [(name, t)] with P_state clones. Returns a report dict."""
    rep = {"equal": True, "owned_equal": True, "buffers": {}, "views": {}}
    raw_ptrs = {}
    for (name, t), p in zip(st, P_state):
        if _bytes_equal(t, p):
            continue
        rep["equal"] = False
        rep["buffers"][name] = True
        if name.startswith("raw:"):
            raw_ptrs[t.untyped_storage().data_ptr()] = (t, p)
        elif name == "sm90.kv_indices" or name.startswith("topk_indices_buffer"):
            ne = torch.nonzero(t.view(-1) != p.view(-1)).flatten()
            rep["buffers"][name] = {"n": int(ne.numel()), "first": int(ne[0]), "last": int(ne[-1])}
        else:
            n, tot, mad, first = _elem_report(p, t)
            rep["buffers"][name] = {"n": n, "numel": tot, "first": first}
    if raw_ptrs:
        for vname, v in views():
            sp = v.untyped_storage().data_ptr()
            if sp not in raw_ptrs:
                continue
            _t, praw = raw_ptrs[sp]
            base = praw.view(torch.uint8).reshape(-1)
            try:
                pv = base.view(v.dtype).as_strided(v.shape, v.stride(), v.storage_offset())
            except Exception as e:  # noqa: BLE001
                rep["views"][vname] = {"error": repr(e)}
                continue
            hf, hp = view_row_hashes(v), view_row_hashes(pv)
            rows = torch.nonzero(hf != hp).flatten().tolist()
            if not rows:
                continue
            owned = _owned_rows(vname, int(v.shape[0]))
            mine = [r for r in rows if r in owned]
            det = []
            hint = torch.float8_e4m3fn if v.dtype == torch.uint8 and "indexer" not in vname else None
            for r in (mine or rows)[:max_rows]:
                n, tot, mad, first = _elem_report(pv[r], v[r], hint)
                det.append({"row": r, "owned": r in owned, "n": n, "numel": tot, "max_abs": mad, "first": first,
                            "row_shape": list(v[r].shape), "dtype": str(v.dtype)})
            rep["views"][vname] = {"rows": rows[:64], "n_rows": len(rows), "owned_rows": mine[:64],
                                   "n_owned": len(mine), "detail": det}
            if mine:
                rep["owned_equal"] = False
    return rep


def _cmp_out(P_out, out):
    res = []
    for (name, a), (_n2, b) in zip(P_out, _out_list(out)):
        if _bytes_equal(a, b):
            res.append({"name": name, "equal": True})
        else:
            n, tot, mad, first = _elem_report(a, b)
            res.append({"name": name, "equal": False, "n": n, "numel": tot, "max_abs": mad, "first": first})
    return res


def _hook_modules(model):
    pat = re.compile(r"^(?:model\.)?layers\.(\d+)(\.self_attn|\.mlp)?$")

    def mk(name):
        def hook(_m, _inp, out):
            tr = SH["trace"]
            if tr is None:
                return
            hs = []
            for _, x in _out_list(out):
                hs.append(tensor_hash(x) if x.numel() else 0)
            tr.append((name, hs))
        return hook

    n = 0
    for name, m in model.named_modules():
        if pat.match(name):
            m.register_forward_hook(mk(name))
            n += 1
    log(f"shadow: forward hooks on {n} modules")


def _first_trace_diff(tp, tf):
    for (na, ha), (nb, hb) in zip(tp, tf):
        if na != nb:
            return {"module": na, "why": f"order differs ({nb})"}
        if ha != hb:
            return {"module": na, "outputs_differ": [i for i, (x, y) in enumerate(zip(ha, hb)) if x != y]}
    if len(tp) != len(tf):
        return {"why": f"trace lengths {len(tp)} vs {len(tf)}"}
    return None


def _wrap_model_forward():
    runner = H["runner"]
    m = runner.model
    for attr in ("runnable", "model", "module"):
        inner = getattr(m, attr, None)
        if isinstance(inner, torch.nn.Module) and type(m).__name__ != "Glm5NextForCausalLM":
            m = inner
    cls = type(m)
    orig = cls.forward
    if getattr(orig, "_handoff_shadow", False):
        return
    log(f"shadow: wrapping {cls.__module__}.{cls.__name__}.forward")

    def forward(self, input_ids, positions, *a, **k):
        T = int(positions.shape[0])
        if (not ARGS.shadow or not H["active"] or T < ARGS.shadow_min_t
                or torch.cuda.is_current_stream_capturing()):
            return orig(self, input_ids, positions, *a, **k)
        if not SH["hooked"]:
            _hook_modules(self)
            SH["hooked"] = True
        torch.cuda.synchronize()
        st = [(f"raw:{ln}", t) for ln, t in uniq_raw()] + side_tensors()
        snap = [t.clone() for _, t in st]
        c0 = fast_path_counters()
        SH["trace"] = []
        SH["phase"] = "P"
        with ProdMode():
            out_p = orig(self, input_ids, positions, *a, **k)
        torch.cuda.synchronize()
        tr_p, SH["trace"] = SH["trace"], None
        c1 = fast_path_counters()
        P_out = [(n, x.clone()) for n, x in _out_list(out_p)]
        del out_p
        P_state = [t.clone() for _, t in st]
        rec = {"step": H["step"], "tokens": T, "fast_P": _delta(c0, c1)}
        if ARGS.shadow_control:
            for (_, t), s in zip(st, snap):
                t.copy_(s)
            SH["trace"] = []
            SH["phase"] = "C"
            with ProdMode():
                out_c = orig(self, input_ids, positions, *a, **k)
            torch.cuda.synchronize()
            tr_c, SH["trace"] = SH["trace"], None
            rec["control"] = {"out": _cmp_out(P_out, out_c), "state": _cmp_state(st, P_state),
                              "first_module": _first_trace_diff(tr_p, tr_c)}
            del out_c
        for (_, t), s in zip(st, snap):
            t.copy_(s)
        del snap
        c2 = fast_path_counters()
        SH["trace"] = []
        SH["phase"] = "F"
        out_f = orig(self, input_ids, positions, *a, **k)
        SH["phase"] = None
        torch.cuda.synchronize()
        tr_f, SH["trace"] = SH["trace"], None
        c3 = fast_path_counters()
        rec["fast_F"] = _delta(c2, c3)
        rec["feature"] = {"out": _cmp_out(P_out, out_f), "state": _cmp_state(st, P_state),
                          "first_module": _first_trace_diff(tr_p, tr_f)}
        del P_state, P_out
        SH["steps"].append(rec)
        _log_shadow(rec)
        return out_f

    forward._handoff_shadow = True
    cls.forward = forward
    SH["wrapped"] = True
    _install_mla_probe()


def _log_shadow(rec):
    def verdict(r):
        if r is None:
            return "-"
        oe = all(x["equal"] for x in r["out"])
        st = r["state"]
        sv = ("IDENTICAL" if st["equal"] else "DIFFERS IN REQUEST-OWNED ROWS" if not st["owned_equal"]
              else "differs only in side buffers / rows of no active request")
        return (f"outputs {'IDENTICAL' if oe else 'DIFFER'}, state {sv}"
                f"{'' if r['first_module'] is None else ', first module ' + json.dumps(r['first_module'])}")
    log(f"shadow step {rec['step']} ({rec['tokens']} tokens): fast paths in F {rec['fast_F']} (in P {rec['fast_P']})")
    log(f"   control P vs P: {verdict(rec.get('control'))}")
    log(f"   feature P vs F: {verdict(rec['feature'])}")
    for tag in ("control", "feature"):
        r = rec.get(tag)
        if r is None:
            continue
        for o in r["out"]:
            if not o["equal"]:
                log(f"     [{tag}] output {o['name']}: {o['n']}/{o['numel']} elements, max|diff| {o['max_abs']:.4g}, "
                    f"first {o['first']}")
        for b, v in r["state"]["buffers"].items():
            if not b.startswith("raw:"):
                log(f"     [{tag}] buffer {b}: {v}")
        for vn, v in list(r["state"]["views"].items())[:40]:
            if "error" in v:
                log(f"     [{tag}] view {vn}: {v['error']}")
                continue
            log(f"     [{tag}] view {vn}: {v['n_rows']} rows differ {v['rows'][:12]}; request-owned {v['owned_rows'][:12]}")
            for d in v["detail"][:3]:
                log(f"         row {d['row']}{' (owned)' if d['owned'] else ''}: {d['n']}/{d['numel']} elements, "
                    f"max|diff| {d['max_abs']:.4g}, first {d['first']} (row {d['row_shape']} {d['dtype']})")


# ------------------------------------------------------------------------------------------------ per-step records
def snapshot(rec):
    torch.cuda.synchronize()
    nb = H["runner"].kv_cache_config.num_blocks
    vh = {}
    for name, t in views():
        vh[name] = view_row_hashes(t).cpu()
    rh = {}
    for ln, raw in uniq_raw():
        rh[ln] = raw_block_hashes(raw, nb).cpu()
    rec["side"] = side_buffers(rec["tokens"])
    rec["fast"] = fast_path_counters()
    rec["view_hash"] = vh
    rec["raw_hash"] = rh
    if rec["dump"]:
        d = {}
        nl = int(H["runner"].model_config.hf_config.num_hidden_layers)
        for name, t in views():
            if _layer_idx(name) >= nl:        # drafter KV: hashed every step, not dumped (size)
                continue
            h = vh[name]
            rows = torch.nonzero(h != 0).flatten().tolist()
            if rows:
                idx = torch.tensor(rows, device=t.device)
                d[name] = {"rows": rows, "data": t.index_select(0, idx).contiguous().cpu(), "shape": tuple(t.shape),
                           "dtype": str(t.dtype)}
        p = os.path.join(ARGS.out, f"dump_step{rec['step']:03d}.pt")
        torch.save(d, p)
        H["dumped"].append(p)
    torch.cuda.synchronize()


_orig_exec = MR.GPUModelRunner.execute_model
_orig_sample = MR.GPUModelRunner.sample_tokens


def _track_blocks(so):
    for r in so.scheduled_new_reqs:
        H["blocks"][r.req_id] = [list(b) for b in r.block_ids]
    cr = so.scheduled_cached_reqs
    for i, rid in enumerate(cr.req_ids):
        nb = cr.new_block_ids[i] if cr.new_block_ids is not None else None
        if nb is None:
            continue
        cur = H["blocks"].setdefault(rid, [[] for _ in nb])
        for g, b in enumerate(nb):
            if b:
                cur[g].extend(b)
    for rid in list(H["blocks"]):            # prefixhit: block ids per request, kept after the request finishes
        H.setdefault("bh", {})[rid] = [list(g) for g in H["blocks"][rid]]
    for rid in getattr(so, "finished_req_ids", ()) or ():
        H["blocks"].pop(rid, None)


def _exec(self, scheduler_output, *a, **k):
    H["runner"] = self
    if os.environ.get("HANDOFF_TAIL_PROBE", "").strip() == "1":
        try:
            _install_tail_probe()
        except Exception as e:  # noqa: BLE001
            log("tail probe install:", repr(e))
        if H["active"]:
            _bt = None
            try:
                _bt = {rid: [list(b)[:1] for b in v][1] for rid, v in H["blocks"].items()}
            except Exception:  # noqa: BLE001
                pass
            if H["step"] < 400:
                log(f"tail probe step {H['step']}: own tail blocks {_bt}")
    if ARGS.shadow and not SH["wrapped"] and getattr(self, "model", None) is not None:
        _wrap_model_forward()
    try:
        _track_blocks(scheduler_output)
    except Exception as e:  # noqa: BLE001
        log("block tracking:", repr(e))
    if H["active"] and not k.get("dummy_run", False) and scheduler_output.total_num_scheduled_tokens > 0:
        so = scheduler_output
        comp = {}
        for r in so.scheduled_new_reqs:
            comp[r.req_id] = int(r.num_computed_tokens)
        cr = so.scheduled_cached_reqs
        for rid, n in zip(cr.req_ids, cr.num_computed_tokens):
            comp[rid] = int(n)
        spec = {rid: len(v) for rid, v in (so.scheduled_spec_decode_tokens or {}).items()}
        sched = {rid: int(n) for rid, n in so.num_scheduled_tokens.items()}
        is_prefill = any(sched[r] - spec.get(r, 0) > 1 for r in sched)
        if not is_prefill:
            H["decode_steps"] += 1
        dump = bool(ARGS.dump_decode) and (is_prefill or H["decode_steps"] <= ARGS.dump_decode)
        H["cur"] = {"step": H["step"], "sched": sched, "computed": comp, "spec": spec, "prefill": is_prefill,
                    "tokens": int(so.total_num_scheduled_tokens), "dump": dump, "t": time.time(),
                    "fast0": fast_path_counters()}
        if H.get("ph_dump") and is_prefill and len(sched) == 1:
            try:
                _ph_dump(so, sched, comp)
            except Exception as e:  # noqa: BLE001
                log("ph dump:", repr(e))
            rid0, n0 = next(iter(sched.items()))
            if os.environ.get("HANDOFF_PH_TRACE", "").strip() == "1" and comp[rid0] + n0 == H.get("ph_target"):
                _install_ph_trace()
                _install_ph_trace2()
                H["ph_trace"] = {"kda": {}, "mla": {}, "n": n0, "computed": comp[rid0],
                                 "path": H["ph_dump"].replace("phdump_", "phtrace_")}
    return _orig_exec(self, scheduler_output, *a, **k)


def _install_tail_probe():
    """prefixhit (HANDOFF_TAIL_PROBE=1): log, per step, the tail-ring block ids the kpool tail writers use
    (slot // kpool of the prefill tail tokens / of every decode token) next to each request's own tail block."""
    import vllm.model_executor.layers.sparse_attn_indexer_kpool as KP
    if getattr(KP, "_ph_tail_probe", False):
        return
    KP._ph_tail_probe = True
    seed0, dec0 = KP.kpool_seed_tail_cache, KP.kpool_decode_update_and_maybe_write_cache_batched

    def seed(tail_kv_cache, k, gate, slot_mapping, kpool, head_dim, *a, **kw):
        st = H.setdefault("tail_probe", {})
        if st.get("step") != H["step"] and not torch.cuda.is_current_stream_capturing():
            sm = slot_mapping[-kpool:].to(torch.int64).cpu().tolist()
            st.update(step=H["step"], logged=True)
            log(f"tail probe step {H['step']}: prefill seed of the batch's last {kpool} tokens -> tail blocks "
                f"{sorted(set(x // kpool for x in sm if x >= 0))} (slots {sm}); n_tokens {int(slot_mapping.shape[0])}")
        return seed0(tail_kv_cache, k, gate, slot_mapping, kpool, head_dim, *a, **kw)

    def dec(kv_raw, tail_kv_cache, dec_tail_slot, *a, **kw):
        st = H.setdefault("tail_probe", {})
        if st.get("dstep") != H["step"] and not torch.cuda.is_current_stream_capturing():
            st["dstep"] = H["step"]
            sm = dec_tail_slot.to(torch.int64).cpu()
            log(f"tail probe step {H['step']}: decode tail slots per request -> blocks "
                f"{[sorted(set(int(x) // 4 for x in row.flatten().tolist() if x >= 0)) for row in sm]}")
        return dec0(kv_raw, tail_kv_cache, dec_tail_slot, *a, **kw)

    KP.kpool_seed_tail_cache = seed
    KP.kpool_decode_update_and_maybe_write_cache_batched = dec
    # which logical blocks' indexer page 0 (bytes [b*P, b*P + 8448) of the shared indexer allocation) each writer
    # changes, for the FIRST indexer allocation seen (one MLA layer)
    W8 = {}

    def _pages(t):
        if torch.cuda.is_current_stream_capturing() or not H["active"]:
            return None
        if H["runner"] is None or getattr(H["runner"], "kv_cache_config", None) is None or not H["kv"]:
            return None
        nb = int(H["runner"].kv_cache_config.num_blocks)
        if "v" not in W8:
            W8["v"] = dict(views())["model.layers.2.self_attn.indexer.k_cache"]
        v = W8["v"]
        return v.reshape(nb, -1, v.shape[1] * v.shape[2])[:, 0].to(torch.int32).sum(1)

    def wrap(name, fn, argi):
        def w(*a, **kw):
            t = a[argi]
            before = _pages(t)
            out = fn(*a, **kw)
            if before is not None:
                after = _pages(t)
                ch = (after != before).nonzero().flatten().tolist()
                if ch:
                    log(f"write probe step {H['step']}: {name} changed indexer page0 of blocks {ch[:24]}")
            return out
        return w

    KP.kpool_seed_tail_cache = wrap("tail_seed", KP.kpool_seed_tail_cache, 0)
    KP.kpool_decode_update_and_maybe_write_cache_batched = wrap("decode_update", KP.kpool_decode_update_and_maybe_write_cache_batched, 1)
    KP.kpool_compress_and_write_cache = wrap("prefill_compress", KP.kpool_compress_and_write_cache, 0)
    log("tail probe: installed")


def _install_ph_trace2():
    """prefixhit: module-global hooks (fire inside breakable-graph replays too, unlike class-method patches, because
    the recorded eager-break callables look their callees up at run time)."""
    if H.get("ph_trace2_installed"):
        return
    H["ph_trace2_installed"] = True
    import vllm.models.glm5next.nvidia.kda as KM
    import vllm.model_executor.layers.sparse_attn_indexer_kpool as KP

    def rec(kind, x):
        tr = H.get("ph_trace")
        if tr is None or torch.cuda.is_current_stream_capturing():
            return
        if isinstance(x, (tuple, list)):
            x = x[0]
        tr.setdefault(kind, []).append(x.detach().float().cpu().clone())

    def wrap(mod, name, kind, pick_out=True):
        f = getattr(mod, name, None)
        if f is None:
            log(f"ph trace2: {mod.__name__}.{name} absent")
            return

        def w(*a, **kw):
            out = f(*a, **kw)
            try:
                rec(kind, out)
            except Exception as e:  # noqa: BLE001
                log("ph trace2 rec:", kind, repr(e))
            return out
        setattr(mod, name, w)

    wrap(KM, "causal_conv1d_fn", "kda_conv")
    wrap(KM, "gather_initial_states", "kda_init")
    wrap(KM, "chunk_kda_with_fused_gate", "kda_chunk")
    wrap(KM, "fused_recurrent_kda", "kda_rec")
    try:
        import glm53_flashkda as FK
        wrap(FK, "chunk_prefill", "kda_fkda")
    except Exception as e:  # noqa: BLE001
        log("ph trace2: no glm53_flashkda:", repr(e))
    wrap(KP, "expand_pools_and_append_tail", "idx_topk")
    log("ph trace2: hooks installed")


def _install_ph_trace():
    """prefixhit: per-layer capture of the KDA layers' read state + output and the sparse-MLA layers' top-k + output
    during the traced step (H['ph_trace'] not None). Both are eager-break ops (Python runs every call)."""
    if H.get("ph_trace_installed"):
        return
    H["ph_trace_installed"] = True
    from vllm.models.glm5next.nvidia.kda import Glm5NextLinearAttention as KDA
    kf = KDA._forward

    def _forward(self, qkv_proj_states, g1, beta, core_attn_out):
        tr = H.get("ph_trace")
        if tr is None or torch.cuda.is_current_stream_capturing():
            return kf(self, qkv_proj_states, g1, beta, core_attn_out)
        md = get_forward_context().attn_metadata[self.prefix]
        conv_state, rec = self.kv_cache
        idx = md.non_spec_state_indices_tensor
        i0 = int(idx[0]) if idx is not None else -1
        pre = {"slot": i0, "conv": conv_state[i0].float().cpu().clone(), "rec": rec[i0].float().cpu().clone(),
               "has_init": md.has_initial_state.cpu().clone() if md.has_initial_state is not None else None,
               "num_prefills": int(md.num_prefills), "num_decodes": int(md.num_decodes),
               "qkv_in": qkv_proj_states[: tr["n"]].float().cpu().clone()}
        out = kf(self, qkv_proj_states, g1, beta, core_attn_out)
        pre["out"] = core_attn_out[0, : tr["n"]].float().cpu().clone()
        pre["rec_after"] = rec[i0].float().cpu().clone()
        tr["kda"][self.prefix] = pre
        return out

    KDA._forward = _forward
    from vllm.forward_context import get_forward_context  # noqa: F401
    globals()["get_forward_context"] = get_forward_context
    sm = sys.modules.get(M_SM90)
    if sm is not None:
        cls = sm.FlashInferMLASparseSM90Impl
        cur = cls.forward_mqa

        def forward_mqa(self, q, kv_cache, md, layer):
            tr = H.get("ph_trace")
            out = cur(self, q, kv_cache, md, layer)
            if tr is not None and not torch.cuda.is_current_stream_capturing():
                n = tr["n"]
                o = out[0] if isinstance(out, tuple) else out
                tr["mla"][getattr(layer, "layer_name", str(len(tr["mla"])))] = {
                    "topk": self.topk_indices_buffer[:n].cpu().clone(),
                    "q": (q[0] if isinstance(q, tuple) else q)[:n].float().cpu().clone(),
                    "out": o[:n].float().cpu().clone()}
            return out

        cls.forward_mqa = forward_mqa
    log("ph trace: hooks installed")


def _ph_dump(so, sched, comp):
    """prefixhit (HANDOFF_PH_DUMP=1): before the LAST prefill step of a PH request (the one that reaches the prompt
    end), save every KV row that step reads for the prefix: group-0 blocks [0, computed/bs), the mamba column holding
    the state at `computed`, the request's kpool tail block. One file per PH arm (H['ph_dump'] = path prefix)."""
    rid, n = next(iter(sched.items()))
    c = comp[rid]
    tgt = H.get("ph_target")
    if tgt is None or c + n != tgt:
        return
    kvc = H["runner"].kv_cache_config
    blocks = H["blocks"].get(rid)
    l2g = {}
    for gi, g in enumerate(kvc.kv_cache_groups):
        for ln in g.layer_names:
            l2g[ln] = (gi, g.kv_cache_spec)
    torch.cuda.synchronize()
    d = {"rid": rid, "computed": c, "n": n, "blocks": [list(b) for b in blocks], "views": {}}
    for name, t in views():
        base = name.split("[")[0]
        if base not in l2g:
            continue
        gi, spec = l2g[base]
        bs = int(spec.block_size)
        if type(spec).__name__ == "MambaSpec":
            cols = [c // bs - 1, (c + n - 1) // bs]
        elif bs == 4:
            cols = [0]
        elif _layer_idx(name) >= int(H["runner"].model_config.hf_config.num_hidden_layers):
            continue                                    # drafter
        else:
            cols = list(range(min(len(blocks[gi]), (c + bs - 1) // bs)))
        ids = [blocks[gi][x] for x in cols if x < len(blocks[gi])]
        if not ids:
            continue
        m = int(t.shape[0]) // int(kvc.num_blocks)      # kernel blocks per logical block (MLA 72, kpool indexer 18)
        rows = [b * m + i for b in ids for i in range(m)]
        idx = torch.tensor(rows, device=t.device)
        d["views"][name] = {"gid": gi, "cols": cols, "ids": ids, "m": m,
                            "data": t.index_select(0, idx).contiguous().cpu()}
    torch.save(d, H["ph_dump"])
    log(f"ph dump: {H['ph_dump']} rid {rid} computed {c} n {n} views {len(d['views'])}")


def _stale_probe(rec):
    """After a step: FA2's plan (kv_len per row) against the kv_indices row stride W. A row with kv_len > W reads
    kv_len - W entries past its row; the last row of the step reads them past the rows this step wrote, i.e. whatever
    an earlier forward_mqa left in the process-wide kv_indices. Records those slot ids and whether they belong to the
    request (its MLA blocks) or not."""
    sm = sys.modules.get(M_SM90)
    st = getattr(sm, "_SM90_STATE", None) if sm is not None else None
    if st is None or getattr(st, "_lens_cpu", None) is None:
        return None
    W = int(st.topk_width)
    M = int(rec["tokens"])
    lens = st._lens_cpu[:M].tolist()
    over = [max(0, int(x) - W) for x in lens]
    r = over[-1] if over else 0
    slots = st.kv_indices[M * W: M * W + r].tolist() if r else []
    own_blocks = set()
    try:
        kvc = H["runner"].kv_cache_config
        g0 = [gi for gi, g in enumerate(kvc.kv_cache_groups) if any(n.endswith(".attn") and _layer_idx(n) < int(
            H["runner"].model_config.hf_config.num_hidden_layers) for n in g.layer_names)]
        bs = int(kvc.kv_cache_groups[g0[0]].kv_cache_spec.block_size) if g0 else 0
        for blocks in H["blocks"].values():
            if g0 and g0[0] < len(blocks):
                own_blocks.update(blocks[g0[0]])
    except Exception as e:  # noqa: BLE001
        bs = 0
        log("stale probe:", repr(e))
    own = [bool(bs) and (int(x) // bs) in own_blocks and int(x) > 0 for x in slots]
    return {"W": W, "rows": M, "rows_over": sum(1 for x in over if x), "last_row_over": r, "stale_slots": slots,
            "stale_own": own}


def _sample(self, grammar_output, *a, **k):
    out = _orig_sample(self, grammar_output, *a, **k)
    tr = H.get("ph_trace")
    if tr is not None:
        torch.cuda.synchronize()
        torch.save(tr, tr["path"])
        log(f"ph trace: {len(tr['kda'])} kda + {len(tr['mla'])} mla layers; "
            f"{ {k: len(v) for k, v in tr.items() if isinstance(v, list)} } -> {tr['path']}")
        H["ph_trace"] = None
    rec = H["cur"]
    if rec is not None:
        H["cur"] = None
        try:
            rec["stale"] = _stale_probe(rec)
        except Exception as e:  # noqa: BLE001
            rec["stale"] = {"error": repr(e)}
        snapshot(rec)
        rec["fast_step"] = _delta(rec.pop("fast0"), rec["fast"])
        H["records"].append(rec)
        log(f"step {rec['step']:3d} {'PREFILL' if rec['prefill'] else 'decode '} sched {rec['sched']} "
            f"computed {rec['computed']} drafts {rec['spec']} fast paths this step {rec['fast_step']}"
            + (f" | FA2 plan: {rec['stale']['rows_over']}/{rec['stale']['rows']} rows read past their row; last row "
               f"reads {rec['stale']['last_row_over']} entries past the step: slots {rec['stale']['stale_slots']} "
               f"own {rec['stale']['stale_own']}" if rec.get("stale") and "W" in rec["stale"] else ""))
        H["step"] += 1
    return out


# opt-decode-rev: HANDOFF_HOSTLOOP_UNWRAP=1 exposes the original execute_model to glm53_hostloop's source-fingerprint
# check (its getattr(fn, "_glm53_orig", fn)); without it the check sees this harness wrapper and GLM53_DEC_HOSTLOOP
# never installs in this rig ("unverified vLLM source: GPUModelRunner.execute_model=..."). Default: unchanged.
if os.environ.get("HANDOFF_HOSTLOOP_UNWRAP", "").strip() == "1":
    _exec._glm53_orig = _orig_exec
    _sample._glm53_orig = _orig_sample
MR.GPUModelRunner.execute_model = _exec
MR.GPUModelRunner.sample_tokens = _sample


# ------------------------------------------------------------------------------------------------ prefixhit
def _prefixhit_b(llm, reqs, prompts):
    """prefixhit: HANDOFF_B_LENS="L1,L2,..." -> for every absolute length L, seq = A0's text extended to L tokens
    (prompt_ids_ext; seq[:len(A0 prompt)] == A0's prompt), run as
      fresh  (unique salt: no prefix hit)        hit  (A0's salt: hits what A0 / earlier B's cached)
    and, with HANDOFF_B_REPEAT=1, also fresh2 (another unique salt) and hit2 (A0's salt again) = the noise floors.
    HANDOFF_B_GEN=N (default 1): greedy tokens per request, top-HANDOFF_B_LOGPROBS (default 20) logprobs each.
    HANDOFF_B_HITSALT=<s>: the salt of the hit arm (default A0's 'handoff-A0').
    Steps are logged (sched / computed per step) so the chunk structure of every arm is visible."""
    from vllm import SamplingParams
    lens = [int(x) for x in os.environ["HANDOFF_B_LENS"].split(",") if x.strip()]
    gen_n = int(os.environ.get("HANDOFF_B_GEN", "1") or 1)
    nlp = int(os.environ.get("HANDOFF_B_LOGPROBS", "20") or 20)
    rep = os.environ.get("HANDOFF_B_REPEAT", "").strip() == "1"
    hit_salt = os.environ.get("HANDOFF_B_HITSALT", "").strip() or "handoff-A0"
    order = os.environ.get("HANDOFF_B_ORDER", "fresh,hit").split(",")
    if rep:
        order = order + ["fresh2", "hit2"]
    i0, n0 = prompts[0]
    ext = prompt_ids_ext(n0, i0, max(lens + [n0]))
    assert ext[:n0] == reqs[0]["prompt_ids"]
    sp = SamplingParams(temperature=0.0, max_tokens=gen_n, ignore_eos=True, logprobs=nlp, detokenize=False, seed=0)
    out = []
    for j, L in enumerate(lens):
        seq = ext[:L]
        rec = {"len": L}
        for arm in order:
            salt = hit_salt if arm.startswith("hit") else f"handoff-PH{j}-{arm}"
            H["active"] = True
            if os.environ.get("HANDOFF_PH_DUMP", "").strip() == "1" and arm in ("fresh", "hit", "hit2", "fresh2"):
                H["ph_dump"] = os.path.join(ARGS.out, f"phdump_L{L}_{arm}.pt")
                H["ph_target"] = L
            s0 = H["step"]
            ro = llm.generate([{"prompt_token_ids": seq, "cache_salt": salt}], sp, use_tqdm=False)[0]
            H["active"] = False
            H["ph_dump"] = None
            ob = ro.outputs[0]
            _rid = str(ro.request_id)
            _bh = [v for k, v in H.get("bh", {}).items() if k == _rid or k.startswith(_rid + "-")]
            rec[arm] = {"cached": int(getattr(ro, "num_cached_tokens", -1) or 0), "req_id": _rid,
                        "blocks": _bh[-1] if _bh else None,
                        "gen": [int(x) for x in ob.token_ids],
                        "lp": [{str(int(t)): float(v.logprob) for t, v in d.items()} for d in (ob.logprobs or [])],
                        "steps": [s0, H["step"]],
                        "sched": [(r["sched"], r["computed"]) for r in H["records"] if s0 <= r["step"] < H["step"]]}
            log(f"PH L={L} {arm:6s} salt {salt}: cached {rec[arm]['cached']} gen {rec[arm]['gen'][:4]} "
                f"steps {s0}..{H['step'] - 1}")
        out.append(rec)
    return out


def _pool_entries(blocks_g0, npos):
    """prefixhit: the pooled indexer entries (values 128 B + scale 4 B per pool) of positions [0, npos) for every
    MLA layer, read through the group-0 block list (pool p -> logical block p // ppb, entry p % ppb; a kernel page
    holds 64 entries as [64 x 128 B values][64 x 4 B scales])."""
    out = {}
    nb = int(H["runner"].kv_cache_config.num_blocks)
    for name, t in views():
        if not name.endswith("indexer.k_cache"):
            continue
        m = int(t.shape[0]) // nb                     # kernel pages per logical block (18)
        ppb = m * 64                                  # pools per logical block (1152)
        page = t.reshape(t.shape[0], -1)              # [pages, 8448]
        npool = npos // 4
        pidx = torch.arange(npool, device=t.device)
        blk = torch.tensor(blocks_g0, device=t.device)[pidx // ppb]
        e = pidx % ppb
        row = blk * m + e // 64
        ein = e % 64
        vals = page[row[:, None], (ein * 128)[:, None] + torch.arange(128, device=t.device)[None, :]]
        scl = page[row[:, None], (64 * 128 + ein * 4)[:, None] + torch.arange(4, device=t.device)[None, :]]
        out[name] = torch.cat([vals, scl], 1).cpu()
    return out


def _prefixhit_conc(llm, reqs):
    """prefixhit HANDOFF_CONC_TEST=1: the pooled indexer keys every A request WROTE (prefill + decode, possibly while
    another request ran) against the keys a fresh solo re-prefill of prompt + generated tokens computes (prefill
    compress straight from k, no tail ring). Per request: pools that differ, split prompt region / generated region."""
    from vllm import SamplingParams
    torch.cuda.synchronize()
    got = []
    for r in reqs:
        rid = r.get("rid")
        bl = H["bh"].get(rid) if rid else None
        n = len(r["prompt_ids"]) + len(r["gen"]) - 1          # last generated token has no KV yet
        got.append(_pool_entries(bl[0], n) if bl else None)
    res = []
    sp = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True, detokenize=False)
    for i, r in enumerate(reqs):
        seq = r["prompt_ids"] + r["gen"]
        H["active"] = True
        ro = llm.generate([{"prompt_token_ids": seq, "cache_salt": f"handoff-CONC{i}"}], sp, use_tqdm=False)[0]
        H["active"] = False
        _rid = str(ro.request_id)
        bl = [v for k, v in H["bh"].items() if k == _rid or k.startswith(_rid + "-")][-1]
        n = len(r["prompt_ids"]) + len(r["gen"]) - 1
        ref = _pool_entries(bl[0], n)
        torch.save({"got": got[i], "ref": ref, "prompt_len": len(r["prompt_ids"]), "n": n},
                   os.path.join(ARGS.out, f"conc_req{i}.pt"))
        P = len(r["prompt_ids"]) // 4
        row = {"req": i, "prompt_pools": P, "gen_pools": n // 4 - P}
        for name in sorted(ref):
            a, b = got[i][name], ref[name]
            d = (a != b).any(1)
            row[name] = {"prompt_diff": int(d[:P].sum()), "gen_diff": int(d[P:].sum()),
                         "gen_diff_idx": d[P:].nonzero().flatten().tolist()[:12]}
        res.append(row)
        log(f"conc test req {i}: prompt pools {P} gen pools {row['gen_pools']}: " + "; ".join(
            f"{k.split('.')[2]} prompt {v['prompt_diff']} gen {v['gen_diff']}" for k, v in row.items() if isinstance(v, dict)))
    return res


# ------------------------------------------------------------------------------------------------ run
def main():
    from vllm import LLM, SamplingParams
    flags = {k: v for k, v in sorted(os.environ.items()) if k.startswith(("GLM53_", "TF_EXL3", "VLLM_"))}
    log(f"== handoff run {ARGS.label!r}: args {json.dumps(vars(ARGS))}")
    log(f"   flags {json.dumps(flags)}")
    k_set = sorted(int(x) for x in os.environ.get("GLM53_ADAPTIVE_K_SET", "4,5,7").split(",") if x.strip())
    sizes = {1, 2, 4, 8, 16, 24, 32}
    sizes.update(n * (k + 1) for n in range(1, 5) for k in k_set + [7])
    spec = {"method": "dflash", "model": ARGS.model + "-dflash2", "num_speculative_tokens": 7,
            "kv_cache_dtype": "auto", "draft_sample_method": "probabilistic", "rejection_sample_method": "standard"}
    # prefixhit: HANDOFF_NO_SPEC=1 -> no speculative decoding at all (isolates DFlash2 from the prefix-hit path)
    if os.environ.get("HANDOFF_NO_SPEC", "").strip() == "1":
        spec = None
    # opt-decode: HANDOFF_SPEC_SYNTHETIC="0.9,0.8,..." accepts drafts with those unconditional per-position rates
    # (vLLM's synthetic rejection mode) so decode steps with many accepted rows run on the mini model too
    _syn = os.environ.get("HANDOFF_SPEC_SYNTHETIC", "").strip()
    if _syn and spec is not None:
        spec["rejection_sample_method"] = "synthetic"
        spec["synthetic_acceptance_rates"] = [float(x) for x in _syn.split(",")]
    # decode5 (GLM53_DEC_DLMH rig): the mini model is TP=1, so its lm_head shard is the whole 154880-row vocabulary,
    # a shape fp8_gemv's TABLE has no entry for (production's TP=2 shard is 77440 rows) -> production's code runs
    # Marlin there and the DLMH candidate head declines. HANDOFF_FP8_FULLVOCAB=1 gives (154880, 4096) the 77440-row
    # entry in BOTH arms, so the rig's lm_head calls run fp8_gemv exactly as production's shard does.
    if os.environ.get("HANDOFF_FP8_FULLVOCAB", "").strip() == "1":
        import fp8_gemv as _FG
        _FG.TABLE[(154880, 4096)] = dict(_FG.TABLE[(77440, 4096)])
        log(f"   HANDOFF_FP8_FULLVOCAB: fp8_gemv TABLE[(154880, 4096)] = {_FG.TABLE[(154880, 4096)]}")
    _install_topk_determinizer()
    if os.environ.get("HANDOFF_TAILPTR", "").strip() == "1":
        _install_tailptr_probe()
    t0 = time.time()
    _xk = {}
    if os.environ.get("HANDOFF_MP", "").strip() == "1":          # prefixhit: workers in their own process (as production)
        _xk["distributed_executor_backend"] = "mp"
    if os.environ.get("HANDOFF_EAGER", "").strip() == "1":       # prefixhit: no CUDA graphs at all
        _xk["enforce_eager"] = True
    if os.environ.get("HANDOFF_CAPTURE_SIZES", "").strip():     # prefixhit: override the capture sizes
        sizes = {int(x) for x in os.environ["HANDOFF_CAPTURE_SIZES"].split(",") if x.strip()}
    llm = LLM(model=ARGS.model, skip_tokenizer_init=True, tensor_parallel_size=1, dtype="bfloat16", **_xk,
              max_model_len=ARGS.max_model_len, max_num_seqs=4, max_num_batched_tokens=ARGS.mnbt,
              enable_prefix_caching=True, kv_cache_dtype="fp8", kv_cache_memory_bytes=ARGS.kv_bytes,
              gpu_memory_utilization=ARGS.gpu_util, speculative_config=spec, enable_flashinfer_autotune=False, seed=0,
              compilation_config={"cudagraph_capture_sizes": sorted(sizes)})
    log(f"engine up in {time.time() - t0:.0f}s; capture sizes {sorted(sizes)}; fast paths during boot/capture "
        f"{fast_path_counters()}")
    try:
        import glm53_prefill_quickwins as Q
        log(f"quickwins: items {sorted(Q._STATE['items'])} installed {Q._STATE['installed']} refused {Q._STATE['refused']}")
    except Exception as e:  # noqa: BLE001
        log("quickwins not importable:", repr(e))
    try:
        import glm53_mla_prefill as M
        if not ARGS.mla_kv_indices:
            M.STATE.write_kv_indices = False
        log(f"mla prefill: installed {M.STATE.installed} min_tokens {M.STATE.min_tokens} variant {M.STATE.variant} "
            f"write_kv_indices {getattr(M.STATE, 'write_kv_indices', 'n/a (module without the fix)')}")
    except Exception as e:  # noqa: BLE001
        log("mla prefill not importable:", repr(e))
    if H["runner"] is None:                   # prefixhit HANDOFF_MP=1: the runner lives in the worker process
        class _K:
            kv_cache_groups = []
            num_blocks = 0
        kvc = _K()
        log("HANDOFF_MP: worker in its own process; no hooks, no KV views, no zeroing")
    else:
        kvc = H["runner"].kv_cache_config
    groups = []
    for gi, g in enumerate(kvc.kv_cache_groups):
        s = g.kv_cache_spec
        groups.append({"gid": gi, "spec": type(s).__name__, "block_size": getattr(s, "block_size", None),
                       "page_size_bytes": getattr(s, "page_size_bytes", None), "layers": list(g.layer_names)})
    meta = {"num_blocks": kvc.num_blocks, "groups": groups,
            "views": [(n, list(t.shape), list(t.stride()), str(t.dtype)) for n, t in views()],
            "raw": [(ln, int(t.numel())) for ln, t in uniq_raw()], "args": vars(ARGS), "flags": flags}
    for g in groups:
        log(f"kv group {g['gid']}: {g['spec']} block {g['block_size']} page {g['page_size_bytes']} "
            f"layers {g['layers'][:3]}{'...' if len(g['layers']) > 3 else ''} ({len(g['layers'])})")
    log(f"num_blocks {kvc.num_blocks}; {len(meta['views'])} layer views; {len(meta['raw'])} raw allocations")
    torch.cuda.synchronize()
    for _ln, t in uniq_raw():                 # every request starts from the same bytes in every run
        t.zero_()
    torch.cuda.synchronize()
    # opt-decode: HANDOFF_TEMPERATURE (default 0 = greedy, the harness's original setting); production samples at 1.0
    sp = SamplingParams(temperature=float(os.environ.get("HANDOFF_TEMPERATURE", "0") or 0), max_tokens=ARGS.gen,
                        ignore_eos=True, logprobs=5, detokenize=False, seed=0)
    reqs = []
    reqs = []
    prompts = [(i, n) for i, n in enumerate(int(x) for x in ARGS.prompts.split(",") if x.strip())]
    batches = ([prompts[:ARGS.batch_a]] + [[p] for p in prompts[ARGS.batch_a:]]) if ARGS.batch_a else [[p] for p in prompts]
    _prof_b = os.environ.get("HANDOFF_PROF_BATCH", "").strip()   # decode6: SIGUSR2 -> GLM53_TF_PROFILE before batch N
    for bi, grp in enumerate(batches):
        grp = list(grp)
        req_inputs = [{"prompt_token_ids": prompt_ids(n, i), "cache_salt": f"handoff-A{i}"} for i, n in grp]
        if _prof_b and int(_prof_b) == bi:
            import signal as _sig
            log(f"decode6: arming the GLM53_TF_PROFILE profiler before batch {bi}")
            os.kill(os.getpid(), _sig.SIGUSR2)
        H["active"] = True
        t0 = time.time()
        s0 = H["step"]
        outs = llm.generate(req_inputs, sp, use_tqdm=False)
        H["active"] = False
        for (i, n), o in zip(grp, outs):
            o = o.outputs[0]
            gen = [int(x) for x in o.token_ids]
            lps = [{str(int(t)): float(v.logprob) for t, v in d.items()} for d in (o.logprobs or [])]
            log(f"A{i}: {n} prompt tokens -> {len(gen)} generated in {time.time() - t0:.1f}s; steps {s0}..{H['step'] - 1}; "
                f"first tokens {gen[:8]}")
            reqs.append({"prompt_ids": prompt_ids(n, i), "gen": gen, "logprobs": lps, "steps": [s0, H["step"]],
                         "rid": next((k for k in H.get("bh", {}) if k.split("-")[0] == str(outs[grp.index((i, n))].request_id)), None)})
    # decode4 review: HANDOFF_BLOCK_CKSUM=1 -> per-(raw allocation, block) checksums after the A phase (and after B)
    def _block_cksums():
        torch.cuda.synchronize()
        out = {}
        nb = int(meta.get("num_blocks", 0) or 0) if isinstance(meta, dict) else 0
        for ln, t in uniq_raw():
            b = t.reshape(-1).view(torch.uint8)
            n = b.numel()
            nbk = nb if (nb and n % nb == 0) else None
            if nbk is None:
                continue
            w = b.view(nbk, n // nbk)
            if (n // nbk) % 4 == 0:
                w = w.view(torch.int32)
            co = _coef(w.shape[1], "ck")
            out[ln] = [int((w[i].to(torch.int64) * co).sum()) for i in range(nbk)]
        return out
    _ck = os.environ.get("HANDOFF_BLOCK_CKSUM", "").strip() == "1"
    cks_A = _block_cksums() if _ck else None
    res = {"label": ARGS.label, "requests": reqs, "meta": meta,
           "prompt_ids": reqs[0]["prompt_ids"], "gen": reqs[0]["gen"], "logprobs": reqs[0]["logprobs"]}
    if os.environ.get("HANDOFF_CONC_TEST", "").strip() == "1":
        res["CONC"] = _prefixhit_conc(llm, reqs)
    if os.environ.get("HANDOFF_B_LENS", "").strip():
        res["PH"] = _prefixhit_b(llm, reqs, prompts)
    if ARGS.consistency:
        B = []
        sp1 = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True, logprobs=5, detokenize=False, seed=0)
        ids, gen = reqs[0]["prompt_ids"], reqs[0]["gen"]
        # decode4 review: HANDOFF_B_SALT=A -> every B_j shares A0's cache_salt, so B_j HITS the prefix cache at the
        # mamba-aligned blocks A0's decode saved (the align post-copy of the lazy commit's bias column is then READ)
        # HANDOFF_B_PAIR=1: every B_j twice - fresh (own salt, no hit) into B and with A0's salt (prefix hit) into
        # B_hit, so the hit's saved state is judged against a fresh prefill of the SAME tokens (no decode noise).
        # HANDOFF_B_BASE=<n>: B_j = prompt[:n + j] (a boundary saved during A0's PREFILL) instead of prompt + gen[:j].
        b_salt_a = os.environ.get("HANDOFF_B_SALT", "").strip() == "A"
        b_pair = os.environ.get("HANDOFF_B_PAIR", "").strip() == "1"
        b_base = os.environ.get("HANDOFF_B_BASE", "").strip()
        B_cached, B_hit, B_hit_cached = [], [], []
        def _one(seq, salt):
            ro = llm.generate([{"prompt_token_ids": seq, "cache_salt": salt}], sp1, use_tqdm=False)[0]
            ob = ro.outputs[0]
            return (int(getattr(ro, "num_cached_tokens", -1) or 0),
                    {str(int(t)): float(v.logprob) for t, v in (ob.logprobs or [{}])[0].items()})
        nb = ARGS.consistency if b_base else min(ARGS.consistency, len(gen))
        for j in range(nb):
            seq = ids[:int(b_base) + j] if b_base else ids + gen[:j]
            c, lp = _one(seq, "handoff-A0" if (b_salt_a and not b_pair) else f"handoff-B{j}")
            B_cached.append(c); B.append(lp)
            if b_pair:
                c, lp = _one(seq, "handoff-A0")
                B_hit_cached.append(c); B_hit.append(lp)
        res["B"] = B
        res["B_cached"] = B_cached
        res["B_hit"] = B_hit
        res["B_hit_cached"] = B_hit_cached
        if _ck:
            res["cks_B"] = _block_cksums()
        log(f"B prefills: salt {'A0 (prefix hits allowed)' if b_salt_a and not b_pair else 'own'}; cached tokens {B_cached}"
            + (f"; paired hits cached {B_hit_cached}" if b_pair else "") + (f"; base prompt[:{b_base}+j]" if b_base else ""))
    if _ck:
        res["cks_A"] = cks_A
    decode_fast = {k: v for r in H["records"] if not r["prefill"] for k, v in r["fast_step"].items()}
    shadow_ok = None
    if ARGS.shadow:
        def same(r):
            return r["state"]["equal"] and all(o["equal"] for o in r["out"])
        ctrl = all(s.get("control") is None or same(s["control"]) for s in SH["steps"])
        feat = all(same(s["feature"]) for s in SH["steps"])
        shadow_ok = {"steps": len(SH["steps"]), "control_identical": ctrl, "feature_identical": feat,
                     "decode_fast_paths": decode_fast}
    decode_stale = [dict(step=r["step"], **r["stale"]) for r in H["records"]
                    if not r["prefill"] and r.get("stale") and "W" in r["stale"]]
    if ARGS.shadow:
        json.dump({"summary": shadow_ok, "steps": SH["steps"], "mla_probe": SH["mla"], "decode_stale": decode_stale,
                   "topk_probe": TOPK, "args": vars(ARGS)},
                  open(os.path.join(ARGS.out, "shadow.json"), "w"), default=str)
    torch.save({"records": H["records"], "dumps": H["dumped"]}, os.path.join(ARGS.out, "records.pt"))
    res["bh"] = H.get("bh", {})
    json.dump(res, open(os.path.join(ARGS.out, "result.json"), "w"))
    log(f"fast paths during decode steps: {decode_fast or 'none'}")
    st_dec = [r["stale"] for r in H["records"] if not r["prefill"] and r.get("stale") and "W" in r["stale"]]
    n_over = sum(1 for x in st_dec if x["last_row_over"])
    n_foreign = sum(1 for x in st_dec if x["last_row_over"] and not all(x["stale_own"]))
    log(f"decode FA2 stale reads: {len(st_dec)} decode steps; last row read past the step in {n_over}; "
        f"of those, slots NOT of the request in {n_foreign}")
    if ARGS.topk != "prod" and ARGS.topk_probe:
        log(f"top-k probe totals: {TOPK['calls']} calls, {TOPK['rows']} rows; production's op drifted in order on "
            f"{TOPK['order_drift']} rows, in the selected set on {TOPK['set_drift']} rows")
    if shadow_ok is not None:
        log(f"SHADOW {ARGS.label}: {shadow_ok['steps']} prefill forwards; control (P vs P) "
            f"{'IDENTICAL' if shadow_ok['control_identical'] else 'DIFFERENT'}; feature (P vs F) "
            f"{'IDENTICAL' if shadow_ok['feature_identical'] else 'DIFFERENT'}")
    log(f"wrote {ARGS.out}: result.json, shadow.json, records.pt")


if __name__ == "__main__":
    main()
