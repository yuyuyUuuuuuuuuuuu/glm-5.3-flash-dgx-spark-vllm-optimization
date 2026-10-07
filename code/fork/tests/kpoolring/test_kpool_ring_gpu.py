#!/usr/bin/env python3
"""GPU tests for the vLLM #58454 backport (overlay/patch_kpool_tail_ring.py), in the production image on nodeC.

    tests/gpu_run.sh python3 tests/kpoolring/test_kpool_ring_gpu.py

Nothing here imports a hand-copied kernel. Three builds of the image's own files are made in a temp dir by running the
SHIPPED overlay scripts on copies (env-path overrides):
  prod  = image + patch_kpool_tail_seed_stride.py              (what production runs today, GLM53_KPOOL_RING unset)
  ring  = image + seed-stride + patch_kpool_tail_ring.py         (GLM53_KPOOL_RING=1)
and a third kernel module, upstream vLLM's kpool_compress.py at the #58454 merge commit 2617fe93
(tests/kpoolring/ref/kpool_compress_vllm58454.py, verbatim), as the cross-check for the ring build's seed-at-call-site
approach. Each build's ring size comes from ITS attention.py (Glm5NextTailCache.get_kv_cache_spec at production
settings: block 4608, index_kpool 4, num_speculative_tokens 7), and each build's seed kpool argument is the expression
its sparse_attn_indexer_kpool.py passes at the seed call (read with ast, evaluated on the test's tail tensor). The tail
cache is an as_strided view with production's padded block stride (idx page 152064 B at block 4608), over storage
pre-filled with garbage (a reused block). Prefill uses the image's _kpool_compress_insert; decode tokens are grouped
with the image's _build_decode_scatter_indices / _scatter_decode_tokens_by_request and tail slots come from the
image's compute_kpool_tail_slot_mapping (the path the V2 model runner's KpoolTailMetadataBuilder takes).

The reference for every complete pool is the prefill writer (kpool_compress_and_write_cache) over the ACCEPTED token
stream; test_decode_writer_matches_prefill_writer shows the decode writer is bitwise equal to it, and the directed
case also checks a sequential accepted-only decode (one token per step) reference.
"""
from __future__ import annotations

import ast
import importlib.util
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parents[2]
SP = Path("/usr/local/lib/python3.12/dist-packages/vllm")
UPSTREAM = REPO / "tests/kpoolring/ref/kpool_compress_vllm58454.py"
DEV = "cuda"
HEAD = 128
KPOOL = 4
PAGE = 64  # indexer cache page (pools per page) used by the tests
ROUND = True
NUM_SPEC = 7
PROD_BLOCK = 4608
TAIL_BLOCK_ELEMS = (PROD_BLOCK // KPOOL) * (HEAD + 4) // 2  # 152064 B idx page -> bf16 elements (76032)

import vllm.models.glm5next  # noqa: E402,F401  (package first: importing the indexer module first is a circular import)
from vllm.model_executor.layers import sparse_attn_indexer_kpool as IDX  # noqa: E402  (image module)
from vllm.v1.attention.backends.mla.indexer import (  # noqa: E402
    KpoolTailMetadataBuilder,
    compute_kpool_tail_slot_mapping,
)
from vllm.v1.attention.backend import CommonAttentionMetadata  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_variants() -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="kpoolring-"))
    files = {
        "kc": "models/glm5next/nvidia/ops/kpool_compress.py",
        "idx": "model_executor/layers/sparse_attn_indexer_kpool.py",
        "attn": "models/glm5next/nvidia/attention.py",
    }
    out = {}
    for name in ("prod", "ring"):
        d = tmp / name
        d.mkdir()
        paths = {k: d / Path(v).name for k, v in files.items()}
        for k, v in files.items():
            shutil.copy(SP / v, paths[k])
        env = dict(
            os.environ,
            GLM53_KPOOL_COMPRESS_PY=str(paths["kc"]),
            GLM53_GLM5NEXT_ATTENTION_PY=str(paths["attn"]),
            GLM53_SPARSE_INDEXER_KPOOL_PY=str(paths["idx"]),
            GLM53_KPOOL_SEED_STRIDE="1",
        )
        scripts = ["patch_kpool_tail_seed_stride.py"] + (["patch_kpool_tail_ring.py"] if name == "ring" else [])
        if name == "ring":
            env["GLM53_KPOOL_RING"] = "1"
        for s in scripts:
            r = subprocess.run([sys.executable, str(REPO / "overlay" / s)], env=env, capture_output=True, text=True)
            print(f"[build {name}] {s}: rc={r.returncode} {r.stdout.strip()} {r.stderr.strip()[-300:]}")
            assert r.returncode == 0
        kc = load(paths["kc"], f"kr_{name}_kpool_compress")
        attn = load(paths["attn"], f"kr_{name}_attention")
        tree = ast.parse(paths["idx"].read_text())
        calls = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "kpool_seed_tail_cache"
        ]
        assert len(calls) == 1, len(calls)
        seed_expr = ast.unparse(calls[0].args[4])
        from vllm.config import VllmConfig, set_current_vllm_config

        with set_current_vllm_config(VllmConfig()):
            cache = attn.Glm5NextTailCache(
                head_dim=HEAD, dtype=torch.bfloat16, prefix=f"kr.{name}.tail",
                cache_config=SimpleNamespace(block_size=PROD_BLOCK), index_kpool=KPOOL,
            )
        spec = cache.get_kv_cache_spec(SimpleNamespace(num_speculative_tokens=NUM_SPEC))
        out[name] = SimpleNamespace(name=name, kc=kc, ring=spec.block_size, seed_expr=seed_expr, spec=spec)
        print(f"[build {name}] tail spec block_size={spec.block_size} sliding_window={spec.sliding_window} "
              f"unpadded page={spec.page_size_bytes} B; seed call kpool argument: `{seed_expr}`")
    up = load(UPSTREAM, "kr_upstream_kpool_compress")
    out["upstream"] = SimpleNamespace(name="upstream", kc=up, ring=out["ring"].ring, seed_expr="index_kpool", spec=None)
    return out


V: dict = {}


def seed_kpool(v, tail) -> int:
    return int(eval(v.seed_expr, {}, {"tail_kv_cache": tail, "index_kpool": KPOOL}))  # noqa: S307 (test-only)


def test(fn):
    def run():
        t0 = time.time()
        try:
            msg = fn() or ""
            RESULTS.append((fn.__name__, True, msg))
            print(f"PASS {fn.__name__} ({time.time() - t0:.1f}s) {msg}")
        except Exception as exc:  # noqa: BLE001
            RESULTS.append((fn.__name__, False, f"{type(exc).__name__}: {exc}"))
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    run.__name__ = fn.__name__
    TESTS.append(run)
    return run


TESTS: list = []


# ----------------------------------------------------------------------------------------------------- helpers
def new_kv(num_pages: int) -> torch.Tensor:
    return torch.zeros(num_pages, PAGE, HEAD + 4, dtype=torch.uint8, device=DEV)


def new_tail(num_blocks: int, ring: int, garbage_seed: int | None = 1234) -> tuple[torch.Tensor, torch.Tensor]:
    """Tail view with production's padded block stride over (garbage-filled) storage."""
    g = torch.Generator(device=DEV).manual_seed(garbage_seed or 0)
    backing = (
        torch.randn(num_blocks * TAIL_BLOCK_ELEMS, dtype=torch.float32, device=DEV, generator=g).to(torch.bfloat16)
        if garbage_seed is not None
        else torch.zeros(num_blocks * TAIL_BLOCK_ELEMS, dtype=torch.bfloat16, device=DEV)
    )
    tail = torch.as_strided(backing, (num_blocks, 2, ring, HEAD), (TAIL_BLOCK_ELEMS, ring * HEAD, HEAD, 1))
    return backing, tail


def pool_bytes(kv: torch.Tensor, loc: int) -> torch.Tensor:
    flat = kv[loc // PAGE].reshape(-1)
    r = loc % PAGE
    return torch.cat([flat[r * HEAD:(r + 1) * HEAD], flat[PAGE * HEAD + 4 * r:PAGE * HEAD + 4 * (r + 1)]])


def copy_pool(dst: torch.Tensor, src: torch.Tensor, loc: int) -> None:
    """Copy one pool entry (its 128 fp8 K bytes and its fp32 scale) between caches (page = [K rows | scales])."""
    d, s_ = dst[loc // PAGE].reshape(-1), src[loc // PAGE].reshape(-1)
    r = loc % PAGE
    d[r * HEAD:(r + 1) * HEAD] = s_[r * HEAD:(r + 1) * HEAD]
    d[PAGE * HEAD + 4 * r:PAGE * HEAD + 4 * (r + 1)] = s_[PAGE * HEAD + 4 * r:PAGE * HEAD + 4 * (r + 1)]


def reference_pools(kc, K, S, ape, n_pools: int, loc_base: int, num_pages: int) -> torch.Tensor:
    kv = new_kv(num_pages)
    if n_pools:
        kc.kpool_compress_and_write_cache(
            kv, K[: n_pools * KPOOL].view(n_pools, KPOOL, HEAD), S[: n_pools * KPOOL].view(n_pools, KPOOL, HEAD), ape,
            torch.arange(loc_base, loc_base + n_pools, dtype=torch.int64, device=DEV), pool_size=KPOOL,
            head_dim=HEAD, round_scale=ROUND,
        )
    return kv


def pool_slot(p: int, loc_base: int) -> int:
    return loc_base + p // KPOOL if p % KPOOL == KPOOL - 1 else -1


class Req:
    def __init__(self, rid, block, loc_base, total, prompt, cached=0, chunk=10**9, join=0, plain=False, seed=0):
        g = torch.Generator(device=DEV).manual_seed(1000 + seed)
        self.rid, self.block, self.loc_base = rid, block, loc_base
        self.K = torch.randn(total + 16, HEAD, device=DEV, generator=g).to(torch.bfloat16)
        self.S = torch.randn(total + 16, HEAD, device=DEV, generator=g).to(torch.bfloat16)
        self.total, self.prompt, self.cached, self.chunk, self.join, self.plain = total, prompt, cached, chunk, join, plain
        self.done_prefill = cached  # tokens whose indexer state exists
        self.acc = 0  # accepted (committed) length once decoding


def make_schedule(reqs, steps: int, seed: int, k_choices=(4, 5, 7)):
    """Per step: one uniform k (adaptive K, uniform batch) and per request an accepted count and draft garbage."""
    rng = random.Random(seed)
    g = torch.Generator(device=DEV).manual_seed(seed)
    sched = []
    for _ in range(steps):
        k = rng.choice(k_choices)
        per = {}
        for r in reqs:
            kk = 0 if r.plain else k
            # skew towards early rejections (the corrupting case) but keep full acceptances
            a = min(kk, int(rng.expovariate(0.45))) if kk else 0
            garbage_k = torch.randn(kk + 1, HEAD, device=DEV, generator=g).to(torch.bfloat16)
            garbage_s = torch.randn(kk + 1, HEAD, device=DEV, generator=g).to(torch.bfloat16)
            per[r.rid] = (kk, a, garbage_k, garbage_s)
        sched.append(per)
    return sched


def run_sim(v, ring, reqs_spec, sched, ape, num_pages, num_tail_blocks):
    """Drive one build through prefill (chunked, prefix-resume) + decode (spec verify with rejections)."""
    reqs = [Req(**r) for r in reqs_spec]
    kv = new_kv(num_pages)
    _, tail = new_tail(num_tail_blocks, ring)
    for r in reqs:  # cached prefix: its pools exist (prefix-cache hit); the tail block holds garbage
        if r.cached:
            ref = reference_pools(v.kc, r.K, r.S, ape, r.cached // KPOOL, r.loc_base, num_pages)
            for p in range(r.cached // KPOOL):
                copy_pool(kv, ref, r.loc_base + p)
    decoding: dict[int, bool] = {}
    for t, per in enumerate(sched):
        # ---- prefill chunks of this step (one batch, requests concatenated) ----
        pk, ps, pslot, ppos, pblk = [], [], [], [], []
        for r in reqs:
            if r.join <= t and r.done_prefill < r.prompt:
                a0, e0 = r.done_prefill, min(r.prompt, r.done_prefill + r.chunk)
                pos = list(range(a0, e0))
                pk.append(r.K[a0:e0]); ps.append(r.S[a0:e0])
                pslot += [pool_slot(p, r.loc_base) for p in pos]
                ppos += pos; pblk += [r.block] * len(pos)
                r.done_prefill = e0
                if e0 == r.prompt:
                    r.acc = r.prompt
                    decoding[r.rid] = False  # starts decoding next step
        if pk:
            k_cat, s_cat = torch.cat(pk), torch.cat(ps)
            slot = torch.tensor(pslot, dtype=torch.int64, device=DEV)
            IDX._kpool_compress_insert(k_cat, s_cat, ape, kv, slot, KPOOL, HEAD, round_scale=ROUND)
            tslot = torch.tensor([b * ring + p % ring for b, p in zip(pblk, ppos)], dtype=torch.int64, device=DEV)
            v.kc.kpool_seed_tail_cache(tail, k_cat, s_cat, tslot, seed_kpool(v, tail), HEAD)
        # ---- decode / verify for requests that finished prefill in an earlier step ----
        active = [r for r in reqs if decoding.get(r.rid) and r.acc < r.total]
        for r in reqs:
            if r.rid in decoding:
                decoding[r.rid] = True
        if not active:
            continue
        flat_k, flat_s, flat_slot, flat_pos, lens = [], [], [], [], []
        for r in active:
            kk, a, gk, gs = per[r.rid]
            p0 = r.acc
            keys = [r.K[p0]] + [r.K[p0 + j] if j <= a else gk[j] for j in range(1, kk + 1)]
            scores = [r.S[p0]] + [r.S[p0 + j] if j <= a else gs[j] for j in range(1, kk + 1)]
            flat_k += keys; flat_s += scores
            flat_pos += list(range(p0, p0 + kk + 1))
            flat_slot += [pool_slot(p, r.loc_base) for p in range(p0, p0 + kk + 1)]
            lens.append(kk + 1)
        n, B = len(flat_pos), len(active)
        k_t, s_t = torch.stack(flat_k), torch.stack(flat_s)
        pos_t = torch.tensor(flat_pos, dtype=torch.int64, device=DEV)
        slot_t = torch.tensor(flat_slot, dtype=torch.int32, device=DEV)
        qsl = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32, device=DEV)
        bt = torch.zeros(B, 8, dtype=torch.int32, device=DEV)
        bt[:, 0] = torch.tensor([r.block for r in active], dtype=torch.int32, device=DEV)
        tail_flat = compute_kpool_tail_slot_mapping(
            torch.full((n,), -1, dtype=torch.int64, device=DEV), bt, qsl, pos_t, n, B, ring
        )
        lmax = max(lens)
        if all(x == lmax for x in lens):
            shp = (B, lmax)
            dk, ds = k_t.view(B, lmax, HEAD), s_t.view(B, lmax, HEAD)
            dslot, dpos, dtail = slot_t.view(shp), pos_t.to(torch.int32).view(shp), tail_flat.view(shp)
        else:
            dl = torch.tensor(lens, dtype=torch.int32, device=DEV)
            si = IDX._build_decode_scatter_indices(dl, B, n)
            dk = IDX._scatter_decode_tokens_by_request(k_t, 0, B, lmax, si)
            ds = IDX._scatter_decode_tokens_by_request(s_t, 0, B, lmax, si)
            dslot = IDX._scatter_decode_tokens_by_request(slot_t, -1, B, lmax, si)
            dpos = IDX._scatter_decode_tokens_by_request(pos_t.to(torch.int32), -1, B, lmax, si)
            dtail = IDX._scatter_decode_tokens_by_request(tail_flat, -1, B, lmax, si)
        v.kc.kpool_decode_update_and_maybe_write_cache_batched(
            kv, tail, dtail, dk, ds, ape, dslot, dpos, KPOOL, HEAD, round_scale=ROUND
        )
        for r in active:
            kk, a, _, _ = per[r.rid]
            r.acc = min(r.acc + a + 1, r.total)
    return kv, reqs


def compare(v, kv, reqs, ape, num_pages):
    bad, total, decode_bad = 0, 0, 0
    for r in reqs:
        n_pools = r.acc // KPOOL
        ref = reference_pools(v.kc, r.K, r.S, ape, n_pools, r.loc_base, num_pages)
        for p in range(n_pools):
            loc = r.loc_base + p
            ok = torch.equal(pool_bytes(kv, loc), pool_bytes(ref, loc))
            total += 1
            if not ok:
                bad += 1
                decode_bad += int(p * KPOOL >= r.prompt)
    return bad, total, decode_bad


# ----------------------------------------------------------------------------------------------------- tests
@test
def test_build_ring_sizes():
    p, r = V["prod"], V["ring"]
    assert p.ring == KPOOL and r.ring == 16, (p.ring, r.ring)
    assert p.seed_expr == "index_kpool" and r.seed_expr == "tail_kv_cache.shape[2]", (p.seed_expr, r.seed_expr)
    assert PROD_BLOCK % r.ring == 0
    return f"prod ring {p.ring} (seed kpool `{p.seed_expr}`), patched ring {r.ring} (seed kpool `{r.seed_expr}`)"


@test
def test_prod_kernel_rejects_a_ring():
    """A spec-only fix is not enough: the pre-#58454 decode wrapper asserts ring == pool_size."""
    _, tail = new_tail(2, 16)
    kv = new_kv(2)
    one = torch.zeros(1, 1, HEAD, dtype=torch.bfloat16, device=DEV)
    t = torch.zeros(1, 1, dtype=torch.int32, device=DEV)
    try:
        V["prod"].kc.kpool_decode_update_and_maybe_write_cache_batched(
            kv, tail, t, one, one, torch.zeros(KPOOL, HEAD, device=DEV), t - 1, t, KPOOL, HEAD)
    except AssertionError:
        return "AssertionError as expected"
    raise AssertionError("prod decode wrapper accepted a 16-slot ring")


@test
def test_decode_writer_matches_prefill_writer():
    """Upstream test, ported: the ring build's decode writer (one token per step) equals the prefill writer for pool
    sizes 4 and 16 and rings of 1 and 2 pools."""
    msgs = []
    for pool_size in (4, 16):
        for ring_pools in (1, 2):
            ring = ring_pools * pool_size
            n_pools, nblk = 8, 4
            n_tok = n_pools * pool_size
            torch.manual_seed(0)
            k = torch.randn(n_tok, HEAD, dtype=torch.bfloat16, device=DEV)
            s = torch.randn(n_tok, HEAD, dtype=torch.bfloat16, device=DEV)
            ape = torch.randn(pool_size, HEAD, dtype=torch.float32, device=DEV)
            kv_p = new_kv(nblk)
            V["ring"].kc.kpool_compress_and_write_cache(
                kv_p, k.view(n_pools, pool_size, HEAD), s.view(n_pools, pool_size, HEAD), ape,
                torch.arange(n_pools, dtype=torch.int64, device=DEV), pool_size=pool_size, head_dim=HEAD,
                round_scale=ROUND)
            kv_d = torch.zeros_like(kv_p)
            tail = torch.zeros(nblk, 2, ring, HEAD, dtype=torch.bfloat16, device=DEV)
            for t in range(n_tok):
                done = t % pool_size == pool_size - 1
                V["ring"].kc.kpool_decode_update_and_maybe_write_cache_batched(
                    kv_d, tail, torch.tensor([[t % ring]], dtype=torch.int32, device=DEV), k[t].view(1, 1, HEAD),
                    s[t].view(1, 1, HEAD), ape,
                    torch.tensor([[t // pool_size if done else -1]], dtype=torch.int32, device=DEV),
                    torch.tensor([[t]], dtype=torch.int32, device=DEV), pool_size, HEAD, round_scale=ROUND)
            diff = [p for p in range(n_pools) if not torch.equal(pool_bytes(kv_p, p), pool_bytes(kv_d, p))]
            assert not diff, (pool_size, ring, diff)
            msgs.append(f"pool {pool_size}/ring {ring}: 0/{n_pools} differ")
    return "; ".join(msgs)


@test
def test_rejected_pool_completing_draft_directed():
    """The PR's case at production shape: kpool 4, num_spec 7. Positions 0..5 accepted; the verify of 6..13 accepts
    only 6, so the pool-completing draft 7 is rejected while drafts 8..13 are stashed; the next verify redoes pool 1
    (4..7). Compared with the prefill writer AND with a sequential accepted-only decode (one token per step)."""
    torch.manual_seed(7)
    K = torch.randn(24, HEAD, dtype=torch.bfloat16, device=DEV)
    S = torch.randn(24, HEAD, dtype=torch.bfloat16, device=DEV)
    D = torch.randn(8, HEAD, dtype=torch.bfloat16, device=DEV)
    DS = torch.randn(8, HEAD, dtype=torch.bfloat16, device=DEV)
    ape = torch.randn(KPOOL, HEAD, dtype=torch.float32, device=DEV)
    ref = reference_pools(V["ring"].kc, K, S, ape, 5, 0, 1)
    blk = 3

    def step(v, kv, tail, ring, positions, keys, scores):
        n = len(positions)
        pos = torch.tensor([positions], dtype=torch.int32, device=DEV)
        v.kc.kpool_decode_update_and_maybe_write_cache_batched(
            kv, tail, (blk * ring + pos % ring).to(torch.int32), torch.stack(keys).view(1, n, HEAD),
            torch.stack(scores).view(1, n, HEAD), ape,
            torch.tensor([[pool_slot(p, 0) for p in positions]], dtype=torch.int32, device=DEV), pos, KPOOL, HEAD,
            round_scale=ROUND)

    # sequential accepted-only reference (prod kernel, plain decode): tokens 0..19 one per step
    kv_seq = new_kv(1)
    _, tail_seq = new_tail(8, KPOOL)
    for p in range(20):
        step(V["prod"], kv_seq, tail_seq, KPOOL, [p], [K[p]], [S[p]])
    seq_ok = all(torch.equal(pool_bytes(kv_seq, p), pool_bytes(ref, p)) for p in range(5))
    assert seq_ok, "sequential decode reference disagrees with the prefill writer"
    out = {}
    for name, ring in (("prod", KPOOL), ("ring", V["ring"].ring), ("upstream", V["ring"].ring), ("ring@4", KPOOL)):
        v = V["ring"] if name == "ring@4" else V[name]
        kv = new_kv(1)
        _, tail = new_tail(8, ring)
        for p in range(6):
            step(v, kv, tail, ring, [p], [K[p]], [S[p]])
        step(v, kv, tail, ring, list(range(6, 14)), [K[6]] + [D[j] for j in range(1, 8)],
             [S[6]] + [DS[j] for j in range(1, 8)])
        step(v, kv, tail, ring, list(range(7, 15)), [K[p] for p in range(7, 15)], [S[p] for p in range(7, 15)])
        for p in range(15, 20):
            step(v, kv, tail, ring, [p], [K[p]], [S[p]])
        pb = pool_bytes(kv, 1)
        out[name] = (torch.equal(pb, pool_bytes(ref, 1)), int((pb[:HEAD] != pool_bytes(ref, 1)[:HEAD]).sum()),
                     sum(not torch.equal(pool_bytes(kv, p), pool_bytes(ref, p)) for p in range(5)))
    assert not out["prod"][0] and not out["ring@4"][0], out
    assert out["ring"] == (True, 0, 0) and out["upstream"] == (True, 0, 0), out
    return ("pool 1 vs reference -> " + ", ".join(f"{k}: {'exact' if a else f'WRONG ({b}/128 fp8 key bytes differ)'}"
            f" [{c}/5 pools differ]" for k, (a, b, c) in out.items()) + "; sequential accepted-only decode == prefill writer")


@test
def test_rejected_draft_redo_needs_ring_slots_upstream_case():
    """Upstream test_rejected_draft_redo_needs_ring_slots (num_spec 3, ring 4 vs 8), on the ring build's kernel."""
    pool, spec, nblk = 4, 3, 2
    res = {}
    for ring_pools in (1, 2):
        ring = ring_pools * pool
        torch.manual_seed(1)
        k = torch.randn(3 * pool, HEAD, dtype=torch.bfloat16, device=DEV)
        s = torch.randn(3 * pool, HEAD, dtype=torch.bfloat16, device=DEV)
        ape = torch.randn(pool, HEAD, dtype=torch.float32, device=DEV)
        kv_ref = new_kv(nblk)
        V["ring"].kc.kpool_compress_and_write_cache(
            kv_ref, k.view(3, pool, HEAD), s.view(3, pool, HEAD), ape, torch.arange(3, dtype=torch.int64, device=DEV),
            pool_size=pool, head_dim=HEAD, round_scale=ROUND)
        kv = torch.zeros_like(kv_ref)
        tail = torch.zeros(nblk, 2, ring, HEAD, dtype=torch.bfloat16, device=DEV)

        def st(positions, keys, scores):
            pos = torch.tensor([positions], dtype=torch.int32, device=DEV)
            slots = [(p // pool) if p % pool == pool - 1 else -1 for p in positions]
            V["ring"].kc.kpool_decode_update_and_maybe_write_cache_batched(
                kv, tail, pos % ring, keys.view(1, -1, HEAD), scores.view(1, -1, HEAD), ape,
                torch.tensor([slots], dtype=torch.int32, device=DEV), pos, pool, HEAD, round_scale=ROUND)

        for t in range(7):
            st([t], k[t], s[t])
        drafts = torch.randn(spec, HEAD, dtype=torch.bfloat16, device=DEV)
        dsc = torch.randn(spec, HEAD, dtype=torch.bfloat16, device=DEV)
        st([7, 8, 9, 10], torch.cat([k[7:8], drafts]), torch.cat([s[7:8], dsc]))
        st([8, 9, 10, 11], k[8:12], s[8:12])
        ctrl = all(torch.equal(pool_bytes(kv, p), pool_bytes(kv_ref, p)) for p in (1, 2))
        kv.zero_(); tail.zero_()
        for t in range(6):
            st([t], k[t], s[t])
        st([6, 7, 8, 9], torch.cat([k[6:7], drafts]), torch.cat([s[6:7], dsc]))
        st([7, 8, 9, 10], k[7:11], s[7:11])
        res[ring] = (ctrl, torch.equal(pool_bytes(kv, 1), pool_bytes(kv_ref, 1)))
    assert res[4] == (True, False) and res[8] == (True, True), res
    return f"(control ok, pool1 ok): ring 4 {res[4]}, ring 8 {res[8]}"


def torch_reference(kv, tail, tail_slot, key, score, ape, slot_map, pos, pool, ring):
    """Upstream's independent reference, generalised to a ring of `ring` slots (phys = pos % ring, block = slot //
    ring). With ring == pool it is upstream's _torch_reference verbatim."""
    import math

    kv, tail = kv.clone(), tail.clone()
    B, next_n = pos.shape
    page_bytes, k_region = PAGE * (HEAD + 4), HEAD * PAGE
    ts, sm, pc = tail_slot.cpu().tolist(), slot_map.cpu().tolist(), pos.cpu().tolist()
    kc_, sc_, ap_, tl_ = key.float().cpu(), score.float().cpu(), ape.cpu(), tail.float().cpu()
    kvf = kv.view(torch.uint8).reshape(-1).cpu()
    h = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    while h.shape[0] < HEAD:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    h = h / math.sqrt(HEAD)
    fp8 = torch.float8_e4m3fn
    fmax = torch.finfo(fp8).max
    for b in range(B):
        for t in range(next_n):
            loc, p = sm[b][t], pc[b][t]
            valid = loc >= 0 and p >= 0
            sp = max(p, 0)
            slot, phys_slot = sp % pool, sp % ring
            blk = max(ts[b][t], 0) // ring
            ck, cs = kc_[b, t], sc_[b, t]
            if valid and slot == pool - 1:
                start = sp - slot
                ss, ks = [], []
                for q in range(pool):
                    if q == slot:
                        s_, k_ = cs, ck
                    else:
                        ph = (start + q) % ring
                        s_, k_ = tl_[blk, 1, ph], tl_[blk, 0, ph]
                    ss.append(s_ + ap_[q]); ks.append(k_)
                ss, ks = torch.stack(ss), torch.stack(ks)
                pr = torch.exp(ss - ss.max(0).values)
                x = ((ks * pr).sum(0) / pr.sum(0)).to(torch.bfloat16).float()
                x = (x @ h).to(torch.bfloat16).float()
                sc = torch.exp2(torch.ceil(torch.log2(torch.clamp(x.abs().max(), min=1e-4) / fmax)))
                qz = torch.clamp(x / sc, -fmax, fmax).to(fp8)
                base = (loc // PAGE) * page_bytes
                kvf[base + (loc % PAGE) * HEAD + torch.arange(HEAD)] = qz.view(torch.uint8)
                so = base + k_region + (loc % PAGE) * 4
                kvf[so:so + 4] = sc.reshape(1).view(torch.uint8)
            if p >= 0 and ts[b][t] >= 0:
                tl_[blk, 0, phys_slot] = ck
                tl_[blk, 1, phys_slot] = cs
    return kvf.view(kv.shape).to(DEV), tl_.to(torch.bfloat16).to(DEV)


@test
def test_batched_matches_reference_cases_and_fuzz():
    """Upstream test_batched_matches_reference (5 cases incl. plain decode and non-uniform padding) + 20-seed fuzz,
    POOL_SIZE 16 as upstream, each at ring = pool (stock sizing: ring build must equal prod bitwise) and ring = 2 pools."""
    POOL, NB = 16, 32
    n_cases = 0

    def tslot(blocks, pos, ring):
        blk = torch.tensor(blocks, device=DEV, dtype=torch.int32).unsqueeze(1)
        return (blk * ring + pos % ring).to(torch.int32)

    def seed_prior(tail, blocks, n_prior, seed=42):
        if n_prior <= 0:
            return
        g = torch.Generator(device=DEV).manual_seed(seed)
        pk = torch.randn(len(blocks), n_prior, HEAD, dtype=torch.bfloat16, device=DEV, generator=g)
        ps = torch.randn(len(blocks), n_prior, HEAD, dtype=torch.bfloat16, device=DEV, generator=g)
        for i, b in enumerate(blocks):
            tail[b, 0, :n_prior] = pk[i]
            tail[b, 1, :n_prior] = ps[i]

    def case(cid, ring):
        torch.manual_seed(0)
        if cid == "no_completion":
            B, nn, blocks = 3, 4, [0, 1, 2]
            pos = torch.arange(nn, device=DEV, dtype=torch.int32).unsqueeze(0).expand(B, -1).contiguous()
            sm = torch.full((B, nn), -1, dtype=torch.int32, device=DEV); prior = 0
        elif cid == "completion_at_end":
            B, nn, blocks = 2, 4, [0, 1]
            pos = torch.tensor([[12, 13, 14, 15]] * 2, dtype=torch.int32, device=DEV)
            sm = torch.full((B, nn), -1, dtype=torch.int32, device=DEV)
            sm[:, 3] = torch.tensor([15, PAGE + 15], dtype=torch.int32, device=DEV); prior = POOL - nn
        elif cid == "completion_mid_batch":
            B, nn, blocks = 3, 4, [0, 1, 2]
            pos = torch.tensor([[13, 14, 15, 16]] * B, dtype=torch.int32, device=DEV)
            sm = torch.full((B, nn), -1, dtype=torch.int32, device=DEV)
            sm[:, 2] = torch.tensor([15, PAGE + 15, 2 * PAGE + 15], dtype=torch.int32, device=DEV); prior = 13
        elif cid == "non_uniform_padding":
            B, nn, blocks = 2, 4, [0, 1]
            pos = torch.tensor([[12, 13, 14, 15], [12, 13, -1, -1]], dtype=torch.int32, device=DEV)
            sm = torch.full((B, nn), -1, dtype=torch.int32, device=DEV); sm[0, 3] = 15; prior = POOL - 4
        else:  # plain_decode
            B, nn, blocks = 4, 1, [0, 1, 2, 3]
            pos = torch.tensor([[5], [6], [7], [8]], dtype=torch.int32, device=DEV)
            sm = torch.full((B, nn), -1, dtype=torch.int32, device=DEV); prior = 0
        safe = torch.where(pos >= 0, pos, 0)
        ts = torch.where(pos >= 0, tslot(blocks, safe, ring), 0) if cid == "non_uniform_padding" else tslot(blocks, pos, ring)
        key = torch.randn(B, nn, HEAD, dtype=torch.bfloat16, device=DEV)
        sc = torch.randn(B, nn, HEAD, dtype=torch.bfloat16, device=DEV)
        ape = torch.randn(POOL, HEAD, dtype=torch.float32, device=DEV)
        return blocks, pos, sm, prior, ts, key, sc, ape

    def fuzz(seed, ring):
        g = torch.Generator(device=DEV).manual_seed(seed)
        B = int(torch.randint(1, 6, (1,), generator=g, device=DEV).item())
        nn = int(torch.randint(1, 8, (1,), generator=g, device=DEV).item())
        blocks = list(range(B))
        starts = torch.randint(0, 33, (B,), generator=g, device=DEV, dtype=torch.int32)
        pos = starts.unsqueeze(1) + torch.arange(nn, device=DEV, dtype=torch.int32).unsqueeze(0)
        ts = tslot(blocks, pos, ring)
        comp = pos % POOL == POOL - 1
        blk = torch.tensor(blocks, device=DEV, dtype=torch.int32).unsqueeze(1)
        sm = torch.where(comp, blk * PAGE + (POOL - 1), torch.full_like(pos, -1))
        key = torch.randn(B, nn, HEAD, dtype=torch.bfloat16, device=DEV, generator=g)
        sc = torch.randn(B, nn, HEAD, dtype=torch.bfloat16, device=DEV, generator=g)
        ape = torch.randn(POOL, HEAD, dtype=torch.float32, device=DEV, generator=g)
        return blocks, pos, sm, starts, ts, key, sc, ape

    for ring_pools in (1, 2):
        ring = ring_pools * POOL
        jobs = [("case", c) for c in ("no_completion", "completion_at_end", "completion_mid_batch",
                                      "non_uniform_padding", "plain_decode")] + [("fuzz", s) for s in range(20)]
        for kind, arg in jobs:
            if kind == "case":
                blocks, pos, sm, prior, ts, key, sc, ape = case(arg, ring)
                kv = new_kv(NB); tail = torch.zeros(NB, 2, ring, HEAD, dtype=torch.bfloat16, device=DEV)
                seed_prior(tail, blocks, prior)
            else:
                blocks, pos, sm, starts, ts, key, sc, ape = fuzz(arg, ring)
                kv = new_kv(NB); tail = torch.zeros(NB, 2, ring, HEAD, dtype=torch.bfloat16, device=DEV)
                pg = torch.Generator(device=DEV).manual_seed(arg + 1000)
                for b in range(len(blocks)):
                    npr = int(starts[b].item()) % POOL
                    if npr:
                        tail[blocks[b], 0, :npr] = torch.randn(npr, HEAD, dtype=torch.bfloat16, device=DEV, generator=pg)
                        tail[blocks[b], 1, :npr] = torch.randn(npr, HEAD, dtype=torch.bfloat16, device=DEV, generator=pg)
            rk, rt = torch_reference(kv, tail, ts, key, sc, ape, sm, pos, POOL, ring)
            kk, tt = kv.clone(), tail.clone()
            V["ring"].kc.kpool_decode_update_and_maybe_write_cache_batched(kk, tt, ts, key, sc, ape, sm, pos, POOL, HEAD,
                                                                          round_scale=ROUND)
            assert torch.equal(rk, kk) and torch.equal(rt, tt), (kind, arg, ring)
            if ring == POOL:  # stock sizing: the patched kernel is bitwise the production kernel
                pk, pt = kv.clone(), tail.clone()
                V["prod"].kc.kpool_decode_update_and_maybe_write_cache_batched(pk, pt, ts, key, sc, ape, sm, pos, POOL,
                                                                              HEAD, round_scale=ROUND)
                assert torch.equal(pk, kk) and torch.equal(pt, tt), ("prod != ring at ring == pool", kind, arg)
            n_cases += 1
    return f"{n_cases} kernel runs == reference (ring 16 and 32); ring == pool runs bitwise equal to the prod kernel"


@test
def test_leading_invalid_tail_slot():
    """Upstream test_leading_invalid_tail_slot at ring = 2 pools (pool 16)."""
    POOL, ring, NB = 16, 32, 32
    torch.manual_seed(0)
    B, nn, blocks = 2, 4, [3, 5]
    pos = torch.tensor([[-1, 13, 14, 15], [4, 5, 6, 7]], dtype=torch.int32, device=DEV)
    safe = torch.where(pos >= 0, pos, 0)
    ts = (torch.tensor(blocks, device=DEV, dtype=torch.int32).unsqueeze(1) * ring + safe % ring).to(torch.int32)
    ts[0, 0] = -1
    sm = torch.full((B, nn), -1, dtype=torch.int32, device=DEV); sm[0, 3] = 15
    key = torch.randn(B, nn, HEAD, dtype=torch.bfloat16, device=DEV)
    sc = torch.randn(B, nn, HEAD, dtype=torch.bfloat16, device=DEV)
    ape = torch.randn(POOL, HEAD, dtype=torch.float32, device=DEV)
    kv = new_kv(NB); tail = torch.zeros(NB, 2, ring, HEAD, dtype=torch.bfloat16, device=DEV)
    g = torch.Generator(device=DEV).manual_seed(42)
    for b in blocks:
        tail[b, 0, :13] = torch.randn(13, HEAD, dtype=torch.bfloat16, device=DEV, generator=g)
        tail[b, 1, :13] = torch.randn(13, HEAD, dtype=torch.bfloat16, device=DEV, generator=g)
    rk, rt = torch_reference(kv, tail, ts, key, sc, ape, sm, pos, POOL, ring)
    V["ring"].kc.kpool_decode_update_and_maybe_write_cache_batched(kv, tail, ts, key, sc, ape, sm, pos, POOL, HEAD,
                                                                  round_scale=ROUND)
    assert torch.equal(rk, kv) and torch.equal(rt, tail)


@test
def test_prefill_seed_ring_padded_stride_multi_request():
    """Prefill seed at production stride through the ring build's call-site argument: a 3-request prefill batch
    (lengths 3, 10, 21 -> in-progress pools of 3, 2, 1 tokens). Rows pos % ring of each request's last min(len, ring)
    tokens hold its K/score, the in-progress pool's rows equal upstream #58454's seed (which seeds the last kpool
    tokens), and nothing outside the requests' own blocks' ring rows is touched."""
    v, up = V["ring"], V["upstream"]
    ring = v.ring
    lens, blocks = [3, 10, 21], [2, 5, 1]
    torch.manual_seed(3)
    ks = [torch.randn(n, HEAD, dtype=torch.bfloat16, device=DEV) for n in lens]
    ss = [torch.randn(n, HEAD, dtype=torch.bfloat16, device=DEV) for n in lens]
    tsl = torch.cat([torch.tensor([b * ring + p % ring for p in range(n)], dtype=torch.int64, device=DEV)
                     for b, n in zip(blocks, lens)])
    k_cat, s_cat = torch.cat(ks), torch.cat(ss)
    SENT = -77.0
    bk_r, t_r = new_tail(8, ring, garbage_seed=None)
    bk_r.fill_(SENT)
    bk_u, t_u = new_tail(8, ring, garbage_seed=None)
    bk_u.fill_(SENT)
    v.kc.kpool_seed_tail_cache(t_r, k_cat, s_cat, tsl, seed_kpool(v, t_r), HEAD)
    up.kc.kpool_seed_tail_cache(t_u, k_cat, s_cat, tsl, KPOOL, HEAD)  # upstream call site passes index_kpool
    torch.cuda.synchronize()
    written = torch.zeros_like(bk_r, dtype=torch.bool)
    for b, n, k, s in zip(blocks, lens, ks, ss):
        for p in range(max(0, n - ring), n):
            assert torch.equal(t_r[b, 0, p % ring], k[p]) and torch.equal(t_r[b, 1, p % ring], s[p]), (b, p)
            for half in (0, 1):
                o = b * TAIL_BLOCK_ELEMS + half * ring * HEAD + (p % ring) * HEAD
                written[o:o + HEAD] = True
        for p in range(n - n % KPOOL if n % KPOOL else n, n):  # in-progress pool rows
            assert torch.equal(t_r[b, 0, p % ring], t_u[b, 0, p % ring]), ("K differs from upstream", b, p)
            assert torch.equal(t_r[b, 1, p % ring], t_u[b, 1, p % ring]), ("score differs from upstream", b, p)
    untouched = bool((bk_r[~written] == SENT).all())
    assert untouched, "seed wrote outside the requests' ring rows"
    dense = (blocks[1] * 2 * ring + 3) * HEAD  # where a dense [blocks, 2, ring, head] layout would have put row 3
    assert bool((bk_r[dense:dense + HEAD] == SENT).all())
    rows_up = int((bk_u != SENT).view(-1, HEAD).any(1).sum())
    rows_r = int(written.view(-1, HEAD).any(1).sum())
    return f"ring build wrote {rows_r} rows (last min(len,{ring}) tokens x K/score), upstream {rows_up}; in-progress pool rows identical"


SIM_REQS = [
    # rid, tail block, pool-slot base, final length, prompt, cached prefix, prefill chunk, join step, plain decode
    dict(rid=0, block=1, loc_base=0, total=260, prompt=37, seed=0),
    dict(rid=1, block=4, loc_base=128, total=300, prompt=64, chunk=32, seed=1),  # chunked prefill
    dict(rid=2, block=6, loc_base=256, total=240, prompt=101, cached=64, chunk=32, join=3, seed=2),  # prefix resume
    dict(rid=3, block=2, loc_base=384, total=220, prompt=2, join=5, seed=3),
]


def sim_all(reqs_spec, sched_seed, steps, k_choices=(4, 5, 7), label=""):
    num_pages, n_tail = 8, 8
    ape = torch.randn(KPOOL, HEAD, dtype=torch.float32, device=DEV, generator=torch.Generator(device=DEV).manual_seed(9))
    proto = [Req(**r) for r in reqs_spec]
    sched = make_schedule(proto, steps, sched_seed, k_choices)
    out = {}
    runs = (("prod", V["prod"], V["prod"].ring), ("ring", V["ring"], V["ring"].ring),
            ("upstream", V["upstream"], V["upstream"].ring), ("ring-kernel@ring4", V["ring"], KPOOL))
    kvs = {}
    for name, v, ring in runs:
        kv, reqs = run_sim(v, ring, reqs_spec, sched, ape, num_pages, n_tail)
        out[name] = compare(v, kv, reqs, ape, num_pages)
        kvs[name] = kv
        lens = [r.acc for r in reqs]
    rejections = sum(1 for per in sched for (kk, a, _, _) in per.values() if kk and a < kk)
    return out, kvs, lens, rejections


@test
def test_spec_decode_simulation_production_shape():
    """4 concurrent requests (one chunked, one prefix-resumed from a 64-token cached prefix into a garbage tail block,
    one with a 2-token prompt joining late), adaptive k in {4,5,7}, random acceptance. prod ring 4: corrupted pools;
    ring 16 and upstream: every pool exact, and the two KV caches bitwise identical (including pools written by
    rejected drafts)."""
    msgs = []
    for seed in (11, 12, 13):
        out, kvs, lens, rej = sim_all(SIM_REQS, seed, 90)
        assert out["ring"][0] == 0 and out["upstream"][0] == 0, out
        assert torch.equal(kvs["ring"], kvs["upstream"]), "ring build != upstream kernels"
        assert out["prod"][0] > 0 and out["ring-kernel@ring4"][0] > 0, out
        assert torch.equal(kvs["prod"], kvs["ring-kernel@ring4"]), "ring kernel at ring 4 != prod kernel"
        msgs.append(f"seed {seed}: lens {lens}, {rej} rejecting verifies; pools wrong/compared (decode-built wrong): "
                    + ", ".join(f"{k} {b}/{t} ({d})" for k, (b, t, d) in out.items()))
    return " | ".join(msgs)


@test
def test_mixed_plain_and_spec_batch():
    """Non-uniform batch every step: request 3 decodes one token per step (next_n = 1) while the others verify 1+k
    (padded [B, lmax] layout from the image's scatter helpers)."""
    reqs = [dict(r) for r in SIM_REQS]
    reqs[3]["plain"] = True
    out, kvs, lens, rej = sim_all(reqs, 21, 80)
    assert out["ring"][0] == 0 and out["upstream"][0] == 0 and torch.equal(kvs["ring"], kvs["upstream"]), out
    assert out["prod"][0] > 0, out
    return ", ".join(f"{k} {b}/{t}" for k, (b, t, d) in out.items())


@test
def test_plain_decode_only():
    """next_n = 1 only (no speculation): every build exact, and the ring build's KV is bitwise the prod build's."""
    out, kvs, lens, rej = sim_all(SIM_REQS, 31, 120, k_choices=(0,))
    assert rej == 0 and all(b == 0 for b, t, d in out.values()), out
    assert torch.equal(kvs["prod"], kvs["ring"]) and torch.equal(kvs["ring"], kvs["upstream"])
    return f"lens {lens}; " + ", ".join(f"{k} {b}/{t}" for k, (b, t, d) in out.items())


@test
def test_prefix_resume_one_token_suffix():
    """Prefix-cache resume with the shortest fresh suffix: cached 64 tokens (pool-aligned, like every scheduler-block
    hit), prompt 65 -> the fresh prefill is ONE token and seeds the new pool; the tail block starts as garbage."""
    reqs = [dict(rid=0, block=3, loc_base=0, total=200, prompt=65, cached=64, seed=5),
            dict(rid=1, block=7, loc_base=128, total=200, prompt=128, cached=128, seed=6)]  # suffix 0 -> clamp to 1
    reqs[1]["prompt"], reqs[1]["cached"] = 129, 128
    out, kvs, lens, rej = sim_all(reqs, 41, 80)
    assert out["ring"][0] == 0 and out["upstream"][0] == 0 and torch.equal(kvs["ring"], kvs["upstream"]), out
    return ", ".join(f"{k} {b}/{t}" for k, (b, t, d) in out.items())


@test
def test_ring_size_bound():
    """The correctness bound is ring >= kpool + num_spec (11 here), not "any ring > kpool": the ring build's kernels
    and call-site seed, driven at ring 8 (< 11) corrupt pools like prod; at ring 12 (>= 11, not a power of two, so
    not upstream's choice) and 16 every pool is exact and the KV cache is bitwise upstream's. Upstream rounds to a
    power of two only so that the ring divides the attention block (tested in test_build_ring_sizes)."""
    res = {}
    ape = torch.randn(KPOOL, HEAD, dtype=torch.float32, device=DEV, generator=torch.Generator(device=DEV).manual_seed(9))
    proto = [Req(**r) for r in SIM_REQS]
    sched = make_schedule(proto, 90, 11, (4, 5, 7))
    kvs = {}
    for ring in (8, 12, 16):
        kv, reqs = run_sim(V["ring"], ring, SIM_REQS, sched, ape, 8, 8)
        res[ring] = compare(V["ring"], kv, reqs, ape, 8)
        kvs[ring] = kv
    kv_up, _ = run_sim(V["upstream"], 16, SIM_REQS, sched, ape, 8, 8)
    assert res[8][0] > 0, res
    assert res[12][0] == 0 and res[16][0] == 0, res
    assert torch.equal(kvs[12], kv_up) and torch.equal(kvs[16], kv_up), "exact rings disagree with upstream"
    return ", ".join(f"ring {r}: {b}/{t} pools wrong" for r, (b, t, _) in res.items()) + "; ring 12/16 == upstream"


# ---------------------------------------------------------------------------------------- slot mapping (ported)
def _cam(per_req, own_blocks):
    positions = torch.cat([torch.tensor(p, dtype=torch.int64) for p in per_req])
    n = positions.numel()
    lens = [len(p) for p in per_req]
    qsl = torch.zeros(len(per_req) + 1, dtype=torch.int64)
    torch.cumsum(torch.tensor(lens, dtype=torch.int64), 0, out=qsl[1:])
    bt = torch.zeros(len(per_req), 64, dtype=torch.int32)
    bt[:, 0] = torch.tensor(own_blocks, dtype=torch.int32)
    seq = torch.tensor([max(p) + 1 for p in per_req], dtype=torch.int64)
    return CommonAttentionMetadata(
        query_start_loc=qsl.to(DEV), query_start_loc_cpu=qsl, seq_lens=seq.to(DEV), num_reqs=len(per_req),
        num_actual_tokens=n, max_query_len=max(lens), max_seq_len=int(seq.max()), block_table_tensor=bt.to(DEV),
        slot_mapping=torch.full((n + 4,), -1, dtype=torch.int64, device=DEV), positions=positions.to(DEV),
    )


@test
def test_tail_slot_mapping_follows_ring():
    """compute_kpool_tail_slot_mapping / KpoolTailMetadataBuilder.build (image code, V2 path) with the ring spec:
    every token lands in its own block at pos % ring; padding keeps -1; requests never share a slot."""
    ring = V["ring"].ring
    per_req, own = [list(range(10)), list(range(40, 75))], [5, 9]
    cam = _cam(per_req, own)
    builder = object.__new__(KpoolTailMetadataBuilder)
    builder.kv_cache_spec = SimpleNamespace(block_size=ring)
    meta = KpoolTailMetadataBuilder.build(builder, 0, cam)
    out = meta.slot_mapping.cpu()
    off, sets = 0, []
    for req, prompt in enumerate(per_req):
        s = set()
        for i, pos in enumerate(prompt):
            slot = int(out[off + i])
            assert slot // ring == own[req] and slot % ring == pos % ring, (req, pos, slot)
            s.add(slot)
        sets.append(s)
        off += len(prompt)
    assert not sets[0] & sets[1]
    assert bool((out[cam.num_actual_tokens:] == -1).all())
    return f"ring {ring}: {off} tokens mapped to own blocks, padding intact"


@test
def test_ring_mirror_rejected_completing_draft():
    """Upstream TailRingMirror test (CPU arithmetic) at kpool 4 with ring 4 vs the production ring."""
    def run(ring_size):
        class M:
            def __init__(self):
                self.k = torch.full((2, ring_size, 3), float("nan")); self.s = torch.full((2, ring_size, 3), float("nan"))

            def stash(self, slot, pos, k, s):
                self.k[slot // ring_size, pos % ring_size] = k; self.s[slot // ring_size, pos % ring_size] = s

            def complete(self, slot, pos, k, s):
                b, start = slot // ring_size, pos - (KPOOL - 1)
                kk = torch.stack([self.k[b, (start + i) % ring_size] for i in range(KPOOL)])
                ss = torch.stack([self.s[b, (start + i) % ring_size] for i in range(KPOOL)])
                kk[-1], ss[-1] = k, s
                return (kk * torch.softmax(ss, 0)).sum(0)

        def kvt(req, pos):
            return (torch.tensor([pos + 100.0 * req, pos + 0.5, 2.0 * pos + 0.25]),
                    torch.tensor([0.1 * (pos + 1) + req, 0.2 * pos, 0.05 * pos]))
        truth, ring = M(), M()
        slot = lambda pos: ring_size + pos % ring_size  # noqa: E731
        for pos in range(4, 7):
            truth.stash(slot(pos), pos, *kvt(0, pos)); ring.stash(slot(pos), pos, *kvt(0, pos))
        exp = truth.complete(slot(7), 7, *kvt(0, 7))
        for pos in range(7, 7 + NUM_SPEC + 1):  # draft 7 completes the pool and is rejected; drafts to 7 + num_spec
            k, s = kvt(9, pos)
            if pos % KPOOL == KPOOL - 1:
                ring.complete(slot(pos), pos, k, s)
            ring.stash(slot(pos), pos, k, s)
        return torch.allclose(ring.complete(slot(7), 7, *kvt(0, 7)), exp)
    r4, r16 = run(KPOOL), run(V["ring"].ring)
    assert not r4 and r16, (r4, r16)
    return f"redo exact: ring 4 {r4}, ring {V['ring'].ring} {r16}"


# ---------------------------------------------------------------------------------------- non-KV memory (V2 runner)
@test
def test_v2_block_table_memory():
    """The V2 model runner sizes every group's block table as cdiv(max_model_len, block_size) columns
    (model_runner.py: get_block_table_width), the one-block tail included. Measure the GPU bytes BlockTables
    allocates for production's groups (MLA/indexer 4608, tail, 4 x KDA 4608 align, drafter 1152) at max_model_len
    1,000,000 and max_num_seqs 4, with the tail at 4 and at 16."""
    from vllm.utils.math_utils import cdiv
    from vllm.v1.worker.block_table import get_block_table_width
    from vllm.v1.worker.gpu.block_table import BlockTables

    def alloc(tail_bs, max_len=1_000_000, reqs=4):
        sizes = [4608, tail_bs, 4608, 4608, 4608, 4608, 1152]
        widths = []
        for i, bs in enumerate(sizes):
            n = cdiv(max_len, bs)
            if 2 <= i <= 5:  # MambaSpec align: +num_speculative_blocks, token_alignment=None
                widths.append(get_block_table_width(n + NUM_SPEC, bs, token_alignment=None))
            else:
                widths.append(get_block_table_width(n, bs))
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        bt = BlockTables(block_sizes=sizes, max_num_reqs=reqs, max_num_batched_tokens=16384,
                         max_num_blocks_per_group=widths, device=torch.device(DEV), kernel_block_sizes=sizes)
        torch.cuda.synchronize()
        used = torch.cuda.memory_allocated() - before
        tail_w = bt.block_tables[1].gpu.shape[1]
        del bt
        torch.cuda.empty_cache()
        return used, tail_w
    (u4, w4), (u16, w16) = alloc(4), alloc(16)
    assert w16 * 4 <= w4 + 32 and u16 < u4, (u4, w4, u16, w16)
    return (f"tail width {w4} -> {w16} columns; BlockTables GPU bytes {u4:,} -> {u16:,} "
            f"({(u4 - u16) / 2**20:.2f} MiB less per rank)")


def main() -> int:
    print(f"torch {torch.__version__}; device {torch.cuda.get_device_name(0)}; image vllm "
          f"{__import__('vllm').__version__}")
    V.update(build_variants())
    for t in TESTS:
        t()
    torch.cuda.synchronize()
    print(f"peak GPU memory allocated: {torch.cuda.max_memory_allocated() / 2**20:.1f} MiB")
    ok = sum(1 for _, p, _ in RESULTS if p)
    print(f"== {ok}/{len(RESULTS)} passed")
    for n, p, m in RESULTS:
        print(f"  {'PASS' if p else 'FAIL'} {n}: {m}")
    return 0 if ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
