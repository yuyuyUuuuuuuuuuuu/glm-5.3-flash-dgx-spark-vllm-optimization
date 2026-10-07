"""GLM53_PREFILL_QUICKWINS (glm53_prefill_quickwins.py, docs/PREFILL_QUICKWINS.md): wiring and bitwise parity.

Run in the production image with production's module set (tests/qw/prod_env.sh: launcher overlay exl3.py, SM90_KV=1
mounts). Every parity check is torch.equal against production's own statement on the same inputs.

W.0 plugin path (fresh process, as vLLM loads it): integrate.plugin_register() with GLM53_PREFILL_QUICKWINS=all before
    any target module is imported -> each target is patched when first imported (import hook); unset -> nothing patched.
W.1 wiring: env parsing; the production functions' fingerprints are the verified ones and every edit anchor occurs once;
    install patches all functions (flag _glm53_qw), the recompiled ASTs differ from production's only in the edited
    statements; uninstall restores production's function objects; an unverified function is refused (nothing set).
W.2 mla_bmm: W_UK / W_UV absorption bmm with production's layouts (views of kv_b_proj.weight), T in TS, 3 seeds:
    Triton == torch.bmm; the patched _v_up_proj == production's; below MIN_T / during capture -> torch.bmm.
W.3 mla_index: the patched forward_mqa (stub wrapper) fills kv_indices exactly like production's, realistic top-k rows
    (ctx <= 2048 all prior tokens + -1 tail; longer: 2048 unique positions), several requests, out-of-range blocks.
W.4 kda_conv: the helper's q, k, v and the conv state == production's merged conv + split (then .contiguous()),
    varlen batches with / without initial state, vLLM's own conv metadata; q, k, v contiguous.
W.5 mhc_aux / mhc_mean: fused post->pre == post then standalone pre (all 4 outputs), T in {17, 64, 257, 1791, 13824};
    the aux helper returns production's value and hands over exactly post's output; the Triton post(+mean) kernel ==
    production's tilelang post and aten mean (streams scaled 2^-12..2^11).
W.6 idx_gate: this fork's head-gate op at M >= GATE_MIN_M == production's torch.mm(x.float(), w32), through the
    registered custom op; M below the gate -> production; opt-dense: a non-finite production result leaves the M
    undecided, a forced mismatch keeps only that M on production (GATE_MAX_BAD_M distinct ones turn it off), fp32 mode.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import os
import sys
import textwrap
import types

import torch

sys.path.insert(0, "/w")
import glm53_prefill_quickwins as Q  # noqa: E402

DEV = "cuda"
TS = tuple(int(v) for v in os.environ.get("QW_TS", "256,1791,4608,13824").split(","))
FAILS = []
NCHK = [0]


def ck(cond, msg):
    NCHK[0] += 1
    if not cond:
        FAILS.append(msg)
        print("  FAIL:", msg, flush=True)


def eq(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(a, b))


# ---------------------------------------------------------------------------------------------------------------------
W0_CHILD = r"""
import importlib, os, sys, logging
sys.path.insert(0, "/w")
logging.basicConfig(level=logging.INFO)
targets = {"vllm.model_executor.layers.attention.mla_attention": ["MLAAttention.forward_impl", "MLAAttention._v_up_proj"],
           "vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90": ["FlashInferMLASparseSM90Impl.forward_mqa"],
           "vllm.models.glm5next.nvidia.kda": ["Glm5NextLinearAttention._forward"],
           "vllm.models.glm5next.nvidia.model": ["Glm5NextModel.forward", "Glm5NextDecoderLayer.forward"]}
pre = [m for m in targets if m in sys.modules]
import integrate
integrate.plugin_register()
res = {}
for m, quals in targets.items():
    mod = importlib.import_module(m)
    for q in quals:
        o = mod
        for p in q.split("."):
            o = getattr(o, p)
        res[q] = bool(getattr(o, "_glm53_qw", False))
import glm53_gemv_install as G
res["_head_gate_impl"] = bool(getattr(G._head_gate_impl, "_glm53_qw", False))
print("W0 preimported", pre)
print("W0 RESULT", sorted(res.items()))
"""


def w0_plugin_path():
    import subprocess
    print("W.0 plugin path", flush=True)
    for env_val, want in (("all", True), (None, False)):
        env = dict(os.environ)
        env.pop("TF_EXL3_MOE", None)
        env.pop(Q.ENV, None)
        if env_val is not None:
            env[Q.ENV] = env_val
        r = subprocess.run([sys.executable, "-c", W0_CHILD], env=env, capture_output=True, text=True, timeout=600)
        lines = [l for l in r.stdout.splitlines() if l.startswith("W0 ")]
        print("  ", env_val, lines, flush=True)
        res = [l for l in lines if l.startswith("W0 RESULT")]
        ok = r.returncode == 0 and res and all(v == want for _k, v in eval(res[0][len("W0 RESULT "):]))
        pre = [l for l in lines if l.startswith("W0 preimported")]
        ck(bool(pre) and pre[0].endswith("[]"), f"W.0 targets imported before the plugin: {pre}")
        ck(bool(ok), f"W.0 {Q.ENV}={env_val!r}: {lines} rc={r.returncode} {r.stderr[-800:]}")
        if env_val and not ok:
            print(r.stderr[-3000:])


def w1_wiring():
    print("W.1 wiring", flush=True)
    for raw, want in ((None, set()), ("", set()), ("0", set()), ("off", set()), ("all", set(Q.ITEMS)),
                      ("mla_bmm, kda_conv", {"mla_bmm", "kda_conv"}), ("MHC_AUX", {"mhc_aux"})):
        env = {} if raw is None else {Q.ENV: raw}
        ck(set(Q.parse_env(env)) == want, f"parse_env({raw!r})")
    for raw in ("bogus", "mla_bmm,nope"):
        try:
            Q.parse_env({Q.ENV: raw})
            ck(False, f"parse_env({raw!r}) did not raise")
        except ValueError:
            pass
    ck(Q.parse_min_t({}) == Q.DEFAULT_MIN_T, "min_t default")
    for raw in ("64", "1"):
        try:
            Q.parse_min_t({Q.ENV_MIN_T: raw})
            ck(False, f"parse_min_t({raw}) did not raise")
        except ValueError:
            pass
    mods = {}
    for item, entries in Q.PLAN.items():
        for modname, qual, edits in entries:
            m = mods.setdefault(modname, importlib.import_module(modname))
            _o, _n, fn = Q._get_attr(m, qual)
            fp = Q.source_fingerprint(fn)
            src = inspect.getsource(fn)
            print(f"  {item:9s} {modname}.{qual}: fingerprint {fp}, anchors {[src.count(o) for o, _ in edits]}")
            ck(fp in Q.VERIFIED[(item, qual)], f"{qual} fingerprint {fp} not verified")
            ck(all(src.count(o) == 1 for o, _ in edits), f"{qual} anchors")
    conv = importlib.import_module(Q.M_CONV)
    fp = Q.source_fingerprint(conv.causal_conv1d_fn)
    print(f"  kda_conv  {Q.M_CONV}.causal_conv1d_fn: fingerprint {fp}")
    ck(fp in Q.VERIFIED[("kda_conv", "causal_conv1d_fn")], "causal_conv1d_fn fingerprint")

    origs = {}
    for item, entries in Q.PLAN.items():
        for modname, qual, _e in entries:
            origs[(modname, qual)] = Q._get_attr(mods[modname], qual)[2]
    r = Q.install(Q.ITEMS, 256)
    ck(not r["pending"], f"pending {r['pending']}")
    ck(not Q._STATE["refused"], f"refused {Q._STATE['refused']}")
    for item, entries in Q.PLAN.items():
        for modname, qual, edits in entries:
            new = Q._get_attr(mods[modname], qual)[2]
            ck(getattr(new, "_glm53_qw", False) and new._glm53_qw_orig is origs[(modname, qual)], f"{qual} installed")
            # the recompiled function = production's source with exactly the edits applied (AST level)
            src = inspect.getsource(origs[(modname, qual)])
            for o, n in edits:
                src = src.replace(o, n)
            src = textwrap.dedent(src)
            ck(ast.dump(ast.parse(src)) == ast.dump(ast.parse(textwrap.dedent(inspect.getsource(new)))),
               f"{qual} recompiled AST")
            ck(new.__globals__ is mods[modname].__dict__, f"{qual} globals")
            # every name the new statements use exists: in the original statement, the function's own names, the
            # module globals (incl. the injected helpers) or builtins -> no NameError on the paths tests do not run
            fn_tree = ast.parse(textwrap.dedent(inspect.getsource(origs[(modname, qual)])))
            own = {n.id for n in ast.walk(fn_tree) if isinstance(n, ast.Name)}
            own |= {a.arg for n in ast.walk(fn_tree) if isinstance(n, ast.arguments)
                    for a in n.args + n.kwonlyargs + n.posonlyargs}
            known = own | set(mods[modname].__dict__) | set(dir(__builtins__) if not isinstance(__builtins__, dict)
                                                             else __builtins__)
            for _o, n_ in edits:
                used = {x.id for x in ast.walk(ast.parse(textwrap.dedent(n_))) if isinstance(x, ast.Name)}
                ck(used <= known, f"{qual}: edit uses unknown names {sorted(used - known)}")
    # kda _forward keeps production's eager_break_during_capture decoration
    k = mods[Q.M_KDA].Glm5NextLinearAttention
    ck(inspect.unwrap(k._forward).__code__.co_filename.startswith("<glm53-quickwins"), "kda _forward recompiled")
    ck(inspect.getsource(k._forward).lstrip().startswith("@eager_break_during_capture"), "kda decorator kept")
    ck(Q._CONV_OUT.get("fn") is not None and "out=None" in inspect.getsource(Q._CONV_OUT["fn"]), "conv out variant")
    for item in Q.ITEMS:
        Q.uninstall_item(item)
    for (modname, qual), fn in origs.items():
        ck(Q._get_attr(mods[modname], qual)[2] is fn, f"{qual} restored")
    # refusal: a function whose fingerprint is not verified -> nothing set
    saved = dict(Q.VERIFIED)
    try:
        Q.VERIFIED[("mla_bmm", "MLAAttention._v_up_proj")] = frozenset({"deadbeefdeadbeef"})
        try:
            Q.install_item("mla_bmm", mods[Q.M_MLA])
            ck(False, "unverified install did not refuse")
        except Q.Refused:
            pass
        ck(not getattr(mods[Q.M_MLA].MLAAttention.forward_impl, "_glm53_qw", False), "refusal is all-or-nothing")
    finally:
        Q.VERIFIED.clear()
        Q.VERIFIED.update(saved)
    Q.install(Q.ITEMS, 256)
    return mods


# ---------------------------------------------------------------------------------------------------------------------
def w2_mla_bmm(mods):
    print("W.2 mla_bmm", flush=True)
    N, P, L, V = 32, 256, 512, 256
    MLA = mods[Q.M_MLA].MLAAttention
    for seed in range(3):
        g = torch.Generator(device=DEV).manual_seed(100 + seed)
        w = (torch.randn(N * (P + V), L, device=DEV, generator=g) * 0.05).to(torch.bfloat16)  # kv_b_proj.weight
        kvb = w.T.view(L, N, P + V)
        W_UK, W_UV = kvb.split([P, V], dim=-1)
        W_UK_T = W_UK.permute(1, 2, 0)
        W_UV_t = W_UV.transpose(0, 1)
        for T in TS + (100,):
            qfull = torch.randn(T, N * P + 64, device=DEV, generator=g).to(torch.bfloat16)   # q as a padded view
            q = qfull[:, : N * P].view(T, N, P) if seed == 2 else qfull[:, : N * P].contiguous().view(T, N, P)
            a_nt = q.transpose(0, 1)
            ref = torch.empty(N, T, L, device=DEV, dtype=torch.bfloat16)
            torch.bmm(a_nt, W_UK_T, out=ref)
            got = torch.full_like(ref, 7.0)
            before = dict(Q.STATS)
            Q._qw_mla_bmm(a_nt, W_UK_T, got)
            fast = Q.STATS["mla_bmm_fast"] - before["mla_bmm_fast"]
            ck(eq(got, ref), f"W_UK T={T} seed={seed}: max diff {(got.float() - ref.float()).abs().max().item()}")
            ck(fast == (1 if T >= 256 else 0), f"W_UK T={T} fast path taken {fast}")
            # W_UV through the patched _v_up_proj vs production's
            attn = torch.randn(T, N, L, device=DEV, generator=g).to(torch.bfloat16)
            me = types.SimpleNamespace(num_heads=N, kv_lora_rank=L, v_head_dim=V, W_UV=W_UV_t,
                                       is_aiter_triton_fp4_bmm_enabled=False, is_aiter_triton_fp8_bmm_enabled=False)
            o_ref = torch.empty(T, N * V, device=DEV, dtype=torch.bfloat16)
            MLA._v_up_proj._glm53_qw_orig(me, attn, o_ref)
            o_got = torch.full_like(o_ref, 3.0)
            MLA._v_up_proj(me, attn, o_got)
            ck(eq(o_got, o_ref), f"W_UV T={T} seed={seed}: max diff {(o_got.float() - o_ref.float()).abs().max().item()}")
        print(f"  seed {seed}: T {TS + (100,)} W_UK / W_UV bit-identical: {not FAILS}", flush=True)
    # capture -> production's torch.bmm (no Triton launch inside the graph)
    T = 512
    q = torch.randn(T, N, P, device=DEV).to(torch.bfloat16)
    out = torch.empty(N, T, L, device=DEV, dtype=torch.bfloat16)
    Q._qw_mla_bmm(q.transpose(0, 1), W_UK_T, out)          # warm (Triton compiled, cuBLAS handle)
    torch.bmm(q.transpose(0, 1), W_UK_T, out=out)
    torch.cuda.synchronize()
    gph = torch.cuda.CUDAGraph()
    before = Q.STATS["mla_bmm_fast"]
    with torch.cuda.graph(gph):
        Q._qw_mla_bmm(q.transpose(0, 1), W_UK_T, out)
    ck(Q.STATS["mla_bmm_fast"] == before, "fast path taken during capture")


# ---------------------------------------------------------------------------------------------------------------------
def _topk_rows(ctx_list, W, g, block_size, n_blocks_req, bad_frac=0.0):
    """Top-k slot rows like the kpool indexer's: ctx <= W -> positions 0..ctx-1 then -1; else W unique positions < ctx.
    bad_frac: fraction of rows with a few entries pointing past the request's block table (-> invalid)."""
    rows = torch.full((len(ctx_list), W), -1, dtype=torch.int32)
    for i, c in enumerate(ctx_list):
        if c <= W:
            rows[i, :c] = torch.arange(c, dtype=torch.int32)
        else:
            rows[i] = torch.randperm(c, generator=g)[:W].to(torch.int32)
        if bad_frac and (i % int(1 / bad_frac)) == 0:
            rows[i, 5:9] = n_blocks_req * block_size + 3          # block id out of range -> invalid, compacted out
            rows[i, 11] = -1                                        # interior -1 gap
    return rows


def w3_mla_index(mods):
    print("W.3 mla_index", flush=True)
    sm = mods[Q.M_SM90]
    Impl = sm.FlashInferMLASparseSM90Impl
    conv = sm.triton_convert_req_index_to_global_index
    W, BS = 2048, 64
    g = torch.Generator().manual_seed(7)
    for reqs in ([13824], [4096, 5000, 4728], [1791], [300, 1491]):
        n = sum(reqs)
        ctx_all, req_ids = [], []
        start_ctx = [0 if i == 0 else 2000 * i for i in range(len(reqs))]   # later requests have prior context
        for r, (L_, s0) in enumerate(zip(reqs, start_ctx)):
            ctx_all += [s0 + j + 1 for j in range(L_)]
            req_ids += [r] * L_
        max_blocks = (max(ctx_all) + BS - 1) // BS + 2
        block_table = torch.stack([torch.randperm(4 * max_blocks, generator=g)[:max_blocks] for _ in reqs]).to(
            torch.int32).to(DEV)
        rows = _topk_rows(ctx_all, W, g, BS, max_blocks, bad_frac=0.05).to(DEV)
        req_id = torch.tensor(req_ids, dtype=torch.int32, device=DEV)
        md = types.SimpleNamespace(req_id_per_token=req_id, block_table=block_table, block_size=BS)
        cap_tokens = n + 64
        # production's statements, verbatim, into a reference buffer
        ref_buf = torch.full((cap_tokens * W,), 12345, dtype=torch.int32, device=DEV)
        slots, _vc = conv(md.req_id_per_token[:n], md.block_table, rows, BLOCK_SIZE=BS, NUM_TOPK_TOKENS=W,
                          return_valid_counts=True)
        ref_buf[: n * W].copy_(slots.reshape(-1).clamp_(min=0).to(torch.int32))
        # the patched forward_mqa with a stub wrapper (records what run() sees)
        seen = {}

        class _Wrapper:
            def run(self, q_nope, q_pe, ckv, kpe, **kw):
                seen["kv"] = st.kv_indices.clone()
                return torch.zeros(1, device=DEV)

        st = types.SimpleNamespace(kv_indices=torch.full((cap_tokens * W,), 12345, dtype=torch.int32, device=DEV),
                                   wrapper=_Wrapper())
        me = types.SimpleNamespace(topk_indices_buffer=torch.full((cap_tokens, W), -1, dtype=torch.int32, device=DEV),
                                   num_heads=32, qk_rope_head_dim=0, kv_lora_rank=512, head_size=512,
                                   use_fp8_kv_cache=True)
        me.topk_indices_buffer[:n].copy_(rows)
        layer = types.SimpleNamespace(_k_scale_float=1.0)
        kv_cache = torch.zeros(4, BS, 512, dtype=torch.uint8, device=DEV)
        q = (torch.zeros(n, 32, 512, device=DEV, dtype=torch.bfloat16), torch.zeros(n, 32, 0, device=DEV,
                                                                                    dtype=torch.bfloat16))
        saved = sm._SM90_STATE
        sm._SM90_STATE = st
        try:
            before = Q.STATS["mla_index_fast"]
            Impl.forward_mqa(me, q, kv_cache, md, layer)
            fast = Q.STATS["mla_index_fast"] - before
            ck(eq(seen["kv"], ref_buf), f"kv_indices reqs={reqs}: {(seen['kv'] != ref_buf).sum().item()} differ")
            ck(fast == (1 if n >= 256 else 0), f"mla_index fast path reqs={reqs}: {fast}")
            # production's own forward_mqa gives the same buffer
            st.kv_indices.fill_(12345)
            Impl.forward_mqa._glm53_qw_orig(me, q, kv_cache, md, layer)
            ck(eq(seen["kv"], ref_buf), f"production forward_mqa reqs={reqs}")
        finally:
            sm._SM90_STATE = saved
        print(f"  reqs {reqs}: kv_indices bit-identical ({n} rows x {W})", flush=True)


# ---------------------------------------------------------------------------------------------------------------------
def w4_kda_conv(mods):
    print("W.4 kda_conv", flush=True)
    conv_mod = importlib.import_module(Q.M_CONV)
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata
    P, WID = 4096, 4
    g = torch.Generator(device=DEV).manual_seed(11)
    layer = types.SimpleNamespace(local_projection_size=P)
    for reqs, init, sd in (([13824], [False], False), ([13824], [True], False), ([13824], [True], True),
                           ([1791], [True], False), ([5000, 8824], [True, False], True),
                           ([300, 1491], [False, True], False)):
        T = sum(reqs)
        proj = torch.randn(T, 3 * P + 288, device=DEV, generator=g).to(torch.bfloat16)   # in_proj output (padded)
        qkv = proj[:, : 3 * P]
        wgt = (torch.randn(3 * P, WID, device=DEV, generator=g) * 0.3).to(torch.bfloat16).contiguous()
        n_lines = 8
        if sd:   # conv state stored (lines, width-1, dim): production's SD layout, transposed view (kda.py _forward)
            state0 = torch.randn(n_lines, WID - 1, 3 * P, device=DEV, generator=g).to(torch.bfloat16).transpose(-1, -2)
        else:    # DS layout (lines, dim, width-1)
            state0 = torch.randn(n_lines, 3 * P, WID - 1, device=DEV, generator=g).to(torch.bfloat16)
        qsl_cpu = torch.tensor([0] + list(torch.tensor(reqs).cumsum(0).tolist()), dtype=torch.int32)
        qsl = qsl_cpu.to(DEV)
        cache_idx = torch.tensor([3, 5][: len(reqs)], dtype=torch.int32, device=DEV)
        has_init = torch.tensor(init, dtype=torch.bool, device=DEV)
        nums, bptr, tptr = compute_causal_conv1d_metadata(qsl_cpu, device=torch.device(DEV))
        md = types.SimpleNamespace(nums_dict=nums, batch_ptr=bptr, token_chunk_offset_ptr=tptr)
        # production: merged conv, transpose, split, then FLA's .contiguous()
        st_ref = state0.clone() if not sd else state0.transpose(-1, -2).clone().transpose(-1, -2)
        out = conv_mod.causal_conv1d_fn(qkv.transpose(0, 1), wgt, None, activation="silu", conv_states=st_ref,
                                        has_initial_state=has_init, cache_indices=cache_idx, query_start_loc=qsl,
                                        metadata=md).transpose(0, 1)
        qr, kr, vr = (t.contiguous() for t in out.split(P, dim=-1))
        st_got = state0.clone() if not sd else state0.transpose(-1, -2).clone().transpose(-1, -2)
        before = Q.STATS["kda_conv_fast"]
        qg, kg, vg = Q._qw_kda_conv(layer, qkv, wgt, None, st_got, has_init, cache_idx, qsl, md,
                                    conv_mod.causal_conv1d_fn)
        ck(Q.STATS["kda_conv_fast"] - before == 1, f"kda_conv fast path reqs={reqs}")
        for nm, a, b in (("q", qg, qr), ("k", kg, kr), ("v", vg, vr)):
            ck(eq(a, b), f"kda {nm} reqs={reqs} init={init} sd={sd}")
            ck(a.is_contiguous(), f"kda {nm} contiguous")
        ck(eq(st_got, st_ref) and st_got.stride() == st_ref.stride(), f"kda conv state reqs={reqs} init={init} sd={sd}")
        print(f"  reqs {reqs} init {init} state layout {'SD' if sd else 'DS'}: q/k/v and conv state bit-identical, "
              f"q/k/v contiguous", flush=True)
    # below MIN_T: production's expression (views of one merged output)
    T = 100
    qkv = torch.randn(T, 3 * P, device=DEV).to(torch.bfloat16)
    st = torch.zeros(2, 3 * P, WID - 1, device=DEV, dtype=torch.bfloat16)
    qsl = torch.tensor([0, T], dtype=torch.int32, device=DEV)
    before = Q.STATS["kda_conv_prod"]
    qg, _k, _v = Q._qw_kda_conv(layer, qkv, wgt, None, st, torch.tensor([False], device=DEV),
                                torch.tensor([0], dtype=torch.int32, device=DEV), qsl, None, conv_mod.causal_conv1d_fn)
    ck(Q.STATS["kda_conv_prod"] - before == 1 and not qg.is_contiguous(), "kda_conv below MIN_T -> production")


# ---------------------------------------------------------------------------------------------------------------------
def w5_mhc_aux(mods):
    print("W.5 mhc_aux", flush=True)
    import vllm.model_executor.kernels.mhc.tilelang as TL
    from vllm.model_executor.layers.mhc import hc_contract
    H, HC = 4096, 4
    g = torch.Generator(device=DEV).manual_seed(5)
    fn = (torch.randn(HC * 2 + HC * HC, HC * H, device=DEV, generator=g) * 0.02).float()
    hc_scale = torch.tensor([0.9, 1.1, 0.7], device=DEV)
    hc_base = (torch.randn(HC * 2 + HC * HC, device=DEV, generator=g) * 0.1).float()
    nw = (1 + 0.1 * torch.randn(H, device=DEV, generator=g)).to(torch.bfloat16)
    kw = dict(rms_eps=1e-5, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6, hc_post_mult_value=2.0, sinkhorn_repeat=20)
    for T in (17, 64, 257, 1791, 13824):
        x = torch.randn(T, H, device=DEV, generator=g).to(torch.bfloat16)
        # streams with per-token/stream scales 2^-12..2^11 (rounding of the fp32 sums is exercised)
        res = (torch.randn(T, HC, H, device=DEV, generator=g) * torch.exp2(
            torch.randint(-12, 12, (T, HC, 1), device=DEV, generator=g).float())).to(torch.bfloat16)
        post = torch.rand(T, HC, 1, device=DEV, generator=g) * 2
        comb = torch.softmax(torch.randn(T, HC, HC, device=DEV, generator=g), -1)
        r1, p1, c1, l1 = TL.mhc_fused_post_pre_tilelang(x, res, post, comb, fn, hc_scale, hc_base, kw["rms_eps"],
                                                        kw["hc_pre_eps"], kw["hc_sinkhorn_eps"],
                                                        kw["hc_post_mult_value"], kw["sinkhorn_repeat"], 1, 1, nw, 1e-5)
        full = TL.mhc_post_tilelang(x, res, post, comb)
        p2, c2, l2 = TL.mhc_pre_tilelang(full, fn, hc_scale, hc_base, kw["rms_eps"], kw["hc_pre_eps"],
                                         kw["hc_sinkhorn_eps"], kw["hc_post_mult_value"], kw["sinkhorn_repeat"], 1,
                                         nw, 1e-5)
        same = eq(r1, full) and eq(p1, p2) and eq(c1, c2) and eq(l1, l2)
        print(f"  T={T}: fused post->pre == post + standalone pre: residual {eq(r1, full)} post {eq(p1, p2)} "
              f"comb {eq(c1, c2)} layer_input {eq(l1, l2)}", flush=True)
        if T > 16:
            ck(same, f"mhc fused vs post+pre T={T}")
        # the aux helper: production's value, and hands over post's output
        lay = types.SimpleNamespace(n=HC, hc_post=lambda a, b, c, d: TL.mhc_post_tilelang(a, b, c, d))
        nxt = types.SimpleNamespace(mhc=True, is_mtp_layer=False, layer_idx=8, hc_pre=lambda *a, **k: None)
        model = types.SimpleNamespace(layers=[None] * 7 + [lay, nxt], end_layer=45, is_sequence_parallel=False)
        v, hs, rs, pp, cc = Q._qw_aux_post(model, 7, lay, x, res, post, comb, hc_contract)
        ck(eq(v, hc_contract(full, HC)), f"aux value T={T}")
        if T >= 256:
            ck(eq(hs, full) and pp is None and cc is None, f"aux handover T={T}")
        else:
            ck(hs is x and pp is post and cc is comb, f"aux below MIN_T T={T}")
        # mhc_mean: the Triton post(+mean) kernel == production's tilelang post and aten mean, bit for bit
        mref = hc_contract(full, HC)
        for want in (True, False):
            r = Q.qw_post_mean(x, res, post, comb, want)
            ok = r is not None and eq(r[1], mref) and ((r[0] is not None and eq(r[0], full)) if want else r[0] is None)
            ck(ok, f"qw_post_mean T={T} full={want}")
        fin = Q._qw_final_post(lay, x, res, post, comb, hc_contract)
        ck(eq(fin, mref), f"final post T={T}")
        print(f"  T={T}: mhc_mean post+mean / mean-only / last-layer helper bit-identical: "
              f"{all(m.find(f'T={T}') < 0 for m in FAILS)}", flush=True)


def w6_idx_gate(mods):
    print("W.6 idx_gate", flush=True)
    import glm53_gemv_install as G
    ck(getattr(G._head_gate_impl, "_glm53_qw", False), "idx_gate installed")
    print(f"  {Q.M_GEMV}._head_gate_impl (original) fingerprint {Q.source_fingerprint(G._head_gate_impl._glm53_qw_orig)}")
    G.register_ops()
    g = torch.Generator(device=DEV).manual_seed(9)
    wb = (torch.randn(160, 4096, device=DEV, generator=g) * 0.02).to(torch.bfloat16)     # wk_weights_proj rows
    w32 = wb[128:, :].t().contiguous().float()                                           # production's _wp_fp32
    for M in (Q.GATE_MIN_M, 10752, 12288, 13824, 15360, 16384, 9216, 1791):
        x = (torch.randn(M, 4096, device=DEV, generator=g) * 0.7).to(torch.bfloat16)
        ref = torch.mm(x.float(), w32)
        before = Q.STATS["idx_gate_fast"]
        got = torch.ops.glm53_gemv.head_gate(x, wb, w32, -777, 128)
        fast = Q.STATS["idx_gate_fast"] - before
        ck(eq(got, ref), f"idx_gate M={M}: {(got != ref).sum().item()} differ")
        ck(fast == (1 if M >= Q.GATE_MIN_M else 0), f"idx_gate M={M} fast {fast}")
        print(f"  M={M}: head gate bit-identical {eq(got, ref)} (fast path {bool(fast)})", flush=True)
    # opt-dense: non-finite production result (vLLM's profile run on dummy activations) decides nothing
    xn = torch.randn(16384, 4096, device=DEV, generator=g).to(torch.bfloat16)
    xn[5, 7] = float("nan")
    Q._HG["checked"].discard(16384)
    b0, n0 = Q.STATS["idx_gate_fast"], Q.STATS["idx_gate_nonfinite"]
    got = torch.ops.glm53_gemv.head_gate(xn, wb, w32, -777, 128)
    refn = torch.mm(xn.float(), w32)
    ck(torch.equal(torch.nan_to_num(got, 7.0), torch.nan_to_num(refn, 7.0)) and Q.STATS["idx_gate_nonfinite"] == n0 + 1
       and Q.STATS["idx_gate_fast"] == b0 and 16384 not in Q._HG["checked"] and not Q._HG["off"]
       and 16384 not in Q._HG["bad"], "idx_gate NaN input -> production result, M left undecided, item stays on")
    x = torch.randn(16384, 4096, device=DEV, generator=g).to(torch.bfloat16)
    got = torch.ops.glm53_gemv.head_gate(x, wb, w32, -777, 128)
    ck(eq(got, torch.mm(x.float(), w32)) and Q.STATS["idx_gate_fast"] == b0 + 1 and 16384 in Q._HG["checked"],
       "idx_gate M=16384 after the NaN call: checked on the next finite call and served")
    print(f"  NaN profile-run case: undecided, then served at the next finite call: {not FAILS}", flush=True)
    # a mismatch at the first call of a new M -> only THAT M stays on production; other M keep the fast path;
    # GATE_MAX_BAD_M distinct bad M -> off
    saved = Q.qw_head_gate
    try:
        Q.qw_head_gate = lambda x, w: saved(x, w) + 1e-3
        x = torch.randn(11000, 4096, device=DEV).to(torch.bfloat16)
        got = torch.ops.glm53_gemv.head_gate(x, wb, w32, -777, 128)
        ck((not Q._HG["off"]) and 11000 in Q._HG["bad"] and eq(got, torch.mm(x.float(), w32)),
           "idx_gate mismatch at M=11000 -> that M on production, item stays on")
        Q.qw_head_gate = saved
        b0 = Q.STATS["idx_gate_fast"]
        x = torch.randn(13824, 4096, device=DEV).to(torch.bfloat16)
        got = torch.ops.glm53_gemv.head_gate(x, wb, w32, -777, 128)
        ck(eq(got, torch.mm(x.float(), w32)) and Q.STATS["idx_gate_fast"] == b0 + 1, "M=13824 still fast after M=11000 bad")
        got = torch.ops.glm53_gemv.head_gate(torch.randn(11000, 4096, device=DEV).to(torch.bfloat16), wb, w32, -777, 128)
        ck(Q.STATS["idx_gate_fast"] == b0 + 1, "M=11000 stays on production after its mismatch")
        Q.qw_head_gate = lambda x, w: saved(x, w) + 1e-3
        for mm in range(11001, 11001 + Q.GATE_MAX_BAD_M):
            torch.ops.glm53_gemv.head_gate(torch.randn(mm, 4096, device=DEV).to(torch.bfloat16), wb, w32, -777, 128)
        ck(Q._HG["off"], f"idx_gate off after {Q.GATE_MAX_BAD_M} distinct bad M")
    finally:
        Q.qw_head_gate = saved
        Q._HG["off"] = False
        Q._HG["bad"].clear()
    # fp32 mode (GLM53_PREFILL_QUICKWINS_GATE=fp32): M < GATE_MIN_M served within the fp32-order bound
    ck(Q.parse_gate_mode({}) == "exact" and Q.parse_gate_mode({Q.ENV_GATE: "fp32"}) == "fp32", "gate mode parse")
    try:
        Q.parse_gate_mode({Q.ENV_GATE: "fast"})
        ck(False, "gate mode: invalid value accepted")
    except ValueError:
        pass
    Q._HG["mode"] = "fp32"
    Q._HG["checked"].clear()
    try:
        for M in (1791, 4289, 6912, 9216, 13824):
            x = (torch.randn(M, 4096, device=DEV, generator=g) * 0.7).to(torch.bfloat16)
            ref = torch.mm(x.float(), w32)
            b0 = Q.STATS["idx_gate_fast"]
            got = torch.ops.glm53_gemv.head_gate(x, wb, w32, -777, 128)
            bound = torch.mm(x.float().abs(), w32.abs())
            rel = ((got - ref).abs() / bound).max().item()
            ck(Q.STATS["idx_gate_fast"] == b0 + 1 and rel <= Q.GATE_FP32_TOL, f"fp32 mode M={M}: fast, rel {rel:.2e}")
            print(f"  fp32 mode M={M}: fast path, max |fast-prod|/sum|xw| {rel:.2e}, bitwise {eq(got, ref)}", flush=True)
        x = torch.randn(100, 4096, device=DEV).to(torch.bfloat16)
        b0 = Q.STATS["idx_gate_fast"]
        torch.ops.glm53_gemv.head_gate(x, wb, w32, -777, 128)
        ck(Q.STATS["idx_gate_fast"] == b0, "fp32 mode: M < GATE_SMALL_M stays on the gemv/production op")
    finally:
        Q._HG["mode"] = "exact"
        Q._HG["checked"].clear()


def main():
    torch.manual_seed(0)
    w0_plugin_path()
    mods = w1_wiring()
    w2_mla_bmm(mods)
    w3_mla_index(mods)
    w4_kda_conv(mods)
    w5_mhc_aux(mods)
    w6_idx_gate(mods)
    print(f"STATS {Q.STATS}")
    if FAILS:
        print(f"FAILED {len(FAILS)} of {NCHK[0]} checks")
        sys.exit(1)
    print(f"ALL PASSED ({NCHK[0]} checks)")


if __name__ == "__main__":
    main()
