"""MHC-SP wiring tests on nodeC (one GPU, single process): the patch, its fingerprint chain, and a faithful
2-rank emulation of the PATCHED model forward (docs/MHC_SP.md sections 4-5).

Runs inside the production image, which it patches and RESTORES (the container is --rm; pristine bytes are
kept in memory and rewritten on exit):

1. patch mechanics: GLM53_MHC_SP=0 touches nothing; =1 patches the image's vllm/models/glm5next/nvidia/
   model.py and adds the fingerprints to glm53_prefill_quickwins.py / glm53_moeglue.py (the patch's test-env
   overrides point those two at the repo's files); a second run is a no-op ("already present", byte-identical).
2. fingerprint chain: integrate.source_fingerprint of the live patched forwards equals the patch's values;
   glm53_prefill_quickwins.install(["mhc_aux", "mhc_mean"]) transplants SUCCEED on the patched source (a
   refusal would silently disarm production's quickwins mHC pair); the quickwins-transplanted layer
   forward's fingerprint is accepted by glm53_moeglue's WARM_VERIFIED (else the decode L2 warm disarms).
3. two-rank emulation: the real patched Glm5NextModel.forward / Glm5NextDecoderLayer.forward run for rank 0
   and rank 1 in two threads whose collectives rendezvous (all-gather / reduce-scatter / all-reduce, with
   rank-sharded projections and a CROSS-TOKEN attention core, so a missing or misplaced gather is a wrong
   answer, not a missed one). The SP run (T = 4,096) must be BITWISE equal to the TP run for every layer's
   mHC tensors, the aux hidden states and the final hidden states; a decode-sized forward (T = 8) must stay
   on the TP path and be unchanged; the reduction toggles must round-trip. Quickwins mhc_aux/mhc_mean
   installed on top must not change a bit.

Run: flock /tmp/tf-gpu-bench.lock tests/gpu_run.sh python3 tests/mhc_sp/test_mhc_sp_all.py
"""
from __future__ import annotations

import importlib
import io
import math
import os
import shutil
import sys
import tempfile
import threading

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "overlay"))
SITEPKG = os.environ.get("GLM53_SITEPKG", "/usr/local/lib/python3.12/dist-packages")
MODEL_IMG = os.path.join(SITEPKG, "vllm/models/glm5next/nvidia/model.py")
QW_SRC = os.path.join(ROOT, "glm53_prefill_quickwins.py")     # the repo copies of the two fork modules the
MG_SRC = os.path.join(ROOT, "glm53_moeglue.py")               # patch extends (kit site/ in production)


def _load(name: str, path: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
H, N, TP = 4096, 4, 2
H3 = 2 * N + N * N
CORE, INT = 512, 512                       # core dim; per-rank intermediate (gate/up 2xINT per rank)
T_PREFILL, T_DECODE = 4096, 8
LAYERS = 4                                 # 3 MoE + 1 dense (exercises both reduce toggles)
AUX_AT = (2,)                              # 1-based: the quickwins mhc_aux / drafter layer
SEED = 11

FAILURES: list[str] = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg, flush=True)
    if not cond:
        FAILURES.append(msg)


def eq_maybe_none(a, b):
    if a is None or b is None:
        return a is None and b is None
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if torch.equal(a, b):
        return True
    # fp32 sinkhorn mixes on random weights can be NaN; a NaN produced identically on both paths is the
    # same output (report it rather than silently passing)
    if a.dtype.is_floating_point:
        both_nan = a.isnan() & b.isnan()
        if bool(both_nan.any()):
            print(f"       (note: {int(both_nan.sum())} NaN element(s), identical on both sides)")
            a = torch.nan_to_num(a)
            b = torch.nan_to_num(b)
        return torch.equal(a, b)
    return False


# ----------------------------------------------------------------------------------------------------------- collectives
class RankTL(threading.local):
    rank = -1


TL = RankTL()


class Exchange:
    """Rendezvous for the emulated TP collectives (two threads; both ranks call the same op sequence)."""

    def __init__(self):
        self.cv = threading.Condition()
        self.reset()

    def reset(self):
        self.deposits, self.seq, self.counts = {}, [0] * TP, {"ag": 0, "rs": 0, "ar": 0}
        self.results, self.hist = {}, []

    def _meet(self, kind: str, rank: int, x: torch.Tensor):
        seq = self.seq[rank]
        self.seq[rank] += 1
        with self.cv:
            self.hist.append((rank, kind, seq, tuple(x.shape)))
            if os.environ.get("MHC_SP_DBG"):
                import traceback as _tb
                st = _tb.extract_stack()[-4:-1]
                site = " <- ".join(f"{f.filename.split('/')[-1]}:{f.lineno}:{f.name}" for f in reversed(st))
                print(f"      meet r{rank} {kind}#{seq} {tuple(x.shape)}  via {site}", flush=True)
            self.deposits[(seq, kind, rank)] = x.contiguous()
            self.cv.notify_all()
            key = (seq, kind, rank)
            while key not in self.results:
                if (seq, kind, 1 - rank) in self.deposits:
                    d0 = self.deposits.pop((seq, kind, 0))
                    d1 = self.deposits.pop((seq, kind, 1))
                    if kind == "ag":
                        r0 = r1 = torch.cat([d0, d1], 0)
                    else:
                        total = d0 + d1                  # one bf16 pairwise sum for both ranks (NCCL ring)
                        if kind == "ar":
                            r0 = r1 = total
                        else:
                            half = total.shape[0] // TP
                            r0, r1 = total[:half].contiguous(), total[half:].contiguous()
                    self.results[(seq, kind, 0)], self.results[(seq, kind, 1)] = r0, r1
                    self.cv.notify_all()
                    continue
                if not self.cv.wait(120.0):
                    if os.environ.get("MHC_SP_DBG"):
                        for tid, fr in sys._current_frames().items():
                            print(f"      THREAD {tid}: " +
                                  "".join(traceback.format_stack(fr)).replace("\n", "\n      "), flush=True)
                    raise RuntimeError(f"2-rank emulation deadlock at op {kind} #{seq} "
                                       f"(history: {self.hist})")
            out = self.results.pop(key)
            self.counts[kind] += 1
        return out

    def all_gather(self, rank: int, x: torch.Tensor) -> torch.Tensor:
        return self._meet("ag", rank, x)

    def reduce_scatter(self, rank: int, x: torch.Tensor) -> torch.Tensor:
        return self._meet("rs", rank, x)

    def all_reduce(self, rank: int, x: torch.Tensor) -> torch.Tensor:
        return self._meet("ar", rank, x)


# ---------------------------------------------------------------------------------------------------------------- stubs
class StubLinear(nn.Module):
    """A TP linear. kind: "column" = the rank's own output columns (gate/up, no reduce); "row" = a rank
    partial over a replicated input, reduced by all-reduce whenever the flag says so (the flag
    _mhc_sp_set_reductions flips; mirrors RowParallelLinear.forward); "moe" = the runner's late all-reduce
    over a rank-local input (the rank's own intermediate columns)."""

    def __init__(self, w: torch.Tensor, rank: int, ex: Exchange, kind: str, input_local: bool = False):
        super().__init__()
        self.weight = torch.nn.Parameter(w, requires_grad=False)   # fp32, shared by both ranks
        self.rank, self.ex, self.kind, self.input_local = rank, ex, kind, input_local
        self.reduce_results = True
        self.moe_config = type("Cfg", (), {"skip_final_all_reduce": False})()
        self.half = (w.shape[1] if kind == "column" else w.shape[0]) // TP

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lo = self.rank * self.half
        if self.kind == "column":
            return (x.float() @ self.weight[:, lo:lo + self.half]).to(torch.bfloat16)
        w = self.weight[lo:lo + self.half, :]
        p = (x.float()[:, lo:lo + self.half] @ w if not self.input_local else x.float() @ w)
        p = p.to(torch.bfloat16)
        skip = self.moe_config.skip_final_all_reduce if self.kind == "moe" else False
        if self.reduce_results and not skip:
            return self.ex.all_reduce(self.rank, p)
        return p


class StubSelfAttn(nn.Module):
    """Cross-token attention core (the all-gather is load-bearing) + a row-sharded o_proj."""

    def __init__(self, w_down: torch.Tensor, w_out: torch.Tensor, rank: int, ex: Exchange):
        super().__init__()
        self.w_down = w_down                        # (H, CORE) fp32, shared (attention core is replicated)
        self.o_proj = StubLinear(w_out, rank, ex, "row")

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor):
        x = hidden_states.float() @ self.w_down
        core = torch.softmax(x @ x.transpose(0, 1) * (1.0 / math.sqrt(CORE)), -1) @ x
        return self.o_proj(core.to(torch.bfloat16))


class _MLPBase(nn.Module):
    def _h(self, x):
        h = self.gate(x)
        return torch.nn.functional.silu(h[:, :INT]) * h[:, INT:]

    def forward(self, x, already_sequence_parallel: bool = False):
        return self._tail(self._h(x))


class StubMoE(_MLPBase):
    """MoE-shaped MLP: routed partial per rank, the late all-reduce honoured at call time."""

    def __init__(self, w1: torch.Tensor, w2: torch.Tensor, rank: int, ex: Exchange):
        super().__init__()
        self.gate = StubLinear(w1, rank, ex, "column")
        self.experts = StubLinear(w2, rank, ex, "moe", input_local=True)
        self.moe_config = self.experts.moe_config

    def _tail(self, h):
        return self.experts(h)


class StubDense(_MLPBase):
    """The dense MLP: down_proj carries the reduce flag."""

    def __init__(self, w1: torch.Tensor, w2: torch.Tensor, rank: int, ex: Exchange):
        super().__init__()
        self.gate = StubLinear(w1, rank, ex, "column")
        self.down_proj = StubLinear(w2, rank, ex, "row", input_local=True)

    def _tail(self, h):
        return self.down_proj(h)


class ProbeLayer:
    """Records what the patched layer forward returns (the sharded mHC state) without touching its code."""

    def __init__(self, lay):
        self._lay, self.last = lay, None

    def __getattr__(self, name):
        return getattr(self._lay, name)

    def __call__(self, positions, hidden_states, residual, post, comb):
        self.last_in = tuple(t.detach().clone() if torch.is_tensor(t) else t
                             for t in (hidden_states, residual, post, comb))
        out = self._lay(positions, hidden_states, residual, post, comb)
        self.last = tuple(t.detach().clone() if torch.is_tensor(t) else t for t in out[:4])
        return out


# ------------------------------------------------------------------------------------------------------- model fixtures
def make_weights(seed: int = 7):
    g = torch.Generator(device="cuda").manual_seed(seed)
    mk = lambda *sh: torch.randn(*sh, generator=g, device="cuda")
    return {
        "emb": mk(T_PREFILL, H).bfloat16(),          # the post-all-reduce embedding, identical per rank
        "fn_a": mk(H3, N * H) * 0.02,
        "fn_f": mk(H3, N * H) * 0.02,
        "scale": torch.rand(3, generator=g, device="cuda") + 0.9,
        "base": mk(H3) * 0.1,
        "norm": (torch.rand(H, generator=g, device="cuda") + 0.5).bfloat16(),
        "nw": (torch.rand(H, generator=g, device="cuda") + 0.5).bfloat16(),
        "w_down": mk(H, CORE * TP) * 0.05,           # attention core: TP * CORE dims, replicated compute
        "w_out": mk(CORE * TP, H) * 0.05,            # o_proj: row-sharded on the replicated core
        "w1": mk(H, 2 * INT * TP) * 0.05,            # gate/up: column-sharded (rank's own columns)
        "w2": mk(INT * TP, H) * 0.05,                # down/experts: row-sharded on the intermediate
    }


def make_layer(gm, w, rank, ex, layer_idx, is_moe):
    """A real patched Glm5NextDecoderLayer (uninitialized) with the production mHC ops and stub attention /
    MLP whose reduce contracts mirror RowParallelLinear / the MoE runner."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.mhc import MHCFusedPostPreOp, MHCPostOp, MHCPreOp

    lay = gm.Glm5NextDecoderLayer.__new__(gm.Glm5NextDecoderLayer)
    torch.nn.Module.__init__(lay)
    lay.mhc, lay.is_mtp_layer, lay.layer_idx = True, False, layer_idx
    lay.num_hidden_layers, lay.n, lay.hidden_size = LAYERS, N, H
    lay.rms_norm_eps = lay.hc_eps = 1e-5
    lay.mhc_post_mult_value, lay.mhc_sinkhorn_iterations = 2.0, 20
    lay.hc_attn_fn, lay.hc_attn_scale, lay.hc_attn_base = w["fn_a"], w["scale"], w["base"]
    lay.hc_ffn_fn, lay.hc_ffn_scale, lay.hc_ffn_base = w["fn_f"], w["scale"], w["base"]
    with set_current_vllm_config(VllmConfig()):   # CustomOp dispatch needs a config context in tests
        lay.mhc_pre_op, lay.mhc_fused_post_pre_op, lay.mhc_post_op = (
            MHCPreOp(), MHCFusedPostPreOp(), MHCPostOp())
    lay.input_layernorm = RMSNorm(H, eps=1e-5).to("cuda")
    lay.input_layernorm.weight.data.copy_(w["norm"])
    lay.post_attention_layernorm = RMSNorm(H, eps=1e-5).to("cuda")
    lay.post_attention_layernorm.weight.data.copy_(w["nw"])
    lay.is_sequence_parallel = False
    lay.self_attn = StubSelfAttn(w["w_down"], w["w_out"], rank, ex)
    lay._mlp_is_moe = is_moe
    lay.mlp = StubMoE(w["w1"], w["w2"], rank, ex) if is_moe else StubDense(w["w1"], w["w2"], rank, ex)
    return lay


def make_model(gm, w, rank, ex):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import RMSNorm

    model = gm.Glm5NextModel.__new__(gm.Glm5NextModel)
    torch.nn.Module.__init__(model)
    model.is_sequence_parallel = False
    model.start_layer, model.end_layer = 0, LAYERS
    with set_current_vllm_config(VllmConfig()):   # CustomOp dispatch needs a config context in tests
        model._active_layers = [ProbeLayer(make_layer(gm, w, rank, ex, i, is_moe=i < LAYERS - 1))
                                for i in range(LAYERS)]
        model.aux_hidden_state_layers = AUX_AT
        model.norm = RMSNorm(H, eps=1e-5).to("cuda")
    model.norm.weight.data.copy_(w["norm"])
    model.embed_input_ids = lambda input_ids: w["emb"]
    return model


def run_both_ranks(gm, w, ex, T, sp):
    """One forward per rank in two threads; returns (per-rank (hidden, aux, per-layer states), counts, flags)."""
    ex.reset()
    gm.MHC_SP_ACTIVE = False
    gm.MHC_SP_MIN_TOKENS = T_PREFILL if sp else 1 << 30
    models = [make_model(gm, w, r, ex) for r in range(TP)]
    outs, errs, flag_list = [None] * TP, [None] * TP, [None] * TP

    def body(rank):
        try:
            TL.rank = rank
            torch.manual_seed(SEED)
            positions = torch.arange(T, device="cuda")
            ids = torch.zeros(T, dtype=torch.long, device="cuda")
            out = models[rank](input_ids=ids, positions=positions, intermediate_tensors=None)
            torch.cuda.synchronize()
            hidden, aux = out if isinstance(out, tuple) else (out, [])
            states = tuple(lay.last for lay in models[rank]._active_layers)
            states_in = tuple(lay.last_in for lay in models[rank]._active_layers)
            flag_list[rank] = [(int(lay._lay.self_attn.o_proj.reduce_results),
                                int(lay._lay.mlp.experts.moe_config.skip_final_all_reduce))
                               for lay in models[rank]._active_layers[: LAYERS - 1]]
            outs[rank] = (hidden, aux, states, states_in)
        except Exception as exc:  # noqa: BLE001
            errs[rank] = exc
            with ex.cv:
                ex.cv.notify_all()

    ths = [threading.Thread(target=body, args=(r,)) for r in range(TP)]
    for t in ths:
        t.start()
    for t in ths:
        t.join(900)
    if errs[0] or errs[1]:
        import traceback
        for r in range(TP):
            if errs[r]:
                traceback.print_exception(errs[r])
        raise RuntimeError(f"2-rank emulation failed: {errs[0]!r} / {errs[1]!r}")
    return outs, {k: v // TP for k, v in ex.counts.items()}, flag_list   # counts: one per rank


# ---------------------------------------------------------------------------------------------------------------- main
def main():
    tdir = tempfile.mkdtemp(prefix="mhcsp-")
    # Nothing inside the image is written (site-packages is root-owned under gpu_run.sh): the patch runs on
    # copies under tdir (the image's model.py and the repo's two fork modules), and the patched model.py is
    # exec'd into the freshly imported live module (linecache-seeded, so inspect.getsource sees the patched
    # text exactly like a real patched install).
    os.environ["GLM53_GLM5NEXT_MODEL_PY"] = os.path.join(tdir, "model.py")
    os.environ["GLM53_QUICKWINS_PY"] = os.path.join(tdir, "qw.py")
    os.environ["GLM53_MOEGLUE_PY"] = os.path.join(tdir, "mg.py")
    shutil.copy(MODEL_IMG, os.environ["GLM53_GLM5NEXT_MODEL_PY"])
    shutil.copy(QW_SRC, os.environ["GLM53_QUICKWINS_PY"])
    shutil.copy(MG_SRC, os.environ["GLM53_MOEGLUE_PY"])
    try:
        # ---- 1. patch mechanics ------------------------------------------------------------------------------
        import patch_mhc_sp as P
        os.environ["GLM53_MHC_SP"] = "0"
        check(P.main() == 0 and open(os.environ["GLM53_GLM5NEXT_MODEL_PY"]).read() ==
              open(MODEL_IMG).read(), "GLM53_MHC_SP=0: stock, files untouched")

        os.environ["GLM53_MHC_SP"] = "1"
        check(P.main() == 0, "patch installs (GLM53_MHC_SP=1)")
        MODEL_PY = os.environ["GLM53_GLM5NEXT_MODEL_PY"]
        after = {p: open(p).read() for p in (MODEL_PY, os.environ["GLM53_QUICKWINS_PY"],
                                             os.environ["GLM53_MOEGLUE_PY"])}
        check(after[MODEL_PY] != open(MODEL_IMG).read()
              and after[os.environ["GLM53_QUICKWINS_PY"]] != open(QW_SRC).read()
              and after[os.environ["GLM53_MOEGLUE_PY"]] != open(MG_SRC).read(), "all three files changed")
        check(P.prepare(after[MODEL_PY], P.MODEL_HUNKS) == after[MODEL_PY],
              "model.py is recognised as already-applied")
        for tgt, name in ((os.environ["GLM53_QUICKWINS_PY"], "qw2.py"),
                          (os.environ["GLM53_MOEGLUE_PY"], "mg2.py")):
            shutil.copy(tgt, os.path.join(tdir, name))
        buf, saved = io.StringIO(), sys.stdout
        sys.stdout = buf
        try:
            os.environ["GLM53_QUICKWINS_PY"] = os.path.join(tdir, "qw2.py")
            os.environ["GLM53_MOEGLUE_PY"] = os.path.join(tdir, "mg2.py")
            rc = P.main()
        finally:
            sys.stdout = saved
            os.environ["GLM53_QUICKWINS_PY"] = os.path.join(tdir, "qw.py")
            os.environ["GLM53_MOEGLUE_PY"] = os.path.join(tdir, "mg.py")
        check(rc == 0 and "already present" in buf.getvalue()
              and open(os.path.join(tdir, "qw2.py")).read() == after[os.path.join(tdir, "qw.py")],
              f"second run: no-op, byte-identical ({buf.getvalue().strip()[:90]})")

        # ---- 2. fingerprint chain ---------------------------------------------------------------------------
        import integrate
        importlib.invalidate_caches()
        gm = importlib.import_module("vllm.models.glm5next.nvidia.model")
        import linecache
        fname = "<glm53-mhc-sp patched model.py>"
        linecache.cache[fname] = (len(after[MODEL_PY]), None, after[MODEL_PY].splitlines(True), fname)
        exec(compile(after[MODEL_PY], fname, "exec"), gm.__dict__)
        fps = P.sp_fingerprints(after[MODEL_PY])
        check(integrate.source_fingerprint(gm.Glm5NextDecoderLayer.forward) == fps["layer_forward_sp"],
              "live Glm5NextDecoderLayer.forward fingerprint == the patch's layer_forward_sp")
        check(integrate.source_fingerprint(gm.Glm5NextModel.forward) == fps["model_forward_sp"],
              "live Glm5NextModel.forward fingerprint == the patch's model_forward_sp")

        os.environ["GLM53_PREFILL_QUICKWINS"] = "mhc_aux,mhc_mean"
        QW = _load("qw_patched", os.environ["GLM53_QUICKWINS_PY"])
        r = QW.install(["mhc_aux", "mhc_mean"])
        check("mhc_aux" in r["items"] and "mhc_mean" in r["items"]
              and not ({"mhc_aux", "mhc_mean"} & set(QW._STATE["refused"])),
              f"quickwins mhc_aux+mhc_mean transplant succeeds on the SP-patched source "
              f"(refused: {QW._STATE.get('refused')})")
        check(hasattr(gm.Glm5NextDecoderLayer.forward, "_glm53_qw"), "the quickwins transplant is live")
        MG = _load("mg_patched", os.environ["GLM53_MOEGLUE_PY"])
        bad = [k for k, v in MG._vllm_fingerprints().items() if v not in MG.WARM_VERIFIED[k]]
        check(not bad, f"moeglue warm fingerprints accepted after SP+quickwins (bad: {bad})")

        # ---- 3. the 2-rank emulation -------------------------------------------------------------------------
        import vllm.distributed as vd
        real = (vd.get_tensor_model_parallel_world_size, vd.get_pp_group)
        vd.get_tensor_model_parallel_world_size = lambda: TP
        pp = type("PP", (), {"is_first_rank": True, "is_last_rank": True, "world_size": 1})()
        vd.get_pp_group = lambda: pp
        gm.get_pp_group = lambda: pp
        gm.get_tensor_model_parallel_world_size = lambda: TP
        # the real collectives need an initialized TP group; the emulation rendezvous replaces them
        gm.sp_all_gather = lambda x: ex.all_gather(TL.rank, x)
        gm.sp_reduce_scatter = lambda x: ex.reduce_scatter(TL.rank, x)
        def _emul_shard(x):
            if os.environ.get("MHC_SP_DBG"):
                print(f"      shard TL.rank={TL.rank} shape={tuple(x.shape)}", flush=True)
            return x[TL.rank * x.shape[0] // TP:(TL.rank + 1) * x.shape[0] // TP]

        gm.sp_shard = _emul_shard

        w = make_weights()
        ex = Exchange()
        sp_out, sp_counts, sp_flags = run_both_ranks(gm, w, ex, T_PREFILL, True)
        tp_out, tp_counts, tp_flags = run_both_ranks(gm, w, ex, T_PREFILL, False)
        ag_pairs = LAYERS * 2 + len(AUX_AT) + 1
        check(sp_counts["ag"] == ag_pairs and sp_counts["rs"] == LAYERS * 2,
              f"SP run: {sp_counts['ag']} all-gather / {sp_counts['rs']} reduce-scatter pairs "
              f"(expected {ag_pairs} / {LAYERS * 2})")
        (h_sp, aux_sp, st_sp, sti_sp), (h_tp, aux_tp, st_tp, sti_tp) = sp_out[0], tp_out[0]
        check(all(int(sp_flags[0][i][0]) == 0 and int(sp_flags[0][i][1]) == 1 for i in range(LAYERS - 1)),
              f"SP run toggled the reductions off (o_proj flags {[f[0] for f in sp_flags]}, "
              f"moe skip {[f[1] for f in sp_flags]})")
        st_sp1 = sp_out[1][2]          # rank 1's sharded states
        # the SP run's per-layer states are this rank's shard: compare each rank against its own slice
        def state_eq(i, j, a, b):
            if not torch.is_tensor(a) or not torch.is_tensor(b):
                return a is None and b is None
            n = a.shape[0]
            ok0 = eq_maybe_none(a, b[:n])
            ok1 = eq_maybe_none(st_sp1[i][j], b[n:])
            if not (ok0 and ok1):
                a2 = st_sp1[i][j]
                why = [f"rank0 {tuple(a.shape)} vs tp{tuple(b[:n].shape)}",
                       f"rank1 {tuple(a2.shape) if torch.is_tensor(a2) else a2!r} vs tp{tuple(b[n:].shape)}"]
                for aa, bb in ((a, b[:n]), (st_sp1[i][j], b[n:])):
                    if torch.is_tensor(aa) and torch.is_tensor(bb) and aa.shape == bb.shape:
                        d = (aa.float() - bb.float()).abs().max().item()
                        why.append(f"maxabs {d:.3e} nan {int(aa.isnan().sum())}/{int(bb.isnan().sum())}")
                print(f"       layer {i} field {j}: " + "; ".join(why))
            return ok0 and ok1
        # where does a divergence enter? compare the layer INPUTS (the wired tensors) first, then the states
        def input_eq(i, j):
            a, b = sti_sp[i][j], sti_tp[i][j]
            if not torch.is_tensor(a) or not torch.is_tensor(b):
                return a is None and b is None
            n = a.shape[0]
            a1 = sp_out[1][3][i][j]    # rank 1's sharded inputs
            ok0 = eq_maybe_none(a, b[:n])
            ok1 = eq_maybe_none(a1, b[n:])
            if not (ok0 and ok1) and torch.is_tensor(a1):
                d0 = (a.float() - b[:n].float()).abs().max().item()
                d1 = (a1.float() - b[n:].float()).abs().max().item()
                extra = ""
                if j == 0 and i == 0 and b.shape == (T_PREFILL, H):
                    extra = (f" [vs emb[0:{n}] {(a1.float() - w['emb'][:n].float()).abs().max().item():.3e}; "
                             f"vs emb[{n}:] {(a1.float() - w['emb'][n:].float()).abs().max().item():.3e}]")
                print(f"       layer {i} IN field {j}: rank0 maxabs {d0:.3e}; rank1 maxabs {d1:.3e} "
                      f"shapes {tuple(a.shape)}/{tuple(a1.shape)} vs tp {tuple(b.shape)}{extra}")
            return ok0 and ok1
        same_in = [i for i in range(LAYERS) if all(input_eq(i, j) for j in range(4))]
        check(len(same_in) == LAYERS,
              f"every layer's inputs: SP shards == TP bitwise ({len(same_in)}/{LAYERS})")
        same = [i for i in range(LAYERS) if all(state_eq(i, j, st_sp[i][j], st_tp[i][j])
                                                for j in range(4))]
        check(len(same) == LAYERS,
              f"every layer's (x, residual, post, comb): SP shards == TP bitwise ({len(same)}/{LAYERS})")
        for i in set(range(LAYERS)) - set(same):
            for j, nm in enumerate(("x", "residual", "post", "comb")):
                a, b = st_sp[i][j], st_tp[i][j]
                if torch.is_tensor(a) and torch.is_tensor(b):
                    print(f"       layer {i} {nm}: max abs diff "
                          f"{(a.float() - b[:a.shape[0]].float()).abs().max().item():.3e}")
        check(torch.equal(h_sp, h_tp) and torch.equal(sp_out[1][0], tp_out[1][0]),
              "final hidden states, both ranks: SP == TP bitwise (T=4,096)")
        check(len(aux_sp) == len(AUX_AT) and all(torch.equal(a, b) for a, b in zip(aux_sp, aux_tp))
              and all(torch.equal(a, b) for a, b in zip(sp_out[1][1], tp_out[1][1])),
              f"aux hidden states, both ranks: SP == TP bitwise ({len(aux_sp)} aux layers)")

        dec_sp, dec_counts, _ = run_both_ranks(gm, w, ex, T_DECODE, True)
        dec_tp, _, _ = run_both_ranks(gm, w, ex, T_DECODE, False)
        check(torch.equal(dec_sp[0][0], dec_tp[0][0]),
              f"decode T={T_DECODE}: the SP-enabled module == the TP module bitwise")
        check(dec_counts["ag"] == 0 and dec_counts["rs"] == 0,
              f"decode used no SP collectives (counts {dec_counts})")
        check(gm.MHC_SP_ACTIVE is False, "MHC_SP_ACTIVE flipped back off after the decode-sized step")
        check(all(f[0] == 1 and f[1] == 0 for pair in tp_flags for f in pair),
              "the decode-sized step restored every reduction flag")

        # SP must never be recorded in a graph: the gate is False while a CUDA graph is being captured
        cap = []
        s_stream = torch.cuda.Stream()
        s_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s_stream):
            _ = torch.zeros(8, device="cuda")
        torch.cuda.current_stream().wait_stream(s_stream)
        torch.cuda.synchronize()
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            cap.append(bool(gm._mhc_sp_now(4096)) or bool(gm._mhc_sp_now(T_PREFILL)))
        check(cap == [False],
              f"the SP gate is False inside a CUDA-graph capture (no graph records an SP branch): {cap}")

        # ---- restore ------------------------------------------------------------------------------------------
        os.environ["GLM53_PREFILL_QUICKWINS"] = ""
        QW.uninstall_item("mhc_aux")
        QW.uninstall_item("mhc_mean")
        vd.get_tensor_model_parallel_world_size, vd.get_pp_group = real
    finally:
        os.environ.pop("GLM53_MHC_SP", None)
        os.environ.pop("GLM53_GLM5NEXT_MODEL_PY", None)
        os.environ.pop("GLM53_QUICKWINS_PY", None)
        os.environ.pop("GLM53_MOEGLUE_PY", None)
        os.environ["GLM53_PREFILL_QUICKWINS"] = ""
        shutil.rmtree(tdir, ignore_errors=True)
        print("nothing inside the image was written (all patched files were copies under /tmp)")
    print(f"\n{'ALL OK' if not FAILURES else f'{len(FAILURES)} FAILURES: ' + '; '.join(FAILURES)}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
