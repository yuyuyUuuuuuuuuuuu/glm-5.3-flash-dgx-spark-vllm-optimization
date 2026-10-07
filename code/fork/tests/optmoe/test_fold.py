"""opt-moe GLM53_MOE_E4M3_FOLD_SHARED=1 (routed sum accumulated into the shared experts' bf16 output): real layer-10
experts (TP=2 rank-0 shard), production shapes.

  1. numerics: run(fold_into=S) (S = a shared-expert-like bf16 output) vs S + routed in fp64 (fp32 accumulator): the
     folded result's error is the bf16 accumulator's class (rel-L2 < 6e-3, and <= 1.25 x the unfolded bf16 path's
     bf16(S + routed_bf16) error); the result IS S's storage; fold_into is ignored (S untouched) with the fp32
     accumulator, a wrong shape, a non-contiguous or a non-bf16 buffer
  2. the runner protocol on a stand-in moe_runner module (same forward / _apply_quant_method lines as vLLM's,
     fingerprints included): the hook folds a served call, the runner's result is the shared buffer itself (no add),
     == S + routed; a runner with routed_scaling_factor != 1, a routed output transform, a pre-reduced fused output,
     sequence parallel or no pending shared output does NOT fold (result == shared + routed by the add); the
     pending flag never leaks into the next forward; uninstall restores forward / _unpack
  3. the real vllm moe_runner of the image: fingerprints present, the wrappers install and uninstall
  4. invalid values refused; FOLD without ACC=bf16 refused
  5. timing at T = 13,824 / 4,289: run(bf16) + vLLM's add (and gather2's zeroing) vs run(fold_into)
Run: GPU_RUN_RO=$TF_EXL3_ASSETS/moee4m3 tests/gpu_run.sh python3 tests/optmoe/test_fold.py
"""
from __future__ import annotations

import importlib.util
import os
import statistics
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "moee4m3"))
sys.path.insert(0, HERE)
import torch  # noqa: E402

import prefill_cap_common as C  # noqa: E402
import harness as H  # noqa: E402
from real_layer import make_real_layer  # noqa: E402

CHK = H.Checks()

STUB = '''
import types
import torch


def _unpack(result):
    if isinstance(result, tuple):
        return result
    else:
        return (None, result)


class SE:
    def __init__(self):
        self._output = [None, None]
        self.enable_dbo = False

    @property
    def _output_idx(self):
        return 0

    @property
    def output(self):
        assert self._output[self._output_idx] is not None
        output = self._output[self._output_idx]
        self._output[self._output_idx] = None
        return output


class MoERunner:
    def __init__(self, apply_fn, shared_fn, scale=1.0, transform=None, reduced=False, sp=False, shared=True):
        self._shared_experts = SE() if shared else None
        self.routed_scaling_factor = scale
        self.routed_output_transform = transform
        self._reduced = reduced
        self.moe_config = types.SimpleNamespace(is_sequence_parallel=sp)
        self.router = None
        self.apply_fn = apply_fn
        self.shared_fn = shared_fn
        self.adds = 0

    @property
    def _fused_output_is_reduced(self):
        return self._reduced

    def _apply_quant_method(self, x):
        if self._shared_experts is not None:
            self._shared_experts._output[0] = self.shared_fn(x)
        fused_out = self.apply_fn(x)
        return (
            self._shared_experts.output if self._shared_experts is not None else None,
            fused_out,
        )

    def forward(self, x):
        result = self._apply_quant_method(x)
        shared_output, fused_output = _unpack(result)
        if self.routed_scaling_factor != 1.0:
            fused_output = fused_output * self.routed_scaling_factor
        if shared_output is not None:
            result = shared_output + fused_output
            self.adds += 1
        else:
            result = fused_output
        return result
'''


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


def inputs(T, kind, seed, dev):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, 4096, generator=g).to(torch.bfloat16).to(dev)
    s = (torch.randn(T, 4096, generator=g) * 0.05).to(torch.bfloat16).to(dev)
    ids = C.routing(kind, T, seed, dev)
    w = C.weights_for(T, seed, dev).float()
    return x, s, ids, w


def sample(fn, n):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize()
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n


def main():
    H.gpu_guard(8.0)
    prod = H.load_prod()
    H.load_xl()
    import _ext as OX
    OX.preload()
    import glm53_moe_e4m3 as M
    import glm53_prefill_cap as PC

    dev = torch.device("cuda", 0)
    L = make_real_layer(prod, dev)
    emap = prod.pin_exl3_expert_map(L, dev)
    assert PC.install(prodmod=prod, n=1)["installed"]
    # 1. numerics
    for T, kind, seed in ((13824, "real", 31), (4289, "real", 32), (2048, "collapsed", 33)):
        x, s, ids, w = inputs(T, kind, seed, dev)
        tag = f"T={T} {kind}"
        rf = M.run(prod, x, ids, w, L, C.LIMIT, expert_map=emap, sched={"acc": "f32"}).clone()
        rb = M.run(prod, x, ids, w, L, C.LIMIT, expert_map=emap, sched={"acc": "bf16"}).clone()
        ref = s.double() + rf.double()
        buf = s.clone()
        fo = M.run(prod, x, ids, w, L, C.LIMIT, expert_map=emap, sched={"acc": "bf16"}, fold_into=buf)
        CHK(fo is buf and fo.dtype == torch.bfloat16 and bool(torch.isfinite(fo).all()),
            f"[{tag}] the folded result is the shared buffer itself, bf16, finite")
        e_fold, e_plain = rel(fo, ref), rel(s + rb, ref)
        CHK(e_fold < 6e-3 and e_fold <= 1.25 * e_plain,
            f"[{tag}] folded vs S + routed(fp32 acc) {e_fold:.2e} < 6e-3, <= 1.25 x unfolded bf16 {e_plain:.2e}")
        print(f"  [{tag}] fold {e_fold:.3e} | bf16(S + routed bf16) {e_plain:.3e} | routed share "
              f"{float(rf.double().norm() / ref.norm()):.3f}", flush=True)
        # ignored cases: S untouched, plain result returned
        for name, sch, fb in (("fp32 accumulator", {"acc": "f32"}, s.clone()),
                              ("wrong shape", {"acc": "bf16"}, s[:-1].clone()),
                              ("non-contiguous", {"acc": "bf16"}, s.t().contiguous().t()),
                              ("fp16 buffer", {"acc": "bf16"}, s.half())):
            keep = fb.clone()
            o = M.run(prod, x, ids, w, L, C.LIMIT, expert_map=emap, sched=sch, fold_into=fb)
            exp = rf if sch["acc"] == "f32" else rb
            CHK(o is not fb and torch.equal(fb, keep) and rel(o, exp) < 6e-3,
                f"[{tag}] fold_into ignored with a {name}: buffer untouched, plain result")
        del x, s, rf, rb, ref, buf, fo
        torch.cuda.empty_cache()

    # 2. runner protocol on a stand-in module
    for bad in ("2", "true", "yes"):
        rep = M.install(prod, environ={"GLM53_MOE_E4M3": "1", "GLM53_MOE_E4M3_ACC": "bf16",
                                       "GLM53_MOE_E4M3_FOLD_SHARED": bad})
        CHK(not rep["installed"] and not M.FOLD["on"], f"[hook] GLM53_MOE_E4M3_FOLD_SHARED={bad!r} refused")
    rep = M.install(prod, environ={"GLM53_MOE_E4M3": "1", "GLM53_MOE_E4M3_FOLD_SHARED": "1"})
    CHK(not rep["installed"] and not M.FOLD["on"], "[hook] FOLD_SHARED=1 without ACC=bf16 refused")
    tmpd = tempfile.mkdtemp(dir=os.environ.get("OPTMOE_TMP", "/w/tests/optmoe"))
    stub_py = os.path.join(tmpd, "stub_moe_runner.py")
    with open(stub_py, "w") as f:
        f.write(STUB)
    spec = importlib.util.spec_from_file_location("stub_moe_runner", stub_py)
    R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R)
    orig_fwd, orig_unpack = R.MoERunner.forward, R._unpack
    why = M._install_fold(R)            # reads the stub's source (inspect): the file must still exist
    os.unlink(stub_py)
    os.rmdir(tmpd)
    CHK(why == "ok" and R.MoERunner.forward is not orig_fwd and R._unpack is not orig_unpack,
        f"[stub] fingerprints found, forward / _unpack wrapped ({why})")
    if why != "ok":
        CHK.summary()
        return
    rep = M.install(prod, environ={"GLM53_MOE_E4M3": "1", "GLM53_MOE_E4M3_ACC": "bf16",
                                   "GLM53_MOE_E4M3_FOLD_SHARED": "1"}, load_selftest=False)
    CHK(rep["installed"] and rep.get("fold_shared") is True and M.FOLD["on"], f"[hook] FOLD installed: {rep}")
    T = 4289
    x, s, ids, w = inputs(T, "real", 41, dev)
    rb = M.run(prod, x, ids, w, L, C.LIMIT, expert_map=emap, sched={"acc": "bf16"}).clone()
    rf = M.run(prod, x, ids, w, L, C.LIMIT, expert_map=emap, sched={"acc": "f32"}).clone()
    ref = s.double() + rf.double()

    def apply_fn(xx):
        return prod.apply_exl3_experts(xx, ids, w, L, limit=C.LIMIT)

    holder = {}

    def shared_fn(xx):
        holder["s"] = s.clone()
        return holder["s"]

    f0 = M.STATS["folded"]
    run1 = R.MoERunner(apply_fn, shared_fn)
    out = run1.forward(x)
    CHK(out is holder["s"] and run1.adds == 0 and M.STATS["folded"] == f0 + 1,
        f"[stub] served call folded: result is the shared buffer, no add (adds {run1.adds})")
    CHK(rel(out, ref) < 6e-3 and rel(out, ref) <= 1.25 * rel(s + rb, ref),
        f"[stub] folded result == S + routed ({rel(out, ref):.2e} vs unfolded {rel(s + rb, ref):.2e})")
    CHK(M._FOLD_CTX["se"] is None and not M._FOLD_CTX["pending"], "[stub] fold context cleared after forward")
    for name, kw in (("routed_scaling_factor 2.5", {"scale": 2.5}), ("output transform", {"transform": abs}),
                     ("fused output pre-reduced", {"reduced": True}), ("sequence parallel", {"sp": True})):
        rr = R.MoERunner(apply_fn, shared_fn, **kw)
        o = rr.forward(x)
        exp = s.double() + (rf.double() * (2.5 if "scaling" in name else 1.0))
        CHK(rr.adds == 1 and M.STATS["folded"] == f0 + 1 and o is not holder["s"] and rel(o, exp) < 6e-3,
            f"[stub] {name}: not folded, vLLM's add ({rel(o, exp):.2e})")
    # no pending shared output (shared runs after the routed experts, e.g. the aux-stream order)
    rr = R.MoERunner(apply_fn, lambda xx: None)
    rr._apply_quant_method = lambda xx: (None, apply_fn(xx))
    o = rr.forward(x)
    CHK(M.STATS["folded"] == f0 + 1 and rel(o, rf) < 6e-3, "[stub] no pending shared output: not folded")
    # the pending flag never leaks: a folded call followed by a plain _unpack
    CHK(R._unpack((s, rb))[0] is s, "[stub] _unpack outside a folded call returns the shared half")
    # a call at/below the fused cap passes to production (no fold)
    small = 64
    rs = R.MoERunner(lambda xx: prod.apply_exl3_experts(xx, ids[:small], w[:small], L, limit=C.LIMIT),
                     lambda xx: s[:small].clone())
    o = rs.forward(x[:small])
    CHK(rs.adds == 1 and M.STATS["folded"] == f0 + 1, "[stub] decode-sized call (<= cap): production path, add")
    M.uninstall(prod)
    CHK(R.MoERunner.forward is orig_fwd and R._unpack is orig_unpack and not M.FOLD["on"]
        and not M.FOLD["wrapped"], "[stub] uninstall restores forward / _unpack")

    # 3. the image's vllm moe_runner
    try:
        import vllm.model_executor.layers.fused_moe.runner.moe_runner as VR
        vf, vu = VR.MoERunner.forward, VR._unpack
        why = M._install_fold()
        CHK(why == "ok" and VR.MoERunner.forward is not vf and VR._unpack is not vu,
            f"[vllm] the image's MoERunner: fingerprints present, wrappers installed ({why})")
        M._uninstall_fold()
        CHK(VR.MoERunner.forward is vf and VR._unpack is vu, "[vllm] uninstall restores the image's functions")
    except Exception as exc:  # noqa: BLE001
        CHK(False, f"[vllm] import / install failed: {exc!r}")

    # 5. timing
    rounds, n = 7, 5
    for T in (13824, 4289):
        x, s, ids, w = inputs(T, "real", 50 + T, dev)
        buf = s.clone()
        fns = {
            "bf16 + add": lambda: s + M.run(prod, x, ids, w, L, C.LIMIT, expert_map=emap, sched={"acc": "bf16"}),
            "fold": lambda: M.run(prod, x, ids, w, L, C.LIMIT, expert_map=emap, sched={"acc": "bf16"},
                                  fold_into=buf),
            "add alone": lambda: s + buf,
        }
        for f in fns.values():
            f()
        tt = {k: [] for k in fns}
        for r in range(rounds):
            for k in (list(fns) if r % 2 == 0 else list(fns)[::-1]):
                tt[k].append(sample(fns[k], n))
        med = {k: statistics.median(v) for k, v in tt.items()}
        print(f"  [T={T} timing] " + " | ".join(f"{k} {v:.2f}" for k, v in med.items())
              + f" ms  (fold - unfolded {med['fold'] - med['bf16 + add']:+.2f})", flush=True)
        CHK(med["fold"] < med["bf16 + add"], f"[T={T} timing] fold faster than bf16 + add")
        del x, s, buf
        torch.cuda.empty_cache()
    H.report_peak(8.0)
    CHK.summary()


if __name__ == "__main__":
    H.run_main(main)
