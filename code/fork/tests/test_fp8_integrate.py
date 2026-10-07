"""fp8_gemv.install() against the real production module (the image's exl3.py, or the launcher overlay's via
GPU_RUN_BIND): the wrapper around Glm53DenseFp8Method.apply / process_weights_after_loading. Exit 1 on any failure.

I.1  inert: GLM53_FP8_GEMV unset / 0 / off -> install() and plugin_register() patch nothing, the extension is not
     imported; the custom op tf_fp8::linear (always registered) then returns Marlin's result bitwise.
I.2  env parsing (GLM53_FP8_GEMV, GLM53_FP8_GEMV_MAX_M): invalid MAX_M refuses with a WARNING, never raises.
I.3  install wraps both methods once (idempotent); integrate.plugin_register() installs it with only GLM53_FP8_GEMV set.
I.4  per-layer self-test at load for every production shape (process_weights_after_loading), INFO once per shape.
I.5  apply(): output shape x.shape[:-1] + (N,), dtype; served by the new kernel exactly where select_config says
     (counters) and then bitwise equal to the direct kernel call and within 1 bf16 ulp of Marlin; everywhere else
     (M > table limit, M > 64, unknown shape) bitwise equal to production's apply; 3-D inputs; bias.
I.6  fallbacks, each bitwise == production's apply: fp16 layer (fp16 scales), misaligned x, failed self-test
     (WARNING), kernel pre-check error (WARNING once; strict mode raises; CUDA-error text re-raised), method not
     ready, KDA large-M BF16 copy present, GLM53_FP8_GEMV_MAX_M=8 at M = 16.
I.7  a layer first seen during CUDA-graph capture (the drafter lm_head copy is built outside
     process_weights_after_loading): Marlin inside the capture (no sync, capture succeeds), lazy self-test at the
     next eager call, then the new kernel.
I.8  CUDA graph capture of apply (M = 5, 8, 16): replay == eager, bitwise; the captured kernel is fp8_gemv.
I.9  torch.compile(fullgraph=True, dynamic=True) of a module calling apply (the drafter is compiled): no graph break,
     the FX graph holds tf_fp8.linear (never the kernel choice), results == eager at M = 5 and M = 100 (Marlin).
I.10 an instance created before install (the drafter lm_head's method object) is served by the wrapper.
I.11 uninstall() restores both methods; the custom op falls back to Marlin.
"""
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Checks, run_main, gpu_guard, load_prod  # noqa: E402
import inspect  # noqa: E402
import torch  # noqa: E402

dev = "cuda"


class Cap(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records = []

    def emit(self, r):
        self.records.append((r.levelno, r.getMessage()))

    def has(self, level, text):
        return any(lv == level and text in m for lv, m in self.records)

    def clear(self):
        self.records.clear()


def single_rank_tp():
    import socket

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
    from vllm.distributed import parallel_state as ps
    if ps.model_parallel_is_initialized():
        return
    with socket.socket() as s_:
        s_.bind(("127.0.0.1", 0))
        port = s_.getsockname()[1]
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                     distributed_init_method=f"tcp://127.0.0.1:{port}", backend="nccl")
        ensure_model_parallel_initialized(1, 1)


class L(torch.nn.Module):
    def __init__(self, w, bias=None):
        super().__init__()
        self.weight = torch.nn.Parameter(w, requires_grad=False)
        self.output_size_per_partition, self.input_size_per_partition = w.shape
        self.bias = None if bias is None else torch.nn.Parameter(bias, requires_grad=False)


def main():
    gpu_guard(6.0)
    ck = Checks()
    for v in ("GLM53_FP8_GEMV", "GLM53_FP8_GEMV_MAX_M", "TF_EXL3_MOE", "GLM53_KDA_BF16_LARGE_M"):
        os.environ.pop(v, None)
    single_rank_tp()
    prod = load_prod()
    import fp8_gemv as F
    import integrate
    cap = Cap()
    logging.getLogger("vllm.tf_fp8_gemv").addHandler(cap)
    logging.getLogger("vllm.tf_fp8_gemv").setLevel(logging.DEBUG)
    cls = prod.Glm53DenseFp8Method
    orig_apply, orig_pwal = cls.apply, cls.process_weights_after_loading
    takes_prefix = "prefix" in inspect.signature(cls.__init__).parameters
    print(f"Glm53DenseFp8Method{inspect.signature(cls.__init__)}")

    def method(group, prefix):
        return cls(group, prefix) if takes_prefix else cls(group)

    g = torch.Generator(device=dev).manual_seed(0)

    def build(n, k, group="dense", prefix="model.layers.1.mlp.gate_up_proj", dtype=torch.bfloat16, bias=False):
        w = (torch.randn(n, k, device=dev, generator=g) * 0.02).to(dtype)
        b = (torch.randn(n, device=dev, generator=g) * 0.1).to(dtype) if bias else None
        lay, m = L(w, b), method(group, prefix)
        m.process_weights_after_loading(lay)
        return lay, m

    def orig(m, lay, x, bias=None):
        return orig_apply(m, lay, x, bias)

    def agree(y, yo):
        rl2 = (y.float() - yo.float()).norm().item() / max(yo.float().norm().item(), 1e-30)
        return rl2, (y == yo).float().mean().item()

    def xr(M, k, dtype=torch.bfloat16):
        return torch.randn(M, k, device=dev, generator=g).to(dtype)

    # ---------------------------------------------------------------- I.1 / I.2 inert and env parsing
    print("== I.1 inert")
    for val in (None, "", "0", "off", "false", "no"):
        if val is None:
            os.environ.pop("GLM53_FP8_GEMV", None)
        else:
            os.environ["GLM53_FP8_GEMV"] = val
        rep = F.install(prod)
        F.plugin_register()
        integrate.plugin_register()
        ck(not rep["installed"] and cls.apply is orig_apply and cls.process_weights_after_loading is orig_pwal,
           f"I.1 {val!r}: patched")
    ck(F.STATE.ext is None, "I.1 extension imported while disabled")
    lay, m = build(4096, 4096)
    x = xr(8, 4096)
    y_op = torch.ops.tf_fp8.linear(x, lay.weight, lay.weight_scale, lay.workspace, None, 4096, 4096)
    ck(torch.equal(y_op, orig(m, lay, x)), "I.1 custom op while disabled != Marlin")
    print(f"   install/plugin_register inert for unset/''/0/off/false/no; extension not imported; op == Marlin")

    print("== I.2 env parsing")
    ck(all(F.env_enabled({"GLM53_FP8_GEMV": v}) for v in ("1", "on", "true", "yes", " ON ")), "I.2 on values")
    ck(not any(F.env_enabled({"GLM53_FP8_GEMV": v}) for v in ("", "0", "off", "2", "enable")), "I.2 off values")
    for bad in ("0", "65", "x"):
        os.environ["GLM53_FP8_GEMV_MAX_M"] = bad
        cap.clear()
        rep = F.install(prod, force=True)
        ck(not rep["installed"] and cap.has(logging.WARNING, "NOT installed") and cls.apply is orig_apply,
           f"I.2 MAX_M={bad!r} not refused")
    os.environ.pop("GLM53_FP8_GEMV_MAX_M")
    print("   GLM53_FP8_GEMV on: 1/on/true/yes; invalid GLM53_FP8_GEMV_MAX_M (0, 65, x) refused with a WARNING")

    # ---------------------------------------------------------------- I.3 install
    print("== I.3 install")
    os.environ["GLM53_FP8_GEMV"] = "1"
    cap.clear()
    early = method("draft_lm_head", "draft.glm53_candidate_head")        # created before install (I.10)
    integrate.plugin_register()
    ck(getattr(cls.apply, "_tf_fp8_hook", False) and getattr(cls.process_weights_after_loading, "_tf_fp8_hook", False),
       "I.3 integrate.plugin_register did not install")
    ck(cap.has(logging.INFO, "tf_fp8_gemv installed"), "I.3 no install INFO")
    wrapped = cls.apply
    rep = F.install(prod)
    ck(rep["reason"] == "already installed" and cls.apply is wrapped and cls.apply._tf_fp8_orig is orig_apply,
       "I.3 not idempotent")
    print(f"   installed via integrate.plugin_register (TF_EXL3_MOE unset); second install: {rep['reason']}")

    # ---------------------------------------------------------------- I.4 self-test at load
    print("== I.4 per-layer self-test at load (every production shape)")
    from fp8_bench_common import PROD_SHAPES
    groups = {"kda": ("kda", "model.layers.0.self_attn.in_proj_qkvbfg_a"), "mla": ("mla", "model.layers.3.self_attn.q_b_proj"),
              "dense": ("dense", "model.layers.1.mlp.down_proj"), "shared": ("shared", "model.layers.5.mlp.shared_experts.down_proj"),
              "draft": ("draft", "model.layers.45.mlp.down_proj")}
    built = {}
    cap.clear()
    for name, (n, k, _) in PROD_SHAPES.items():
        if (n, k) in built or name == "draft.lmhead":
            continue
        grp, pre = groups[name.split(".")[0]]
        lay, m = build(n, k, grp, pre)
        key = F._key(lay.weight, lay.weight_scale, k)
        ok = F.VERDICT.get(key, (False, "missing"))
        ck(ok[0], f"I.4 {name} {n}x{k}: self-test {ok}")
        built[(n, k)] = (lay, m)
        print(f"   {name:16s} {n:6d}x{k:5d}: {ok[1][:110]}")
    npass = sum(1 for lv, msg in cap.records if lv == logging.INFO and "self-test passed" in msg)
    ck(npass == len(built), f"I.4 INFO lines {npass} != {len(built)}")

    # ---------------------------------------------------------------- I.5 apply equivalence
    print("== I.5 apply: new kernel where selected, production's apply elsewhere")
    E = F.ext()
    for (n, k), (lay, m) in built.items():
        npad = lay.weight.shape[1] // 4
        line = f"   {n:6d}x{k:5d}:"
        for M in (1, 5, 8, 12, 16, 24, 32, 48, 64, 65, 200):
            x = xr(M, k)
            before = dict(F.COUNTERS)
            y = m.apply(lay, x)
            yo = orig(m, lay, x)
            served = F.COUNTERS.get("new_eager", 0) - before.get("new_eager", 0)
            cfg = F.select_config(npad, k, M)
            ck(y.shape == (M, n) and y.dtype == torch.bfloat16, f"I.5 {n}x{k} M={M}: shape/dtype {y.shape} {y.dtype}")
            if cfg is not None:
                yd = E.fp8_gemv(x, lay.weight, lay.weight_scale.view(-1), None, n, k, *cfg)
                rl2, same = agree(y, yo)
                ck(served == 1 and torch.equal(y, yd), f"I.5 {n}x{k} M={M}: not served by the selected kernel")
                ck(rl2 <= 1e-3 and same >= 0.99, f"I.5 {n}x{k} M={M}: vs Marlin rel_l2 {rl2:.1e}, identical {same:.4f}")
                line += f" {M}:new"
            else:
                ck(served == 0 and torch.equal(y, yo), f"I.5 {n}x{k} M={M}: fallback != production apply")
                line += f" {M}:marlin"
        print(line)
    lay, m = built[(4096, 4096)]
    x3 = torch.randn(2, 4, 4096, device=dev, generator=g).to(torch.bfloat16)
    y3 = m.apply(lay, x3)
    ck(y3.shape == (2, 4, 4096) and torch.equal(y3.reshape(8, 4096), m.apply(lay, x3.reshape(8, 4096))), "I.5 3-D input")
    layb, mb_ = build(4096, 4096, bias=True)
    for M in (1, 8, 16):
        x = xr(M, 4096)
        yb, yob = mb_.apply(layb, x, layb.bias), orig(mb_, layb, x, layb.bias)
        rl2, same = agree(yb, yob)
        ck(rl2 <= 1e-3 and same >= 0.99, f"I.5 bias M={M}: rel_l2 {rl2:.1e} identical {same:.4f}")
    print("   3-D input (2, 4, K) ok; bias (Marlin-permuted layer.bias) agrees with production's apply")

    # ---------------------------------------------------------------- I.6 fallbacks
    print("== I.6 fallbacks (each bitwise == production's apply)")

    def outcome(fn):
        try:
            return fn()
        except RuntimeError as exc:
            return f"raised {type(exc).__name__}: {str(exc).splitlines()[0][-50:]}"

    def fallback(name, m, lay, x, counter, bias=None):
        before = F.COUNTERS.get(counter, 0)
        y = outcome(lambda: m.apply(lay, x, bias))
        yo = outcome(lambda: orig(m, lay, x, bias))
        same = (y == yo) if isinstance(y, str) or isinstance(yo, str) else torch.equal(y, yo)
        ok = same and (counter is None or F.COUNTERS.get(counter, 0) == before + 1)
        if isinstance(yo, str):
            name += f" (production's apply itself {yo}; the wrapper the same)"
        ck(ok, f"I.6 {name}")
        print(f"   {name}: {'ok' if ok else 'FAILED'}")

    lay16, m16 = build(4096, 4096, dtype=torch.float16)
    fallback("fp16 layer / fp16 activations", m16, lay16, xr(8, 4096, torch.float16), "marlin_dtype")
    lay, m = built[(4096, 4096)]
    big = torch.randn(8 * 4096 + 1, device=dev, generator=g).to(torch.bfloat16)
    fallback("misaligned x", m, lay, big[1:].view(8, 4096), "marlin_x_layout")
    odd = torch.randn(8, 4097, device=dev, generator=g).to(torch.bfloat16)[:, :4096]
    fallback("odd row stride (4097)", m, lay, odd, "marlin_x_layout")
    layu, mu = build(1024, 4096)
    fallback("shape not in the table (1024x4096)", mu, layu, xr(8, 4096), "marlin_m")
    tol = F.SELFTEST_TOL
    F.SELFTEST_TOL = -1.0
    cap.clear()
    layf, mf = build(2048, 4096, "shared", "model.layers.9.mlp.shared_experts.gate_up_proj")
    F.SELFTEST_TOL = tol
    ck(cap.has(logging.WARNING, "self-test FAILED"), "I.6 no WARNING for a failed self-test")
    fallback("failed self-test", mf, layf, xr(8, 4096), "marlin_selftest_failed")

    class Boom:
        def __init__(self, real, msg):
            self.real, self.msg = real, msg

        def fp8_gemv(self, *a):
            raise RuntimeError(self.msg)

    real = F.STATE.ext
    F.STATE.ext = Boom(real, "fp8_gemv: synthetic pre-check failure")
    cap.clear()
    fallback("kernel pre-check error", m, lay, xr(8, 4096), "marlin_error")
    fallback("kernel pre-check error (again)", m, lay, xr(8, 4096), "marlin_error")
    ck(sum(1 for lv, msg in cap.records if lv == logging.WARNING and "pre-check failed" in msg) == 1,
       "I.6 pre-check WARNING not exactly once")
    F.CFG.strict = True
    try:
        m.apply(lay, xr(8, 4096))
        ck(False, "I.6 strict mode did not raise")
    except RuntimeError:
        pass
    F.CFG.strict = False
    F.STATE.ext = Boom(real, "CUDA error: an illegal memory access was encountered")
    try:
        m.apply(lay, xr(8, 4096))
        ck(False, "I.6 CUDA-error text not re-raised")
    except RuntimeError:
        pass
    F.STATE.ext = real
    print("   strict mode raises; CUDA-error text re-raised")
    mnr = method("dense", "model.layers.1.mlp.down_proj")          # never processed: ready False -> BF16 path
    laynr = L((torch.randn(4096, 4096, device=dev, generator=g) * 0.02).to(torch.bfloat16))
    fallback("method not ready (BF16 weight)", mnr, laynr, xr(8, 4096), None)
    lay, m = built[(12576, 4096)]
    lay.glm53_bf16_lm_w = torch.zeros(1, device=dev)                 # KDA large-M BF16 copy present -> production
    lay.glm53_bf16_lm_min_m = 512
    n0 = F.COUNTERS.get("new_eager", 0)
    lay_w = lay.glm53_bf16_lm_w
    fallback("KDA large-M BF16 copy present (M=8)", m, lay, xr(8, 4096), None)
    ck(F.COUNTERS.get("new_eager", 0) == n0, "I.6 large-M layer served by the new kernel")
    del lay.glm53_bf16_lm_w, lay.glm53_bf16_lm_min_m, lay_w
    F.CFG.max_m = 8
    lay, m = built[(4096, 4096)]
    fallback("GLM53_FP8_GEMV_MAX_M=8, M=16", m, lay, xr(16, 4096), "marlin_m")
    F.CFG.max_m = F.MAX_M

    # ---------------------------------------------------------------- I.7 first seen in capture
    print("== I.7 layer first seen during CUDA-graph capture")
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_layer_for_marlin
    from fp8_bench_common import quantize_like_prod
    head, _, _ = quantize_like_prod((torch.randn(12288, 4096, device=dev, generator=g) * 0.02).to(torch.bfloat16))
    early.ready = True
    x = xr(7, 4096)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    before = F.COUNTERS.get("marlin_untested_in_capture", 0)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        with torch.cuda.graph(gr, stream=s):
            yg = early.apply(head, x)
    torch.cuda.synchronize()
    ck(F.COUNTERS.get("marlin_untested_in_capture", 0) == before + 1, "I.7 capture did not fall back")
    gr.replay()
    torch.cuda.synchronize()
    ck(torch.equal(yg, orig(early, head, x)), "I.7 captured fallback != Marlin")
    n0 = F.COUNTERS.get("new_eager", 0)
    ye = early.apply(head, x)
    ck(F.VERDICT.get(F._key(head.weight, head.weight_scale, 4096), (False,))[0] and
       F.COUNTERS.get("new_eager", 0) == n0 + 1, "I.7 lazy self-test / new kernel after capture")
    print("   capture: Marlin (no sync, capture ok); next eager call: lazy self-test passed, new kernel")

    # ---------------------------------------------------------------- I.8 CUDA graphs
    print("== I.8 CUDA graph capture of apply")
    lay, m = built[(4096, 4096)]
    for M in (5, 8, 16):
        x = xr(M, 4096)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            m.apply(lay, x)
        torch.cuda.current_stream().wait_stream(s)
        n0 = F.COUNTERS.get("new_captured", 0)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            yg = m.apply(lay, x)
        ok = F.COUNTERS.get("new_captured", 0) == n0 + 1
        for _ in range(2):
            x.copy_(xr(M, 4096))
            gr.replay()
            ok &= torch.equal(yg, m.apply(lay, x))
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            gr.replay()
            torch.cuda.synchronize()
        names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        ok &= any("fp8_gemv_kernel" in nm for nm in names) and not any("Marlin" in nm for nm in names)
        ck(ok, f"I.8 M={M}")
        print(f"   M={M}: replay == eager (bitwise), captured kernel fp8_gemv {ok}")

    # ---------------------------------------------------------------- I.9 torch.compile
    print("== I.9 torch.compile(fullgraph=True, dynamic=True)")
    lay, m = built[(3072, 4096)]
    targets = []

    def backend(gm, example_inputs):
        targets.extend(str(nd.target) for nd in gm.graph.nodes if nd.op == "call_function")
        return gm.forward

    class Mod(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lay = lay

        def forward(self, x):
            return m.apply(self.lay, x) * 1.0

    torch._dynamo.reset()
    cm = torch.compile(Mod(), backend=backend, fullgraph=True, dynamic=True)
    for M in (5, 100, 8):
        x = xr(M, 4096)
        n0 = F.COUNTERS.get("new_eager", 0)
        yc = cm(x)
        want = F.select_config(3072, 4096, M) is not None
        ref = m.apply(lay, x) if want else orig(m, lay, x)
        ck(torch.equal(yc, ref), f"I.9 M={M}: compiled != eager")
        ck((F.COUNTERS.get("new_eager", 0) - n0 >= 1) == want, f"I.9 M={M}: runtime choice wrong")
    ck(any("tf_fp8.linear" in t for t in targets) and not any("marlin" in t.lower() for t in targets),
       f"I.9 graph targets {targets}")
    print(f"   one graph, targets {sorted(set(targets))}; M=5/8 new kernel, M=100 Marlin, all == eager")

    # ---------------------------------------------------------------- I.10 / I.11
    print("== I.10 instance created before install is wrapped:", early.apply.__func__ is cls.apply)
    ck(early.apply.__func__ is cls.apply, "I.10")
    print("== I.11 uninstall")
    rep = F.uninstall()
    lay, m = built[(4096, 4096)]
    x = xr(8, 4096)
    ok = rep["restored"] and cls.apply is orig_apply and cls.process_weights_after_loading is orig_pwal
    ok &= torch.equal(torch.ops.tf_fp8.linear(x, lay.weight, lay.weight_scale, lay.workspace, None, 4096, 4096),
                      orig(m, lay, x))
    ck(ok, "I.11 uninstall")
    print(f"   restored {rep['restored']}; op -> Marlin")
    print("counters:", F.summary())
    ck.summary()


if __name__ == "__main__":
    run_main(main)
