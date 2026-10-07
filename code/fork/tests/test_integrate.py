"""integrate.py / tf_exl3_moe control paths (docs/DESIGN.md §C.3, §C.5, §D).

D.1  TF_EXL3_MOE parsing; install is a no-op when disabled, and refuses with a WARNING (the reason) when production
     would never call exl3_moe: EXL3_FUSED_MOE=0, or GLM53_EXL3_MX on with the image module (the launcher overlay
     has no MX path: there GLM53_EXL3_MX is ignored by production and install proceeds)
D.6  production module versions (docs/STATUS.md): the loaded module is one of integrate.KNOWN_PROD_MODULES; a module
     lacking a required symbol is refused with a WARNING naming it; a production decode branch whose source is not
     one K2 was verified against gets the dispatcher path only, with a WARNING; GLM53_EXL3_MOE_FAST=1 without the
     thin-decode build, with a glm53_fast_moe_version() other than 1 or one that raises, and an invalid value, are
     reported (WARNING) and never raise out of install(); with the launcher overlay module, install()'s "will refuse
     to load" WARNING is given exactly when production's own process_weights_after_loading refuses (review F3)
D.7  production's build_exl3_fused_state failing for a layer (the overlay catches it and runs the python loop): the
     build hook logs a WARNING and re-raises, production's fallback is unchanged, the layer is not registered (F4)
D.8  observability: a call on a registered layer handed to production for a contract reason logs a WARNING once per
     reason (expected hand-offs, B > R, do not), calls are counted by category, a summary is logged at 10^k calls
     and after CUDA graph capture with the batch sizes captured with TF / with production (review safety-F3)
D.2  idempotent install (never wraps a wrapper), dispatcher keeps orig's doc, no __wrapped__
D.3  build hook: a raising pre-flight never reaches production's load path
D.4  `pip install --no-build-isolation` of this repo, then vLLM's own load_general_plugins() in a fresh
     interpreter installs the dispatcher AND the build hook (TF_EXL3_MOE=1); a layer finished by production's
     own process_weights_after_loading in that interpreter is registered through the plugin's hook, the
     production apply runs the TF path from the pip-installed AOT module and matches orig (E.1); nothing is
     installed when TF_EXL3_MOE is unset, and VLLM_PLUGINS is honored; the plugin logs "plugin loaded ... off" when
     TF_EXL3_MOE is unset (a TP rank without the variable is visible, review safety-F1)
D.5  uninstall restores only our objects (a later third-party patch survives), frees scratch (no capture);
     a dispatcher with nothing registered says so once (TF inert warning) and delegates
C.3  pointer sanity rejects a table entry outside / misaligned in its weight tensor; a failing self-test
     leaves the layer unregistered and disables TF process-wide
C.4  TF_EXL3_TOKENS windows (1:4, 2:64, bare 64, :8, 16:), TF_EXL3_MAX_PAIRS 32 and 4, and a layer with
     R = 4 temps (EXL3_TEMP_ROWS_FUSED < 8): the layer still registers with its self-test, TF stays enabled, and
     plan() serves exactly the window / domain (inclusive edges), delegating the rest with orig's result;
     malformed TF_EXL3_* values make install() refuse (and register() reject) instead of raising
C.5  a pre-launch check failure falls back to orig (same result) and disables TF (sticky); a CUDA-error
     message is re-raised; STRICT re-raises everything
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys

import torch

import harness as H

K, N, NEXP, TOPK = 4096, 1024, 16, 8


def main():
    H.gpu_guard(6.0)                     # + 2 GiB for the fresh-interpreter plugin probe (D.4) = 8 GiB per test
    xl = H.load_xl()
    prod = H.load_prod()
    tf = H.load_tf()
    import integrate

    ck = H.Checks()
    dev = torch.device("cuda", 0)
    real = xl.exl3_moe
    warned = []

    class _Warn(logging.Handler):
        def emit(self, record):
            warned.append(record.getMessage())

    hwarn = _Warn(level=logging.WARNING)
    logging.getLogger("vllm.tf_exl3_moe").addHandler(hwarn)

    def new_warnings(since):
        return warned[since:]

    # ---------------- D.1 env parsing / no-op installs ------------------------------------------------------
    on = ["1", "on", "true", "yes", " TRUE ", "Yes\n"]
    off = [None, "0", "", "false", "off", "2", "enable", "no"]
    res_on = [integrate.env_enabled({} if v is None else {"TF_EXL3_MOE": v}) for v in on]
    res_off = [integrate.env_enabled({} if v is None else {"TF_EXL3_MOE": v}) for v in off]
    print(f"D.1 enabled for {on}: {res_on}; for {off}: {res_off}")
    ck(all(res_on) and not any(res_off), "TF_EXL3_MOE parsing")
    saved = {k: os.environ.get(k) for k in ("TF_EXL3_MOE", "GLM53_EXL3_MX", "EXL3_FUSED_MOE")}
    try:
        os.environ.pop("TF_EXL3_MOE", None)
        r = integrate.install(prodmod=prod, ext=xl)
        ck(not r["installed"] and xl.exl3_moe is real, f"install with TF_EXL3_MOE unset: {r}")
        os.environ["TF_EXL3_MOE"] = "0"
        r0 = integrate.install(prodmod=prod, ext=xl)
        ck(not r0["installed"] and xl.exl3_moe is real, "install with TF_EXL3_MOE=0")
        ck(not new_warnings(0), f"TF_EXL3_MOE unset / 0 must stay silent: {warned}")
        os.environ["TF_EXL3_MOE"] = "1"
        os.environ["GLM53_EXL3_MX"] = "1"
        w0 = len(warned)
        r1 = integrate.install(prodmod=prod, ext=xl)
        if hasattr(prod, "mx_enabled"):          # image module: MX on -> production never calls exl3_moe
            ck(not r1["installed"] and xl.exl3_moe is real and any("NOT installed" in m and "GLM53_EXL3_MX" in m
                                                                   for m in new_warnings(w0)),
               f"install with MX on (image module): {r1}, warnings {new_warnings(w0)}")
        else:                                    # launcher overlay: no MX path, production ignores the variable
            ck(r1["installed"] and xl.exl3_moe is not real and not any("NOT installed" in m for m in new_warnings(w0)),
               f"install with GLM53_EXL3_MX=1 on a module without an MX path must proceed: {r1}")
            integrate.uninstall(prodmod=prod, ext=xl)
            ck(xl.exl3_moe is real and not getattr(prod.build_exl3_fused_state, "_tf_exl3_hook", False),
               "uninstall after the MX-ignored install")
        os.environ.pop("GLM53_EXL3_MX")
        os.environ["EXL3_FUSED_MOE"] = "0"
        w0 = len(warned)
        r2 = integrate.install(prodmod=prod, ext=xl)
        ck(not r2["installed"] and xl.exl3_moe is real and any("NOT installed" in m and "EXL3_FUSED_MOE=0" in m
                                                               for m in new_warnings(w0)),
           f"install with EXL3_FUSED_MOE=0: {r2}, warnings {new_warnings(w0)}")
        os.environ.pop("EXL3_FUSED_MOE")
        print(f"D.1 no-op installs: unset -> {r['reason']!r}; =0 -> {r0['reason']!r}; MX -> {r1['reason']!r} "
              f"(module {'has' if hasattr(prod, 'mx_enabled') else 'has no'} MX path); fused off -> {r2['reason']!r} "
              f"(WARNING logged)")
        # ---------------- D.6 production module versions -----------------------------------------------------
        ident = integrate.prod_identity(prod)
        print(f"D.6 production module: {ident}")
        ck(ident["known"] is not None, f"production module under test is not a known version: {ident}")
        k2ok, k2why = integrate.k2_compatibility(prod)
        ck(k2ok, f"K2 must be verified for both known module versions: {k2why}")
        import types

        def proxy(drop=(), **over):
            m_ = types.ModuleType("prod_proxy")
            for k_ in dir(prod):
                if k_ not in drop and not k_.startswith("__"):
                    setattr(m_, k_, getattr(prod, k_))
            m_.__file__ = prod.__file__
            for k_, v_ in over.items():
                setattr(m_, k_, v_)
            return m_

        def fake_ext(**extra):
            return types.SimpleNamespace(exl3_moe=real, **extra)

        w0 = len(warned)
        fx = fake_ext()
        rm = integrate.install(prodmod=proxy(drop=("build_exl3_fused_state",)), ext=fx)
        ck(not rm["installed"] and fx.exl3_moe is real and any("lacks build_exl3_fused_state" in m
                                                              for m in new_warnings(w0)),
           f"module lacking build_exl3_fused_state: {rm}")
        orig_apply_fn = prod.apply_exl3_fused_moe

        def apply_exl3_fused_moe(x2d, ids, weights, layer, inners, expert_map, limit):   # a changed decode branch
            return orig_apply_fn(x2d, ids, weights, layer, inners, expert_map, limit * 1.0)

        w0 = len(warned)
        fx, pm_ = fake_ext(), proxy(apply_exl3_fused_moe=apply_exl3_fused_moe)
        rk = integrate.install(prodmod=pm_, ext=fx)
        k2w = [m for m in new_warnings(w0) if "K2 apply path NOT installed" in m]
        ck(rk["installed"] and getattr(fx.exl3_moe, "_tf_exl3_dispatch", False) and not rk["apply_hook"]
           and pm_.apply_exl3_fused_moe is apply_exl3_fused_moe and len(k2w) == 1,
           f"unverified decode branch: dispatcher only + WARNING expected: {rk}, {k2w}")
        integrate.uninstall(prodmod=pm_, ext=fx)
        print(f"D.6 module lacking a required symbol -> refused ({rm['reason'][-60:]!r}); changed decode branch -> "
              f"dispatcher installed, K2 off: {rk['apply_off_reason'][:90]!r}")
        fast_cases = []
        saved_fast = os.environ.get("GLM53_EXL3_MOE_FAST")

        def ver_raises():
            raise RuntimeError("synthetic glm53_fast_moe_version failure")

        try:
            for val, ext_extra, expect_kind, expect_warn in (
                    ("1", {}, "stock exl3_moe (thin-decode requested but not built in)", "glm53_fast_moe_version"),
                    ("1", {"glm53_fast_moe_version": lambda: 1}, "native thin-decode exl3_moe (", None),
                    ("1", {"glm53_fast_moe_version": lambda: 2}, "native thin-decode exl3_moe of an untested version",
                     "returned 2, not 1"),
                    ("1", {"glm53_fast_moe_version": ver_raises},
                     "native thin-decode exl3_moe of an untested version", "raised (RuntimeError"),
                    ("0", {}, "stock exl3_moe (GLM53_EXL3_MOE_FAST=0", None),
                    ("2", {}, ("unknown" if hasattr(prod, "exl3_moe_fast_requested") else "stock exl3_moe"),
                     ("GLM53_EXL3_MOE_FAST='2'" if hasattr(prod, "exl3_moe_fast_requested") else None))):
                os.environ["GLM53_EXL3_MOE_FAST"] = val
                w0 = len(warned)
                fx, pm_ = fake_ext(**ext_extra), proxy()
                try:
                    rf = integrate.install(prodmod=pm_, ext=fx)
                except Exception as exc:  # noqa: BLE001
                    rf = {"installed": False, "raised": repr(exc)}
                ws = new_warnings(w0)
                ok_f = (rf["installed"] and rf.get("orig_kernel", "").startswith(expect_kind)
                        and (any(expect_warn in m for m in ws) if expect_warn else not ws))
                fast_cases.append((val, bool(ext_extra), rf.get("orig_kernel"), ok_f))
                ck(ok_f, f"GLM53_EXL3_MOE_FAST={val} thin-build={bool(ext_extra)}: {rf}, warnings {ws}")
                if rf.get("installed"):
                    integrate.uninstall(prodmod=pm_, ext=fx)
        finally:
            if saved_fast is None:
                os.environ.pop("GLM53_EXL3_MOE_FAST", None)
            else:
                os.environ["GLM53_EXL3_MOE_FAST"] = saved_fast
        print(f"D.6 GLM53_EXL3_MOE_FAST (value, thin-decode build, kernel TF replaces, ok): {fast_cases}")
        # install()'s "will refuse to load" WARNING vs what production's own load does, same env and extension
        if hasattr(prod, "exl3_moe_fast_requested"):
            agree = []
            had_ver = hasattr(xl, "glm53_fast_moe_version")
            try:
                os.environ["GLM53_EXL3_MOE_FAST"] = "1"
                for tag, fn in (("absent", None), ("1", lambda: 1), ("2", lambda: 2), ("raises", ver_raises)):
                    if fn is None:
                        if hasattr(xl, "glm53_fast_moe_version"):
                            delattr(xl, "glm53_fast_moe_version")
                    else:
                        xl.glm53_fast_moe_version = fn
                    _, problem = integrate.exl3_moe_kind(prod, xl)
                    warns_refuse = bool(problem) and "refuse to load" in problem
                    try:
                        H.make_layer(prod, H.Weights(4, K, N, dev, seed=33))
                        refused = None
                    except RuntimeError as exc:
                        refused = str(exc)[:70]
                    agree.append((tag, warns_refuse, refused is not None))
                    ck(warns_refuse == (refused is not None),
                       f"thin build {tag}: install() warns 'refuse to load' {warns_refuse}, production refused: {refused}")
            finally:
                if saved_fast is None:
                    os.environ.pop("GLM53_EXL3_MOE_FAST", None)
                else:
                    os.environ["GLM53_EXL3_MOE_FAST"] = saved_fast
                if not had_ver and hasattr(xl, "glm53_fast_moe_version"):
                    delattr(xl, "glm53_fast_moe_version")
            print(f"D.6 GLM53_EXL3_MOE_FAST=1, thin build (glm53_fast_moe_version, install() warns refuse, production "
                  f"refused): {agree}")
        tf.set_enabled(True)
        # ---------------- D.2 idempotent install --------------------------------------------------------------
        ra = integrate.install(prodmod=prod, ext=xl)
        d1, b1 = xl.exl3_moe, prod.build_exl3_fused_state
        rb = integrate.install(prodmod=prod, ext=xl)
        ok = (ra["installed"] and rb["reason"] == "already installed" and xl.exl3_moe is d1
              and d1._tf_exl3_orig is real and prod.build_exl3_fused_state is b1
              and not getattr(b1._tf_exl3_orig, "_tf_exl3_hook", False)
              and d1.__doc__ == real.__doc__ and not hasattr(d1, "__wrapped__"))
        print(f"D.2 install twice: second -> {rb['reason']!r}; single wrapper: {ok}")
        ck(ok, "idempotent install")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    disp = xl.exl3_moe
    tf.CFG.strict = False

    # ---------------- layer through the hooked production build -------------------------------------------
    W = H.Weights(NEXP, K, N, dev, seed=31)
    layer = H.make_layer(prod, W)
    key = (0, layer._exl3_ptrs["gate_trellis"].data_ptr())
    ck(key in tf.REG, "layer not registered")
    g = torch.Generator().manual_seed(2)
    T = 8
    x = torch.randn(T, K, generator=g).half().to(dev)
    args = H.capture_args(prod, xl, x, H.random_ids(T, NEXP, TOPK, g, dev), H.random_weights(T, TOPK, g, dev), layer, 10.0)
    ref = torch.zeros(T, K, dtype=torch.float32, device=dev)
    real(*H.with_out(args, ref))
    torch.cuda.synchronize()

    # ---------------- D.3 a raising pre-flight never reaches production -----------------------------------
    orig_pf = tf.preflight

    def boom(layer, orig=None):
        raise RuntimeError("synthetic pre-flight failure")

    tf.preflight = boom
    try:
        W2 = H.Weights(4, K, N, dev, seed=32)
        l2 = H.make_layer(prod, W2)
        ok = bool(l2._exl3_ptrs) and (0, l2._exl3_ptrs["gate_trellis"].data_ptr()) not in tf.REG
    except Exception as e:  # noqa: BLE001
        ok = False
        print("  raised into production:", e)
    finally:
        tf.preflight = orig_pf
    print(f"D.3 raising pre-flight: production built its pointer tables and the layer stays on orig: {ok}")
    ck(ok, "pre-flight exception escaped into production")

    # ---------------- D.7 production's own fused build fails for a layer (review F4) ---------------------------
    real_conc = xl.exl3_moe_max_concurrency
    w0, f0, n_reg = len(warned), tf.COUNTERS["prod_build_failed"], len(tf.REG)

    def conc_fails(idx):
        raise RuntimeError("synthetic: exl3_moe_max_concurrency unavailable")

    xl.exl3_moe_max_concurrency = conc_fails
    try:
        l7 = H.make_layer(prod, H.Weights(4, K, N, dev, seed=34), require_ptrs=False)
        raised7 = None
    except Exception as exc:  # noqa: BLE001
        l7, raised7 = None, repr(exc)
    finally:
        xl.exl3_moe_max_concurrency = real_conc
    w7 = [m for m in new_warnings(w0) if "production's build_exl3_fused_state raised" in m]
    ok7 = (raised7 is None and l7 is not None and not getattr(l7, "_exl3_ptrs", None) and len(tf.REG) == n_reg
           and tf.COUNTERS["prod_build_failed"] == f0 + 1 and len(w7) == 1 and "python loop" in w7[0])
    print(f"D.7 production build fails: production fell back (no pointer tables, load not raised: {raised7 is None}), "
          f"layer registered {len(tf.REG) != n_reg}, WARNING {w7[:1]}")
    ck(ok7, f"production build failure must be loud and change nothing else: raised {raised7}, warnings {w7}")

    # ---------------- C.3 pointer sanity -------------------------------------------------------------------
    regions = tf._regions_of_layer(layer)
    temps = layer._exl3_fused_temps
    bad = dict(layer._exl3_ptrs)
    bad["gate_trellis"] = bad["gate_trellis"].clone()
    bad["gate_trellis"][3] += 128                                  # inside the tensor, not at a matrix boundary
    r_mis = tf.register(bad, temps, K, N, regions, selftest=False, name="misaligned")
    bad2 = dict(layer._exl3_ptrs)
    bad2["up_suh"] = layer._exl3_ptrs["down_svh"]                   # points into w2_svh, not w13_suh
    r_out = tf.register(bad2, temps, K, N, regions, selftest=False, name="outside")
    print(f"C.3 pointer sanity: misaligned entry registered={r_mis}; foreign-tensor table registered={r_out}")
    ck(not r_mis and not r_out, "pointer sanity accepted a bad table")
    ck(tf.STATE.enabled, "pointer rejection must not disable TF globally")

    # ---------------- C.3 failing self-test disables TF --------------------------------------------------
    real_launch = tf._launch_kernels

    def noisy(p, a):
        real_launch(p, a)
        a[1].add_(0.05 * a[1].abs().max())                          # a systematic TF error

    tf.REG.pop(key)
    tf._launch_kernels = noisy
    try:
        r_st = tf.register(layer._exl3_ptrs, temps, K, N, regions, orig=real, name="selftest-teeth")
    finally:
        tf._launch_kernels = real_launch
    dis = not tf.STATE.enabled
    print(f"C.3 self-test with an injected TF error: registered={r_st}, TF disabled={dis} "
          f"({(tf.STATE.disabled_reason or '')[:60]}...)")
    ck(not r_st and dis and key not in tf.REG, "self-test did not catch the injected error")
    o = torch.zeros_like(ref)
    c0 = tf.COUNTERS["tf_calls"]
    disp(*H.with_out(args, o))
    ck(tf.COUNTERS["tf_calls"] == c0, "disabled TF still ran")
    tf.set_enabled(True)
    ck(tf.register(layer._exl3_ptrs, temps, K, N, regions, orig=real), "re-register after re-enable")

    # ---------------- C.5 exception policy ------------------------------------------------------------------
    class Proxy:
        def __init__(self, inner, msg):
            self.inner, self.msg = inner, msg

        def moe_forward(self, *a):
            raise RuntimeError(self.msg)

        def __getattr__(self, k):
            return getattr(self.inner, k)

    ext = tf._EXT
    tf._EXT = Proxy(ext, "expert_count: expected a contiguous CUDA tensor of dtype Long")
    o = torch.zeros_like(ref)
    try:
        disp(*H.with_out(args, o))
        torch.cuda.synchronize()
        fell_back = float((o - ref).norm() / ref.norm()) <= 1e-6 and not tf.STATE.enabled
        sticky = tf.plan(H.with_out(args, o)) is None
    finally:
        tf._EXT = ext
    print(f"C.5 pre-check error: fell back to orig with its result and disabled TF: {fell_back}; sticky: {sticky}")
    ck(fell_back and sticky, "pre-check error policy")
    tf.set_enabled(True)
    tf._EXT = Proxy(ext, "CUDA error: an illegal memory access was encountered")
    raised = False
    try:
        disp(*H.with_out(args, torch.zeros_like(ref)))
    except RuntimeError:
        raised = True
    finally:
        tf._EXT = ext
    print(f"C.5 CUDA-error message: re-raised {raised}, TF still enabled {tf.STATE.enabled}")
    ck(raised and tf.STATE.enabled, "CUDA error policy")
    tf.CFG.strict = True
    tf._EXT = Proxy(ext, "expert_count: expected ...")
    raised = False
    try:
        disp(*H.with_out(args, torch.zeros_like(ref)))
    except RuntimeError:
        raised = True
    finally:
        tf._EXT = ext
        tf.CFG.strict = False
    print(f"C.5 STRICT: pre-check error re-raised {raised}")
    ck(raised and tf.STATE.enabled, "strict policy")
    o = torch.zeros_like(ref)
    disp(*H.with_out(args, o))
    torch.cuda.synchronize()
    ck(tf.passes_e1(tf.compare(o, ref)), "TF result after the policy tests")

    # ---------------- C.4 token window / pair cap / small R never switch TF off (F1) -------------------------
    c1 = H.c1_temps(layer)
    gw = torch.Generator().manual_seed(64)
    argsB = {}
    for B in (1, 2, 4, 5, 8, 9, 15, 16, 64, 65, 128):
        xb_ = torch.randn(B, K, generator=gw).half().to(dev)
        argsB[B] = H.with_temps(H.capture_args(prod, xl, xb_, H.random_ids(B, NEXP, TOPK, gw, dev),
                                               H.random_weights(B, TOPK, gw, dev), layer, 10.0), c1)

    def reregister(tag, temps_=None):
        tf.REG.pop(key, None)
        ok_ = tf.register(layer._exl3_ptrs, temps if temps_ is None else temps_, K, N, regions, orig=real, name=tag)
        info_ = tf.REG.get(key)
        return ok_ and info_ is not None and tf.STATE.enabled, info_

    def serves(B, temps_=None):
        a_ = argsB[B] if temps_ is None else H.with_temps(argsB[B], temps_)
        return tf.plan(H.with_out(a_, torch.zeros(B, K, dtype=torch.float32, device=dev))) is not None

    def delegates_like_orig(B):
        o1 = torch.zeros(B, K, dtype=torch.float32, device=dev)
        o2 = torch.zeros_like(o1)
        c0 = tf.COUNTERS["tf_calls"]
        disp(*H.with_out(argsB[B], o1))
        real(*H.with_out(argsB[B], o2))
        torch.cuda.synchronize()
        return tf.COUNTERS["tf_calls"] == c0 and float((o1 - o2).norm()) <= 1e-6 * float(o2.norm())

    windows = [("1:4", {1: True, 4: True, 5: False, 8: False}),
               ("2:64", {1: False, 2: True, 64: True, 65: False}),
               ("64", {1: True, 64: True, 65: False, 128: False}),
               (":8", {1: True, 8: True, 9: False}),
               ("16:", {15: False, 16: True, 128: True})]
    for val, expect in windows:
        cfg = tf.configure({"TF_EXL3_TOKENS": val})
        ok_, info_ = reregister(f"window {val}")
        got = {B: serves(B) for B in expect}
        outside = min(B for B, e in expect.items() if not e)
        dl = delegates_like_orig(outside)
        print(f"C.4 TF_EXL3_TOKENS={val!r} -> [{cfg.tokens_lo}, {cfg.tokens_hi if cfg.tokens_hi < 1 << 30 else 'inf'}]: "
              f"registered with self-test {ok_} (B={info_.selftest.get('B') if info_ else None}, "
              f"{info_.selftest.get('cases') if info_ else 0} cases), TF enabled {tf.STATE.enabled}; plan served {got}; "
              f"B={outside} delegated with orig's result {dl}")
        ck(not cfg.invalid and ok_ and info_.selftest.get("cases", 0) >= 2, f"window {val}: layer not registered / TF off")
        ck(got == expect, f"window {val}: plan() served {got}, expected {expect}")
        ck(dl, f"window {val}: B={outside} not delegated with orig's result")
    for val, lo_hi in (("abc", None), ("4:2", None), ("0:4", None), ("1:x", None), ("5:-1", None)):
        cfg = tf.configure({"TF_EXL3_TOKENS": val})
        rj = tf.register(layer._exl3_ptrs, temps, K, N, regions, orig=real, name="invalid cfg", selftest=False)
        ck(bool(cfg.invalid) and not rj and tf.STATE.enabled, f"malformed TF_EXL3_TOKENS={val!r} accepted: {cfg}")
    saved_env = {k: os.environ.get(k) for k in ("TF_EXL3_MOE", "TF_EXL3_TOKENS", "TF_EXL3_MAX_PAIRS")}
    try:
        os.environ["TF_EXL3_MOE"] = "1"
        refused = []
        for k_, v_ in (("TF_EXL3_TOKENS", "4:2"), ("TF_EXL3_MAX_PAIRS", "lots")):
            os.environ[k_] = v_
            r_ = integrate.install(prodmod=prod, ext=xl)
            refused.append((not r_["installed"]) and k_ in (r_["reason"] or "") and xl.exl3_moe is disp)
            os.environ.pop(k_)
        print(f"C.4 malformed values (abc, 4:2, 0:4, 1:x, 5:-1) listed as invalid and rejected by register(); "
              f"install() refuses with the reason for TF_EXL3_TOKENS=4:2 / TF_EXL3_MAX_PAIRS=lots: {refused}")
        ck(all(refused), "install() did not refuse a malformed TF_EXL3_* value")
    finally:
        for k_, v_ in saved_env.items():
            if v_ is None:
                os.environ.pop(k_, None)
            else:
                os.environ[k_] = v_
    for mp, expect in (("32", {4: True, 5: False}), ("4", {1: False})):
        cfg = tf.configure({"TF_EXL3_MAX_PAIRS": mp})
        ok_, info_ = reregister(f"max_pairs {mp}")
        got = {B: serves(B) for B in expect}
        print(f"C.4 TF_EXL3_MAX_PAIRS={mp}: P_cap {info_.P_cap if info_ else None}, registered {ok_} (self-test "
              f"B={info_.selftest.get('B') if info_ else None}), TF enabled {tf.STATE.enabled}; plan served {got}")
        ck(ok_ and got == expect, f"TF_EXL3_MAX_PAIRS={mp}: registered {ok_}, served {got}")
    tf.configure({})
    temps4 = tuple(torch.empty((t.shape[0], 4, t.shape[2]), dtype=t.dtype, device=dev) for t in temps)
    ok_, info_ = reregister("R=4", temps4)
    got = {4: serves(4, temps4), 5: serves(5, temps4)}
    print(f"C.4 temps with R = 4 (EXL3_TEMP_ROWS_FUSED=4): P_cap {info_.P_cap if info_ else None}, registered {ok_} "
          f"(self-test B={info_.selftest.get('B') if info_ else None}); plan served {got}")
    ck(ok_ and got == {4: True, 5: False}, f"R=4: registered {ok_}, served {got}")
    tf.configure()
    ok_, info_ = reregister("defaults again")
    ck(ok_ and serves(128) and info_.selftest.get("B") == (1, 8) and info_.selftest.get("L") == (10.0, 0.0),
       f"default re-registration: {info_.selftest if info_ else None}")

    # ---------------- D.8 observability (review safety-F3) -----------------------------------------------------
    info_msgs = []

    class _Info(logging.Handler):
        def emit(self, record):
            info_msgs.append(record.getMessage())

    hinfo = _Info(level=logging.INFO)
    lg = logging.getLogger("vllm.tf_exl3_moe")
    old_level = lg.level
    lg.setLevel(logging.INFO)
    lg.addHandler(hinfo)
    try:
        # (a) a contract hand-off on a registered layer: same pointer values in another tensor (production accepts
        # the call, TF only serves the tables it validated) -> production's result, one WARNING for two calls
        a_c = list(args)
        a_c[14] = args[14].clone()
        a_c = tuple(a_c)
        w0, dc0, t0 = len(warned), tf.COUNTERS["delegated_contract"], tf.COUNTERS["tf_calls"]
        o1, o2 = torch.zeros_like(ref), torch.zeros_like(ref)
        disp(*H.with_out(a_c, o1))
        disp(*H.with_out(a_c, o2))
        torch.cuda.synchronize()
        wc = [m for m in new_warnings(w0) if "handed to production's exl3_moe" in m]
        r1, r2 = float((o1 - ref).norm() / ref.norm()), float((o2 - ref).norm() / ref.norm())
        same_c = r1 <= 1e-6 and r2 <= 1e-6
        print(f"D.8 contract hand-off (a copied gate_suh table): production's result {same_c} (rel {r1:.1e}, {r2:.1e}), counted "
              f"{tf.COUNTERS['delegated_contract'] - dc0}, WARNING x{len(wc)}: {wc[:1]}")
        ck(same_c and tf.COUNTERS["tf_calls"] == t0 and tf.COUNTERS["delegated_contract"] == dc0 + 2 and len(wc) == 1
           and "gate_suh" in wc[0], f"contract hand-off not reported once: {wc}")
        # (b) an expected hand-off (B = R + 1, production's prefill branch through its own apply): counted, no WARNING
        R_ = int(layer._exl3_fused_temps[0].shape[1])
        gp = torch.Generator().manual_seed(81)
        xp = torch.randn(R_ + 1, K, generator=gp).to(torch.bfloat16).to(dev)
        w0, db0, ab0 = len(warned), tf.COUNTERS["delegated_b_gt_r"], tf.COUNTERS["apply_delegated_b_gt_r"]
        prod.apply_exl3_fused_moe(xp, H.random_ids(R_ + 1, NEXP, TOPK, gp, dev), H.random_weights(R_ + 1, TOPK, gp, dev),
                                  layer, layer._exl3_inners, None, 10.0)
        torch.cuda.synchronize()
        wb = [m for m in new_warnings(w0) if "handed to production" in m]
        print(f"D.8 B = R + 1 = {R_ + 1} through production's apply: K2 hand-offs b_gt_r "
              f"{tf.COUNTERS['apply_delegated_b_gt_r'] - ab0}, exl3_moe hand-offs b_gt_r "
              f"{tf.COUNTERS['delegated_b_gt_r'] - db0}, WARNINGs {wb}")
        ck(tf.COUNTERS["apply_delegated_b_gt_r"] == ab0 + 1 and not wb, f"B > R hand-off: warnings {wb}")
        # (c) summary at 10^k calls
        n0 = len(info_msgs)
        tf._OBS["next_summary"] = tf.COUNTERS["tf_calls"] + tf.COUNTERS["delegated"] + 1
        disp(*H.with_out(args, torch.zeros_like(ref)))
        torch.cuda.synchronize()
        sm = [m for m in info_msgs[n0:] if "MoE calls: TF" in m]
        print(f"D.8 summary at the next 10^k: {sm[:1]}")
        ck(len(sm) == 1 and f"contract {tf.COUNTERS['delegated_contract']}" in sm[0]
           and f"TF {tf.COUNTERS['tf_calls']}" in sm[0], f"summary line: {sm}")
        # (d) CUDA graph capture: one TF call (B=T) and one contract hand-off (B=T) captured -> summary after capture
        tf.CAPTURE_QUIET_S = 0.5
        n0 = len(info_msgs)
        gtf0, gd0 = tf.COUNTERS["graph_tf_calls"], tf.COUNTERS["graph_delegated"]
        oc1, oc2 = torch.zeros_like(ref), torch.zeros_like(ref)
        a1, a2 = H.with_out(args, oc1), H.with_out(a_c, oc2)
        s_ = torch.cuda.Stream()
        s_.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s_):                                  # warm up outside the capture
            disp(*a1)
            disp(*a2)
        torch.cuda.current_stream().wait_stream(s_)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            disp(*a1)
            disp(*a2)
        oc1.zero_()
        oc2.zero_()
        gr.replay()
        torch.cuda.synchronize()
        import time as _t
        deadline = _t.monotonic() + 10.0
        cap_lines = []
        while _t.monotonic() < deadline and not cap_lines:
            _t.sleep(0.25)
            cap_lines = [m for m in info_msgs[n0:] if "after CUDA graph capture" in m]
        replay_ok = tf.passes_e1(tf.compare(oc1, ref)) and float((oc2 - ref).norm()) <= 1e-6 * float(ref.norm())
        print(f"D.8 capture of one TF call and one hand-off (B={T}): graph TF +{tf.COUNTERS['graph_tf_calls'] - gtf0}, "
              f"production +{tf.COUNTERS['graph_delegated'] - gd0}; replay results ok {replay_ok}; {cap_lines[:1]}")
        ck(tf.COUNTERS["graph_tf_calls"] == gtf0 + 1 and tf.COUNTERS["graph_delegated"] == gd0 + 1 and replay_ok
           and len(cap_lines) == 1 and f"B={T};" in cap_lines[0] and f"contract B={T}" in cap_lines[0],
           f"capture summary: {cap_lines}")
        del gr
        torch.cuda.synchronize()
        tf.STATE.captured = False             # the only graph that captured a TF launch is gone (D.5 frees scratch)
    finally:
        tf.CAPTURE_QUIET_S = 10.0
        lg.removeHandler(hinfo)
        lg.setLevel(old_level)

    # ---------------- D.5 uninstall ----------------------------------------------------------------------------
    def third_party(*a):
        return disp(*a)

    xl.exl3_moe = third_party
    u = integrate.uninstall(prodmod=prod, ext=xl)
    ok = (xl.exl3_moe is third_party and not u["restored_exl3_moe"] and u["restored_build"]
          and not getattr(prod.build_exl3_fused_state, "_tf_exl3_hook", False) and not tf.STATE.enabled
          and not tf.REG and not tf._SCRATCH)
    print(f"D.5 uninstall with a later third-party patch: patch kept {xl.exl3_moe is third_party}, build hook restored "
          f"{u['restored_build']}, TF disabled {not tf.STATE.enabled}, registry/scratch cleared {not tf.REG and not tf._SCRATCH}")
    ck(ok, f"uninstall: {u}")
    xl.exl3_moe = real
    u2 = integrate.uninstall(prodmod=prod, ext=xl)
    ck(xl.exl3_moe is real and not u2["restored_exl3_moe"], "uninstall not idempotent")

    # a dispatcher that is called while nothing is registered says so (once) and still delegates exactly
    msgs = []

    class _Cap(logging.Handler):
        def emit(self, record):
            msgs.append(record.getMessage())

    hcap = _Cap(level=logging.WARNING)
    logging.getLogger("vllm.tf_exl3_moe").addHandler(hcap)
    try:
        d_inert = integrate._make_dispatcher(tf, real)
        o1 = torch.zeros_like(ref)
        o2 = torch.zeros_like(ref)
        d_inert(*H.with_out(args, o1))
        d_inert(*H.with_out(args, o2))
        torch.cuda.synchronize()
    finally:
        logging.getLogger("vllm.tf_exl3_moe").removeHandler(hcap)
    inert = [m_ for m_ in msgs if "no layer is registered" in m_]
    same = float((o1 - ref).norm() / ref.norm()) <= 1e-6 and float((o2 - ref).norm() / ref.norm()) <= 1e-6
    print(f"D.5 dispatcher with an empty registry: inert warning logged {len(inert)}x over 2 calls; results == orig {same}")
    ck(len(inert) == 1 and same, f"inert warning {inert}, same {same}")

    # ---------------- D.4 packaging + vLLM plugin loader (fresh interpreters) ------------------------------
    tgt = "/tmp/tf_exl3_pkg"
    pip = subprocess.run([sys.executable, "-m", "pip", "install", "--no-build-isolation", "--no-deps", "--no-index",
                          "--target", tgt, str(H.REPO)], capture_output=True, text=True)
    print(f"D.4 pip install --no-build-isolation --target {tgt}: rc={pip.returncode} "
          f"{(pip.stdout.strip().splitlines() or [''])[-1][:100]}")
    if pip.returncode:
        print(pip.stderr[-2000:])
    ck(pip.returncode == 0, "pip install failed")
    probe = r"""
import os, sys, json
import torch
from importlib.metadata import entry_points
eps = [f"{e.name}={e.value}" for e in entry_points(group="vllm.general_plugins")]
import logging
import vllm.logger  # noqa: F401 - vLLM's dictConfig resets handlers of existing vllm.* loggers: configure it first
_msgs = []
class _H(logging.Handler):
    def emit(self, r):
        _msgs.append(r.getMessage())
_lg = logging.getLogger("vllm.tf_exl3_moe")
_lg.addHandler(_H(level=logging.INFO))
_lg.setLevel(logging.INFO)
from vllm.plugins import load_general_plugins
load_general_plugins()
import exllamav3_ext, tf_exl3_moe, integrate
import vllm.model_executor.layers.quantization.exl3 as prodmod
fn = exllamav3_ext.exl3_moe
res = {"eps": eps, "installed": bool(getattr(fn, "_tf_exl3_dispatch", False)),
       "hook": bool(getattr(prodmod.build_exl3_fused_state, "_tf_exl3_hook", False)), "ext": None,
       "tf_file": tf_exl3_moe.__file__, "integrate_file": integrate.__file__,
       "plugin_log": [m for m in _msgs if "plugin loaded" in m]}
import fp8_gemv   # GLM53_FP8_GEMV (independent of TF_EXL3_MOE), installed by the same entry point
res["fp8_installed"] = bool(getattr(prodmod.Glm53DenseFp8Method.apply, "_tf_fp8_hook", False))
res["fp8_ext"] = getattr(fp8_gemv.STATE.ext, "__file__", None)
res["fp8_file"] = fp8_gemv.__file__
if res["installed"]:
    tf_exl3_moe.load_ext(); res["ext"] = tf_exl3_moe.EXT_SOURCE   # before the harness puts /w on sys.path
if res["installed"] and os.environ.get("TF_PROBE_LAYER") == "1":
    sys.path.append("/w/tests")
    import harness as H
    H.gpu_guard(2.0)
    dev = torch.device("cuda", 0)
    W = H.Weights(16, 4096, 1024, dev, seed=41)
    layer = H.make_layer(prodmod, W)          # production process_weights_after_loading -> the plugin's hook
    res["registered"] = len(tf_exl3_moe.REG)
    res["selftest"] = {k: v for k, v in next(iter(tf_exl3_moe.REG.values())).selftest.items() if k != "per_case"} \
        if tf_exl3_moe.REG else None
    g = torch.Generator().manual_seed(4)
    x = torch.randn(8, 4096, generator=g).to(torch.bfloat16).to(dev)   # apply_exl3_fused_moe returns its fp32 out
    ids = H.random_ids(8, 16, 8, g, dev)
    w = H.random_weights(8, 8, g, dev)
    c0 = tf_exl3_moe.COUNTERS["tf_calls"]
    y_tf = prodmod.apply_exl3_fused_moe(x, ids, w, layer, layer._exl3_inners, None, 10.0)
    torch.cuda.synchronize()
    res["tf_calls"] = tf_exl3_moe.COUNTERS["tf_calls"] - c0
    exllamav3_ext.exl3_moe = fn._tf_exl3_orig
    try:
        y_ref = prodmod.apply_exl3_fused_moe(x, ids, w, layer, layer._exl3_inners, None, 10.0)
        torch.cuda.synchronize()
    finally:
        exllamav3_ext.exl3_moe = fn
    m = tf_exl3_moe.compare(y_tf, y_ref)
    res["rel_l2"], res["e1"] = m["rel_l2"], tf_exl3_moe.passes_e1(m)
    res["ext_after"] = tf_exl3_moe.EXT_SOURCE
    res["tf_file_after"] = sys.modules["tf_exl3_moe"].__file__
print(json.dumps(res))
"""
    import json

    def run_probe(env_extra):
        env = {k: v for k, v in os.environ.items()
               if k not in ("TF_EXL3_MOE", "VLLM_PLUGINS", "TF_EXL3_JIT", "GLM53_FP8_GEMV", "GLM53_FP8_GEMV_MAX_M")}
        env.update({"PYTHONPATH": tgt, "HOME": "/tmp"})
        env.update(env_extra)
        pr = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=env, cwd="/tmp")
        line = [l for l in pr.stdout.splitlines() if l.startswith("{")]
        if pr.returncode or not line:
            print(pr.stdout[-1500:], pr.stderr[-1500:])
            return None
        return json.loads(line[-1])

    p_on = run_probe({"TF_EXL3_MOE": "1", "TF_PROBE_LAYER": "1"})
    p_off = run_probe({})
    p_allow = run_probe({"TF_EXL3_MOE": "1", "VLLM_PLUGINS": "some_other_plugin"})
    print(f"D.4 fresh interpreter, TF_EXL3_MOE=1: {p_on}")
    print(f"D.4 fresh interpreter, TF_EXL3_MOE unset: installed={p_off and p_off['installed']}")
    print(f"D.4 fresh interpreter, VLLM_PLUGINS excludes us: installed={p_allow and p_allow['installed']}")
    ck(p_on is not None and "tf_exl3_moe=integrate:plugin_register" in p_on["eps"] and p_on["installed"]
       and p_on["hook"] and p_on["ext"].startswith(f"aot:{tgt}") and p_on["tf_file"].startswith(tgt)
       and p_on["integrate_file"].startswith(tgt), "plugin install via vLLM loader (dispatcher + build hook)")
    ck(p_on is not None and p_on.get("registered", 0) >= 1 and p_on.get("tf_calls") == 1 and p_on.get("e1")
       and p_on.get("ext_after", "").startswith(f"aot:{tgt}") and p_on.get("tf_file_after", "").startswith(tgt),
       "plugin path: layer not registered through the hook, or the production apply did not run TF correctly")
    ck(p_off is not None and not p_off["installed"] and not p_off["hook"], "plugin installed although TF_EXL3_MOE unset")
    print(f"D.4 plugin log lines: TF_EXL3_MOE=1 {p_on and p_on['plugin_log']}; unset {p_off and p_off['plugin_log']}")
    ck(p_off is not None and len(p_off["plugin_log"]) == 1 and "off" in p_off["plugin_log"][0],
       "a process that loads the plugin without TF_EXL3_MOE must say so (INFO 'plugin loaded ... off')")
    ck(p_on is not None and len(p_on["plugin_log"]) == 1 and "installing" in p_on["plugin_log"][0],
       "plugin loaded line with TF_EXL3_MOE=1")
    ck(p_allow is not None and not p_allow["installed"] and not p_allow["hook"], "VLLM_PLUGINS allowlist not honored")
    # GLM53_FP8_GEMV rides the same entry point and is independent of TF_EXL3_MOE (fp8_gemv.py)
    p_fp8 = run_probe({"GLM53_FP8_GEMV": "1"})
    print(f"D.4 fresh interpreter, GLM53_FP8_GEMV=1 only: fp8 hook {p_fp8 and p_fp8['fp8_installed']}, "
          f"TF {p_fp8 and p_fp8['installed']}, ext {p_fp8 and p_fp8['fp8_ext']}")
    ck(p_fp8 is not None and p_fp8["fp8_installed"] and not p_fp8["installed"]
       and str(p_fp8["fp8_ext"]).startswith(tgt) and p_fp8["fp8_file"].startswith(tgt),
       "GLM53_FP8_GEMV=1: FP8 hook not installed from the plugin's AOT build")
    ck(p_on is not None and not p_on["fp8_installed"] and p_on["fp8_ext"] is None
       and p_off is not None and not p_off["fp8_installed"] and p_off["fp8_ext"] is None,
       "FP8 hook installed (or its extension imported) although GLM53_FP8_GEMV unset")

    logging.getLogger("vllm.tf_exl3_moe").removeHandler(hwarn)
    H.report_peak()
    ck.summary()


if __name__ == "__main__":
    H.run_main(main)
