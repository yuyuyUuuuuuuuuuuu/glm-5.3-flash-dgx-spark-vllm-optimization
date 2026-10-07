"""Put tf_exl3_moe into production's EXL3 decode path (docs/DESIGN.md §D).

Production decode calls ``exllamav3_ext.exl3_moe`` through ``_exl3_moe_launch(fn, ...)`` with
``fn = exllamav3_ext.exl3_moe`` read on every call (docs/prod_exl3_reference.py:1636), and builds its pointer
tables and temps once per layer in ``build_exl3_fused_state`` (:1438), called by module-global name from
``process_weights_after_loading`` (:2293-2294). install() wraps both:

  (1) build hook: production's build_exl3_fused_state first (its exceptions propagate exactly as before),
      then tf_exl3_moe.preflight(layer) — never raises; a rejected layer simply stays on the original kernel.
  (2) dispatcher ``exl3_moe(*args)``: tf.plan(args) is None -> orig(*args) with exactly the args received
      (29, or 30 incl. num_active); else tf.launch(plan, args, orig). The dispatcher carries orig's __doc__
      and no __wrapped__, so production's num_active detector (:912-921) answers the same for both.
  (3) apply hook (K2, docs/OPTIMIZATION.md; TF_EXL3_APPLY, default on): production's apply_exl3_fused_moe is
      called by module-global name from apply_exl3_experts (:1818). While exllamav3_ext.exl3_moe is our dispatcher,
      tf.apply_fused() serves the decode call from the router ids (one kernel instead of production's ~12-kernel
      routing prelude); None -> production's own apply with exactly the args received (which then reaches the
      dispatcher as before). With any other exl3_moe in place (uninstalled, a third-party patch, a test swapping
      in the original kernel) the hook always delegates.

Enable: ``TF_EXL3_MOE`` in {1, on, true, yes} (read once, at install). Production-safe revert: unset it and
restart vLLM (already-captured CUDA graphs keep the TF kernels baked in). Plugin: entry point group
``vllm.general_plugins`` -> ``plugin_register`` (loaded in every vLLM process, incl. workers, before model load).

Two versions of the production module exist (docs/STATUS.md "Production module versions"): the image's own
quantization/exl3.py (docs/prod_exl3_reference.py, has the GLM53_EXL3_MX path) and the launcher's overlay exl3.py
that production installs over it at container start (docs/ref/prod_live/overlay_exl3.py, no MX path, adds
GLM53_EXL3_MOE_FAST). install() depends on neither version's extras: every optional symbol is feature-detected,
and whenever TF is enabled but cannot engage, install() logs a WARNING with the reason (never silently a no-op).

What production logs (docs/STATUS.md "Observability"; docs/PRODUCTION_PLAN.md Phase 2 checks it on BOTH ranks):
  - every process that loads the plugin: "tf_exl3_moe plugin loaded ..." (INFO) with TF_EXL3_MOE on or off, so a
    rank whose env did not carry TF_EXL3_MOE says "off" instead of staying silent;
  - install: "tf_exl3_moe installed: replaces <kernel>; production module <version>; K2 apply path on|off" (INFO),
    or a WARNING with the reason it did not install / what production will do instead;
  - per layer: "layer N registered, self-test ..." (INFO); a layer production could not build a fused state for
    (production then runs it on its python loop, or refuses to load) -> WARNING from the build hook;
  - calls: which calls TF served and which it handed to production, by reason, and what the captured CUDA graphs
    contain ("after CUDA graph capture: ..." and at 10^3, 10^4, ... calls, INFO); a call on a registered layer
    handed to production for an unexpected reason -> WARNING (once per reason).
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import logging
import os
import textwrap

_log = logging.getLogger("vllm.tf_exl3_moe")
_TRUE = frozenset({"1", "on", "true", "yes"})
PROD_MODULE = "vllm.model_executor.layers.quantization.exl3"

# Production module files this fork was tested against (sha256 of the whole file). Informational: an unknown file
# is still served (the exl3_moe path is contract-checked per call and self-tested per layer), with a WARNING.
# To add a version (docs/PRODUCTION_PLAN.md Phase 0 item 1): put the file under docs/ref/, add its sha256 here, run
# tests/run_all.sh and the drafter suite with GPU_RUN_BIND=<file>=<vLLM path> until both print ALL PASSED, commit, and
# build/install that commit. An unknown file is served too but logs "not a version this fork was tested against".
KNOWN_PROD_MODULES = {
    "2656a699a91aad3657a46f5cdca47494b6afb417ddb203fb58607c5e3bfc5501":
        "image quantization/exl3.py (docs/prod_exl3_reference.py)",
    "849e25882ab7901fbdd7227990a4f125809e1f79288ce6506311b6e6a53e6fb2":
        "launcher overlay exl3.py df864b5 (docs/ref/prod_live/overlay_exl3.py)",
}
# Symbols the exl3_moe dispatcher path needs from the production module (both known versions have them).
REQUIRED_SYMBOLS = ("fused_moe_enabled", "load_exllamav3_ext", "_exl3_moe_accepts_num_active",
                    "build_exl3_fused_state", "map_topk_to_local")
# K2 (the apply hook) re-implements production's whole decode branch: map_topk_to_local, the prelude of
# apply_exl3_fused_moe and the argument list of _exl3_moe_launch. It is installed only when these functions are
# one of the versions tests/test_apply_fused.py verified (fingerprint = sha256 of ast.dump of the function source:
# comments and formatting do not count, code and docstrings do). Both known module versions have identical
# code here. Any other version -> K2 is not installed (WARNING); the exl3_moe dispatcher path is unaffected.
K2_VERIFIED_FINGERPRINTS = {
    "apply_exl3_fused_moe": frozenset({"fa19d59307dd88bf"}),
    "map_topk_to_local": frozenset({"8d447eb03b60d92f"}),
    "_exl3_moe_launch": frozenset({"44903919b0ba7727"}),
}


def env_enabled(environ: dict | None = None) -> bool:
    """D.1: enabled iff TF_EXL3_MOE.strip().lower() in {1, on, true, yes}; unset/0/''/false/other -> off."""
    v = (os.environ if environ is None else environ).get("TF_EXL3_MOE")
    return v is not None and v.strip().lower() in _TRUE


def source_fingerprint(fn) -> str | None:
    """sha256 (first 16 hex) of ast.dump of the function's source; None if the source is not available."""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except Exception:  # noqa: BLE001 - no source (C function, stripped install, lambda...) = unknown
        return None
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()[:16]


def prod_identity(prodmod) -> dict:
    """Which production module is loaded: file, sha256, known label (or None)."""
    path = getattr(prodmod, "__file__", None)
    sha = None
    try:
        with open(path, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
    except Exception:  # noqa: BLE001
        pass
    return {"file": path, "sha256": sha, "known": KNOWN_PROD_MODULES.get(sha)}


def k2_compatibility(prodmod) -> tuple[bool, str]:
    """(True, "") when production's decode branch is one K2 was verified against, else (False, reason)."""
    bad = []
    for name, ok in K2_VERIFIED_FINGERPRINTS.items():
        fn = getattr(prodmod, name, None)
        if not callable(fn):
            bad.append(f"{name} missing")
            continue
        fp = source_fingerprint(getattr(fn, "_tf_exl3_orig", fn))
        if fp not in ok:
            bad.append(f"{name} fingerprint {fp} not in {sorted(ok)}")
    if getattr(prodmod, "MOE_ACT_SILU", None) != 0:
        bad.append(f"MOE_ACT_SILU={getattr(prodmod, 'MOE_ACT_SILU', None)!r} (K2 assumes 0)")
    return (not bad), "; ".join(bad)


# The only thin-decode build version production's overlay accepts (build_exl3_fused_state raises "Unsupported native
# EXL3 decode-pipeline version" for any other, docs/ref/prod_live/overlay_exl3.py:1277-1278) and the one nodeC measured.
THIN_DECODE_VERSION = 1


def exl3_moe_kind(prodmod, ext) -> tuple[str, str | None]:
    """(description of the exllamav3_ext.exl3_moe that TF replaces, problem or None).

    The launcher's overlay (GLM53_EXL3_MOE_FAST=1) mounts a rebuilt exllamav3_ext whose native exl3_moe dispatches
    K=4 / N%256==0 calls to an SM121 thin-decode kernel when the env variable is 1 (read by the native code itself),
    and exl3_moe_fast_requested() in the overlay module makes the load fail closed without that build, or with a
    glm53_fast_moe_version() other than 1 (or one that raises)."""
    raw = os.environ.get("GLM53_EXL3_MOE_FAST")
    ver_fn = getattr(ext, "glm53_fast_moe_version", None)
    ver = None
    ver_err = None
    if callable(ver_fn):
        try:
            ver = ver_fn()
        except Exception as exc:  # noqa: BLE001
            ver, ver_err = "?", f"{type(exc).__name__}: {exc}"
    fast_fn = getattr(prodmod, "exl3_moe_fast_requested", None)
    problem = None
    if callable(fast_fn):
        try:
            requested = bool(fast_fn())
        except Exception as exc:  # noqa: BLE001 - production itself raises at load in this case
            return (f"unknown (GLM53_EXL3_MOE_FAST={raw!r} is invalid)",
                    f"GLM53_EXL3_MOE_FAST={raw!r}: production's exl3_moe_fast_requested() raises "
                    f"({exc}); the model load will fail")
    else:
        requested = raw == "1"
    if requested and ver is None:
        problem = ("GLM53_EXL3_MOE_FAST=1 but this exllamav3_ext has no glm53_fast_moe_version (not the thin-decode "
                   "build)" + ("; production's build_exl3_fused_state will refuse to load" if callable(fast_fn)
                               else "; the stock kernel runs"))
        return "stock exl3_moe (thin-decode requested but not built in)", problem
    if requested and (ver_err is not None or ver != THIN_DECODE_VERSION):
        got = f"raised ({ver_err})" if ver_err is not None else f"returned {ver!r}"
        problem = (f"GLM53_EXL3_MOE_FAST=1 but this exllamav3_ext's glm53_fast_moe_version() {got}, not "
                   f"{THIN_DECODE_VERSION}" + ("; production's build_exl3_fused_state will refuse to load (Unsupported "
                                               "native EXL3 decode-pipeline version)" if callable(fast_fn)
                                               else "; not the thin-decode build this fork was measured against"))
        return f"native thin-decode exl3_moe of an untested version (glm53_fast_moe_version {got})", problem
    if requested:
        return f"native thin-decode exl3_moe (GLM53_EXL3_MOE_FAST=1, glm53_fast_moe_version={ver})", None
    return f"stock exl3_moe (GLM53_EXL3_MOE_FAST={raw or 'unset'}{', thin-decode build present' if ver else ''})", None


def _make_dispatcher(tf, orig):
    plan, launch, counters, reg = tf.plan, tf.launch, tf.COUNTERS, tf.REG
    inert_logged = [False]

    note_call = tf.note_call
    import torch

    capturing = torch.cuda.is_current_stream_capturing

    def exl3_moe(*args):                                  # positional only: production never passes kwargs
        miss = []
        p = plan(args, miss)
        if p is None:
            counters["delegated"] += 1
            try:
                B = args[0].shape[0]
                cap = capturing()
            except Exception:  # noqa: BLE001 - accounting only
                B, cap = "?", False
            note_call(False, B, cap, miss[0] if miss else None)
            if not reg and not inert_logged[0]:           # installed, called, but nothing registered: say so once
                inert_logged[0] = True
                why = (f"TF disabled: {tf.STATE.disabled_reason}" if not tf.STATE.enabled
                       else "the build hook was not reached or every layer was rejected")
                _log.warning("tf_exl3_moe: exl3_moe is being called but no layer is registered (TF is inert: %s); "
                             "delegating to exl3_moe", why)
            return orig(*args)                            # exactly the args received
        return launch(p, args, orig)

    exl3_moe.__doc__ = getattr(orig, "__doc__", None)    # the detector falls back to the doc string
    exl3_moe.__name__ = getattr(orig, "__name__", "exl3_moe")
    exl3_moe.__qualname__ = exl3_moe.__name__
    exl3_moe._tf_exl3_dispatch = True
    exl3_moe._tf_exl3_orig = orig
    # deliberately no __wrapped__: inspect.signature would follow it
    return exl3_moe


def _make_build_hook(tf, orig_build, orig_fn):
    def build_exl3_fused_state(layer, inners):
        try:
            orig_build(layer, inners)                     # production first; its exceptions propagate unchanged
        except Exception as exc:
            # production's process_weights_after_loading catches this and runs the layer on its python loop
            # (never calling exl3_moe), or refuses to load with GLM53_EXL3_MOE_FAST=1: either way TF cannot serve
            # the layer, and production itself says so only in an INFO line. Say it loudly, then re-raise as is.
            tf.COUNTERS["prod_build_failed"] += 1
            tf._log_once(f"prod_build:{exc!r}"[:100], logging.WARNING,
                         "tf_exl3_moe: production's build_exl3_fused_state raised %r: this layer has no fused state, so "
                         "production runs it on its python loop (or refuses to load if GLM53_EXL3_MOE_FAST=1) and TF "
                         "does not serve it (identical failures are counted, not repeated: prod_build_failed)", exc)
            raise
        try:
            tf.preflight(layer, orig_fn)
        except Exception as exc:  # noqa: BLE001 - never demote the layer to the python loop
            _log.warning("tf_exl3_moe pre-flight raised (layer stays on exl3_moe): %r", exc)

    build_exl3_fused_state.__doc__ = getattr(orig_build, "__doc__", None)
    build_exl3_fused_state._tf_exl3_hook = True
    build_exl3_fused_state._tf_exl3_orig = orig_build
    return build_exl3_fused_state


def _make_apply_hook(tf, orig_apply, ext):
    apply_fused, counters = tf.apply_fused, tf.COUNTERS

    note_apply_delegated = tf.note_apply_delegated

    def apply_exl3_fused_moe(x2d, ids, weights, layer, inners, expert_map, limit):
        if getattr(getattr(ext, "exl3_moe", None), "_tf_exl3_dispatch", False):
            miss = []
            out = apply_fused(x2d, ids, weights, layer, inners, expert_map, limit, miss)
            if out is not None:
                return out
            note_apply_delegated(getattr(x2d, "shape", ("?",))[0], miss[0] if miss else None)
        else:
            counters["apply_delegated"] += 1
        return orig_apply(x2d, ids, weights, layer, inners, expert_map, limit)

    apply_exl3_fused_moe.__doc__ = getattr(orig_apply, "__doc__", None)
    apply_exl3_fused_moe._tf_exl3_apply_hook = True
    apply_exl3_fused_moe._tf_exl3_orig = orig_apply
    return apply_exl3_fused_moe


def _refuse(report: dict, reason: str) -> dict:
    """TF is enabled but cannot engage: say so loudly (never a silent no-op)."""
    report["reason"] = reason
    _log.warning("tf_exl3_moe enabled (TF_EXL3_MOE) but NOT installed: %s; production kernels unchanged", reason)
    return report


def install(prodmod=None, ext=None, *, force: bool | None = None) -> dict:
    """Install the build hook, the dispatcher and (K2) the apply hook. No-op unless enabled (env or force=True).
    Idempotent. Every optional production symbol is feature-detected; when TF is enabled but cannot engage, the
    reason is logged as a WARNING and returned in report["reason"]."""
    report: dict = {"installed": False, "reason": None}
    if not (env_enabled() if force is None else bool(force)):
        report["reason"] = "TF_EXL3_MOE not enabled"
        return report
    import tf_exl3_moe as tf

    cfg = tf.configure()
    if cfg.invalid:
        return _refuse(report, "invalid TF_EXL3_* settings: " + "; ".join(cfg.invalid))
    if prodmod is None:
        prodmod = importlib.import_module(PROD_MODULE)
    ident = prod_identity(prodmod)
    report["prod_module"] = ident
    missing = [n for n in REQUIRED_SYMBOLS if not callable(getattr(prodmod, n, None))]
    if missing:
        return _refuse(report, f"production module {ident['file']} (sha256 {ident['sha256']}) lacks "
                               f"{', '.join(missing)}: not a module this fork can hook")
    mx_fn = getattr(prodmod, "mx_enabled", None)      # image module only (GLM53_EXL3_MX); absent in the overlay
    if callable(mx_fn) and mx_fn():
        return _refuse(report, "GLM53_EXL3_MX is on: production runs the MX path and never calls exl3_moe")
    if not prodmod.fused_moe_enabled():
        return _refuse(report, "EXL3_FUSED_MOE=0: production runs the python loop and never calls exl3_moe")
    if ext is None:
        ext = prodmod.load_exllamav3_ext()
    orig = getattr(ext, "exl3_moe", None)
    if orig is None:
        return _refuse(report, "exllamav3_ext.exl3_moe missing (production falls back to the python loop)")
    if getattr(orig, "_tf_exl3_dispatch", False):        # never wrap a wrapper
        report.update(installed=True, reason="already installed")
        return report
    kind, kind_problem = exl3_moe_kind(prodmod, ext)
    report["orig_kernel"] = kind
    if kind_problem:
        _log.warning("tf_exl3_moe: %s", kind_problem)
    accepts = prodmod._exl3_moe_accepts_num_active(orig)
    dispatcher = _make_dispatcher(tf, orig)
    if prodmod._exl3_moe_accepts_num_active(dispatcher) != accepts:
        return _refuse(report, "num_active detection would change")
    # (1) build hook, (2) exl3_moe — layers built before install stay unregistered (-> orig)
    orig_build = prodmod.build_exl3_fused_state
    if not getattr(orig_build, "_tf_exl3_hook", False):
        prodmod.build_exl3_fused_state = _make_build_hook(tf, orig_build, orig)
    ext.exl3_moe = dispatcher
    # (3) apply hook (K2), only if production's decode branch is one K2 was verified against
    orig_apply = getattr(prodmod, "apply_exl3_fused_moe", None)
    k2_reason = None
    if not cfg.apply:
        k2_reason = "TF_EXL3_APPLY=0"
    elif getattr(orig_apply, "_tf_exl3_apply_hook", False):
        k2_reason = None                                   # already hooked (idempotent)
    else:
        k2_ok, why = k2_compatibility(prodmod)
        if k2_ok:
            prodmod.apply_exl3_fused_moe = _make_apply_hook(tf, orig_apply, ext)
        else:
            k2_reason = f"production decode branch not verified for K2 ({why})"
            _log.warning("tf_exl3_moe: K2 apply path NOT installed: %s; the exl3_moe dispatcher path serves decode",
                         k2_reason)
    tf.set_enabled(True)
    apply_hook = bool(getattr(getattr(prodmod, "apply_exl3_fused_moe", None), "_tf_exl3_apply_hook", False))
    report.update(installed=True, reason="ok", accepts_num_active=accepts, orig=orig, apply_hook=apply_hook,
                  apply_off_reason=k2_reason)
    if ident["known"] is None:
        _log.warning("tf_exl3_moe: production module %s (sha256 %s) is not a version this fork was tested against; "
                     "the exl3_moe path relies on its per-call contract checks and the per-layer self-test",
                     ident["file"], ident["sha256"])
    _log.info("tf_exl3_moe installed: replaces %s; production module %s; K2 apply path %s; num_active detection %s",
              kind, ident["known"] or f"UNKNOWN sha256 {ident['sha256']}",
              "on" if apply_hook else f"off ({k2_reason})", accepts)
    return report


def uninstall(prodmod=None, ext=None) -> dict:
    """D.5: restore only what we installed (never clobber a later third-party patch); idempotent."""
    import tf_exl3_moe as tf

    report = {"restored_exl3_moe": False, "restored_build": False, "restored_apply": False}
    if prodmod is None:
        try:
            prodmod = importlib.import_module(PROD_MODULE)
        except Exception:  # noqa: BLE001
            prodmod = None
    if ext is None:
        try:
            loader = getattr(prodmod, "load_exllamav3_ext", None)
            ext = loader() if callable(loader) else importlib.import_module("exllamav3_ext")
        except Exception:  # noqa: BLE001
            ext = None
    fn = getattr(ext, "exl3_moe", None) if ext is not None else None
    if fn is not None and getattr(fn, "_tf_exl3_dispatch", False):
        ext.exl3_moe = fn._tf_exl3_orig
        report["restored_exl3_moe"] = True
    b = getattr(prodmod, "build_exl3_fused_state", None) if prodmod is not None else None
    if b is not None and getattr(b, "_tf_exl3_hook", False):
        prodmod.build_exl3_fused_state = b._tf_exl3_orig
        report["restored_build"] = True
    a = getattr(prodmod, "apply_exl3_fused_moe", None) if prodmod is not None else None
    if a is not None and getattr(a, "_tf_exl3_apply_hook", False):
        prodmod.apply_exl3_fused_moe = a._tf_exl3_orig
        report["restored_apply"] = True
    tf.set_enabled(False)                                 # stale references to the dispatcher now delegate
    tf.STATE.disabled_reason = "uninstalled"
    tf.reset()
    return report


def plugin_register() -> None:
    """vllm.general_plugins entry point (D.4): install() and never raise into vLLM's plugin loader.
    Every process that loads the plugin says so (INFO), with TF_EXL3_MOE on or off: on a TP=2 deployment a rank
    whose env did not carry TF_EXL3_MOE then logs "off" instead of nothing (docs/PRODUCTION_PLAN.md Phase 2)."""
    try:
        on = env_enabled()
        _log.info("tf_exl3_moe plugin loaded (pid %d): TF_EXL3_MOE=%r -> %s", os.getpid(),
                  os.environ.get("TF_EXL3_MOE"), "installing" if on else "off, production kernels unchanged")
        if on:
            install()
    except Exception as exc:  # noqa: BLE001
        _log.warning("tf_exl3_moe plugin install failed (production kernel unchanged): %r", exc)
    try:  # GLM53_DEC_MOEGLUE (glm53_moeglue.py, docs/DEC_MOEGLUE.md): needs TF (above) installed; inert when unset
        import glm53_moeglue
        glm53_moeglue.plugin_install()
    except Exception as exc:  # noqa: BLE001
        _log.warning("glm53_moeglue not loaded: %r", exc)
    try:  # independent of TF_EXL3_MOE: GLM53_PREFILL_FUSED_CAP (glm53_prefill_cap.py, docs/PREFILL_CAP.md), inert when unset
        import glm53_prefill_cap
        glm53_prefill_cap.plugin_install()
    except Exception as exc:  # noqa: BLE001
        _log.warning("glm53_prefill_cap not loaded: %r", exc)
    try:  # independent of TF_EXL3_MOE: GLM53_PREFILL_QUICKWINS (glm53_prefill_quickwins.py, docs/PREFILL_QUICKWINS.md)
        import glm53_prefill_quickwins
        glm53_prefill_quickwins.plugin_install()
    except Exception as exc:  # noqa: BLE001
        _log.warning("glm53_prefill_quickwins not loaded: %r", exc)
    try:  # independent of TF_EXL3_MOE: GLM53_MEM_HYGIENE / GLM53_TF_PROFILE (glm53_runtime.py), inert when unset
        import glm53_runtime
        glm53_runtime.install()
    except Exception as exc:  # noqa: BLE001
        _log.warning("glm53_runtime not installed: %r", exc)
    try:  # independent of TF_EXL3_MOE: GLM53_FP8_GEMV (fp8_gemv.py, FP8-Marlin decode linears), inert when unset
        import fp8_gemv
        fp8_gemv.plugin_register()
    except Exception as exc:  # noqa: BLE001
        _log.warning("tf_fp8_gemv not installed: %r", exc)
    try:  # GLM53_DEC_FP8ROOF (fp8_roof.py, docs/DEC_FP8ROOF.md): needs GLM53_FP8_GEMV installed, inert when unset
        import fp8_roof
        fp8_roof.plugin_register()
    except Exception as exc:  # noqa: BLE001
        _log.warning("tf_fp8_roof not installed: %r", exc)
    try:  # independent of TF_EXL3_MOE: GLM53_BF16_GEMV (glm53_gemv_install.py, docs/BF16_GEMV.md), inert when unset
        import glm53_gemv_install
        glm53_gemv_install.plugin_install()
    except Exception as exc:  # noqa: BLE001
        _log.warning("glm53_gemv_install not loaded: %r", exc)
    try:  # independent of TF_EXL3_MOE: GLM53_MLA_PREFILL (glm53_mla_prefill.py, docs/MLA_PREFILL.md), inert when unset
        import glm53_mla_prefill
        glm53_mla_prefill.plugin_install()
    except Exception as exc:  # noqa: BLE001
        _log.warning("glm53_mla_prefill not loaded: %r", exc)
    try:  # independent of TF_EXL3_MOE: GLM53_DEC_HOSTLOOP / GLM53_DEC_HOSTLOOP_METER (glm53_hostloop.py,
        # docs/DEC_HOSTLOOP.md), inert when unset
        import glm53_hostloop
        glm53_hostloop.plugin_install()
    except Exception as exc:  # noqa: BLE001
        _log.warning("glm53_hostloop not loaded: %r", exc)
    try:  # independent of TF_EXL3_MOE: GLM53_DEC_KDA_LAZY (glm53_kda_lazy.py, docs/DEC_KDA_LAZY.md), inert when unset
        import glm53_kda_lazy
        glm53_kda_lazy.plugin_install()
    except Exception as exc:  # noqa: BLE001
        _log.warning("glm53_kda_lazy not loaded: %r", exc)
    try:  # independent of TF_EXL3_MOE: GLM53_DEC_AR1SHOT (glm53_ar1shot.py, docs/DEC_AR1SHOT.md), inert when unset
        import glm53_ar1shot
        glm53_ar1shot.plugin_install()
    except Exception as exc:  # noqa: BLE001
        _log.warning("glm53_ar1shot not loaded: %r", exc)
    try:  # independent of TF_EXL3_MOE: GLM53_DEC_DLMH (glm53_dlmh.py, docs/DEC_DLMH.md), inert when unset
        import glm53_dlmh
        glm53_dlmh.plugin_install()
    except Exception as exc:  # noqa: BLE001
        _log.warning("glm53_dlmh not loaded: %r", exc)
    try:  # independent of TF_EXL3_MOE: GLM53_MLA_PLAN_PIN (glm53_mla_planpin.py, docs/MLA_PLAN_PIN.md), inert when unset
        import glm53_mla_planpin
        glm53_mla_planpin.plugin_install()
    except Exception as exc:  # noqa: BLE001
        _log.warning("glm53_mla_planpin not loaded: %r", exc)
    try:  # independent of TF_EXL3_MOE: GLM53_DEC_VTRIM_STATS (glm53_vtrim_stats.py, docs/OPT_DECODE.md), inert when unset
        import glm53_vtrim_stats
        glm53_vtrim_stats.plugin_install()
    except Exception as exc:  # noqa: BLE001
        _log.warning("glm53_vtrim_stats not loaded: %r", exc)
    try:  # independent of TF_EXL3_MOE: GLM53_SPEC_VTRIM (glm53_spec_vtrim.py, docs/SPEC_VTRIM.md), inert when unset
        import glm53_spec_vtrim
        glm53_spec_vtrim.plugin_install()
    except Exception as exc:  # noqa: BLE001
        if (os.environ.get("GLM53_SPEC_VTRIM") or "").strip().lower() in ("on", "1"):
            raise     # mode on must never run on one rank only (the TP ranks would accept different drafts)
        _log.warning("glm53_spec_vtrim not loaded: %r", exc)
    try:  # independent of TF_EXL3_MOE: GLM53_DEC_SMALLOPS (glm53_smallops_install.py, docs/DEC_SMALLOPS.md), inert when unset
        import glm53_smallops_install
        glm53_smallops_install.plugin_install()
    except Exception as exc:  # noqa: BLE001
        _log.warning("glm53_smallops_install not loaded: %r", exc)
