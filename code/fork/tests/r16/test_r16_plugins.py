"""deploy-r16: every install of the bundle through vLLM's REAL plugin loader (vllm.plugins.load_general_plugins ->
entry point tf_exl3_moe = integrate:plugin_register, as the bundle's dist-info registers it), in fresh interpreters,
followed by the imports a GPU worker makes while it builds the model (model runner, SM90 sparse-MLA backend, GLM5Next
model / KDA / MLA, causal_conv1d, qwen3_dflash2, exl3 quantization, base_loader). Run with production's SM90_KV mounts
(tests/mla_env.sh) and the deploy-r15 bundle mounted read-only (GPU_RUN_RO += its site dir).

  P1 R15 env (production's .env today), deploy-r15 bundle (the production site copy) -> census A
  P2 R15 env, this bundle (every R16 flag unset)                                          -> census B
     B's patched callables == A's (same targets, same replacement code); the R16 modules are inert
     (nothing installed, no loader hook, no meta-path finder); no WARNING
  P3 R15 env + R16 ship flags (tests/r16/flags.sh R16_ON_ENV)                              -> census C
     every R16 feature installed as designed and composed: quickwins' six items, MLA wrapper over quickwins'
     forward_mqa, fp8roof on fp8_gemv (table + triggers), moeglue warm armed and its vLLM fingerprints verified WITH
     quickwins' recompiled DecoderLayer.forward, hostloop on the model runner + SM90 builder with its fingerprints
     verified, smallops dconv loader hook, loader wrapper chain smallops -> bf16 gemv -> moeglue -> vLLM; no WARNING
  P4 P3 + every other R16 knob passed EMPTY (what the launcher does for unset knobs)       -> census D == C
  (P0 R15 env without the SM90 file would differ: not run.) The CUDA context after plugin loading is reported per
  census (production's R15 plugins already create one: tests/r16/probe_plugin_cuda_ctx.py).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

R15_SITE = os.environ.get("R15_SITE", os.path.join(os.environ.get("TF_EXL3_ASSETS") or os.path.expanduser("~/tf-exl3-assets"), "prod-launcher/overlay/tf/site"))
HERE = Path(__file__).resolve().parent


def envset(s: str) -> dict:
    return dict(kv.split("=", 1) for kv in s.split(";") if kv)


src = (HERE / "flags.sh").read_text()


def _var(name: str) -> str:
    i = src.index(name + '="') + len(name) + 2
    return src[i:src.index('"', i)]


R15 = envset(_var("R16_R15_ENV"))
R16_ON = envset(_var("R16_ON_ENV"))
KNOBS = _var("R16_KNOBS").split()

CHILD = r'''
import importlib, inspect, json, logging, os, sys, types
recs = []
class H(logging.Handler):
    def emit(self, r):
        recs.append((r.levelname, r.name, r.getMessage()[:300]))
root = logging.getLogger(); root.addHandler(H()); root.setLevel(logging.INFO)
logging.getLogger("vllm").addHandler(H())
import torch
from vllm.plugins import load_general_plugins
load_general_plugins()
ctx_after_plugins = torch.cuda.is_initialized()
WORKER = ["vllm.v1.worker.gpu.model_runner", "vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90",
          "vllm.model_executor.layers.attention.mla_attention", "vllm.models.glm5next.nvidia.kda",
          "vllm.model_executor.layers.mamba.ops.causal_conv1d", "vllm.models.glm5next.nvidia.model",
          "vllm.model_executor.models.qwen3_dflash2", "vllm.model_executor.layers.quantization.exl3",
          "vllm.model_executor.model_loader.base_loader", "vllm.model_executor.layers.linear"]
imp_err = {}
for m in WORKER:
    try:
        importlib.import_module(m)
    except Exception as exc:
        imp_err[m] = repr(exc)[:200]
BUNDLE = {"integrate", "tf_exl3_moe", "glm53_runtime", "fp8_gemv", "fp8_large_m_tl", "glm53_bf16_gemv",
          "glm53_gemv_install", "glm53_prefill_cap", "glm53_prefill_quickwins", "glm53_mla_prefill", "fp8_roof",
          "glm53_moeglue", "glm53_hostloop", "glm53_smallops", "glm53_smallops_install"}
def plain(fn):
    # only real Python functions: vllm's lazy PlaceholderModule objects import on attribute access
    if isinstance(fn, (staticmethod, classmethod, types.MethodType)):
        fn = fn.__func__
    if isinstance(fn, property):
        fn = fn.fget
    return fn if isinstance(fn, types.FunctionType) else None
def origin(fn):
    fn = plain(fn)
    if fn is None:
        return None
    f = fn.__code__.co_filename
    if f.startswith("<glm53-quickwins"):
        return "quickwins-recompiled:" + f[len("<glm53-quickwins "):-1]
    b = os.path.basename(f)[:-3] if f.endswith(".py") else None
    return b if b in BUNDLE else None
def markers(fn):
    fn = plain(fn)
    return sorted(k for k in fn.__dict__ if k.startswith(("_glm53", "__glm53", "_tf_"))) if fn is not None else []
patched = {}
for name, mod in sorted(sys.modules.items()):
    if not (name.startswith("vllm") or name.startswith("exllamav3")) or not isinstance(mod, types.ModuleType):
        continue
    try:
        items = list(vars(mod).items())
    except Exception:
        continue
    for attr, val in items:
        if isinstance(val, type) and getattr(val, "__module__", None) == name:
            for a2, v2 in list(vars(val).items()):
                o = origin(v2)
                if o:
                    patched[f"{name}.{val.__name__}.{a2}"] = [o, plain(v2).__qualname__, markers(v2)]
        else:
            o = origin(val)
            if o:
                patched[f"{name}.{attr}"] = [o, plain(val).__qualname__, markers(val)]
def chain(fn, depth=0):
    out = []
    while plain(fn) is not None and depth < 12:
        fn = plain(fn)
        out.append(origin(fn) or fn.__module__)
        nxt = None
        for k, v in fn.__dict__.items():
            if k.endswith("_orig") and callable(v):
                nxt = v
                break
        fn, depth = nxt, depth + 1
    return out
import vllm.model_executor.model_loader.base_loader as BL
ops = sorted(n for n in torch._C._dispatch_get_all_op_names() if n.split("::")[0].startswith(("tf_", "glm53")))
st = {"loader_chain": chain(BL.process_weights_after_loading)}
def g(modname, fn):
    if modname not in sys.modules:
        return None
    try:
        return fn(sys.modules[modname])
    except Exception as exc:
        return f"ERR {exc!r}"[:200]
st["quickwins"] = g("glm53_prefill_quickwins", lambda Q: {"items": sorted(Q._STATE["items"]), "installed": sorted(map(str, Q._STATE["installed"])), "refused": {str(k): str(v)[:120] for k, v in Q._STATE["refused"].items()}, "min_t": Q._STATE["min_t"], "finder": any(type(f).__module__ == "glm53_prefill_quickwins" for f in sys.meta_path)})
def mqa(S):
    fm = S.FlashInferMLASparseSM90Impl.forward_mqa
    inner = fm
    cl = getattr(fm, "__closure__", None) or ()
    for c in cl:
        try:
            v = c.cell_contents
        except ValueError:
            continue
        if callable(v) and getattr(v, "__name__", "") == "forward_mqa":
            inner = v
    return {"mla_wrapper": bool(getattr(fm, "__glm53_mla_prefill__", False)), "outer_origin": origin(fm),
            "inner_origin": origin(inner), "inner_is_quickwins": bool(getattr(inner, "_glm53_qw", False))}
st["forward_mqa"] = g("vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90", mqa)
st["mla"] = g("glm53_mla_prefill", lambda M: {"installed": M.STATE.installed, "variant": M.STATE.variant, "min_tokens": M.STATE.min_tokens, "mixed": M.STATE.mixed, "ext": M.STATE.ext is not None})
st["fp8_gemv"] = g("fp8_gemv", lambda F: {"enabled": F.STATE.enabled, "large": F.STATE.large, "roof_hook": getattr(F, "ROOF_HOOK", None) is not None, "fc_table": (4096, 20480) in F.TABLE, "n_table": len(F.TABLE), "max_m": F.CFG.max_m})
st["fp8_roof"] = g("fp8_roof", lambda R: {"installed": R.STATE.installed, "table": R.STATE.table, "pf": R.STATE.pf, "triggers": sorted(R.STATE.triggers), "mib": R.STATE.mib, "ctas": R.STATE.ctas, "max_m": R.STATE.max_m, "pol": R.STATE.pol, "bk_guard": getattr(R.STATE, "bk_guard", None), "bk_skip": sorted(getattr(R.STATE, "bk_skip", None) or ())})
def mg(MG):
    import integrate
    fps = MG._vllm_fingerprints()
    return {"hooked": MG.STATE.hooked, "glue": MG.STATE.enabled, "warm": MG.STATE.warm, "warm_reason": MG.STATE.warm_reason,
            "loader_hooked": MG.STATE.loader_hooked, "cfg": [MG.CFG.warm_set, MG.CFG.warm_mib, MG.CFG.warm_blocks, MG.CFG.warm_max_m, MG.CFG.prefetch],
            "warm_fps": fps, "warm_fps_ok": all(v in MG.WARM_VERIFIED[k] for k, v in fps.items()),
            "bk_guard": getattr(getattr(MG, "WARM", None), "bk_guard", None)}
st["moeglue"] = g("glm53_moeglue", mg)
def hl(HL):
    R = sys.modules[HL.MR_MODULE].GPUModelRunner if HL.MR_MODULE in sys.modules else None
    B = sys.modules[HL.SM90_MODULE].FlashInferMLASparseSM90Builder if HL.SM90_MODULE in sys.modules else None
    ok, why, _ = HL._verify_sources(sys.modules[HL.MR_MODULE], sys.modules[HL.SM90_MODULE]) if R is not None and B is not None else (None, "not imported", None)
    return {"enabled": HL.ST.enabled, "verify": [HL.ST.verify_first, HL.ST.verify_every], "wake": [HL._WAKE.raw, HL._WAKE.pin, HL._WAKE.tick_us], "meter": HL._METER.every,
            "patched": sorted(n for n in ("prepare_inputs", "postprocess_sampled", "add_requests", "execute_model", "sample_tokens") if R is not None and getattr(getattr(R, n, None), "_glm53_hostloop", False)),
            "kv_lens_host": bool(B is not None and getattr(B._kv_lens_host, "_glm53_hostloop", False)), "fingerprints_ok": ok, "why": why,
            "finder": any(type(f).__module__ == "glm53_hostloop" for f in sys.meta_path)}
st["hostloop"] = g("glm53_hostloop", hl)
st["smallops"] = g("glm53_smallops_install", lambda S: {"loader": S._STATE["loader"], "kinds": list(S._STATE["kinds"]), "ops": S._STATE["ops"], "grouped_conv_fp_ok": S._fingerprint(sys.modules["vllm.model_executor.models.qwen3_dflash2"]._grouped_conv) in S.FP_GROUPED_CONV})
st["bf16_gemv"] = g("glm53_gemv_install", lambda G: {"loader": G._STATE["loader"]})
st["runtime"] = g("glm53_runtime", lambda RT: {"prof_diag": RT._diag_on() if hasattr(RT, "_diag_on") else None})
exl3 = sys.modules.get("exllamav3_ext")
st["tf"] = {"dispatch": bool(getattr(getattr(exl3, "exl3_moe", None), "_tf_exl3_dispatch", False)) if exl3 else None}
warn = [r for r in recs if r[0] in ("WARNING", "ERROR") and (r[1].startswith("vllm.glm53") or r[1].startswith("vllm.tf") or "glm53" in r[2] or "tf_exl3" in r[2] or "tf_fp8" in r[2])]
print("RESULT " + json.dumps({"ctx_after_plugins": ctx_after_plugins, "import_errors": imp_err, "patched": patched, "ops": ops,
                              "states": st, "warnings": warn,
                              "installs": [r[2][:160] for r in recs if r[0] == "INFO" and ("install" in r[2] or "armed" in r[2] or "plugin loaded" in r[2])]}))
'''


def child(label: str, env_add: dict, site: str | None) -> dict:
    env = {k: v for k, v in os.environ.items() if not (k.startswith("GLM53_") or k.startswith("TF_EXL3"))}
    env.update(env_add)
    env["TF_EXL3_JIT"] = "0"
    if site is None:
        d = Path(tempfile.mkdtemp())
        di = d / "tf_exl3_moe-0.1.0.dist-info"
        di.mkdir()
        (di / "METADATA").write_text("Metadata-Version: 2.1\nName: tf_exl3_moe\nVersion: 0.1.0\n")
        (di / "entry_points.txt").write_text("[vllm.general_plugins]\ntf_exl3_moe = integrate:plugin_register\n")
        env["PYTHONPATH"] = f"{d}:/w:" + env.get("PYTHONPATH", "")
    else:
        env["PYTHONPATH"] = f"{site}:" + env.get("PYTHONPATH", "")
    p = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True, timeout=900, cwd="/tmp")
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
    if not line:
        print(f"=== {label}: no RESULT rc={p.returncode}\n{p.stdout[-3000:]}\n{p.stderr[-5000:]}")
        return {}
    r = json.loads(line[0][7:])
    print(f"=== {label}: ctx_after_plugins={r['ctx_after_plugins']} patched={len(r['patched'])} ops={len(r['ops'])} "
          f"warnings={len(r['warnings'])} import_errors={r['import_errors']}")
    return r


def judge(A: dict, B: dict, C: dict, D: dict | None, ck) -> None:
    """The P1..P4 checks on census results (from child() here, or from tools/deploy16/kit_chain.sh's in-container
    runs through the launcher overlay chain + patch_tf_bundle.py: --judge A.json B.json C.json [D.json])."""
    empties = {k: '' for k in KNOBS if k not in R16_ON}
    # ---- P2 vs P1: OFF == deploy-r15
    for lbl, r in (("P1", A), ("P2", B), ("P3", C), ("P4", D or {"import_errors": {}})):
        ck(not r["import_errors"], f"{lbl} worker imports: {r['import_errors']}")
    pa, pb = A["patched"], B["patched"]
    ck(pa == pb, f"P2 patched callables == deploy-r15's ({len(pb)} vs {len(pa)}; only in r16: "
                 f"{sorted(set(pb) - set(pa))}; only in r15: {sorted(set(pa) - set(pb))}; differing: "
                 f"{sorted(k for k in set(pa) & set(pb) if pa[k] != pb[k])})")
    new_ops = sorted(set(B["ops"]) - set(A["ops"]))
    print(f"INFO P2 custom ops registered at import by the new modules (inert until wired): {new_ops}")
    ck(set(A["ops"]) <= set(B["ops"]), f"P2 keeps every deploy-r15 custom op ({sorted(set(A['ops']) - set(B['ops']))})")
    ck(A["states"]["loader_chain"] == B["states"]["loader_chain"],
       f"P2 loader chain {B['states']['loader_chain']} == r15 {A['states']['loader_chain']}")
    s = B["states"]
    inert = (s["quickwins"] is None or (not s["quickwins"]["items"] and not s["quickwins"]["finder"])) and \
        (s["mla"] is None or not s["mla"]["installed"]) and not s["forward_mqa"]["mla_wrapper"] and \
        not s["forward_mqa"]["inner_is_quickwins"] and \
        (s["fp8_roof"] is None or not s["fp8_roof"]["installed"]) and not s["fp8_gemv"]["roof_hook"] and \
        not s["fp8_gemv"]["fc_table"] and (s["moeglue"] is None or not s["moeglue"]["hooked"]) and \
        (s["hostloop"] is None or (not s["hostloop"]["enabled"] and not s["hostloop"]["patched"] and
                                    not s["hostloop"]["kv_lens_host"] and not s["hostloop"]["finder"])) and \
        (s["smallops"] is None or not s["smallops"]["loader"]) and s["runtime"]["prof_diag"] in (False, None)
    ck(inert, f"P2 every R16 module inert: {json.dumps({k: s[k] for k in ('quickwins', 'mla', 'forward_mqa', 'fp8_roof', 'moeglue', 'hostloop', 'smallops', 'runtime')})}")
    ck(A["states"]["fp8_gemv"] == B["states"]["fp8_gemv"], f"P2 fp8_gemv state == r15 {B['states']['fp8_gemv']}")
    ck(A["states"]["tf"] == B["states"]["tf"] and A["states"]["bf16_gemv"] == B["states"]["bf16_gemv"],
       "P2 TF dispatcher / bf16 gemv loader == r15")
    ck(not B["warnings"], f"P2 no WARNING: {B['warnings']}")
    ck(A["ctx_after_plugins"] == B["ctx_after_plugins"],
       f"P2 CUDA context after plugin loading == r15 ({B['ctx_after_plugins']} vs {A['ctx_after_plugins']})")

    # ---- P3: everything installed and composed
    s = C["states"]
    q = s["quickwins"]
    ck(q["items"] == ["idx_gate", "kda_conv", "mhc_aux", "mhc_mean", "mla_bmm", "mla_index"] and not q["refused"],
       f"P3 quickwins items {q}")
    ck(s["forward_mqa"] == {"mla_wrapper": True, "outer_origin": "glm53_mla_prefill",
                            "inner_origin": "quickwins-recompiled:vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90.FlashInferMLASparseSM90Impl.forward_mqa",
                            "inner_is_quickwins": True},
       f"P3 forward_mqa = MLA wrapper over quickwins' recompiled forward_mqa: {s['forward_mqa']}")
    ck(s["mla"]["installed"] and s["mla"]["variant"] == 4 and s["mla"]["min_tokens"] == 256 and s["mla"]["mixed"],
       f"P3 MLA prefill {s['mla']}")
    r = s["fp8_roof"]
    ck(r["installed"] and r["table"] and r["pf"] and r["triggers"] == ["t0", "t1", "t2", "t3", "t4", "t5"] and
       r["ctas"] == 8 and r["max_m"] == 64 and r["pol"] == 0 and s["fp8_gemv"]["roof_hook"] and s["fp8_gemv"]["fc_table"]
       and s["fp8_gemv"]["enabled"] and s["fp8_gemv"]["large"] and s["fp8_gemv"]["max_m"] == 16, f"P3 fp8roof {r} {s['fp8_gemv']}")
    m = s["moeglue"]
    ck(m["hooked"] and m["warm"] and not m["glue"] and m["loader_hooked"] and m["cfg"] == ["frgd", 16.0, 16, 64, True]
       and m["warm_fps_ok"], f"P3 moeglue warm armed, fingerprints verified with quickwins on: {m}")
    h = s["hostloop"]
    ck(h["enabled"] and h["kv_lens_host"] and {"prepare_inputs", "postprocess_sampled"} <= set(h["patched"]) and
       h["fingerprints_ok"] and h["wake"][0] == "" and h["meter"] == 0 and h["verify"] == [64, 1024],
       f"P3 hostloop fast path {h}")
    so = s["smallops"]
    ck(so["loader"] and so["kinds"] == ["dconv"] and so["grouped_conv_fp_ok"], f"P3 smallops dconv {so}")
    ck(s["loader_chain"][:3] == ["glm53_smallops_install", "glm53_gemv_install", "glm53_moeglue"],
       f"P3 loader chain {s['loader_chain']}")
    ck(not C["warnings"], f"P3 no WARNING: {C['warnings']}")
    added = sorted(set(C["patched"]) - set(B["patched"]))
    changed = sorted(k for k in set(C["patched"]) & set(B["patched"]) if C["patched"][k] != B["patched"][k])
    print("INFO P3 callables patched by R16 (on top of R15):")
    for k in added + changed:
        print(f"   {k}: {C['patched'][k][0]} {C['patched'][k][2]}")
    ck(not sorted(set(B["patched"]) - set(C["patched"])), "P3 keeps every R15 patch")

    # ---- P4: empty knobs == defaults
    if D is not None:
        ck(D["states"] == C["states"] and D["patched"] == C["patched"] and D["ops"] == C["ops"] and
           not D["warnings"], f"P4 {len(empties)} empty knobs == unset: states differ in "
           f"{[k for k in C['states'] if C['states'][k] != D['states'].get(k)]}, warnings {D['warnings']}")
    print(f"INFO CUDA context after plugin loading: r15 {A['ctx_after_plugins']}, r16 off {B['ctx_after_plugins']}, "
          f"r16 on {C['ctx_after_plugins']} (production's R15 plugins already create one)")


def main() -> int:
    fails: list[str] = []

    def ck(ok, msg):
        print(("PASS " if ok else "FAIL ") + msg)
        if not ok:
            fails.append(msg)

    if len(sys.argv) > 1 and sys.argv[1] == "--judge":
        res = [json.loads(Path(f).read_text()) for f in sys.argv[2:]]
        A, B, C = res[:3]
        D = res[3] if len(res) > 3 else None
        for lbl, r in zip(("A", "B", "C", "D"), res):
            print(f"=== {lbl} {sys.argv[2 + 'ABCD'.index(lbl)]}: ctx_after_plugins={r['ctx_after_plugins']} "
                  f"patched={len(r['patched'])} ops={len(r['ops'])} warnings={len(r['warnings'])}")
    else:
        have15 = Path(R15_SITE, "integrate.py").is_file()
        A = child("P1 R15 env, deploy-r15 bundle", R15, R15_SITE) if have15 else {}
        ck(have15, f"deploy-r15 site mounted at {R15_SITE}")
        B = child("P2 R15 env, deploy-r16 bundle (R16 flags unset)", R15, None)
        C = child("P3 R15 env + R16 ship flags", {**R15, **R16_ON}, None)
        empties = {k: "" for k in KNOBS if k not in R16_ON}
        D = child(f"P4 P3 + {len(empties)} other R16 knobs passed empty", {**R15, **R16_ON, **empties}, None)
        if not (A and B and C and D):
            ck(False, "a child produced no result")
            print("FAILED")
            return 1
    judge(A, B, C, D, ck)
    print("ALL PASSED" if not fails else f"FAILED ({len(fails)})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
