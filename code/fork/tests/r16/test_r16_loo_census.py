"""deploy-r16 (review): plugin census (vLLM's real plugin loader, fresh interpreters) for every env-only revert state of ROLLOUT §5:
R15 env + env.r16 minus ONE feature (leave-one-out), and R15 env + ONE feature alone. Every other knob of R16_KNOBS is
passed EMPTY as the launcher does. For each state: every enabled plugin feature is installed and composed, every
disabled one inert, no WARNING, no import error. sidestream: fp8roof / moeglue warm count as installed only with their
breakable-CUDA-graph segment guard hooked (fp8roof: t1 / t4 not forked inside a segment); plus production's state of
2026-09-29 (smallops only) and the R16 update's target (smallops + fp8roof + moeglue warm)."""
import sys
sys.path.insert(0, "/w/tests/r16")
import test_r16_plugins as T

FEAT = {  # feature -> env.r16 names (plugin-level; kpoolring/apc are overlays/launcher, not plugin state)
    "quickwins": ["GLM53_PREFILL_QUICKWINS"], "mla": ["GLM53_MLA_PREFILL"], "fp8roof": ["GLM53_DEC_FP8ROOF"],
    "moeglue": ["GLM53_DEC_MOEGLUE_WARM"], "hostloop": ["GLM53_DEC_HOSTLOOP"],
    "smallops": ["GLM53_DEC_SMALLOPS", "GLM53_DEC_SMALLOPS_KINDS"],
}
ALL = {k: v for k, v in T.R16_ON.items() if k != "GLM53_KPOOL_RING"}

def on_state(r, f):
    s = r["states"]
    if f == "quickwins":
        q = s["quickwins"] or {}
        return bool(q.get("items")) and len(q.get("installed", [])) >= 6 and not q.get("refused")
    if f == "mla":
        return bool((s["mla"] or {}).get("installed")) and bool((s["forward_mqa"] or {}).get("mla_wrapper"))
    if f == "fp8roof":
        r = s["fp8_roof"] or {}
        return bool(r.get("installed")) and bool(r.get("pf")) and r.get("bk_guard") == "hooked" and r.get("bk_skip") == ["t1", "t4"]
    if f == "moeglue":
        m = s["moeglue"] or {}
        return bool(m.get("warm")) and bool(m.get("warm_fps_ok")) and not m.get("glue") and m.get("bk_guard") == "hooked"
    if f == "hostloop":
        h = s["hostloop"] or {}
        return bool(h.get("enabled")) and bool(h.get("kv_lens_host")) and bool(h.get("fingerprints_ok"))
    if f == "smallops":
        so = s["smallops"] or {}
        return bool(so.get("loader")) and list(so.get("kinds", [])) == ["dconv"]

fails = []
def run(label, on):
    env = dict(T.R15)
    for f in on:
        for n in FEAT[f]:
            env[n] = ALL[n]
    env.update({k: "" for k in T.KNOBS if k not in env})
    r = T.child(label, env, None)
    if not r:
        fails.append(label); return
    bad = []
    for f in FEAT:
        st = on_state(r, f)
        if (f in on) != bool(st):
            bad.append(f"{f} {'NOT installed' if f in on else 'installed while off'}")
    if r["warnings"]:
        bad.append(f"warnings {r['warnings'][:2]}")
    if r["import_errors"]:
        bad.append(f"import errors {r['import_errors']}")
    q = r["states"]["forward_mqa"] or {}
    print(f"    {label}: patched={len(r['patched'])} mqa={q} warm_fps={((r['states']['moeglue'] or {}).get('warm_fps'))}")
    print(("PASS " if not bad else "FAIL ") + label + (": " + "; ".join(bad) if bad else ""))
    if bad:
        fails.append(label)

feats = list(FEAT)
for f in feats:
    run(f"all but {f}", [g for g in feats if g != f])
for f in feats:
    run(f"only {f}", [f])
run("all six", feats)
run("none", [])
run("update target: smallops + fp8roof + moeglue", ["smallops", "fp8roof", "moeglue"])
run("fp8roof + moeglue", ["fp8roof", "moeglue"])
print("ALL PASSED" if not fails else f"FAILED {fails}")
sys.exit(1 if fails else 0)
