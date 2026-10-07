"""opt-w8a8layers in-container driver (run by tests/handoff/run.sh with HANDOFF_DRIVER=/w/tests/w8a8layers/dvp_driver.py):
per-LAYER (and per layer x projection) sensitivity of the dense W8A8 prefill path, measured with a TEACHER-FORCED
decode-vs-prefill comparison on IDENTICAL token trajectories (the production dvp probe's structure without its
free-running confound).

The production dvp probe (tools/prodcheck/kpool_decode_consistency.py) compares, at every generated position j,
  A_j: the logits the DECODE path produced (prompt prefilled by the prefill path, then one token per step), and
  B_j: the logits of a fresh PREFILL of prompt + the generated tokens.
W8A8 only serves prefill calls (M >= 512), so in A it only touches the prompt's state, while in B it touches every
scored position: dvp measures exactly "W8A8 prefill vs non-W8A8 decode at the same positions". On the 10-layer mini,
free-running greedy makes every arm generate its own text (the opt-decodekit-rev refuter: arms land on different
trajectories, the mean is dominated by a few trajectory-specific flips). Here every arm runs the SAME trajectories:
  * real-text prompts: prompt = a real-text window of P tokens, trajectory = the next G real tokens (teacher forcing)
  * --greedy prompts: trajectory = the W8A8-OFF arm's own greedy continuation, generated once, then forced for every arm
A is produced by the real decode path (one token per step, CUDA-graph replays) with the sampler's output overwritten
by the trajectory token (the logits are captured before that, fp32 log-softmax over the full vocabulary); B by one
prefill of prompt + trajectory[:G-1] (prompt_logprobs capture at positions P-1 .. P+G-2).

All arms run in ONE engine process (one boot with GLM53_DENSE_W8A8=1, every eligible layer self-tested at load):
an arm is a set of (layer, projection) pairs and the driver replaces fp8_w8a8.selected (the per-call ONLY filter,
consulted on every eager call) by membership in that set. The served (layer, projection) pairs of every arm are
counted through fp8_w8a8.try_w8a8 and checked against the arm (a mismatch is logged as SERVED-MISMATCH). The arm
"env" leaves the module's own selection (GLM53_DENSE_W8A8_ONLY / _SKIP_LAYERS) in place: the in-engine check of the
layer filter against the in-process emulation (same numbers expected bit for bit).

Per arm and position k = 1 .. G-1 (k = 0 is the prompt's last position, a prefill row on both sides):
  dvp = KL(A_arm || B_arm)        (production's dvp, teacher forced)
  pf  = KL(B_off || B_arm)        (the W8A8 prefill error at the scored positions, same tokens)
  da  = KL(A_off || A_arm)        (the W8A8 error carried into decode through the prompt's state only)
written per position to dvp.jsonl; the summary line per arm: means, p50/p95/max, top-1 agreement, events (> 0.05).
Arm grammar (--arms, ';'-separated "label=expr" or bare expr): expr = term (('+'|'^') term)*, '^' = set minus;
term = off | all | sub | L<a>[..<b>][:<sel>] | P:<sel>, sel = attn | mlp | fgb | <group>.<projection> | <group>.
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

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--label", default="")
ap.add_argument("--model", default=os.environ.get("KL_MODEL", os.path.join(os.environ.get("TF_EXL3_MODELS") or os.path.expanduser("~/models"), "GLM-5.3-Flash-handoff-moe-mini-tp2r0")))
ap.add_argument("--prompts", default="6000,6000,6000", help="real-text prompt lengths (teacher-forced on real text)")
ap.add_argument("--greedy", default="6000", help="prompt lengths whose trajectory is the off arm's greedy output")
ap.add_argument("--gen", type=int, default=256)
ap.add_argument("--arms", default="off;all;off")
ap.add_argument("--mnbt", type=int, default=13824)
ap.add_argument("--kv-bytes", type=int, default=1 << 30)
ap.add_argument("--gpu-util", type=float, default=0.125)
ap.add_argument("--event", type=float, default=0.05)
ARGS = ap.parse_args()
os.makedirs(ARGS.out, exist_ok=True)
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
for _v in ("VLLM_PREFIX_CACHE_RETENTION_INTERVAL", "VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA", "GLM53_APC_PRIOR_CHECKPOINT"):
    os.environ.pop(_v, None)   # no drafter / no APC here
import torch  # noqa: E402

LOG = open(os.path.join(ARGS.out, "dvp.log"), "a")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


_TEXT = {}


def text_ids():
    if "ids" not in _TEXT:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(os.environ.get("TF_EXL3_MODELS") or os.path.expanduser("~/models"), "GLM-OCR/tokenizer.json"))
        files = sorted(glob.glob("/usr/share/common-licenses/*")) + sorted(glob.glob("/usr/lib/python3.12/*.py"))
        t = "".join("\n\n### %s\n%s" % (os.path.basename(f), open(f, errors="ignore").read()) for f in files)
        _TEXT["ids"] = [int(x) for x in tok.encode(t).ids]
    return _TEXT["ids"]


def window(n, g, i):
    ids = text_ids()
    s = random.Random("w8a8layers-dvp-%d" % i).randrange(0, len(ids) - n - g - 1)
    return ids[s:s + n], ids[s + n:s + n + g]


# ---------------------------------------------------------------- deterministic sparse top-k (as tests/w8a82/kl_driver.py)
def install_topk():
    ns = torch.ops._C
    orig = ns.top_k_per_row_prefill

    def top_k_per_row_prefill(logits, ks, ke, out, num_rows, s0, s1, k):
        n = int(num_rows)
        if torch.cuda.is_current_stream_capturing():
            return orig(logits, ks, ke, out, num_rows, s0, s1, k)
        cols = logits.shape[1]
        ar = torch.arange(cols, device=logits.device)
        valid = (ar[None, :] >= ks[:n, None]) & (ar[None, :] < ke[:n, None])
        v = torch.where(valid, logits[:n].float(), torch.full((1, 1), float("-inf"), device=logits.device))
        kk = min(int(k), cols)
        o = torch.sort(-v, dim=1, stable=True).indices[:, :kk]
        sv = valid.gather(1, o)
        rel = o - ks[:n, None].to(torch.int64)
        out[:n, :kk] = torch.where(sv, rel, torch.full_like(rel, -1)).to(out.dtype)
        if kk < int(k):
            out[:n, kk:int(k)] = -1

    ns.top_k_per_row_prefill = top_k_per_row_prefill


# ---------------------------------------------------------------- B: logits capture at prompt positions [lo, hi]
CAP = {"range": None, "rows": {}}


def install_prompt_capture():
    import vllm.v1.worker.gpu.sample.prompt_logprob as PL
    orig = PL.PromptLogprobsWorker.compute_prompt_logprobs

    def compute_prompt_logprobs(self, logits_fn, hidden_states, input_batch, all_token_ids, num_computed_tokens,
                                prompt_lens):
        rng = CAP["range"]
        if rng is not None:
            qsl = input_batch.query_start_loc_np if hasattr(input_batch, "query_start_loc_np") else \
                input_batch.query_start_loc.cpu().numpy()
            start = int(input_batch.num_computed_prefill_tokens_np[0])
            n = int(qsl[1] - qsl[0])
            rows = [r for r in range(n) if rng[0] <= start + r <= rng[1]]
            if rows:
                with torch.no_grad():
                    lg = logits_fn(hidden_states[int(qsl[0]) + torch.tensor(rows, device=hidden_states.device)])
                    lp = torch.log_softmax(lg.float(), dim=-1)
                for r, row in zip(rows, lp):
                    CAP["rows"][start + r] = row
        return orig(self, logits_fn, hidden_states, input_batch, all_token_ids, num_computed_tokens, prompt_lens)

    PL.PromptLogprobsWorker.compute_prompt_logprobs = compute_prompt_logprobs


# ---------------------------------------------------------------- A: teacher-forced decode through the real sampler
FORCE = {"on": False, "toks": None, "rows": [], "gen": []}


def install_forcing():
    import vllm.v1.worker.gpu.sample.sampler as S
    orig = S.Sampler.__call__

    def __call__(self, logits, input_batch):
        out = orig(self, logits, input_batch)
        if FORCE["on"] and not torch.cuda.is_current_stream_capturing():
            ns = int(out.num_sampled.reshape(-1)[0]) if out.num_sampled is not None else 1
            if logits.shape[0] == 1 and ns == 1:
                FORCE["rows"].append(torch.log_softmax(logits[0].float(), dim=-1))
                st = out.sampled_token_ids
                k = len(FORCE["rows"]) - 1
                if FORCE["toks"] is None:
                    FORCE["gen"].append(int(st.reshape(-1)[0]))
                elif k < len(FORCE["toks"]):
                    st.reshape(-1)[0] = int(FORCE["toks"][k])
        return out

    S.Sampler.__call__ = __call__


# ---------------------------------------------------------------- W8A8 arm control
W8 = {"universe": {}, "cur": None, "served": {}, "orig_selected": None}
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def key_of(method):
    pre = getattr(method, "prefix", "") or ""
    m = _LAYER.search(pre)
    return (int(m.group(1)) if m else -1, f"{method.group}.{pre.rsplit('.', 1)[-1]}")


def install_arm_control(W):
    W8["orig_selected"] = W.selected
    orig_try = W.try_w8a8

    def selected(method):
        if W8["cur"] is None:
            return W8["orig_selected"](method)
        return key_of(method) in W8["cur"]

    def try_w8a8(x, layer, bias, n, k):
        y = orig_try(x, layer, bias, n, k)
        if y is not None:
            kk = W8["universe"].get(id(layer), ("?", "?"))
            W8["served"][kk] = W8["served"].get(kk, 0) + 1
        return y

    W.selected = selected
    W.try_w8a8 = try_w8a8


def build_universe(model, W):
    import vllm.model_executor.layers.quantization.exl3 as Q
    uni = {}
    for _name, mod in model.named_modules():
        qm = getattr(mod, "quant_method", None)
        if isinstance(qm, Q.Glm53DenseFp8Method) and getattr(qm, "ready", False) and qm.group in W.STATE.groups \
                and hasattr(mod, "glm53_fp8_n") and getattr(mod, "glm53_bf16_lm_w", None) is None:
            uni[id(mod)] = key_of(qm)
    W8["universe"] = uni
    return sorted(set(uni.values()))


ATTN = ("kda.", "mla.")
MLP = ("dense.", "shared.")
SUB = {"kda.in_proj_qkvbfg_a", "mla.o_proj"}


def sel_match(proj, sel):
    if sel is None:
        return True
    if sel == "attn":
        return proj.startswith(ATTN)
    if sel == "mlp":
        return proj.startswith(MLP)
    if sel == "fgb":
        return proj in ("kda.f_b_proj", "kda.g_b_proj")
    if "." in sel:
        return proj == sel
    return proj.startswith(sel + ".")


def parse_arm(expr, U):
    toks = re.findall(r"[+^]|[^+^]+", expr.replace(" ", ""))
    cur, op = set(), "+"
    for t in toks:
        if t in "+^":
            op = t
            continue
        if t == "off":
            s = set()
        elif t == "all":
            s = set(U)
        elif t == "sub":
            s = {u for u in U if u[1] in SUB}
        elif t.startswith("P:"):
            s = {u for u in U if sel_match(u[1], t[2:])}
        elif t.startswith("L"):
            m = re.fullmatch(r"L(\d+)(?:\.\.(\d+))?(?::(.+))?", t)
            if not m:
                raise SystemExit(f"bad arm term {t!r}")
            a = int(m.group(1))
            b = int(m.group(2)) if m.group(2) else a
            s = {u for u in U if a <= u[0] <= b and sel_match(u[1], m.group(3))}
        else:
            raise SystemExit(f"bad arm term {t!r}")
        cur = cur | s if op == "+" else cur - s
    return cur


# ---------------------------------------------------------------- metrics
def kl_rows(p_log, q_log):
    """KL(p || q) per row, fp32 log-probs [R, V] on the device."""
    return (p_log.exp() * (p_log - q_log)).sum(-1)


def summ(v):
    t = torch.tensor(v, dtype=torch.float64)
    return {"mean": t.mean().item(), "p50": t.quantile(0.5).item(), "p95": t.quantile(0.95).item(),
            "max": t.max().item(), "events": int((t > ARGS.event).sum().item())}


def main():
    from vllm import LLM, SamplingParams
    import vllm.v1.worker.gpu.model_runner as MR
    install_topk()
    install_prompt_capture()
    install_forcing()
    RUN = {}
    orig_exec = MR.GPUModelRunner.execute_model

    def _exec(self, *a, **k):
        RUN["runner"] = self
        return orig_exec(self, *a, **k)

    MR.GPUModelRunner.execute_model = _exec
    flags = {k: v for k, v in sorted(os.environ.items()) if k.startswith(("GLM53_", "TF_EXL3"))}
    log(f"== dvp run {ARGS.label!r}: {json.dumps(vars(ARGS))}")
    log(f"   flags {json.dumps(flags)}")
    t0 = time.time()
    llm = LLM(model=ARGS.model, skip_tokenizer_init=True, tensor_parallel_size=1, dtype="bfloat16",
              max_model_len=32768, max_num_seqs=4, max_num_batched_tokens=ARGS.mnbt, enable_prefix_caching=False,
              kv_cache_dtype="fp8", kv_cache_memory_bytes=ARGS.kv_bytes, gpu_memory_utilization=ARGS.gpu_util,
              enable_flashinfer_autotune=False, seed=0, max_logprobs=20,
              compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8]})
    log(f"engine up in {time.time() - t0:.0f}s")
    import fp8_w8a8 as W
    log(f"w8a8: enabled {W.STATE.enabled} reason {W.STATE.disabled_reason}; only {W.CFG.only}; "
        f"skip {getattr(W.CFG, 'skip', 'n/a')}; {W.summary()}")
    if not W.STATE.enabled:
        raise SystemExit("GLM53_DENSE_W8A8 is not installed: nothing to measure")
    U = build_universe(RUN["runner"].model, W)
    log(f"universe: {len(U)} (layer, projection) pairs: " + ", ".join(f"{l}:{p}" for l, p in U))
    install_arm_control(W)

    G = ARGS.gen
    traj = []                               # (name, prompt ids, trajectory or None = greedy of the off arm)
    for i, n in enumerate(int(x) for x in ARGS.prompts.split(",") if x.strip()):
        p, c = window(n, G, i)
        traj.append({"name": f"t{i}", "prompt": p, "toks": c})
    for j, n in enumerate(int(x) for x in ARGS.greedy.split(",") if x.strip()):
        p, _ = window(n, G, 100 + j)
        traj.append({"name": f"g{j}", "prompt": p, "toks": None})
    spA = SamplingParams(temperature=0.0, max_tokens=G, ignore_eos=True, detokenize=False, seed=0)
    spB = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True, prompt_logprobs=1, detokenize=False, seed=0)
    REF = {}                                 # name -> (A_off, B_off) device tensors
    arms = []
    for a in [x for x in ARGS.arms.split(";") if x.strip()]:
        lab, _, ex = a.partition("=") if "=" in a else (a, "", a)
        arms.append((lab.strip(), ex.strip()))
    if arms[0][1] != "off":
        raise SystemExit("the first arm must be off (the reference and the greedy trajectories)")
    fout = open(os.path.join(ARGS.out, "dvp.jsonl"), "a")
    seen = {}
    for lab, ex in arms:
        if ex == "env":
            W8["cur"], want = None, None
        else:
            W8["cur"] = want = parse_arm(ex, U)
        n_lab = seen[lab] = seen.get(lab, 0) + 1
        lab_r = lab if n_lab == 1 else f"{lab}#{n_lab}"
        W8["served"] = {}
        c0 = dict(W.COUNTERS)
        t1 = time.time()
        per = {"dvp": [], "pf": [], "da": [], "agreeAB": [], "agreeBoff": []}
        for ti, tr in enumerate(traj):
            P = len(tr["prompt"])
            FORCE.update(on=True, toks=tr["toks"], rows=[], gen=[])
            llm.generate([{"prompt_token_ids": tr["prompt"], "cache_salt": f"A-{lab_r}-{ti}"}], spA, use_tqdm=False)
            FORCE["on"] = False
            if tr["toks"] is None:
                tr["toks"] = FORCE["gen"][:G]
                log(f"trajectory {tr['name']}: greedy of the off arm, first tokens {tr['toks'][:12]}")
            A = torch.stack(FORCE["rows"][:G])
            if A.shape[0] != G:
                raise SystemExit(f"arm {lab_r} {tr['name']}: {A.shape[0]} decode rows, expected {G}")
            CAP.update(range=(P - 1, P + G - 2), rows={})
            llm.generate([{"prompt_token_ids": tr["prompt"] + tr["toks"][:G - 1], "cache_salt": f"B-{lab_r}-{ti}"}],
                         spB, use_tqdm=False)
            CAP["range"] = None
            B = torch.stack([CAP["rows"][P - 1 + k] for k in range(G)])
            if tr["name"] not in REF:
                REF[tr["name"]] = (A.clone(), B.clone())
            A0, B0 = REF[tr["name"]]
            sl = slice(1, G)
            dvp = kl_rows(A[sl], B[sl]).tolist()
            pf = kl_rows(B0[sl], B[sl]).tolist()
            da = kl_rows(A0[sl], A[sl]).tolist()
            agAB = (A[sl].argmax(-1) == B[sl].argmax(-1)).float().tolist()
            agB0 = (B0[sl].argmax(-1) == B[sl].argmax(-1)).float().tolist()
            import hashlib
            b_sha = hashlib.sha256(B.contiguous().cpu().numpy().tobytes()).hexdigest()[:16]
            fout.write(json.dumps({"arm": lab_r, "expr": ex, "traj": tr["name"], "dvp": dvp, "pf": pf, "da": da,
                                   "agreeAB": agAB, "agreeBoff": agB0, "b_sha": b_sha,
                                   "toks_sha": hashlib.sha256(str(tr["toks"]).encode()).hexdigest()[:16]}) + "\n")
            for kname, v in (("dvp", dvp), ("pf", pf), ("da", da), ("agreeAB", agAB), ("agreeBoff", agB0)):
                per[kname].extend(v)
            del A, B
        fout.flush()
        served = dict(W8["served"])
        mism = ""
        if want is not None and set(served) != want:
            mism = f" SERVED-MISMATCH missing {sorted(want - set(served))} extra {sorted(set(served) - want)}"
        dc = {k: W.COUNTERS.get(k, 0) - c0.get(k, 0) for k in W.COUNTERS if W.COUNTERS.get(k, 0) != c0.get(k, 0)}
        res = {"arm": lab_r, "expr": ex, "pairs": len(served), "dvp": summ(per["dvp"]), "pf": summ(per["pf"]),
               "da": summ(per["da"]), "top1_AB": sum(per["agreeAB"]) / len(per["agreeAB"]),
               "top1_Boff": sum(per["agreeBoff"]) / len(per["agreeBoff"]), "positions": len(per["dvp"]),
               "secs": round(time.time() - t1, 1), "served": sorted(f"{l}:{p}" for l, p in served),
               "counters": dc}
        log(f"ARM {lab_r:24s} pairs {len(served):3d} dvp {res['dvp']['mean']:.6f} (ev {res['dvp']['events']:3d}) "
            f"pf {res['pf']['mean']:.6f} da {res['da']['mean']:.6f} top1AB {res['top1_AB']:.4f} "
            f"[{res['secs']}s]{mism}")
        with open(os.path.join(ARGS.out, "arms.jsonl"), "a") as f:
            f.write(json.dumps(res) + "\n")
    log(f"w8a8 counters at the end: {W.summary()}")
    log("DVP-DONE")


if __name__ == "__main__":
    main()
