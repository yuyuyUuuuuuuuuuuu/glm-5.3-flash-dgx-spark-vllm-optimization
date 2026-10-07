"""[hostgap review] Compare the handoff engine runs hl_notrace / hl_trace / hl_notrace2 (tests/handoff/run.sh, GLM53_DEC_HOSTLOOP=1 in every arm; the
harness wraps execute_model itself, so hostloop refuses there): token identity + logprob drift, and the trace content.
Usage: python3 tests/review_engine_trace_compare.py <handoff out dir>"""
import glob, json, os, statistics as S, sys
O = sys.argv[1]
arms = ["hl_notrace", "hl_trace", "hl_notrace2"]
R = {a: json.load(open(os.path.join(O, a, "result.json"))) for a in arms}
def reqs(r):
    return r.get("requests") or r.get("reqs") or [r]
for a in arms:
    print(a, "keys", sorted(R[a].keys())[:12])
def gens(r):
    rq = reqs(r)
    return [x["gen"] for x in rq]
def lps(r):
    return [x["logprobs"] for x in reqs(r)]
g = {a: gens(R[a]) for a in arms}
print("requests per arm:", {a: len(g[a]) for a in arms}, "tokens per request:", [len(x) for x in g[arms[0]]])
print("generated tokens identical notrace == trace:", g["hl_notrace"] == g["hl_trace"])
print("generated tokens identical notrace == notrace2:", g["hl_notrace"] == g["hl_notrace2"])
def lpdiff(a, b):
    mx, npos, ndiff = 0.0, 0, 0
    for ra, rb in zip(lps(R[a]), lps(R[b])):
        for pa, pb in zip(ra, rb):
            npos += 1
            shared = set(pa) & set(pb)
            d = max((abs(pa[k] - pb[k]) for k in shared), default=0.0)
            ndiff += d > 0
            mx = max(mx, d)
    return mx, ndiff, npos
for a, b in (("hl_notrace", "hl_trace"), ("hl_notrace", "hl_notrace2"), ("hl_trace", "hl_notrace2")):
    mx, nd, n = lpdiff(a, b)
    print(f"top-5 logprobs {a} vs {b}: {nd}/{n} positions differ, max diff on shared ids {mx:.3e}")
for a in arms:
    log = open(os.path.join(O, a, "container.log"), errors="replace").read()
    hl = [l for l in log.splitlines() if "fast path ON" in l or "NOT installed" in l or "switched OFF" in l or "dectrace" in l]
    print(f"{a}: hostloop/tracer lines:")
    for l in hl[:6]:
        print("   ", l[:220])
fs = sorted(glob.glob(os.path.join(O, "hl_trace", "dectrace", "*.ndjson")))
print("trace files:", [os.path.basename(f) + f" ({os.path.getsize(f)} B)" for f in fs])
for f in fs:
    steps, rank, stopped, hdr = {}, None, None, None
    for line in open(f):
        line = line.strip()
        if line.startswith("#!glm53-dectrace"):
            hdr = json.loads(line.split(" ", 1)[1])
        elif line.startswith("#! rank"):
            rank = line
        elif line.startswith("#! stopped"):
            stopped = line
        elif line and not line.startswith("#!"):
            p = line.split()
            steps.setdefault(int(p[0]), []).append((int(p[1]), p[2]))
    print("  header gpu:", hdr.get("gpu"), "| rank line:", rank, "| stop:", stopped)
    kinds = {}
    for n, ev in steps.items():
        key = " ".join(e for _, e in ev)
        kinds.setdefault(key, []).append(n)
    print(f"  {len(steps)} steps; event sequences:")
    for k, ns in sorted(kinds.items(), key=lambda x: x[1][0]):
        print(f"   x{len(ns):3d} (steps {ns[0]}..{ns[-1]}): {k}")
    def win(a, b):
        v = []
        for n, ev in steps.items():
            d = {e: t for t, e in reversed(ev)}
            if a in d and b in d and d[b] >= d[a]:
                v.append((d[b] - d[a]) / 1e3)
        return v
    for a, b in (("sched", "prep"), ("prep", "fg_b"), ("fg_b", "fg_e"), ("prep", "tgt_b"), ("tgt_b", "tgt_e"), ("sam_b", "sam_e"), ("dr_b", "dr_e"), ("sched", "exec_e")):
        v = win(a, b)
        if v:
            print(f"   {a}->{b}: n {len(v)} med {S.median(v):.1f} us  min {min(v):.1f}  max {max(v):.1f}")
