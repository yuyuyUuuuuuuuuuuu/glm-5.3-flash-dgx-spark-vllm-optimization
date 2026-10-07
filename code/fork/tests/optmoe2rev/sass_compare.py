import re, subprocess, sys, collections
def funcs(so):
    out = subprocess.run(["cuobjdump", "-sass", so], capture_output=True, text=True).stdout
    fs = collections.OrderedDict(); cur = None
    for line in out.splitlines():
        m = re.match(r"\s*Function : (\S+)", line)
        if m: cur = m.group(1); fs[cur] = []; continue
        if cur is None: continue
        m = re.match(r"\s*/\*[0-9a-f]+\*/\s+(.*?);", line)
        if m: fs[cur].append(re.sub(r"\s+", " ", m.group(1)))
        m2 = re.match(r"\s*/\* (0x[0-9a-f]+) \*/", line)
        if m2: fs[cur].append("enc " + m2.group(1))
    return fs
def dem(n): return subprocess.run(["c++filt", n], capture_output=True, text=True).stdout.strip()
a, b = funcs(sys.argv[1]), funcs(sys.argv[2])
print("base functions", len(a), "new functions", len(b))
bd = {dem(k): k for k in b}
ok = bad = 0
for k, ins in a.items():
    d = dem(k)
    cand = d
    if "me_fused_kernel" in d:   # new build has one extra trailing template arg (MS = 0)
        cand = d.replace(">((anonymous namespace)::FusedArgs)", ", 0>((anonymous namespace)::FusedArgs)")
    if cand not in bd: print("MISSING in new:", d); bad += 1; continue
    nb = b[bd[cand]]
    # strip the encoding words that carry the (relocated) function-address-independent bits? compare both
    same_txt = [x for x in ins if not x.startswith("enc")] == [x for x in nb if not x.startswith("enc")]
    same_enc = ins == nb
    print(("IDENT " if same_txt and same_enc else ("TXTONLY " if same_txt else "DIFF ")), len([x for x in ins if not x.startswith('enc')]), d[:150])
    if same_txt: ok += 1
    else: bad += 1
print("identical", ok, "different/missing", bad)
extra = [dem(k) for k in b if dem(k) not in {(dem(x).replace(">((anonymous namespace)::FusedArgs)", ", 0>((anonymous namespace)::FusedArgs)") if "me_fused_kernel" in dem(x) else dem(x)) for x in a}]
print("new-only kernels:", len(extra)); [print("  ", e[:170]) for e in extra]
