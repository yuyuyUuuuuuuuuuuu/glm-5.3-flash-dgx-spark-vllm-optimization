"""CPU-only: kernels/dlmh_gemv.cu's dlmh_gemv_kernel == kernels/fp8_gemv.cu's fp8_gemv_kernel except the lines marked
[glm53-dlmh] (the weight address and the scale's source column) and the two signature lines; the device helpers
(ldw, ldx, bf2_mul, cvt4, mma_bf16, perm_col, MinBlocks) are byte-identical."""
import difflib
import re
import sys

a = open(sys.argv[1] if len(sys.argv) > 1 else "kernels/fp8_gemv.cu").read()
b = open(sys.argv[2] if len(sys.argv) > 2 else "kernels/dlmh_gemv.cu").read()


def part(s, start, end):
    i = s.index(start)
    return s[i:s.index(end, i)]


ha = part(a, "namespace fp8g {", "template <int WARPS, int MB, int U, int KW, bool EXACT>").replace("namespace fp8g {", "")
hb = part(b, "namespace dlmhg {", "template <int WARPS, int MB, int U, int KW, bool EXACT>").replace("namespace dlmhg {", "")
ok = ha == hb
print(f"device helpers identical: {ok}")
ka = part(a, "template <int WARPS, int MB, int U, int KW, bool EXACT>", "using KernFn").splitlines()
kb = part(b, "template <int WARPS, int MB, int U, int KW, bool EXACT>", "using KernFn").splitlines()
allowed_a = {"fp8_gemv_kernel(", "int ntiles, int ktiles, int pol_on) {", "const size_t kstride = (size_t)ntiles * 256;",
             "const uint32_t* wp = Wq + (size_t)first * kstride + (size_t)(nt < ntiles ? nt : 0) * 256 + lane * 8;",
             "const int p = perm_col(n);"}
bad = []
for op in difflib.SequenceMatcher(None, ka, kb, autojunk=False).get_opcodes():
    tag, i1, i2, j1, j2 = op
    if tag == "equal":
        continue
    for line in ka[i1:i2]:
        if not any(x in line for x in allowed_a):
            bad.append(("removed", line))
    for line in kb[j1:j2]:
        t = line.strip()
        if not (t == "" or "dlmh" in t or "oct" in t or "src_col" in t or "kstride" in t or t.startswith("//")
                or "perm_col(src_col)" in t):
            bad.append(("added", line))
for k, l in bad:
    print(f"  unexpected {k}: {l}")
print(f"kernel body differs only in the marked lines: {not bad}")
sys.exit(0 if ok and not bad else 1)
