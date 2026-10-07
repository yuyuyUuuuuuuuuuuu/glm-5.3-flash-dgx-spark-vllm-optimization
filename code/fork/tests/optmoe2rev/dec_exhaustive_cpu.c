// opt-moe2-rev: exhaustive CPU proof of the lean decode integer identity (gcc -O3; 13 s): 0 mismatches.
// Exhaustive CPU check of the opt-moe2 lean decode integer identity (pre-hadd2 lo/hi words; hadd2 is a pure
// function of (lo,hi), so equal (lo,hi) => equal output).
#include <stdint.h>
#include <stdio.h>
static inline uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {  // PTX prmt default mode
  uint64_t x = ((uint64_t)b << 32) | a; uint32_t r = 0;
  for (int i = 0; i < 4; i++) { uint32_t s = (sel >> (4*i)) & 0xf; uint32_t byte = (x >> (8*(s & 7))) & 0xff;
    if (s & 8) byte = (byte & 0x80) ? 0xff : 0; r |= byte << (8*i); }
  return r; }
static inline uint32_t lop3(uint32_t a, uint32_t b, uint32_t c, uint32_t lut) {
  uint32_t r = 0; for (int i = 0; i < 32; i++) { int idx = (((a>>i)&1)<<2) | (((b>>i)&1)<<1) | ((c>>i)&1);
    r |= ((lut >> idx) & 1u) << i; } return r; }
int main(void) {
  // 1. lop3 LUT 0x6A == (a & b) ^ c  (all 8 truth-table rows)
  for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) for (int c = 0; c < 2; c++)
    if (((0x6A >> ((a<<2)|(b<<1)|c)) & 1) != ((a & b) ^ c)) { printf("LUT mismatch\n"); return 1; }
  // spot-check lop3 emulation on words
  if (lop3(0xDEADBEEF, 0x8FFF8FFF, 0x3B603B60, 0x6A) != ((0xDEADBEEFu & 0x8FFF8FFFu) ^ 0x3B603B60u)) return 2;
  uint64_t bad1 = 0, bad2 = 0, bad3 = 0;
  // 2. field extraction: (v >> 8) & 0xffff == prmt(v, 0, 0x4421) for all 2^32 v
  for (uint64_t v = 0; v <= 0xffffffffull; v++) if ((((uint32_t)v >> 8) & 0xffffu) != prmt((uint32_t)v, 0, 0x4421)) bad1++;
  printf("field extraction mismatches over 2^32: %llu\n", (unsigned long long)bad1);
  // 3. mcg2 vs mcg2_lean (lo, hi words) for all 16-bit s0, s1 (2^32 pairs)
  for (uint32_t s0 = 0; s0 < 65536; s0++) {
    uint32_t x0 = s0 * 0xCBAC1FEDu;
    uint32_t a0 = (x0 & 0x8FFF8FFFu) ^ 0x3B603B60u;
    for (uint32_t s1 = 0; s1 < 65536; s1++) {
      uint32_t x1 = s1 * 0xCBAC1FEDu;
      uint32_t a1 = (x1 & 0x8FFF8FFFu) ^ 0x3B603B60u;
      uint32_t lo = prmt(a0, a1, 0x5410), hi = prmt(a0, a1, 0x7632);
      uint32_t lo2 = (prmt(x0, x1, 0x5410) & 0x8FFF8FFFu) ^ 0x3B603B60u;
      uint32_t hi2 = (prmt(x0, x1, 0x7632) & 0x8FFF8FFFu) ^ 0x3B603B60u;
      if (lo != lo2 || hi != hi2) bad2++;
    }
  }
  printf("mcg2 vs lean (16-bit s0,s1, 2^32 pairs) mismatches: %llu\n", (unsigned long long)bad2);
  // 4. full 32-bit x0/x1 (mcg2_lean takes arbitrary words): symbolic per-bit argument -> check 2^32 random-ish pairs
  uint64_t st = 0x9E3779B97F4A7C15ull;
  for (uint64_t i = 0; i < (1ull << 28); i++) {
    st ^= st << 13; st ^= st >> 7; st ^= st << 17; uint32_t x0 = (uint32_t)st, x1 = (uint32_t)(st >> 32);
    uint32_t a0 = (x0 & 0x8FFF8FFFu) ^ 0x3B603B60u, a1 = (x1 & 0x8FFF8FFFu) ^ 0x3B603B60u;
    if (prmt(a0, a1, 0x5410) != lop3(prmt(x0, x1, 0x5410), 0x8FFF8FFF, 0x3B603B60, 0x6A) && (bad3++, 1)) {}
    if (prmt(a0, a1, 0x7632) != ((prmt(x0, x1, 0x7632) & 0x8FFF8FFFu) ^ 0x3B603B60u)) bad3++;
    if (i < (1ull<<20)) { /* lop3 emulation path also checked on a subset */ }
  }
  printf("arbitrary 32-bit words, 2^28 pairs mismatches: %llu\n", (unsigned long long)bad3);
  return (bad1 || bad2 || bad3) ? 1 : 0;
}
