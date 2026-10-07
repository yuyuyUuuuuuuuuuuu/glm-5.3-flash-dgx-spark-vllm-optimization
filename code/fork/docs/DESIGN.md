# tf-exl3-fork — implementation spec: correct, reachable drop-in for `exllamav3_ext.exl3_moe`

Status: SPEC (nothing implemented yet). Author role: design lead. Date: 2026-09-27.
Paths are relative to `${HOME}/tf-exl3-fork` unless prefixed `image:` (= inside
`ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor`). Evidence format:
`path:line — "verbatim quote"`. `DERIVED` = arithmetic/logic from cited inputs, not executed.
`UNKNOWN` = no evidence. Fact ids in brackets refer to the lens facts (e.g. [prod:CALL-05]).

--------------------------------------------------------------------------------------------
## 0. Ground truth this spec is built on

### 0.1 The only decode call site (the "user surface" of this component)
- Decode chain: `Exl3MoEMethod.apply` → `apply_exl3_experts` → `apply_exl3_fused_moe` →
  (`tokens <= cap`) → `_exl3_moe_launch(fn, ...)` → `fn(*args)`. [prod:FLAG-01, prod:PATH-01]
  - docs/prod_exl3_reference.py:2340 — "limit = getattr(self.moe, \"swiglu_limit\", None) or SWIGLU_LIMIT_DEFAULT"
  - docs/prod_exl3_reference.py:1636 — "fn = exllamav3_ext.exl3_moe"  (module attribute read on EVERY call → patchable)
  - docs/prod_exl3_reference.py:1641 — "cap = int(temps[0].shape[1])"
  - docs/prod_exl3_reference.py:1648 — "if tokens <= cap:"
  - docs/prod_exl3_reference.py:1814 — "use_fused = hasattr(exllamav3_ext, \"exl3_moe\")"
- Routing tensors the drop-in receives (built by production, not by us): [prod:CALL-04/05/06]
  - :1622 — "flat_token = torch.arange(tokens, device=x2d.device, dtype=torch.long).repeat_interleave(topk)"
  - :1624-1626 — "order = local.argsort()" / "token_sorted = flat_token[order]" / "weight_sorted = flat_weight[order]"
  - :1628 — "expert_count = torch.zeros(n_exp + 1, dtype=torch.long, device=local.device)"
  - :1632 — "out = torch.zeros(tokens, hidden, dtype=torch.float32, device=x2d.device)"
  - :1633 — "xh = x2d.contiguous().half()"
  - Consequence: `token_sorted[j]` is a TOKEN index (row of xh/out); the per-token slot is lost
    (only `flat_token[order]` is passed, not `order`). Sentinel id `n_exp` (invalid / non-local)
    sorts last and is counted in `expert_count[n_exp]`, which the kernel ignores.
    docs/ref/xl_exl3_moe.cu:146 — "size_t num_experts = expert_count.size(0) - 1;"
- Positional call shape (29 args, +1 only if the detector says so): [prod:CALL-01, prod:CALL-07 corrected]
  - docs/prod_exl3_reference.py:1503-1533 (args tuple) and :1534-1537 — "if n_active_host is not None:" / "fn(*args, n_active_host)" / "fn(*args)"
  - Index map (0-based): 0 xh fp16[B,K]; 1 out fp32[B,K]; 2 expert_count i64[n+1]; 3 token_sorted i64[P];
    4 weight_sorted fp16[P]; 5..8 temps (C,R,K),(C,R,K),(C,R,N),(C,R,N) fp16; 9 act_function (=0);
    10..12 K_gate,K_up,K_down (=bits); 13..21 ptr tables i64[n] (gate trellis,suh,svh, up …, down …);
    22..27 mcg/mul1 bools (T,F,T,F,T,F); 28 act_limit float; [29] num_active (only if detector true).
  - docs/prod_exl3_reference.py:58 — "MOE_ACT_SILU = 0"; docs/ref/xl_exl3_moe_common.cuh:6 — "#define MOE_ACT_SILU 0"
- The image's exl3_moe has no num_active: 29-param C++ signature, plain `m.def`, 0 occurrences of
  "num_active" in the .so → detector almost certainly false → 29 args. (inferred, not executed) [prod:CALL-07 corrected, xl:MOE-16]

### 0.2 Exact math of exl3_moe (the numerical reference) [xl:MOE-05,07,08,10,12,13,14; prod:MATH-01 corrected]
Per eligible expert e (0 < count_e <= R, R = temp_state_g.size(1)), per pair j in its span, t = token_sorted[j]:
1. xg = f16(H(f16(x[t] ⊙ suh_g[e])) · r), xu likewise with suh_u;  r = 0.088388347648f; H = 128-block Sylvester, fp32 inside.
   - docs/ref/xl_hadamard_inner.cuh:110 — "v.x = __hmul2(v.x, scales.x);"  (fp16 pre-scale)
   - docs/ref/xl_hadamard_inner.cuh:130 — "v.x = __floats2half2_rn(h0 * r_scale, h1 * r_scale);"
2. g0 = f16(xg · Wq_g[e]), u0 = f16(xu · Wq_u[e]) — mma f16×f16→f32; cross-CTA split-K partials pass through fp16.
   - docs/ref/xl_exl3_gemm_inner.cuh:483 — "float2 interm = __half22float2(*c_ptr);" ; :527 — "half2 sum = __floats2half2_rn(frag_c[n][0], frag_c[n][1]);"
3. g = f16(H(g0)·r) ⊙16 svh_g[e]; u = f16(H(u0)·r) ⊙16 svh_u[e]; g = silu_f16(g) (h2exp/h2rcp);
   if L != 0: u = min(max(u,-L),L), g = min(g,L)  [clamp AFTER SiLU, gate upper only]; a = (g ⊙16 u) ⊙16 suh_d[e];
   xd = f16(H(a)·r).
   - docs/ref/xl_hadamard_inner.cuh:352 — "vg.x = __hmul2(vg.x, scales_g.x);"
   - :320-328 — "half2 e = h2exp(neg_x);" … "half2 r = h2rcp(sum);" … "half2 result = __hmul2(x, r);"
   - :375-382 — "if (act_limit != 0.0f)" … "vu.x = __hmax2(vu.x, __float2half2_rn(-act_limit));" … "vg.x = __hmin2(vg.x, __float2half2_rn(act_limit));"
   - :386 — "vg.x = __hmul2(vg.x, vu.x);" ; :391 — "vg.x = __hmul2(vg.x, scales_d.x);" ; :395 — "vg = had(vg);"
4. d0 = f16(xd · Wq_d[e]).  docs/ref/xl_exl3_moe_kernel.cuh:219 — "gemm_down(temp_intermediate_g, temp_state_g, exp_down_trellis, K_down);"
5. out[t] += (H(d0) · (r·float(w_j))) ⊙ float(svh_d[e])  in fp32, atomicAdd.
   - docs/ref/xl_exl3_moe_kernel.cuh:240 — "0.088388347648f * __half2float(weight)"
   - docs/ref/xl_hadamard_inner.cuh:433 — "h0 *= r_scale;" ; :441 — "h0 *= __low2float(scales.x);" ; :455 — "atomicAdd(output_ptr +  0 + t, sh[ 0 + t]);"
- Skip rule: docs/ref/xl_exl3_moe_kernel.cuh:55-56 — "if (token_count == 0) continue;" / "if (token_count > max_tokens_per_expert) continue;"  [xl:MOE-03]
- Spans are a running prefix over ALL buckets 0..n-1 (skipped experts still consume their span):
  docs/ref/xl_exl3_moe_kernel.cuh:50-51 — "start = end;" / "end += expert_count[expert_idx];"  [xl:MOE-02]

### 0.3 Weights and pointer tables [prod:LOAD-01..04, layout:PROD-01/02/04]
- docs/prod_exl3_reference.py:2078 — "num_experts, 2, in_tiles, out_tiles, k_words, dtype=torch.int16" (w13_trellis [E,2,K/16,N/16,64])
- :2083 — "torch.empty(num_experts, 2, hidden_size, dtype=torch.float16)" (w13_suh); w2_trellis :2098 — "num_experts, out_tiles, in_tiles, k_words, dtype=torch.int16"
- :1447-1452 — "[int(getattr(pack[which], attr).data_ptr()) for pack in inners]," → `layer._exl3_ptrs[...]` int64 [n] on device,
  pointing into the stacked Parameters (LinearEXL3 stores the given slices: docs/ref/xl_modules_exl3.py:56 — "self.trellis = trellis").
- Tile bits/ordering/codebook/Hadamard convention of TF == ExLlamaV3 (no bit transform needed): [layout:LAYOUT-01..03, CB-01, HAD-01..03, CONCL-01 corrected]
  kernels/exl3.cu:92 — "const uint32_t* tile = T + (((size_t)e * KT + kt0) * NTILES + nt0) * 32 + lane;" — per-expert block is exactly
  one contiguous int16 [K/16,N/16,64] matrix viewed as uint32 [K/16,N/16,32].

### 0.4 Output contract [prod:OUT-01, OUT-02, CALL-03]
- The call must ADD this rank's partial routed sum (weights already include routed_scaling_factor) into the
  zero-initialized fp32 `out`; no shared expert, no all-reduce. docs/prod_exl3_reference.py:2339 — "del shared_experts, shared_experts_input";
  :1823 — "return out.to(dtype=x.dtype)".

### 0.5 Why pointer tables and not stacked tensors (memory) — DERIVED
- Inputs (NVFP4 sibling config; EXL3 checkpoint config UNKNOWN): ${HOME}/models/GLM-5.3-Flash-Uncensored-NVFP4/config.json:62 — "\"hidden_size\": 4096,";
  :274 — "\"moe_intermediate_size\": 2048,"; :277 — "\"n_routed_experts\": 288,"; :56 — "\"first_k_dense_replace\": 3,"; :282 — "\"num_hidden_layers\": 45,";
  mlp_layer_types (:226-272) = 3 "dense" + 42 "sparse". bits=4 by checkpoint name only: docs/prod_exl3_reference.py:4 — "Checkpoint ABI (brandonmusic/GLM-5.3-Flash-tr3-4bpw):" [prod:K-02].
- Per rank (TP=2, intermediate_local=1024): one matrix 4096×1024×4 bit = 2 MiB; expert (3 matrices) 6 MiB; ×288 = 1728 MiB/layer;
  ×42 = 72,576 MiB ≈ 70.9 GiB/rank of routed trellis. gate+up = 2/3 ≈ 47.3 GiB/rank.
- TF kernels require contiguous [E,…] per matrix kind (kernels/exl3.cpp:15 — "TORCH_CHECK(x.is_cuda() && x.scalar_type() == t && x.is_contiguous(), name,"),
  but production gate/up are interleaved [E,2,…] → `w13_trellis[:,0]` is non-contiguous → a de-interleave copy would add ≈47 GiB/rank
  (≈95 GiB both ranks). Not acceptable. Decision: kernels read per-expert base addresses from the production pointer tables
  (args 13..21), zero copy. Precedent in the same image: Mia's fat kernel does exactly this:
  docs/ref/mia_exl3-fat-kernel/exl3_fat_moe.cu:321 — "const uint16_t* const packed[NS] = { gate_ptrs[e], up_ptrs[e] };"

--------------------------------------------------------------------------------------------
## A. Kernel changes (kernels/exl3.cu, kernels/exl3.cpp) — our fork, MIT

Legend: REQ-C = required for correctness; REQ-S = required for safety (no OOB / no crash); REQ-G = required for
CUDA-graph/host-sync compliance; PAR = precision parity with exl3_moe (default ON, switchable by compile flag
`TF_PARITY`, default 1); OPT = optional (performance).

### A1. Per-expert weights via pointer tables — REQ-C (fixes AUDIT-04, AUDIT-18, layout:PROD-01/04)
- Replace every `[E,...]`-tensor access by a pointer-table lookup. Pointer tables are int64 [n] CUDA tensors
  (production args 13..21), reinterpreted as `const T* const*`.
  - grouped: `const uint32_t* Tm = reinterpret_cast<const uint32_t*>(tptr_mat[e]);`
    `tile = Tm + ((size_t)kt0 * NTILES + nt0) * 32 + lane;` (was kernels/exl3.cu:92 — "T + (((size_t)e * KT + kt0) * NTILES + nt0) * 32 + lane")
  - rot_in: `const half* suh = suh_ptr_mat[e] + blk*128;` (was :169 — "(mat ? suh1 : suh0) + (size_t)e * K + blk * 128 + 4 * lane")
  - gateup_epilogue: `svh_g_ptr[e] + n`, `svh_u_ptr[e] + n`, `suh_d_ptr[e] + n` (was :211-214 `svh_g[(size_t)e * N + n + j]` …)
  - down_epilogue: `svh_d_ptr[e] + n` (was :243 — "svh_d[(size_t)e * D + n + j]").
- Validity: the int16 trellis [K/16,N/16,64] and uint32 [K/16,N/16,32] are the same bytes [layout:CONCL-01]; `from_int16_trellis`
  and `Exl3ExpertWeights` are deleted (tf_exl3_moe.py:131-132 claims "view のみ(コピーしない)" which is false for w13[:,0]).
- C++ entry checks (keep as defense; the Python plan duplicates them): every pointer table `is_cuda && is_contiguous && dtype==kLong &&
  dim()==1 && size(0)>=n` (same shape as docs/ref/mia_exl3-fat-kernel/exl3_fat_moe.cu:472 — "TORCH_CHECK(t.scalar_type() == at::kLong && t.dim() == 1 && t.size(0) >= n,").

### A2. fp16 activation input to rot_in — REQ-C (fixes AUDIT-02 / xl:FORK-03 / layout:FORK-03)
- kernels/exl3.cpp:38 — "TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16, \"x: bf16 CUDA\");" → change to `at::kHalf`.
- Kernel reads `const half* x` with element row stride `x_stride` (int64, not int: exl3.cu:282 casts to int).
- Rationale: production passes fp16 (docs/prod_exl3_reference.py:1633 — "xh = x2d.contiguous().half()"); casting to bf16 would lose 3
  mantissa bits [audit:AUDIT-02 corrected].

### A3. Remove the shared-expert slot skip — REQ-C (fixes AUDIT-03 / FORK-02)
- Delete kernels/exl3.cu:166, :191, :229 — "if (slot == slots - 1) return;". The `slots` concept disappears entirely (see A4):
  production has no shared-expert slot (docs/prod_exl3_reference.py:2339 — "del shared_experts, shared_experts_input").

### A4. Row indexing = sorted pair index j; members table replaced by segment table — REQ-C (fixes AUDIT-01, AUDIT-14, xl:FORK-01 extra finding)
- Every per-pair buffer row (xg, xu, Z, xd) is indexed by `j` = position in `token_sorted` (0 ≤ j < P, P = token_sorted.numel()).
  This removes all three inconsistent encodings: kernel `(code >> 5) * slots + (code & 31)` (kernels/exl3.cu:81), fork `row*slots+slot`
  (tf_exl3_moe.py:391 — "pick[code[inb]] = expert_of_row[valid][inb].to(torch.int32)"), production token index.
- grouped program = (segment s, n block, mat*SK+split). A segment is ≤16 consecutive pair rows of ONE expert:
  `seg_expert[s], seg_row0[s], seg_rows[s]` (built by route_prep, §B). No `members`, no `uids`, no `MT`, no `maxm`.
  New kernel prologue (replaces kernels/exl3.cu:64-84):
  ```
  const int s = blockIdx.x; if (s >= nseg[0]) return;
  const int split = blockIdx.z % SK, mat = blockIdx.z / SK;
  const int e = seg_expert[s], r0 = seg_row0[s], rn = seg_rows[s];      // rn in [1,16]
  if (threadIdx.x < 16) rows_sh[threadIdx.x] = threadIdx.x < rn ? r0 + threadIdx.x : -1;
  __syncthreads();
  ```
  Everything after (loads `x0 = X + r0_*K`, mma loop, warp reduction in `red[W][16][NT*16]`, Z store at :139) is unchanged, with
  `Z[(((size_t)mat * SK + split) * P + r) * N + nt0 * 16 + col]` and r = pair row j.
- Grid: `dim3(S_cap, N/(16*nt), mats*SK)` (was :256 — "dim3 grid((unsigned)uids.size(0), …, (unsigned)(mats * SK * MT))").
- Divisibility unchanged: kernels/exl3.cu:255 — "TORCH_CHECK(K % (16 * SK * warps) == 0 && N % (16 * nt) == 0, \"K and N must split evenly\");"

### A5. Routing weight + accumulate into fp32 out[token] inside down_epilogue — REQ-C (fixes AUDIT-09, STATUS item 4)
- New down_epilogue program (j, 128-block of D), 32 threads:
  ```
  e = pair_expert[j]; if (e < 0) return;
  t = token_sorted[j]; if (t < 0 || t >= B) return;          // REQ-S guard
  w = weight_sorted[j] (half);
  s[i] = sum_{k<SK} Z[((size_t)k*P + j)*D + n + i]              // fixed order
  PAR:  s[i] = __half2float(__float2half_rn(s[i]));            // exl3_moe stores d0 as fp16 (0.2 step 4)
  fwht128(s, lane);
  const float rs = 0.088388347648f * __half2float(w);         // PAR: same fp32 product as xl_exl3_moe_kernel.cuh:240
  s[i] *= rs; s[i] *= __half2float(svh_d_ptr[e][n + i]);       // PAR: two separate fp32 multiplies, xl_hadamard_inner.cuh:433,441
  atomicAdd(out + (size_t)t*D + n + i, s[i]);                  // accumulate, never overwrite
  ```
- `y` buffer and `_combine` (tf_exl3_moe.py:395-408) are deleted. Accumulation order across a token's experts is unordered fp32,
  exactly the class of exl3_moe (atomics, non-deterministic when concurrency>1) [xl:MOE-15].
- OPT A5b: coalesced atomics via a warp shuffle / shared reshuffle (as xl_hadamard_inner.cuh:446-458) or 16-byte vector atomics (sm_90+).
- OPT A5c (deterministic mode, flag): write per-pair fp32 rows to a y[P,D] buffer and add them per token in ascending j order in a
  separate kernel. Not needed for parity (exl3_moe itself is non-deterministic).

### A6. Sentinel / non-local / over-cap pairs — REQ-C + REQ-S (fixes AUDIT-07, AUDIT-08, AUDIT-10)
- `pair_expert[j] = -1` for (a) j ≥ n_valid (sentinel bucket rows), (b) rows of experts with count > R (exl3_moe skip rule),
  (c) rows beyond P. rot_in / gateup_epilogue / down_epilogue return immediately on `e < 0`; route_prep never emits a segment for them.
  This replaces the fork's `pick = torch.full(..., E)` default (tf_exl3_moe.py:387) which produced out-of-bounds suh/svh reads
  [audit:AUDIT-08 confirmed].
- Token guard `0 <= t < B` in rot_in and down_epilogue (REQ-S; costs one compare).

### A7. rot_in (new) — REQ-C with PAR details
```
grid (P, K/128, 2), block 32
j = blockIdx.x; blk = blockIdx.y; mat = blockIdx.z; lane = threadIdx.x
e = pair_expert[j]; if (e < 0) return; t = token_sorted[j]; if (t < 0 || t >= B) return;
half4 xv = *(const half4*)(x + t*x_stride + blk*128 + 4*lane);
half4 sv = *(const half4*)(suh_ptr_mat[e] + blk*128 + 4*lane);
PAR:    xv = __hmul2 per half2 (xv, sv); v[i] = __half2float(xv[i])          // == xl_hadamard_inner.cuh:106-111,115-118
NATIVE: v[i] = __half2float(x[i]) * __half2float(suh[i])                     // TF original fp32 prescale (exl3.cu:173)
fwht128(v, lane);  out_mat[j*K + blk*128 + 4*lane + i] = __float2half_rn(v[i] * HAD_SCALE);
```
- Expected bit-identical to exl3_moe stage 1 and to `exllamav3_ext.had_r_128(x_row, out, suh_e, None, 1.0)`:
  fwht128 == shuffle_had_f4x32 bitwise [layout:HAD-02 confirmed]; HAD_SCALE float bits == 0.088388347648f [layout:HAD-01];
  docs/ref/xl_hadamard.cu:107 — "float r_scale = scale * 0.088388347648f; // scale / sqrt(128)"; had_r_128 uses the same inner
  (docs/ref/xl_hadamard.cu:21 — "had_hf_r_128_inner<pre_scale, post_scale>(input_ptr, output_ptr, scale, r_scale);"). Tested in E-U1.

### A8. gateup_epilogue (new) — precision items individually classified
```
grid (P, N/128), block 32; j, blk, lane; e = pair_expert[j]; if (e < 0) return; n = blk*128 + 4*lane
sg[i] = sum_{s<SK} Z[((0*SK+s)*P + j)*N + n+i];  su[i] = sum_{s<SK} Z[((1*SK+s)*P + j)*N + n+i]   // fixed order (unchanged, :199-201)
PAR-e: sg[i] = __half2float(__float2half_rn(sg[i])); su likewise        // exl3_moe keeps g0/u0 in fp16
fwht128(sg); fwht128(su);
half4 g = __floats2half2_rn(sg*r ...), u = ... ;                         // r = HAD_SCALE; PAR-e also
PAR-f: g = __hmul2(g, svh_g[e][n..]); u = __hmul2(u, svh_u[e][n..])      // fp16 post-scale (xl :352-355)
PAR-d: g = silu_h2(g)  where silu_h2(x) = __hmul2(x, h2rcp(__hadd2(one, h2exp(__hneg2(x)))))   (xl :320-328)
REQ-c: if (limit != 0.f) { Lh = __float2half2_rn(limit); u = __hmax2(u, __float2half2_rn(-limit)); u = __hmin2(u, Lh); g = __hmin2(g, Lh); }
PAR-f: a = __hmul2(g, u); a = __hmul2(a, suh_d[e][n..])
fwht128(float(a)); xd[j*N + n + i] = __float2half_rn(a[i] * r)
```
Classification (vs current kernels/exl3.cu:211-214 — "float gg = fminf(bf16r(…), limit);" … "float act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);"):
- A8-b REQ-C: remove all `bf16r` roundings (bf16 = 8-bit mantissa, TF's own GLM recipe; production decode never rounds to bf16).
  With bf16r the TF-vs-exl3_moe noise is ~2^-8 relative per stage instead of ~2^-11 and the E1 tolerance could not separate bugs from noise.
- A8-c REQ-C (semantic parity): clamp AFTER SiLU, gate upper-only, `limit != 0` guard — exl3_moe decode semantics
  [xl:MOE-11 confirmed]. Note (DERIVED): with L=10 in fp16 the two orders coincide almost everywhere (silu_f16(g)=g for g ≳ 7.6
  because 1+e^-g rounds to 1 in fp16; fp16(silu_f32(10)) = 10.0), but the definition must match the reference, and L=1 tests differ.
- A8-d PAR (default ON): fp16 SiLU via h2exp/h2rcp instead of fp32 expf/division. Bitwise identity with the XL binary also needs the
  same CUDA fp16 intrinsic codegen — UNKNOWN (see G8).
- A8-e PAR (default ON): round split-summed GEMM outputs to fp16 before the Hadamard, and round H(·)·r to fp16 (XL stores fp16 temps).
  Also preserves fp16 overflow semantics (inf at |v|>65504) identical to XL.
- A8-f PAR (default ON): svh / suh_d multiplies as __hmul2 (fp16) instead of fp32.
- NOT replicated (by design): XL's extra fp16 rounding of a cross-CTA split-K partial (gate/up only; see E1 tolerance) and XL's fp32
  summation order inside mma/sub_k/CTAs. TF sums its SK=4 splits in fp32 — strictly more accurate.
- NATIVE mode (`TF_PARITY=0`): fp32 epilogue without bf16r and without the fp16 roundings of A8-d/e/f, clamp still post-SiLU. For
  experiments only; production default is PAR.

### A9. route_prep kernel (new) — REQ-G (fixes AUDIT-12, AUDIT-13)
Single block (1024 threads), one launch, no host sync, static grid; exact semantics in §B. Replaces
`build_member_table(_device)`, `_build_pick` (tf_exl3_moe.py:161-258, 376-392), which contain `torch.nonzero`, boolean-mask
indexing, `int(offsets[-1])`, `.item()` loops [audit:AUDIT-12 confirmed]. Requires n ≤ 1024 (n=288 for GLM per sibling config) —
checked at pre-flight; for n > 1024 the kernel loops (block-strided scan) — implement the loop, test n=288 and n=1100.

### A10. Z write-before-read guarantee — REQ-C (fixes AUDIT-22)
- For every j with pair_expert[j] ≥ 0 there is exactly one segment containing j; that segment's grouped blocks write
  Z[mat][split][j][0..N) for all mats, splits and all N/(16nt) column blocks (grid.y covers N, grid.z covers mats*SK). Epilogues read
  only rows with pair_expert[j] ≥ 0 → never stale, regardless of scratch reuse across layers/calls.
- Z capacity check stays: kernels/exl3.cpp:31 — "TORCH_CHECK(Z.numel() >= mats * SK * P * N, \"Z too small\");"

### A11. Divisibility contract (unchanged kernels, documented) [tf:GRP-06]
- gate/up grouped (GATEUP_CFG nt=8,W=4,SK=4): hidden % 256 == 0, intermediate % 128 == 0.
- down grouped (DOWN_CFG nt=8,W=4,SK=1): intermediate % 64 == 0, hidden % 128 == 0.
- rot_in hidden % 128; gateup_epilogue intermediate % 128; down_epilogue hidden % 128 (kernels/exl3.cpp:44, :58, :69).
- Combined gate: `hidden % 256 == 0 and intermediate_local % 128 == 0`. GLM (sibling dims) 4096 / 1024 pass (DERIVED).
- Bits: decode_tile is hard-wired 4-bit (kernels/exl3.cu:34 — "This lane's eight values of a 4-bit tile"); gate K_gate==K_up==K_down==4.

### A12. Build — REQ-S
- Ship an AOT-built extension (setup.py/pyproject, `TORCH_CUDA_ARCH_LIST=12.1a` or equivalent producing sm_121a) inside the image used
  on BOTH ranks; runtime JIT (tf_exl3_moe.py:80-93 `load(...)` on first call) is forbidden in production (ninja/nvcc/header-shim
  failure modes, minutes of latency) [audit:AUDIT-15 corrected]. JIT stays allowed for nodeC tests.
- Flags: `-O3`; `--use_fast_math` not required for PAR (no expf/div left in PAR; FWHT has no mul-add pairs to contract). The XL .so
  was built with ftz/approx semantics (xl:SRC-01 corrected: "mul.ftz.f32 8828 vs mul.f32 0"); ftz only affects fp32 subnormals,
  unreachable from fp16 data except exact zeros (DERIVED).

### A13. OPT performance items (do only after E passes)
- OPT-1 fuse down_epilogue into the down grouped kernel when SK_d==1 and nt==8 (one block owns 16 rows × 128 columns = exactly one
  Hadamard block): saves 1 launch + one P×D fp32 Z round trip.
- OPT-2 multi-warp blocks (4–8 pairs per block) for rot_in/epilogues (current 32-thread blocks, kernels/exl3.cu:281,292,302).
- OPT-3 compute rot_in once when gate/up suh are equal (production flags this per layer: docs/prod_exl3_reference.py:2254-2256
  "layer._exl3_shared_w13_suh = bool(torch.equal(layer.w13_suh[:, 0], layer.w13_suh[:, 1]))").
- OPT-4 SK_g=1 for tiny T to allow gate/up epilogue fusion (trade parallelism; measure).

--------------------------------------------------------------------------------------------
## B. Routing translation (fully on-device, static shapes, no host sync, graph-capturable)

### B.1 Inputs (from the positional args, host knows only SHAPES)
- c = expert_count (i64 [n+1]); ts = token_sorted (i64 [P]); ws = weight_sorted (f16 [P]); B = xh.size(0);
  R = temp_state_g.size(1) (= EXL3_TEMP_ROWS_FUSED, default 128; docs/prod_exl3_reference.py:57 — "TEMP_ROWS_FUSED = 128").
- Host-side static capacities (from shapes only): P = ts.numel(); U_cap = min(P, n); S_cap = U_cap + ceil(P/16).
  Bound proof (DERIVED): Σ_{elig e} ceil(c_e/16) ≤ Σ (c_e/16 + 15/16) ≤ P/16 + U ≤ ceil(P/16) + U_cap.

### B.2 Definitions (exact)
For e ∈ [0,n):
```
start[e] = Σ_{i<e} c[i]                      (exclusive prefix over ALL real buckets; matches xl_exl3_moe_kernel.cuh:50-51)
elig[e]  = (c[e] > 0) ∧ (c[e] ≤ R) ∧ (start[e] + c[e] ≤ P)      (XL skip rule :55-56 + REQ-S bound)
nseg_e   = elig[e] ? ceil(c[e]/16) : 0
soff[e]  = Σ_{i<e} nseg_i ;  nseg = Σ_e nseg_e   (≤ S_cap)
```
Outputs (int32, persistent scratch, capacity-sized):
```
pair_expert[j] = e   if ∃e: elig[e] ∧ start[e] ≤ j < start[e]+c[e];   else -1        (j ∈ [0,P))
for elig e, q ∈ [0, nseg_e):  s = soff[e]+q:
    seg_expert[s] = e ; seg_row0[s] = start[e] + 16q ; seg_rows[s] = min(16, c[e] - 16q)
nseg_dev[0] = nseg
```
Row j of every per-pair buffer belongs to token ts[j] with weight ws[j]. Sentinel pairs (j ≥ Σ_{i<n} c[i]) and
over-cap experts get -1 and no segment. The last bucket c[n] is never read (as in exl3_moe).

### B.3 CUDA implementation (route_prep_kernel, grid 1, block 1024)
```
1  for j = tid; j < P; j += 1024: pair_expert[j] = -1
2  load c[e] (e = tid, tid+1024, …); block exclusive scan → start[e]; flags elig; block exclusive scan of nseg_e → soff[e]; tid 0 writes nseg_dev
3  __syncthreads()                               // orders step-1 writes before step-4 writes within the block
4  for each elig e owned by tid: for q: write seg_* ; for i < c[e]: pair_expert[start[e]+i] = e   (≤ R iterations/thread)
```
All counts fit int32 (P ≤ P_cap ≤ 2^20). No atomics, deterministic output.

### B.4 Torch-op oracle (tests only; identical outputs; also graph-safe but ~15 launches)
```
cnt   = c[:n]
end   = torch.cumsum(cnt, 0);  start = end - cnt
elig  = (cnt > 0) & (cnt <= R) & (end <= P)
nsg   = torch.where(elig, (cnt + 15) // 16, 0)
send  = torch.cumsum(nsg, 0);  soff = send - nsg;  nseg = send[-1]
j     = torch.arange(P);  eo = torch.searchsorted(end, j, right=True)          # n for sentinel rows
ok    = eo < n;  ec = eo.clamp(max=n-1)
pair_expert = torch.where(ok & elig[ec], eo, -1).int()
s     = torch.arange(S_cap);  se = torch.searchsorted(send, s, right=True)
sv    = s < nseg;  sc = se.clamp(max=n-1);  q = s - soff[sc]
seg_expert = torch.where(sv, se, -1); seg_row0 = start[sc] + 16*q; seg_rows = torch.minimum(16, cnt[sc] - 16*q)  (masked by sv)
```
E-U3 requires exact equality kernel == oracle on the valid prefix [0, nseg).

### B.5 Full per-call launch sequence (all on the current stream; grids depend only on shapes)
```
route_prep     grid (1)                           block 1024
rot_in         grid (P, K/128, 2)                 block 32
grouped g/u    grid (S_cap, N/(16*8), 2*4)        block 128     X0=xg, X1=xu, Tp0=gate_trellis_ptrs, Tp1=up_trellis_ptrs
gateup_epi     grid (P, N/128)                    block 32
grouped down   grid (S_cap, K/(16*8), 1*1)        block 128     X0=X1=xd, Tp0=Tp1=down_trellis_ptrs, (K_arg=N, N_arg=K)
down_epi       grid (P, K/128)                    block 32      atomicAdd into out
```
(K = hidden, N = intermediate_local.) Decode T=1, topk 8 (DERIVED): P=8, S_cap=9, grouped g/u grid 9×8×8.

--------------------------------------------------------------------------------------------
## C. Python drop-in (tf_exl3_moe.py rewrite)

### C.1 Public surface
- Delete: `exl3_moe = exl3_moe_tf` alias (tf_exl3_moe.py:411-412 — "exl3_moe = exl3_moe_tf") [AUDIT-05]; `Exl3ExpertWeights`,
  `Scratch` (per-call), `build_member_table`, `build_member_table_device`, `_build_pick`, `_combine`, `_maxm_cap` [AUDIT-13/14/23];
  keyword-only `weights=`/`scratch=` [AUDIT-04]; `MOE_ACT_SILU = 1` (tf_exl3_moe.py:56) → constant `ACT_SILU = 0` matching
  docs/prod_exl3_reference.py:58 and docs/ref/xl_exl3_moe_common.cuh:6 [xl:FORK-03 corrected, tf:FORK-01 corrected].
- Keep one internal function with the exact positional contract:
  ```
  def exl3_moe_tf(hidden_state, output_state, expert_count, token_sorted, weight_sorted,
                  temp_state_g, temp_state_u, temp_intermediate_g, temp_intermediate_u,
                  act_function, K_gate, K_up, K_down,
                  gate_ptrs_trellis, gate_ptrs_suh, gate_ptrs_svh,
                  up_ptrs_trellis, up_ptrs_suh, up_ptrs_svh,
                  down_ptrs_trellis, down_ptrs_suh, down_ptrs_svh,
                  gate_mcg, gate_mul1, up_mcg, up_mul1, down_mcg, down_mul1,
                  act_limit, num_active=None, /) -> None
  ```
  It is never exported under the name `exl3_moe`; only integrate's dispatcher calls it via `plan()`/`launch()`.
  `num_active` (30th positional, only present if production's detector returned True) is accepted and ignored.

### C.2 Where the weights come from
- Exclusively from the positional pointer tables (args 13..21). No layer handle, no copies (A1). The temps (args 5..8) are not used
  for storage; only their shapes: R = temp_state_g.size(1), N = temp_intermediate_g.size(2) (same as
  docs/ref/xl_exl3_moe.cu:158 — "size_t max_tokens_per_expert = temp_state_g.size(1);" and :167 — "size_t intermediate_dim = temp_intermediate_g.size(2);").

### C.3 Load-time pre-flight (once per layer; never per call) — REQ-S
Runs inside the wrapped `build_exl3_fused_state` (see D.3), eagerly, after production built `layer._exl3_ptrs`/temps:
1. Global once: import AOT extension (A12). Failure → TF disabled for the process.
2. Layer eligibility (mirrors production's own per-layer check docs/prod_exl3_reference.py:290,293 — "if bits != 4:" … "if k_words != 64:"):
   `layer._exl3_bits == 4`, `layer._exl3_k_words == 64`, `hidden % 256 == 0`, `intermediate_local % 128 == 0`, n ≤ 4096,
   `layer._exl3_ptrs` has the 9 keys, each int64 [n] CUDA contiguous on the layer device, `layer._exl3_fused_temps` present.
3. Pointer sanity (one host copy of the 9 tables; allowed at load): all nonzero and 16-byte aligned; each trellis pointer p_e satisfies
   `base ≤ p_e ≤ base + nbytes − matrix_bytes` of `layer.w13_trellis` (gate/up) or `layer.w2_trellis` (down); same for suh/svh against
   w13_suh/w13_svh/w2_suh/w2_svh. (Aliasing is PyTorch semantics, not runtime-verified [prod:LOAD-04]; this check verifies it.)
4. Scratch: allocate/grow the module-level shared scratch keyed by (device, K, N): xg,xu f16 [P_cap,K]; xd f16 [P_cap,N];
   Z f32 [max(2·4·P_cap·N, 1·1·P_cap·K)]; pair_expert i32 [P_cap]; seg_expert/seg_row0/seg_rows i32 [S_cap(P_cap)]; nseg i32 [1].
   P_cap = min(R·TOPK_MAX, TF_EXL3_MAX_PAIRS) with TOPK_MAX = env TF_EXL3_TOPK_MAX (default 8, GLM num_experts_per_tok per sibling
   config:281), TF_EXL3_MAX_PAIRS default 1024. Size at K=4096, N=1024, P_cap=1024 (DERIVED): 8+8+2+32 MiB ≈ 50 MiB per device, shared
   by all layers (same "decode is sequential across layers" assumption as production: docs/prod_exl3_reference.py:59 — "# Shared fused scratch: decode is sequential across layers.").
   Must assert `not torch.cuda.is_current_stream_capturing()` (precedent docs/prod_exl3_reference.py:1291-1293).
5. Self-test (env TF_EXL3_SELFTEST, default 1): on the real layer (its pointer tables and temps), seeded synthetic x (f16, B ∈ {1,8}),
   production-identical routing (§E.0 helper) incl. one sentinel id, L = float(10.0) and L = 1.0; run orig exl3_moe → out_ref and TF →
   out_tf; synchronize; require E1 criteria (§E.1). Failure → TF disabled for the whole process (a mismatch is systemic), log stats.
   **[Revised 2026-09-27, review fixes F1/F2]** (a) The self-test calls are planned WITHOUT the per-call `TF_EXL3_TOKENS` window
   (it is a per-call policy, not a property of the layer); the hard domain limits still apply, so the batch sizes are
   B ∈ {1, min(8, R, ⌊P_cap/topk⌋)} with topk = min(8, n, P_cap). Planning the self-test through the window turned any window
   excluding B=1 or B=8 (e.g. 1:4, 2:64) — and MAX_PAIRS < 64 or R < 8 — into a process-wide disable. (b) A self-test call that
   is still not planned is a structural limit of that layer: only that layer stays unregistered (no process-wide disable). Only a
   numerical mismatch or an exception while running disables TF globally. (c) L ∈ {production limit, 0.0} instead of {10.0, 1.0}:
   the production limit is what the layer runs (`quant_method.moe.swiglu_limit or SWIGLU_LIMIT_DEFAULT` = 10.0; production never
   passes 0 or 1), and L = 0 (no clamp) leaves every gate/up weight or scale error visible, while L = 1 clamps ~84% of gate values on the E.0 synthetic magnitudes (std(g) ≈ 5),
   which masks gate/up scale errors and is the noisiest case vs exl3_moe (§E.1 note). (d) Every registered layer logs its self-test
   rel_l2 and margin at INFO, so the real-weight margin (G3) can be read from production logs.
6. Register: `REG[(device_index, gate_ptrs_trellis.data_ptr())] = LayerInfo(K, N, n, P_cap, ok=True)`.
- Exceptions: pre-flight catches `Exception` (ImportError, OSError, subprocess.CalledProcessError, RuntimeError incl. c10 errors and
  torch.OutOfMemoryError, ValueError, AssertionError) → log once, leave the layer unregistered. It never re-raises: a raise would reach
  production's `except Exception as exc: … layer._exl3_ptrs = None` (docs/prod_exl3_reference.py:2298-2300) and silently demote the layer
  to the host-syncing python loop. `BaseException` (KeyboardInterrupt/SystemExit) is not caught.

### C.4 Per-call plan (pure host metadata, O(1), must not raise; any exception inside → None)
`plan(args) -> Plan | None`; None ⇒ dispatcher calls `orig(*args)` unchanged.
- `len(args) in (29, 30)`; `TF enabled`; `args[9] == 0`; `args[10] == args[11] == args[12] == 4`;
  `args[22] and not args[23] and args[24] and not args[25] and args[26] and not args[27]`; `math.isfinite(float(args[28]))`.
- hidden_state: CUDA, f16, dim 2, `stride(1) == 1`, `stride(0) % 4 == 0`, `data_ptr() % 16 == 0` (rot_in loads half4); B = size(0) ≥ 1; K = size(1).
- output_state: f32, same shape, contiguous, same device.
- expert_count: i64, dim 1, n = numel − 1 ≥ 1; token_sorted: i64, dim 1, contiguous, P = numel ≥ 1; weight_sorted: f16, same shape, contiguous
  (exl3_moe does not dtype-check weight_sorted, docs/ref/xl_exl3_moe.cu:150 — "TORCH_CHECK_SHAPES_FULL(token_sorted, weight_sorted);";
  a non-f16 tensor is delegated unchanged).
- temps: temp_state_g f16 dim 3 size(2) == K; temp_intermediate_g f16 dim 3; N = size(2); R = temp_state_g.size(1).
- ptr tables: all 9 int64, dim 1, size(0) == n, contiguous, same device.
- Registry: `REG[(dev, args[13].data_ptr())]` exists, `ok`, and matches (K, N, n).
- Domain: `B ≤ R` (single-launch decode domain; production's fat/row-tile prefill calls have B > R: docs/prod_exl3_reference.py:1648
  and :1585 — "token_sorted.index_select(0, src),") and `P ≤ P_cap`; optional token window `TF_EXL3_TOKENS=lo:hi` (from E5 results).
  **[Revised 2026-09-27, F1]** The window is inclusive and applies to per-call planning only (never to the §C.3 self-test).
  Accepted forms: `lo:hi`, `lo:` (no upper bound), `:hi` and a bare `N` (= `1:N`); 1 ≤ lo ≤ hi. A malformed value of
  `TF_EXL3_TOKENS`, `TF_EXL3_MAX_PAIRS` or `TF_EXL3_TOPK_MAX` makes install() refuse (logged reason) instead of raising.
- K % 256 == 0, N % 128 == 0 (already implied by registry; re-checked cheaply).
- No `.item()`, no `.tolist()`, no tensor allocation, no data-dependent shape anywhere in plan/launch (GRAPH: [prod:GRAPH-01 corrected]).

### C.5 Launch (after plan succeeded)
- `launch(plan, args)` enqueues the six kernels of §B.5 on `torch.cuda.current_stream()` using the persistent scratch. Returns None
  (same as exl3_moe). Only the last kernel (down_epilogue) writes `output_state`.
- Exception policy at call time:
  - Before any TF kernel is enqueued: impossible by construction (plan already validated); if it happens → `orig(*args)`.
  - Exception raised while enqueuing stages 1–5 (route_prep … grouped down) that is a c10 TORCH_CHECK (RuntimeError/ValueError/
    TypeError/IndexError) and whose message does not match `CUDA error|cudaError|illegal|capture`: `output_state` is untouched
    (stages 1–5 write scratch only) → disable TF globally (sticky), log once, `return orig(*args)`. This is safe in eager mode and
    during graph capture (the already-enqueued scratch-only kernels are harmless).
  - Any exception at stage 6 (down_epilogue) or any CUDA error / capture-invalidated error: re-raise. Rationale: stage 6 may have
    partially accumulated into `out` (fallback would double count) and a CUDA error poisons the context for orig as well.
  - `TF_EXL3_STRICT=1` (tests): re-raise everything.
- Because vLLM captures decode into CUDA graphs [prod:GRAPH-02 corrected], the Python decision (TF vs orig) is taken at capture time and
  baked into the graph; replays run the captured kernels without Python. All protection therefore lives in C.3 (load) and C.4 (capture).

--------------------------------------------------------------------------------------------
## D. integrate.py (install / env gating / uninstall)

### D.1 Enable parsing (fixes AUDIT-21)
- `TF_EXL3_MOE`: enabled iff `value.strip().lower() in {"1","on","true","yes"}` (same style as docs/prod_exl3_reference.py:760-766
  mx_enabled). Anything else, including unset, "0", "", "false" → install() is a no-op. Read once at install; the dispatcher never reads env.
- Skip install (no-op) if production would never call exl3_moe: `prodmod.mx_enabled()` or not `prodmod.fused_moe_enabled()`
  (docs/prod_exl3_reference.py:754-757 — "if mx_enabled():" / "return os.environ.get(\"EXL3_FUSED_MOE\", \"1\") != \"0\"").
  **[Revised 2026-09-27 night]** production actually imports the launcher's overlay exl3.py
  (docs/ref/prod_live/overlay_exl3.py), which has no `mx_enabled`; every optional symbol is now feature-detected, the
  required ones are checked up front, and every "enabled but not installed" outcome logs a WARNING with the reason
  (docs/STATUS.md "本番の exl3.py", "Production module versions" below).

### D.2 Dispatcher that preserves how ORIGINAL exl3_moe is invoked (fixes AUDIT-05, AUDIT-06)
```
orig = exllamav3_ext.exl3_moe
if getattr(orig, "_tf_exl3_dispatch", False): return   # idempotent: never wrap a wrapper
accepts = prodmod._exl3_moe_accepts_num_active(orig)
def exl3_moe(*args):                                    # positional only; production never passes kwargs (:1535/:1537)
    p = tf.plan(args)
    if p is None:
        return orig(*args)                              # exactly the args received (29, or 30 incl. num_active)
    return tf.launch(p, args, orig)
exl3_moe.__doc__ = orig.__doc__                         # detector falls back to the doc string: same answer as for orig
exl3_moe.__name__ = getattr(orig, "__name__", "exl3_moe")
exl3_moe._tf_exl3_dispatch = True; exl3_moe._tf_exl3_orig = orig
# do NOT set __wrapped__ (inspect.signature would follow it)
assert prodmod._exl3_moe_accepts_num_active(exl3_moe) == accepts   # else: abort install, log, leave orig in place
exllamav3_ext.exl3_moe = exl3_moe
```
- Detector semantics being preserved: docs/prod_exl3_reference.py:916 — "if \"num_active\" in inspect.signature(fn).parameters:";
  :920-921 — "doc = getattr(fn, \"__doc__\", None) or \"\"" / "return \"num_active\" in doc or \"arg29\" in doc or doc.count(\"arg\") >= 30".
  `(*args)` has no `num_active` parameter → the doc path decides → identical doc → identical decision → production passes the same
  n_active_host to the dispatcher as to orig, and the dispatcher forwards it unchanged. The current fork's `dispatch(*args, **kw)` without
  doc would force False (integrate.py:42 — "def dispatch(*args, **kw):") [audit:AUDIT-06; latent only if a future ext accepts num_active].

### D.3 Load-time hook (pre-flight)
```
orig_build = prodmod.build_exl3_fused_state
def build_exl3_fused_state(layer, inners):
    orig_build(layer, inners)            # production behavior first; its exceptions propagate exactly as before
    try: tf.preflight(layer)             # C.3
    except Exception: log_once(...)      # never propagate (see C.3)
build_exl3_fused_state._tf_exl3_hook = True; ..._tf_exl3_orig = orig_build
prodmod.build_exl3_fused_state = build_exl3_fused_state
```
- Valid because production calls it by module-global name inside process_weights_after_loading:
  docs/prod_exl3_reference.py:2293-2294 — "if hasattr(exllamav3_ext, \"exl3_moe\"):" / "build_exl3_fused_state(layer, inners)".
- Order in install(): (1) patch build hook, (2) patch exllamav3_ext.exl3_moe. Layers built before install stay unregistered → orig.

### D.4 Getting install() into every vLLM worker (both TP ranks, nodeA and nodeB)
- Package the fork (pyproject) with entry point group `vllm.general_plugins`, e.g. `tf_exl3_moe = "integrate:plugin_register"`;
  `plugin_register()` = `install()` wrapped in `try/except Exception: log`.
- Evidence the group runs in worker processes before model load:
  image:/usr/local/lib/python3.12/dist-packages/vllm/plugins/__init__.py:16-18 — "# Default plugins group will be loaded in all processes(process0, engine core" /
  "# process and worker processes)" / "DEFAULT_PLUGINS_GROUP = \"vllm.general_plugins\"";
  image:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/worker_base.py:245-247 — "from vllm.plugins import load_general_plugins" / "load_general_plugins()".
  If `VLLM_PLUGINS` is set in production it must list the plugin name (image:.../vllm/plugins/__init__.py:64 —
  "if allowed_plugins is None or plugin.name in allowed_plugins:").
- Importing `vllm.model_executor.layers.quantization.exl3` inside install is idempotent (registration overwrites):
  image:.../quantization/__init__.py:86-88 — "if quantization in QUANTIZATION_METHODS:" / "logger.debug(" / "\"The quantization method '%s' already exists and will be \"".
- The package + `TF_EXL3_MOE` must be present on both nodes' containers; env propagation to Ray/multiproc workers of the Mia recipe is UNKNOWN (G10).

### D.5 uninstall()
- If `exllamav3_ext.exl3_moe` has `_tf_exl3_dispatch` → restore `._tf_exl3_orig`; if `prodmod.build_exl3_fused_state` has
  `_tf_exl3_hook` → restore its orig; set global `enabled=False` (stale references to the dispatcher then always delegate); clear REG;
  free scratch. Only restores objects we installed (never clobbers a later third-party patch). Idempotent.
- Already-captured CUDA graphs keep TF kernels baked in: a full revert requires re-capture (engine restart). The production-safe revert
  is: unset `TF_EXL3_MOE` and restart vLLM.

--------------------------------------------------------------------------------------------
## E. Test plan (nodeC GB10, synthetic data; GPU stage only — not run by this spec)

Execution: `docker run --rm --gpus all --network none -v ${HOME}/tf-exl3-fork:/w -w /w <image> python3 tests/<t>.py`.
Every test asserts and exits non-zero on failure (fixes AUDIT-19/20: current tests print only; tests/test_pipeline.py:52-54
"except Exception as e: … print(\"PIPELINE FAIL:\"" exits 0). Memory guard: before allocating, `torch.cuda.mem_get_info()`; total test
allocation budget ≤ 8 GiB and abort if free < 2× budget (nodeC hosts other services).

### E.0 Shared harness (tests/harness.py)
- Weights built EXACTLY like production create_weights (stacked, interleaved), not as separate tensors:
  `w13_trellis = randint(int16) [n,2,K/16,N/16,64]`, `w13_suh/svh [n,2,K]/[n,2,N] f16`, `w2_trellis [n,N/16,K/16,64]`, `w2_suh [n,N]`, `w2_svh [n,K]`.
  Pointer tables from slices exactly as production: gate = w13_*[e,0], up = w13_*[e,1], down = w2_*[e] → `data_ptr()` → int64 CUDA.
  Second variant: separate per-expert tensors (tests/prod_baseline.py:24-39 make_experts) to prove pointer-table generality.
- Magnitudes: suh/svh = random sign × scale; calibrate svh_g/svh_u on a probe so std(g) ≈ 5 after svh (≈4.5% of |g| > 10 under a
  Gaussian → clamp exercised at L=10), svh_d so std(out) ≈ 1. x ~ N(0,1) f16.
- Routing: `ids` [T, topk] int64 → production code path verbatim (map_topk_to_local with expert_map None or an EP map, repeat_interleave,
  argsort, scatter_add, `.to(float16)` weights) — import from `vllm.model_executor.layers.quantization.exl3` where possible
  (map_topk_to_local, _exl3_moe_launch) so the true 29-arg call shape and num_active detection are exercised.
- Temps: (C, R, K)×2, (C, R, N)×2 f16 with C = `exl3_moe_max_concurrency(dev)`; deterministic XL reference with C = 1
  (kernel groups = concurrency = temp_state_g.size(0), docs/ref/xl_exl3_moe.cu:159 — "size_t concurrency = temp_state_g.size(0);").
- Dims: K(hidden)=4096, N(intermediate_local)=1024 (= 2048/TP2 from the NVFP4 sibling config — ASSUMPTION, EXL3 checkpoint config
  UNKNOWN); also N=768 (dims used by existing tests/bench_prod_decode.py:8 — "D, NI, E, TOPK, L = 4096, 768, 64, 8, 8").
  n ∈ {32, 288}; topk = 8.

### E.1 Correctness vs production `exllamav3_ext.exl3_moe` on identical inputs (tests/test_e2e_vs_xl.py)
- For each case: out_xl = zeros, orig(*args); out_tf = zeros, tf path(*args) (STRICT mode, assert plan() is not None).
- Metrics: rel_l2 = ‖out_tf − out_xl‖/‖out_xl‖ (global); per-row rel_l2 over rows with ‖ref_row‖ > 1e-3·max_row_norm;
  finite-mask equality (NaN/Inf positions identical).
- Tolerances (u16 = 2^-11 ≈ 4.88e-4, fp16 unit roundoff):
  - global rel_l2 ≤ 2.0e-3 (= 4·u16); per-row rel_l2 ≤ 1.0e-2.
  - Justification (DERIVED from §0.2 and A8): with PAR on, rot_in is bit-identical (E-U1) and every elementwise rounding point is the
    same operation on the same format; differences originate only in the two GEMMs: (a) fp32 summation order, ~√K·2^-24 ≈ 4e-6
    relative, and (b) XL's one extra fp16 rounding of a split-K partial in gate/up — for K=4096, N=1024 XL uses TILESIZE_N=256
    (docs/ref/xl_exl3_moe.cu:205 — "if (hidden_dim % 256 == 0 && intermediate_dim % 256 == 0) N_off = 1;"), tiles_k=128, tiles_n=4,
    512 slices over gridDim.x=8 → 64 per CTA, k fastest (docs/ref/xl_exl3_gemm_inner.cuh:78 — "int slice_beg = tiles_k * tiles_n * blockIdx.x / num_slices;",
    :84 — "auto index_k = [&] (int slice_i) { return (slice_i % tiles_k); };") → each column split over exactly 2 CTAs → ≤ ½u16 relative
    on half of the sum; for down (K=1024, N=4096: tiles_k=32, tiles_n=16, 64 slices = 2 whole columns per CTA) no split-K rounding.
    These ≤1-ulp differences in g0/u0/d0 then propagate as occasional 1-ulp flips through ~8 fp16 rounding points; expected global
    rel_l2 ≈ (0.5–1.5)·u16. 2e-3 leaves ≥2× margin while a real bug is ≥100× larger: dropping 1 of 8 experts ≈ 1/√8 ≈ 0.35 per row,
    wrong expert suh/svh ≈ O(1).
  - **[Corrected 2026-09-27 from measurement, review finding F2/TA-5]** The prediction above is too optimistic, the mechanism is
    not "occasional flips", and the tolerance cannot separate PAR from NATIVE numerics:
    - Mechanism, measured against the XL binary (tests/test_stage_vs_xl.py (e)): TF's up-GEMM output rounded to fp16 differs from
      XL's u0 on **26.9%** of values (5.7% by > 1 ulp, max 771 ulp where the two cross-CTA halves cancel), rel_l2 2.7e-4 ≈ 0.56·u16.
      Every later stage is bit-identical to XL on XL's own intermediates (E-U6), so the whole TF-vs-XL difference is this GEMM
      split-K rounding, spread by each 128-wide Hadamard over the whole block.
    - Measured global rel_l2 TF vs XL (115 E.1 cases): production dims (N=1024) L=10: 6.4–7.7e-4, L=0: 6.6–7.9e-4,
      L=1: 8.1–10.0e-4 (≈ 1.3–2.0·u16; margin to 2e-3 ≥ 2.0×); N=768: L=10 up to 8.4e-4, L=1 up to 1.14e-3 (2.3·u16, margin
      1.75×). Load-time self-test (now L ∈ {10, 0}) on synthetic weights: 7.2–8.2e-4 (margin 2.4–2.8×).
    - NATIVE (TF_PARITY=0) is 3.6–6.6e-4 from float64 and XL 8.4–11.7e-4, so NATIVE-vs-XL is ≲ 1.8e-3 and **passes** this
      tolerance. E.1 is a gross-bug detector (≥ 0.1), not a parity proof. Parity is proven stage by stage against the XL binary
      itself by E-U6 (PAR bit-identical at every stage, NATIVE rejected in every case).
    - The tolerance itself is unchanged (2e-3 global, 1e-2 per row).
  - Noise-floor sanity: XL(C=max) vs XL(C=1) must be ≤ 1e-6 (atomic order only); report it.
  - Mutation check (proves the tolerance has teeth): zero one pair's weight in the TF input only → assert rel_l2 per row > 0.1.
- Cases (each with L ∈ {10.0, 1.0, 0.0}; L=0 exercises the `act_limit != 0` branch):
  - T ∈ {1, 2, 4, 8, 16, 64}, topk = 8, random distinct experts, n = 32 and n = 288.
  - **[Added 2026-09-27, TA-1]** T ∈ {65, 96, 127, 128} (n = 288) and T = 128 (n = 32): plan() serves every B ≤ R = 128, so the
    single-launch decode domain reaches P = P_cap = 1024 (full scratch, S_cap up to 352). T = 128 also skewed (count(e0) = R,
    8 segments) and with sentinel ids. The test asserts that a case at P = P_cap and B = R ran.
  - Skewed: every token includes expert 0 (count_0 = T; T=64 → 4 segments for one expert); every token picks the same 8 experts.
  - Cap edges with duplicate ids per token (synthetic only; production routers give distinct ids — router class UNKNOWN [prod:CALL-06]):
    one expert with count = R (=128, processed) and R+1 (skipped by XL → must be skipped by TF; out rows compared).
  - Sentinel / non-local: ids = −1 and ids ≥ n (expert_map None); EP-style expert_map mapping half the experts to −1; one token with ALL
    ids sentinel (row must stay 0); ALL pairs sentinel (nseg = 0 → TF must launch everything and leave out == 0).
  - Unaligned counts: counts 15, 16, 17, 33 (segment splitting).
- Delegation cases (assert plan() is None and result equals orig within 1e-6 using C=1): B = R+1; N = 1088 (N % 128 ≠ 0);
  K_gate=3; act_function=1; weight_sorted bf16; unregistered pointer table; 30-arg call (num_active=−1) through the dispatcher —
  assert orig receives exactly 30 args when delegated and the TF path accepts 30.

### E.2 Correctness vs `kernels/exl3_format_ref.py` (float64) (tests/test_e2e_vs_f64.py)
- Reference per pair (t, e, w): g = forward(x16[t], gate_e), u = forward(x16[t], up_e) (exl3_format_ref.forward, f64);
  a = (L≠0) ? min(silu(g), L)·clip(u, −L, L) : silu(g)·u  [post-SiLU, XL semantics]; d = forward(a, down_e); out[t] += float(w16)·d.
  Trellis per expert from the stacked tensors (`w13_trellis[e,0].cpu()` etc.), W_q cached per expert (CPU).
- Size limits: n ≤ 16, T ∈ {1, 4, 16} (CPU cost).
- Criteria: e_tf = rel_l2(out_tf, ref), e_xl = rel_l2(out_xl, ref): e_tf ≤ 1.25·e_xl + 2.5e-4 (TF may not be less accurate than production)
  and e_tf ≤ 5e-3 (≈10 fp16 rounding points × u16, first-order worst case, DERIVED).
- Stage unit tests:
  - E-U1 rot_in vs `exllamav3_ext.had_r_128(x_row, out, suh_e, None, 1.0)` → `torch.equal` (bit-identical, PAR on).
  - E-U2 grouped via pointer tables (interleaved w13 layout, e ∈ {0, n−1}, mat ∈ {gate, up}) vs f64 `xg16 @ unpack(trellis_e)`:
    rel ≤ 1e-5 (fp32 accumulation of exact f16 products over K=4096, DERIVED ≈ √4096·6e-8 ≈ 4e-6).
  - E-U3 route_prep vs §B.4 oracle: exact equality over 1000 random routings incl. sentinel/over-cap/empty, n ∈ {288, 1100}.
  - E-U4 gateup_epilogue vs torch half-precision emulation of §0.2 step 3 on identical random Z: ≤ 1 fp16 ulp on ≤ 1% of elements
    (SiLU intrinsics not emulable exactly).
  - Layout: tests/test_layout_equiv.py (XL reconstruct vs format_ref.unpack) must pass bit-equal (its result is not recorded in repo — UNKNOWN).

### E.3 Case matrix summary
tokens {1,2,4,8,16,64} × topk 8 × n {32,288} × L {10,1,0} × routing {random, skewed, same-8, sentinel, EP-map, cap-edge} ×
layout {interleaved-stacked, separate} for E.1; E.2 on the subset above.

### E.4 CUDA graph capture/replay (tests/test_graph.py)
- Warm up eagerly on a side stream, then capture with `torch.cuda.CUDAGraph` a function that runs the REAL production
  `apply_exl3_fused_moe(x2d, ids, weights, layer, inners, None, limit)` (module imported from the image; `layer` = nn.Module holding the
  stacked Parameters, `_exl3_ptrs`, `_exl3_fused_temps`, `_exl3_k`; built through the hooked `build_exl3_fused_state` so pre-flight
  and registry run) for static T ∈ {1, 8, 64}; capture all three graphs (shared scratch); replay interleaved 100× with fresh x/ids/weights
  copied into the static inputs; each replay compared to eager orig within E.1 tolerances.
  **[Added 2026-09-27, TA-1/TA-7]** T = 128 (P = P_cap) is captured too, and one graph holds TWO layers in sequence (n = 288 with
  sentinel ids, then n = 32 through an EP expert_map) sharing the scratch; 20 replays with fresh routing, each layer vs eager orig.
  **[Added 2026-09-27, branch `opt` review F1]** Since K2 (the apply hook) is on by default, the whole capture / replay / two-layer
  suite runs on each serving path production can be in: K2 (default install), the dispatcher with `TF_EXL3_APPLY=0` (apply not
  hooked) and the dispatcher after `disable_apply` (hook installed, K2 off, as after a K2 self-test mismatch). The path that served
  each capture is identified by its own counter (`tf_apply_calls` counts K2 only).
- Assert: TF path chosen at capture (Python counter incremented in launch()); no host sync in the TF path
  (`torch.cuda.set_sync_debug_mode("error")` around an eager TF call on pre-built args); zero TF-side allocations
  (`torch.cuda.memory_stats()["allocation.all.allocated"]` delta == 0 across 100 direct calls with pre-built args); capture of a
  prefill-sized eager call (T = R+1) delegates.

### E.5 Decode A/B timing vs exl3_moe (tests/bench_ab_decode.py)
- n=288, K=4096, N=1024, topk=8, L_layers=2..4 distinct weight sets cycled (≥3.4 GiB of trellis ≫ L2 → cold weights; bounded by E
  memory guard), T ∈ {1,2,4,8,16,32,64}. **[Added 2026-09-27, TA-1]** + T ∈ {96, 128} (128 = R, the largest single-launch call).
  **[TA-7]** Every timed configuration is read back and compared with orig (E.1): bare eager and apply eager for all 12 sets,
  and the timed 12-call / 3-layer TF graphs (bare and apply) replayed once more after timing.
- Measure both (a) the bare call with pre-built args and (b) the production apply incl. routing, each under CUDA-graph replay
  (captures launch overhead of 6 TF kernels vs 1 XL kernel) and eagerly; 200 iterations after 20 warmups; CUDA events; report µs/call,
  touched-expert GB/s (as tests/bench_prod_decode.py:11-12), speedup, per-stage breakdown.
- Output: the T window where speedup ≥ 1.05 → value for `TF_EXL3_TOKENS`. If no T qualifies, TF stays disabled (correct-but-slower is
  not deployed). Existing claim "TF 1.37x vs fat kernel at m=8" (docs/STATUS.md:21) compared against the prefill fat kernel, not the decode path.

--------------------------------------------------------------------------------------------
## F. AUDIT defects → spec items

| # | Defect (source) | Fixed by |
|---|---|---|
| AUDIT-01 | token_sorted token indices misread as codes; 3 incompatible encodings (tf_exl3_moe.py:254, :391, kernels/exl3.cu:81) | A4 (pair-index rows, segment table), B |
| AUDIT-02 | fp16 hidden vs bf16 rot_in check (kernels/exl3.cpp:38) | A2, C.4 dtype plan |
| AUDIT-03 | last routed slot dropped (kernels/exl3.cu:166,191,229) | A3 (no slots concept) |
| AUDIT-04 | weights never passed; attach_exl3_experts undefined (tf_exl3_moe.py:307-311) | A1 (pointer tables from positional args), C.2 |
| AUDIT-05 | `exl3_moe = exl3_moe_tf` alias → num_active=-1 + NotImplementedError w/o fallback (tf_exl3_moe.py:411-412) | C.1 (alias deleted), D.2 |
| AUDIT-06 | dispatcher `(*args, **kw)` without doc changes num_active detection (integrate.py:42) | D.2 (doc copy, `(*args)`, install-time equality assert) |
| AUDIT-07 | expert_count n+1 vs cumsum out= of length n; sentinel bucket fed as expert (tf_exl3_moe.py:236-237, 383-384) | B.2 (only c[:n] read; sentinel rows → -1) |
| AUDIT-08 | pick default E → OOB suh/svh reads (tf_exl3_moe.py:387) | A6, B.2 (pair_expert = −1 + token guard) |
| AUDIT-09 | duplicate-index weight scatter in _combine (tf_exl3_moe.py:404) | A5 (per-pair weight, atomicAdd) |
| AUDIT-10 | cap (count > R) ignored → fat-expert double count | B.2 elig (XL skip rule) + C.4 `B ≤ R` domain |
| AUDIT-11 | slots = numel//rows wrong for row-tile subsets (tf_exl3_moe.py:328) | A4/B (no slots; P = numel) + C.4 `B ≤ R` |
| AUDIT-12 | host syncs / nonzero / dynamic grid / JIT in call path | A9, B.3, C.4, C.5, A12 |
| AUDIT-13 | per-call Scratch/y/members allocations (tf_exl3_moe.py:342-343, 364, 251) | C.3 step 4 (persistent shared scratch) |
| AUDIT-14 | maxm_cap = rows·slots → mostly-empty grid.z | A4/B (≤16-row segments, S_cap bound) |
| AUDIT-15 | non-NotImplementedError exceptions escape fallback | C.3 (load-time catch), C.5 (stage-aware policy), A12 (AOT) |
| AUDIT-16 | K_gate/K_up/K_down never checked (tf_exl3_moe.py:272-274) | C.3 step 2, C.4 (==4) |
| AUDIT-17 | numerical recipe differs (bf16r, clamp-before-SiLU, fp32 SiLU) | A8 (PAR), E.1/E.2 tolerances |
| AUDIT-18 | gate/up non-contiguous → copy would double w13 memory | A1 (zero-copy pointer tables), §0.5 |
| AUDIT-19 | test_pipeline passes by coincidence (P=32 identity, bf16, act=1, no asserts) | E.0–E.1 (production-shaped inputs, P>32, asserts, mutation check) |
| AUDIT-20 | test_fallback never exercises dispatch/num_active/escaping errors | E.1 delegation cases, E.4 |
| AUDIT-21 | env parse loose, env checked only at install, nested wrappers, no worker install | D.1, D.2 (idempotent marker), D.4 (plugin), D.5 |
| AUDIT-22 | stale Z rows read by epilogues on scratch reuse | A10 |
| AUDIT-23 | dead host helper with .item() loop (tf_exl3_moe.py:193-205) | C.1 (deleted) |
| X-1 | fork `MOE_ACT_SILU = 1` vs production 0 (tf_exl3_moe.py:56 vs docs/prod_exl3_reference.py:58) — TF path could never run | C.1 (`ACT_SILU = 0`), C.4 |
| X-2 | from_int16_trellis docstring "no copy" false for w13[:,0] (tf_exl3_moe.py:131-137) | A1 (method deleted) |
| X-3 | JIT build on first call (tf_exl3_moe.py:80-93) | A12, C.3 step 1 |
| X-4 | STATUS item 3 (down layout) open | resolved: w2_trellis [E, inter/16, hidden/16, 64] == TF dt layout [layout:PROD-02 corrected]; with A1 layout is per-pointer anyway |

--------------------------------------------------------------------------------------------
## G. Open risks (cannot be settled without the real checkpoint / production runtime)

G1. Checkpoint bitrate: only the name says 4bpw (docs/prod_exl3_reference.py:4); `bits` value (config.json quantization_config or
    quantization_config.json [prod:K-01 corrected]) UNKNOWN. Mitigation: C.3 gates bits==4, k_words==64; otherwise inert.
G2. Real dims: hidden / moe_intermediate_size / n_routed_experts / top_k of the EXL3 checkpoint UNKNOWN (sibling NVFP4 config used).
    MTP layer (sibling config:284 — "\"num_nextn_predict_layers\": 1,") may have different expert dims or quantization. Mitigation: per-layer pre-flight.
G3. Tile-bit equivalence on REAL weights: established by code identity + numpy emulation on random tiles [layout:CONCL-01]; TensorFold's
    12/12 claim is for Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw (kernels/exl3_format_ref.py:5) while production names brandonmusic/GLM-5.3-Flash-tr3-4bpw.
    Mitigation: C.3 self-test on real weights at every load.
G4. The compiled exllamav3_ext .so vs the reviewed sources: not proven identical [xl:SRC-01 corrected]; the parity analysis (A8, E.1
    tolerance derivation) assumes the source. The self-test/E.1 compare against the binary, so a mismatch shows as a failed tolerance.
G5. Runtime env on nodeA/nodeB (EXL3_FUSED_MOE, EXL3_TEMP_ROWS_FUSED, GLM53_EXL3_MX, EXL3_FAT_*) UNKNOWN [prod:FLAG-02]. If MX is on or
    fused is off, decode never calls exl3_moe → drop-in inert (D.1 skips install).
G6. CUDA-graph mode, capture sizes, max_num_seqs and spec-decode tokens/step UNKNOWN → the actual T distribution (and whether all decode
    batches satisfy B ≤ R) must be read from production logs/config before choosing TF_EXL3_TOKENS.
G7. GB10 SM count (→ exl3_moe concurrency C = num_sms/8) UNKNOWN → baseline speed and atomics non-determinism of production.
G8. fp16 intrinsic codegen (h2exp/h2rcp) of our build vs the XL binary may differ by 1 ulp → E.1 tolerance covers; bit-identity of the
    gate/up epilogue is not promised. **[Resolved 2026-09-27, E-U6]** Measured bit-identical: TF's gate/up epilogue on XL's own
    u0 (= g0, up tables pointed at the gate's) reproduces XL's xd exactly at L ∈ {10, 1, 0}, and down_epilogue on XL's d0
    reproduces XL's out exactly (SK 1 and 2).
G9. Performance: 6 launches per layer (+ route_prep) vs 1 fused XL launch; 32-thread blocks in per-pair kernels. TF may lose at T=1..4
    (docs/STATUS.md:24 — "m=1-4 では届かない(122-187)"). E.5 decides; A13 lists fusions.
G10. Plugin delivery to Ray/multiproc workers on both nodes, env propagation, and any `VLLM_PLUGINS` allowlist in the Mia recipe UNKNOWN.
G11. Concurrency hazards on the shared scratch if production ever overlaps two routed-MoE applies on different streams (e.g. dual-batch
    overlap). Production's shared fused temps carry the same assumption (docs/prod_exl3_reference.py:59); not verified for our config.
G12. Weight relocation after load (sleep mode / offload) would invalidate pointers; production's own pointer tables share this risk.
G13. Model-level validation (greedy-token agreement / perplexity with TP=2 on nodeA+nodeB) requires production hardware and is out of
    scope for nodeC; E.1 per-layer tolerance is necessary but not sufficient evidence of end-to-end quality.
G14. `apply_exl3_python_loop` and fat prefill paths clamp BEFORE SiLU while exl3_moe clamps AFTER [xl:MOE-11]; this spec follows the
    decode kernel. Which one the model owner considers "correct" is a product decision outside this spec.

--------------------------------------------------------------------------------------------
## Appendix — facts relied upon (verdict status)
prod: SRC-01, LOAD-01 (C), LOAD-02 (C), LOAD-03 (C), LOAD-04 (C), LOAD-05 (C), LOAD-06, K-01 (corr), K-02, K-03, CALL-01 (C),
CALL-02 (C), CALL-03 (C), CALL-04 (C), CALL-05 (C), CALL-06 (corr), CALL-07 (corr), CALL-08, MATH-01 (corr), MATH-02, MATH-03,
PATH-01 (C), GRAPH-01 (corr), GRAPH-02 (corr), OUT-01 (C), OUT-02 (C), FLAG-01 (C), FLAG-02, FLAG-03, SUH-01, TP-01, FORK-01 (corr).
xl: SRC-01 (corr), MOE-01, MOE-02 (C), MOE-03 (C), MOE-04, MOE-05 (C), MOE-06 (C), MOE-07 (C), MOE-08 (C), MOE-09, MOE-10 (C),
MOE-11 (C), MOE-12 (C), MOE-13 (C), MOE-14 (C), MOE-15, MOE-16, MOE-17 (C), FORK-01 (corr), FORK-02 (corr), FORK-03 (corr), FORK-04 (C).
tf: ROT-01..04 (C), GRP-01 (C), GRP-02 (C), GRP-03 (corr), GRP-04 (C), GRP-05 (C), GRP-06 (C), GRP-07, GUE-01 (C), GUE-02, DWN-01 (C),
DWN-02 (C), GRPB-01 (C), GRPB-02, WTS-01 (C), WTS-02, WTS-03 (C), FORK-01..05 (corr), FORK-06 (C), LNCH-01.
layout: LAYOUT-01..03 (C), LAYOUT-04, CB-01 (C), CB-02, HAD-01..03 (C), NUM-01, NUM-02, PROD-01 (C), PROD-02 (corr), PROD-03,
PROD-04 (C), FORK-01 (C), FORK-02 (C), FORK-03 (corr), REF-01, REF-02, CKPT-01, CONCL-01 (corr).
audit: AUDIT-01 (C), 02 (corr), 03 (C), 04 (C), 05 (C), 08 (C), 09 (C), 10 (corr), 12 (C), 15 (corr), 19 (C); unverdicted
AUDIT-06/07/11/13/14/16/17/18/20/21/22/23 were re-checked by the design lead against the cited source lines before use.
Facts without a verdict and not re-checked were used only as context, never as the sole basis of a requirement.

--------------------------------------------------------------------------------------------
## Addendum (2026-09-27, main session, measured — supersedes G1, G2, G3-layout and the E.2 "UNKNOWN" layout note)
- G1 RESOLVED: HuggingFace config.json of BOTH Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw and brandonmusic/GLM-5.3-Flash-tr3-4bpw
  (byte-identical files; copy at docs/ref/hf_Mia-AiLab_GLM-5.3-Flash-EXL3-TR3-4bpw_config.json):
  quantization_config = {"bits": 4, "codebook": "mcg", "quant_method": "exl3", "scope": "glm53_routed_experts_only", "version": "0.0.43"}.
- G2 RESOLVED: text_config: hidden_size 4096, moe_intermediate_size 2048 (→ intermediate_local 1024 at TP=2), n_routed_experts 288,
  num_experts_per_tok 8, num_hidden_layers 45, first_k_dense_replace 3 (→ 42 MoE layers), swiglu_limit 10.0,
  routed_scaling_factor 2.5, num_nextn_predict_layers 1. Test dims: K=4096, N=1024, n=288, topk=8, L=10.0.
- Layout equivalence MEASURED on nodeC GB10 (tests/test_layout_equiv.py, commit 7947a9b): exllamav3_ext.reconstruct vs
  kernels/exl3_format_ref.unpack are bit-equal on 3 shapes (6.3M values, 0 differing); full linear suh→had→W→had→svh
  ExLlamaV3 vs TF float64 reference rel err 4.2–4.6e-4.
- Production decode baseline MEASURED (tests/bench_prod_decode.py, commit 470bca1; D=4096 N=768 n=64/layer, 8 layers cycled):
  exl3_moe incl. routing 1 tok 309 µs (122 GB/s), 8 tok 1144 µs (175 GB/s), 64 tok 1764 µs (171 GB/s).
  The image's exl3_moe does NOT accept num_active (29 args); calling with 30 raises TypeError whose pybind11 message stringifies
  tensors (~16 ms) — never probe by exception.
- Production exl3_moe is non-deterministic run-to-run (tests/prod_baseline.py: max diff 5.8e-11 at absmax 8e-4).

--------------------------------------------------------------------------------------------
## Implementation notes (2026-09-27, branch `impl`, implementer; measured on nodeC GB10, synthetic weights)

### What was built (file map)
| Spec | File | Notes |
|---|---|---|
| A1–A11 | `kernels/exl3.cu` | pointer-table `grouped_kernel<NT,W>` over a ≤16-row segment table; `rot_in` (fp16 x, int64 row stride); `gateup_epilogue` / `down_epilogue` with `TF_PARITY` (default 1) exactly as §0.2 / A7 / A8; `down_epilogue` multiplies `0.088388347648f*w` then `svh` and atomically adds into fp32 `out` (16-byte `float4` vector atomics on sm_90+, A5b); `route_prep` = one 1024-thread block, two block scans per 1024-expert chunk (n > 1024 loops), no atomics; `static_assert(HAD_SCALE == 0.088388347648f)` |
| A1/A12 | `kernels/exl3.cpp` | bindings; `moe_forward` = the whole exl3_moe-equivalent call (6 launches, one stream, static grids). Every `TORCH_CHECK` runs before the first launch. Stage entry points (`route_prep`, `rot_in`, `grouped`, `gateup_epilogue`, `down_epilogue`) for unit tests |
| A12 | `setup.py`, `pyproject.toml`, `kernels/exl3_bind.cpp` | AOT `tf_exl3_moe_ext` for sm_121a (`TORCH_CUDA_ARCH_LIST=12.1a` -> `-gencode=arch=compute_121a,code=sm_121a`, cubin verified with `cuobjdump`), `-O3`, same cu13 header shim as the JIT loader; entry point `vllm.general_plugins: tf_exl3_moe = integrate:plugin_register`. Build in the image: `python3 setup.py build_ext --inplace` (tests) or `pip install --no-build-isolation .` |
| C | `tf_exl3_moe.py` | `plan()` / `launch()` / `preflight()` / `register()` / strict `exl3_moe_tf(..., /)`; `ACT_SILU = 0`; persistent per-(device,K,N) scratch (50.0 MiB at K=4096, N=1024, P_cap=1024); loader = AOT import, JIT only with `TF_EXL3_JIT=1` |
| D | `integrate.py` | `env_enabled`, `install` (build hook first, then the `(*args)` dispatcher with orig's `__doc__`, detector equality asserted, idempotent), `uninstall`, `plugin_register` |
| E | `tests/` | `harness.py` (E.0), `test_units.py` (E-U1..E-U5, JIT==AOT), `test_e2e_vs_xl.py` (E.1), `test_e2e_vs_f64.py` (E.2 + NATIVE), `test_graph.py` (E.4), `test_integrate.py` (C.3/C.5/D.1–D.5 incl. packaging + vLLM plugin loader), `bench_ab_decode.py` (E.5), `sweep_grouped_cfg.py` (A13 evidence), `test_layout_equiv.py` (now asserts), `gpu_run.sh` (host rules), `run_all.sh` |

Launch sequence per call (as §B.5, with the A13 tile width): route_prep (1×1024) → rot_in (P, K/128, 2)×32 →
grouped g/u (S_cap, N/64, 2·4)×128 → gateup_epilogue (P, N/128)×32 → grouped down (S_cap, K/64, SK_d)×128 →
down_epilogue (P, K/128)×32, SK_d = 2 if P ≤ 8 else 1.

The harness exercises production code, not copies: the image's `vllm/.../quantization/exl3.py` imports cleanly and is
byte-identical to docs/prod_exl3_reference.py (diff in the image), so no stubs were needed. Layers are built like
`create_weights` (stacked, gate/up interleaved) and finished by the real `Exl3MoEMethod.process_weights_after_loading`
(instantiated with `__new__`, bits=4, to skip `FusedMoEMethodBase(moe)` config plumbing) → real LinearEXL3 inners →
the hooked `build_exl3_fused_state` → `tf.preflight`. The exact 29 positional args are captured by running the real
`apply_exl3_fused_moe` with a recorder (same `__doc__`) in place of `exllamav3_ext.exl3_moe`.

### Deviations from the spec (with evidence)
1. **E.1 delegation case `N = 1088`**: numeric equality is not checkable because production `exl3_moe` itself does not
   support `intermediate % 128 != 0` (its tiles cover 1024 of 1088 columns and the down GEMM reads uninitialized temp
   columns): two identical orig calls differed by rel 1.5e-2 (first E.1 run). The case keeps its `plan() is None` +
   delegation assertion; the numeric "dispatcher == orig" check for an ineligible shape is carried by `K = 3968`
   (hidden % 256 != 0, supported by exl3_moe): rel 0.0.
2. **C.5 stage-aware policy, simplified by construction**: `moe_forward` performs all checks before stage 1, so any
   c10 check failure is necessarily pre-launch (output untouched → disable TF sticky, `orig(*args)`), and anything
   raised later is a CUDA launch error ("CUDA error: …") → re-raised. This satisfies C.5 without per-stage bookkeeping.
3. **C.4 pointer tables**: instead of re-checking dtype/shape/device of the 9 tables per call, `plan()` requires them
   to be the identical tensor objects validated at pre-flight (`args[13+i] is info.tables[i]`). Stricter and cheaper;
   production passes `layer._exl3_ptrs[...]` every call (docs/prod_exl3_reference.py:1503-1533), so identity holds.
4. **D.5 scratch**: `uninstall()` frees the scratch only if no TF launch was ever captured into a CUDA graph; otherwise
   it keeps it alive (a replay of an already-captured graph would otherwise write into freed memory). Outgrown scratch
   is also kept alive for the same reason.
5. **A12 AOT source list**: `kernels/exl3_bind.cpp` (`#include "exl3.cpp"`) is compiled instead of `exl3.cpp` because
   setuptools names objects by basename and `exl3.cpp`/`exl3.cu` both became `exl3.o` (ninja: "multiple rules generate
   …/kernels/exl3.o"). JIT still compiles `exl3.cpp` directly.
6. **A13 (performance, done after everything passed)**: grouped tile width `nt = 4` for gate/up `(4,4,4)` and down
   `(4,4,1)` instead of TensorFold's `nt = 8` — each warp's K range and the split/warp summation order are unchanged,
   so Z is bit-identical (E-U2 asserts it); `tests/sweep_grouped_cfg.py` (median of 5 rounds, graph replay, cold
   weights): gate/up 141.8 vs 149.9 µs at T=1 … 4336.7 vs 4430.9 µs at T=64, down 500.3 vs 515.3 µs at T=8. Down
   K splits `SK_d = 2` when P ≤ 8 (T=1): 70.3 vs 75.6 µs; this changes only the fp32 summation of d0 at T=1 (E-U5b
   bit-exact vs emulation, E.1/E.2 pass). OPT-1 (fusing down_epilogue into grouped down) was not done: it needs
   nt = 8 (one block = one 128-wide Hadamard block), which A13's measured nt = 4 gives up, and under graph replay the
   launch gaps are ~0 (T=1: stages sum 228.2 µs vs 228.3 µs measured). OPT-2/OPT-3/OPT-4 not done (each ≤ ~1% by the
   stage breakdown; OPT-3 needs `_exl3_shared_w13_suh`, false on random test weights).
7. **Memory guard**: on GB10 `torch.cuda.mem_get_info()` "free" excludes reclaimable page cache (reported 4.9 GiB free
   while /proc/meminfo MemAvailable was 85 GiB), so a "free ≥ 2× budget" guard would always abort. `gpu_guard` uses
   max(cuda free, MemAvailable) for the 2× headroom check and enforces the ≤ 8 GiB budget with
   `torch.cuda.set_per_process_memory_fraction` (allocations beyond it raise OOM); every test prints its peak
   (max 5.20 GiB, bench_ab_decode). `tests/gpu_run.sh` additionally refuses to start unless `free -g` available ≥ 40.
8. **E.4 prefill-sized capture**: production `apply_exl3_fused_moe` at T = R+1 takes the prefill branch, which host-
   syncs (`int(counts.max().item())`) and cannot be captured. Tested instead: the eager production apply at T = R+1
   delegates, and a captured bare dispatcher call with B = R+1 delegates and replays equal to orig (rel 1.1e-08).
9. **E.0 magnitudes**: svh_g/svh_u set analytically from the codebook std so std(g) ≈ 5 (measured |g| > 10 on 5.1% of
   elements in E-U4; spec ≈ 4.5%); svh_d calibrated by a probe to std(out) = 1.000; suh = sign × (1 ± 25%).
10. **Removed tests**: `tests/test_pipeline.py`, `tests/test_fallback.py` (exercised the deleted Exl3ExpertWeights /
    NotImplementedError API, AUDIT-19/20). No tolerance of §E was changed.

### Measured results (nodeC, NVIDIA GB10, 48 SMs, exl3_moe concurrency 6; `tests/run_all.sh`, ALL PASSED)
- E-U1 rot_in vs `had_r_128`: bit-identical (40 pairs × 2 mats, 0 differing values; strided x too).
- E-U2 grouped via pointer tables (interleaved w13, e ∈ {0, 287}) vs float64: worst rel 3.44e-07 (gate/up),
  3.43e-07 (down); nt=4 bit-identical to nt=8.
- E-U3 route_prep vs §B.4 oracle: 1000/1000 identical for n = 288 and n = 1100 (incl. sentinel, over-cap, empty,
  inconsistent Σc > P).
- E-U4 gateup_epilogue vs numpy fp16 emulation of §0.2 step 3: max 0 ulp, 0.000% differing at L = 10, 1, 0.
- E-U5/5b down_epilogue (SK 1 and 2) bit-identical to emulation.
- E.1 vs production exl3_moe (94 TF cases: T ∈ {1,2,4,8,16,64} × n ∈ {32,288} × L ∈ {10,1,0}, N=768, skewed,
  same-8, cap R / R+1, sentinel / EP map / all-sentinel row / all pairs sentinel, unaligned 15/16/17/33, separate
  tensors, NaN-poisoned scratch): worst global rel_l2 1.14e-03 (tol 2e-3), worst row rel 1.25e-03 (tol 1e-2), XL
  noise floor (C=6 vs C=1) ≤ 7.4e-08 over the runs (atomics order; 6.9e-08 at 12:13, 7.4e-08 at 12:18); mutation (one pair's weight zeroed) moves the row by 0.730; all 8 delegation
  cases delegate; 30-arg call reaches orig with 30 args; detector answers identical for orig/dispatcher (False on
  the image, True for a num_active-doc orig). 315/315 checks.
- E.2 vs float64 (n=16, T ∈ {1,4,16}, L ∈ {10,1,0}): e_tf/e_xl = 0.893–0.986, worst e_tf 1.103e-03 (TF is slightly
  more accurate than production: fp32 split sums); NATIVE build (TF_PARITY=0) worst 6.568e-04.
- E.4: TF chosen at capture for T ∈ {1,8,64}; 100/100 interleaved replays within E.1 (worst 8.75e-04); no host sync
  (sync-debug "error" silent; positive control `.item()` raised); 0 allocations over 100 calls (control: 1).
- C.3/C.5/D: 23/23 (pointer range/alignment rejection, injected-error self-test disables TF, pre-check fallback
  equals orig and is sticky, CUDA error re-raised, STRICT, uninstall keeps a later third-party patch; `pip install
  --no-build-isolation` + vLLM `load_general_plugins()` in a fresh interpreter installs the dispatcher from the AOT
  package; not with TF_EXL3_MOE unset; not when VLLM_PLUGINS excludes it).
- E.5 decode A/B (n=288, K=4096, N=1024, topk 8, 3 layers = 5.09 GiB cycled × 4 routings, median of 5 alternating
  rounds; µs per call; speedup = XL/TF):

  | T | distinct | bare eager XL / TF | × | bare graph XL / TF | × | apply eager XL / TF | × | apply graph XL / TF | × | TF GB/s |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 1 | 8.0 | 357.6 / 237.9 | 1.50 | 351.9 / 228.3 | 1.54 | 419.2 / 305.2 | 1.37 | 383.4 / 262.5 | 1.46 | 220.4 |
  | 2 | 15.6 | 543.3 / 441.5 | 1.23 | 543.5 / 426.8 | 1.27 | 613.8 / 505.8 | 1.21 | 569.7 / 452.3 | 1.26 | 229.7 |
  | 4 | 30.2 | 1018.2 / 831.4 | 1.22 | 992.4 / 814.5 | 1.22 | 1070.0 / 903.1 | 1.18 | 1020.9 / 840.5 | 1.21 | 233.7 |
  | 8 | 58.5 | 1849.5 / 1582.9 | 1.17 | 1825.1 / 1567.3 | 1.16 | 1899.6 / 1660.3 | 1.14 | 1852.2 / 1606.0 | 1.15 | 234.8 |
  | 16 | 105.2 | 3246.0 / 2844.8 | 1.14 | 3233.8 / 2840.5 | 1.14 | 3315.4 / 2920.7 | 1.14 | 3249.3 / 2865.0 | 1.13 | 233.1 |
  | 32 | 169.0 | 5143.1 / 4630.4 | 1.11 | 5108.8 / 4603.5 | 1.11 | 5228.1 / 4722.8 | 1.11 | 5153.4 / 4637.3 | 1.11 | 231.0 |
  | 64 | 241.2 | 7310.1 / 6686.1 | 1.09 | 7315.5 / 6650.9 | 1.10 | 7399.0 / 6825.6 | 1.08 | 7380.3 / 6707.6 | 1.10 | 228.1 |

  (Table: run_all at 12:13. A repeat run_all at 12:18 gave graph-replay production-apply speedups 1.44, 1.25, 1.21,
  1.16, 1.14, 1.12, 1.10 for T = 1 … 64 — run-to-run variation ≤ 0.02.)
  T window with graph-replay production-apply speedup ≥ 1.05: {1, …, 64} → `TF_EXL3_TOKENS=1:64` (i.e. no window
  needed on this evidence). TF per-stage at T=1: route_prep 2.4, rot_in 2.0, grouped g/u 142.7, gateup_epi 2.5,
  grouped down 76.2, down_epi 2.4 µs; the two GEMVs are 96–99% of the time at every T and run at 220–235 GB/s
  (XL: 143–208 GB/s). Round-to-round spread 2–11% (other GPU clients share nodeC); XL and TF are alternated.

- **Re-run after the review fixes (run_all 12:52, ALL PASSED in 2 min 38 s; checks: units 18, E-U6 19, E.1 379 over
  115 TF cases, E.2 30, E.4 14, integrate 50, bench 579).** E.1 worst unchanged (1.14e-03, N=768 L=1); the new
  T = 65..128 cases 7.25–9.20e-04. E-U6: PAR bit-identical to the XL binary at every stage, NATIVE rejected in 9/9.
  E.4: T = 128 captured, 100/100 replays (worst 7.95e-04), two-layer graph 40/40 (worst 7.65e-04). Bench read-backs of
  every timed configuration within E.1 at every T (worst 7.3–8.0e-04). E.5 with T extended to 128:

  | T | distinct | bare eager XL / TF | × | bare graph XL / TF | × | apply eager XL / TF | × | apply graph XL / TF | × | TF GB/s |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 1 | 8.0 | 361.0 / 244.7 | 1.48 | 351.6 / 235.9 | 1.49 | 422.1 / 307.2 | 1.37 | 387.4 / 263.3 | 1.47 | 213.4 |
  | 2 | 15.6 | 546.8 / 447.3 | 1.22 | 546.2 / 429.0 | 1.27 | 617.4 / 514.2 | 1.20 | 575.5 / 458.2 | 1.26 | 228.6 |
  | 4 | 30.2 | 1014.5 / 828.5 | 1.22 | 994.1 / 811.3 | 1.23 | 1059.6 / 890.0 | 1.19 | 1018.2 / 837.2 | 1.22 | 234.6 |
  | 8 | 58.5 | 1840.2 / 1591.4 | 1.16 | 1852.3 / 1571.6 | 1.18 | 1905.2 / 1645.9 | 1.16 | 1858.2 / 1594.4 | 1.17 | 234.2 |
  | 16 | 105.2 | 3249.0 / 2859.3 | 1.14 | 3262.0 / 2841.5 | 1.15 | 3305.2 / 2916.1 | 1.13 | 3266.4 / 2857.6 | 1.14 | 233.0 |
  | 32 | 169.0 | 5138.9 / 4599.6 | 1.12 | 5143.1 / 4685.9 | 1.10 | 5228.5 / 4777.6 | 1.09 | 5195.1 / 4688.9 | 1.11 | 226.9 |
  | 64 | 241.2 | 7308.3 / 6672.1 | 1.10 | 7306.2 / 6736.5 | 1.08 | 7385.3 / 6820.3 | 1.08 | 7404.3 / 6722.0 | 1.10 | 225.2 |
  | 96 | 266.7 | 8057.2 / 7585.5 | 1.06 | 8096.8 / 7510.7 | 1.08 | 8192.2 / 7673.8 | 1.07 | 8194.2 / 7578.0 | 1.08 | 223.4 |
  | 128 | 280.0 | 8545.4 / 8114.8 | 1.05 | 8611.2 / 8120.1 | 1.06 | 8630.9 / 8211.5 | 1.05 | 8614.2 / 8119.9 | 1.06 | 216.9 |

  (A repeat run_all on the committed tree at 12:55, ALL PASSED in 2 min 36 s, gave graph-replay production-apply
  speedups 1.46, 1.25, 1.21, 1.16, 1.14, 1.11, 1.10, 1.08, 1.07 for T = 1 … 128: variation ≤ 0.02 vs this table.)
  T window with graph-replay production-apply speedup ≥ 1.05: {1, …, 128} (bench: "suggested TF_EXL3_TOKENS=1:128"),
  but T = 96..128 is only 1.05–1.08 (within ~2× of the 2–4% round spread): the per-pair 32-thread kernels (rot_in,
  gateup_epi, down_epi, OPT-2 not done) grow to 311 µs = 3.9% at T = 128 (0.9–1.5% at T = 8..64). Whether T > 64
  matters depends on the production T distribution (G6); if it does, re-measure there before choosing a window.

### Review fixes (2026-09-27, fixer; adversarially reviewed findings, each re-verified against the code first)
Every fix comes with a test that fails on the unfixed code. That was checked by re-introducing each bug on nodeC and
running the test (then restoring the file; `diff` clean).

| Finding | Verdict on re-check | Fix | Test (and the mutant it catches) |
|---|---|---|---|
| F1 / TA-2 (major): the load-time self-test was planned through the per-call `TF_EXL3_TOKENS` window (and fixed B ∈ {1, 8}); a window excluding 1 or 8, `TF_EXL3_MAX_PAIRS` < 64 or R < 8 made the self-test raise → `disable()` → TF off process-wide | real (traced `_plan` → `_selftest` → `_register` → `disable`) | `_plan(..., window=False)` for the self-test only; hard limits (B ≤ R, P ≤ P_cap, contract, registry) kept; self-test B ∈ {1, min(8, R, ⌊P_cap/topk⌋)}, topk = min(8, n, P_cap); a self-test call that is still not planned rejects only that layer; stricter temp-shape eligibility (the 4 temps exactly as plan() needs them). `TF_EXL3_TOKENS`: inclusive, forms `lo:hi`, `lo:`, `:hi`, bare `N` = `1:N` (was parsed as `N:∞`); malformed `TF_EXL3_TOKENS` / `TF_EXL3_MAX_PAIRS` / `TF_EXL3_TOPK_MAX` → `CFG.invalid`, install() refuses with the reason, register() rejects; configure() never raises | test_integrate C.4: windows 1:4, 2:64, 64, :8, 16: (layer registers with its self-test, TF enabled, plan() serves exactly the window incl. both edges, the first excluded B delegates with orig's result), MAX_PAIRS 32 / 4, R = 4 temps, 5 malformed values, install refusal. Mutant (self-test planned through the window): 6 checks fail |
| F2 (uncertain) / TA-5: E.1 derivation too optimistic; self-test used L = 1 (never used in production, noisiest); a NATIVE build passes E.1 and the self-test | real for the derivation and the discrimination gap; the "1.75× at production dims" figure was from N = 768 (production-dim margin is 2.0×) | §E.1 note (measured mechanism and margins); self-test L ∈ {production limit (`quant_method.moe.swiglu_limit` or 10.0), 0.0}; per-layer INFO log of the self-test rel_l2 and margin. New E-U6 `tests/test_stage_vs_xl.py`: every TF stage vs the XL **binary** on its own intermediates (C = 1 temps keep the last expert's xu / u0 / xd / d0; up tables pointed at the gate's so the surviving u0 is also g0; Z fed with XL's fp16 values + a sub-half-ulp perturbation so PAR's split-sum roundings are exercised) | E-U6: PAR bit-identical to XL for rot_in, gate/up epilogue (L = 10, 1, 0) and down epilogue (SK 1, 2); NATIVE rejected in all 9 cases (xd differs on 43–76% of values, out on 100%). GEMM-level: 26.9% of u0 values differ from XL (5.7% by > 1 ulp), rel_l2 2.7e-4 — the whole TF-vs-XL difference. G8 resolved |
| TA-1 (major): no test for B = 65..128 (P up to P_cap = 1024), although plan() serves it and STATUS said no window is needed | real (coverage gap on a reachable path; no bug found by reading) | E.1 + T ∈ {65, 96, 127, 128} (n = 288), T = 128 for n = 32, skewed (count = R) and sentinel at T = 128, and an assertion that P = P_cap / B = R ran; E.4 captures T = 128; E.5 + T ∈ {96, 128} | E.1 21 new cases pass (7.3–9.2e-4); E.4 T = 128 replays pass; E.5 results below |
| TA-3: Python sized the scratch Z from its own copies of the C++ tile constants | real (latent until a retune) | `kernels/exl3.cpp` exports `z_need(P, K, N)` (the pre-launch check's bound, now used by moe_forward itself) and `z_need_max(P_cap, K, N)` (max over every P ≤ P_cap); `Scratch` sizes Z from the extension; `GATEUP_CFG`/`DOWN_CFG`/`DOWN_SMALL` are taken from the loaded extension (`_sync_tile_cfg`, `use_ext`) | test_units: tile constants == extension's; Z ≥ z_need(P) for every P in 1..P_cap. Mutant (old Python formula with a stale split count): Z 4,194,304 < 8,388,608 → fails. E.1 T = 128 runs moe_forward at P = P_cap |
| TA-4 (uncertain): D.4 only checked the dispatcher, not the build hook or a registration through the plugin | test gap real; the failure needs outside interference | the probe also asserts the hook, builds a layer through production's `process_weights_after_loading` in the fresh interpreter (registered via the plugin's hook), runs the production apply through the pip-installed AOT module (TF path taken, E.1 vs orig). The dispatcher logs once when it is called while nothing is registered (with the disabled reason if TF was disabled) | D.4 probe. Mutant (plugin whose build hook is replaced after install): 2 checks fail; the inert warning appears in that run |
| TA-6: AOT==JIT silently skipped when the active extension is JIT | real | `harness.load_tf` asserts the active extension is the in-place AOT build under the repo; test_units asserts AOT before comparing with JIT and restores it with `use_ext` (EXT_SOURCE no longer left pointing at JIT); `tests/gpu_run.sh` defaults `TF_EXL3_JIT=0` (explicit JIT loads in tests are unaffected) | Mutant (`TF_EXL3_LOADER=jit`): test_units fails at load_tf |
| TA-7: bench docstring claimed every timed configuration was checked; only 2 eager bare calls were | real | all 12 sets checked eagerly (bare and production apply); after timing, the 12-call / 3-layer TF graphs (bare and apply) are replayed once more and every output compared with orig (E.1); E.4 adds a two-layer graph (n = 288 with sentinel ids, n = 32 through an EP map) with 20 fresh-routing replays; the hard-coded gate/up split count in the stage timing now comes from the extension | bench: every configuration within E.1 at every T; E.4 two-layer 40/40 |

Deviations from the spec introduced here (all documented in the spec body as "[Revised/Corrected/Added 2026-09-27]"):
C.3 step 5 self-test L ∈ {production limit, 0} (spec: {10, 1}) and batch sizes clamped to the layer's domain; a
not-planned self-test call rejects the layer instead of disabling TF; C.4 window forms and invalid-value handling; E.1/E.4/E.5
case lists extended to T = 128. No tolerance was changed.

### Open (see docs/STATUS.md for the production checklist)
G3 (real checkpoint weights: the per-layer self-test is the guard), G6 (production T distribution / capture sizes),
G10 (plugin + env on both TP ranks' containers), G11 (overlapped MoE applies on different streams would race on the
shared scratch — same assumption as production temps), G12 (weight relocation), G13 (model-level greedy / perplexity
on nodeA+nodeB), G14 (clamp-order product decision) are untouched by nodeC testing.

### Production module versions (2026-09-27 night)
Production installs the launcher's overlay exl3.py (df864b5, docs/ref/prod_live/overlay_exl3.py) over the image's
quantization/exl3.py at container start (patch_dense_fp8.py in the overlay chain; reproduced on nodeC from the 09-24
launcher snapshot, docs/logs/prod_live/chain_launcher_overlays.log). Differences that touch this fork and what changed
(full table in docs/STATUS.md):
- no MX path (`mx_enabled` absent) -> `integrate.install` feature-detects it; `REQUIRED_SYMBOLS` are checked up front;
  `_refuse()` logs "enabled (TF_EXL3_MOE) but NOT installed: <reason>" for every refusal (MX on, EXL3_FUSED_MOE=0,
  exl3_moe missing, missing symbols, invalid TF_EXL3_*, num_active detection change);
- `GLM53_EXL3_MOE_FAST=1` (overlay only): decode still calls `exllamav3_ext.exl3_moe`, which is then the native
  thin-decode dispatcher of a rebuilt extension; with gate/up suh shared, `_exl3_ptrs["up_suh"]` is the gate_suh tensor.
  TF replaces that function as well; `integrate.exl3_moe_kind` reports which kernel it replaced (INFO) and warns when
  FAST=1 is set without the thin-decode build, with an invalid value, or (third review) with a thin build whose
  `glm53_fast_moe_version()` is not 1 or raises;
- K2 re-implements production's decode branch, so it is installed only when `apply_exl3_fused_moe`,
  `map_topk_to_local` and `_exl3_moe_launch` have a verified fingerprint (sha256 of `ast.dump` of their source; identical
  in both known versions) and `MOE_ACT_SILU == 0`; otherwise the dispatcher path alone is installed, with a WARNING;
- `prod_identity()` names the module (sha256 in `KNOWN_PROD_MODULES`); an unknown file is served with a WARNING.
Tests: test_integrate D.1 (MX by version) and D.6 (module versions, FAST cases); both suites pass against both files.

### Review fixes (third review, 2026-09-27 night: compat / safety / evidence lenses)
Nine findings, two of them the same (compat-F2 = EV-1); every one re-checked against the code, the overlay and the logs
first, all real. Code fixes come with a test that fails when the fix is reverted: all four code mutants at once, run on
nodeC against both module versions, fail 7 (image) / 9 (overlay) checks of test_integrate and nothing else
(`docs/logs/review_0927c/mutant_test_integrate_{image,live}.log`). Doc fixes are guarded by `tests/check_docs.py`
(host, last step of run_all), which fails 20 checks on the reviewed commit 27fbd45 (`docs/logs/review_0927c/check_docs_on_27fbd45.log`).

| Finding | Re-check | Fix | Test |
|---|---|---|---|
| compat-F1: an exl3.py whose sha256 is not in `KNOWN_PROD_MODULES` could not reach an approved install: Phase 0 said "re-run and install", run_all's prod_module step fails for any unknown file, and Phase 2 rolls back on the resulting WARNING | real (the procedure had no step that adds the hash) | PRODUCTION_PLAN Phase 0 item 1: the certification procedure (file under docs/ref, `tests/prod_module_check.py` prints the exact `KNOWN_PROD_MODULES` line, both suites ALL PASSED with that bind and run_all without it, commit, build that commit; K2 fingerprints only after test_apply_fused passes). The check is a script now and prints the procedure; `integrate.py` documents it next to the table | nodeC: overlay + one comment line → prod_module exit 1 with the line to add; with the line added → prod_module and test_integrate pass, install names the label, 0 WARNINGs (`review_0927c/f1_*.log`). check_docs: a runbook that rolls back on "not a version …" must name `KNOWN_PROD_MODULES` |
| compat-F2 / EV-1: STATUS said FAST=1 needs no `TF_EXL3_TOKENS` window; the bench's own rule (§E.5, ≥ 1.05 on rand apply graph) printed 1:8 / 1:16 | real (a different criterion was used silently) | the rule is applied: re-run both thin comparisons (1:8 again; indep 1:12, first run 1:16; T = 12–16 sits at 1.045–1.065). PRODUCTION_PLAN Phase 0 table: FAST = 0 → no window (bench: 1:128), FAST = 1 → `1:8` (shared suh) / `1:12` (independent), the range both runs agree on; `TF_EXL3_TOKENS` added to the worker env list | check_docs: each cited E.5 log's printed window must be quoted in STATUS (and, for thin logs, PRODUCTION_PLAN) |
| compat-F3: `exl3_moe_kind` accepted any `glm53_fast_moe_version()` (or one that raises) as the thin build, with no WARNING, while the overlay refuses to load unless it returns 1 | real (overlay :1277-1278 → :2352) | version must be `THIN_DECODE_VERSION` = 1; otherwise WARNING "… will refuse to load (Unsupported native EXL3 decode-pipeline version)" (overlay) / "not the thin-decode build this fork was measured against" (image) | D.6: versions 2 and raising → WARNING; on the overlay, production's own `process_weights_after_loading` refuses exactly for absent / 2 / raising and install() warns "refuse to load" exactly then (4 cases agree). Mutant: 2 checks (image) / 4 (overlay) fail |
| compat-F4: production's `build_exl3_fused_state` failing for a layer (caught by production → python loop) left TF silent: install said "installed", the inert WARNING never fires | real (both module versions catch the exception) | the build hook catches, logs a WARNING (once per distinct exception; `prod_build_failed` counted in the summary) and re-raises unchanged | D.7: `exl3_moe_max_concurrency` made to raise → production falls back (no pointer tables, load not raised), layer not registered, one WARNING naming the python loop. Mutant: fails |
| safety-F1: a rank without `TF_EXL3_MOE` (worker env list, `VLLM_PLUGINS`) stays silently on production; Phase 2 checked TF's lines on one rank only | real (by design install() is silent when disabled; the runbook qualified only the resample line with 両 rank) | `plugin_register` logs `tf_exl3_moe plugin loaded (pid …): TF_EXL3_MOE=… -> installing \| off` in every process that loads the plugin (install() itself stays silent); the registered line carries rank and pid; PRODUCTION_PLAN Phase 2 checks every TF line on both ranks (nodeB's container for rank 1) and treats a missing line as a rollback | D.4 fresh interpreters: unset → exactly one "… -> off" line, on → "… -> installing". Mutant: fails. check_docs: Phase 2 names rank 1 / nodeB |
| safety-F2: no non-KV memory gate: Phase 2 had none, Phase 4's "KV capacity must not shrink" cannot fail while `--kv-cache-memory` pins the pool | real (the gate is vacuous whenever `ENFORCE_EAGER` ≠ 1) | Phase 1 records `Model loading took`, `Graph capturing … took` and host `MemAvailable` after a soak on both nodes; Phases 2–4 compare against it (DRAFTER_IMPL step 4's method); the KV line is used only with `ENFORCE_EAGER=1`. TF's expected cost measured: `tests/probe_tf_memory.py` — +50.0 MiB per rank (the shared scratch, first layer only), +0 per further layer, transient ≤ 1.5 MiB, device code ≤ 12.7 MiB outside the allocator | probe log `review_0927c/probe_tf_memory_image.log` (asserts no per-layer growth). check_docs: Phase 2 has `Model loading took` and `MemAvailable` |
| safety-F3: logs could not tell "TF serves decode" from "TF registered but delegates everything" (plan misses were silent, counters never logged) | real | `plan()` / `plan_apply()` record the reason of every miss ("<category>: <detail>"); calls counted by category (b_gt_r, window, pairs = by design; unregistered, tf_disabled, k2_off, tf_error, contract); a `contract` hand-off (or `unregistered` while other layers are registered) → WARNING once per reason; INFO summary per rank at 10^k calls and 10 s after the last CUDA-graph capture (daemon thread, no CUDA calls) with the batch sizes captured with TF and with production. Python runs only at capture and eagerly, so the capture-time outcome per B is what decode replays | D.8: contract hand-off 2 calls → production's result, counted 2, 1 WARNING; B = R + 1 → no WARNING; 10^k summary; a captured graph with one TF and one hand-off → summary `B=8` / `contract B=8`, replay correct. Mutant (no accounting): 3 checks fail |
| EV-3: the thin E.5 could not be reproduced (the .so and the launcher tree in other sessions' /tmp, no command in the log although STATUS said so, no evidence the thin kernel ran; `thin_calls` counts the prefill path) | real | `tests/run_thin_ab.sh shared\|indep` (refuses a .so with another sha256; log header: command, git HEAD, image ID, both binds with sha256); `tests/probe_native_kernel.py` lists the kernels one production exl3_moe call launches (profiler) and must see `glm53_exl3_moe_fast_kernel<4, 256, true\|false>` under FAST=1 (and `exl3_moe_kernel<4, 256>` for the stock run_all step `native_kernel`); .so, sources and build recipe moved to `artifacts/` (git-ignored), sha256s and provenance in `docs/ref/launcher_0924/PROVENANCE.txt`, patch scripts committed there; `chain_launcher_overlays.sh` records the env file's path, sha256 and variable names (never values) | both thin logs regenerated through the script (first run kept under `prod_live/history/`); check_docs: line 1 is the command, two binds with sha256, the probe line with the right `shared_input` |

Not changed: the 09-24 chain log still does not name its env file (it predates the recording; the file is not in this
repo); whether nodeA/nodeB's `exllamav3_ext.thin.so` is the measured build stays a Phase 0 read.

### Optimization (branch `opt`, 2026-09-27) — supersedes parts of the launch description above
Measured record: docs/OPTIMIZATION.md. What changed relative to the implementation notes:
- grouped GEMV grid is (N/64, S_cap, mats·SK) (n block fastest, K1; falls back to (S_cap, N/64, mats·SK) when
  S_cap > 65535, keeping evict_first: the ORD 0 + evict_first instance exists since the `opt` review, and moe_forward /
  moe_forward_ids check before their first launch that both grouped launches resolve to a compiled instance); weight loads
  carry an L2 evict_first policy (K3); same arithmetic, Z bit-identical.
- split counts come from a table in P (K6/C2, `kernels/exl3.cpp` `kSkTables`, table 6 shipped): gate/up 8 at
  P ≤ 32, 4 up to P = 512, 2 above; down 2 at P ≤ 192, else 1. z_need is the maximum over every table.
- the forward paths drop dead intermediates from L2 after their last read (K3b, `discard.global.L2` in the two
  epilogues); the stage entry points never do.
- production's decode apply is served from the router ids (K2): `route_ids` → rot_in (bf16 x) → … via
  `moe_forward_ids`, installed as a hook on `apply_exl3_fused_moe` by `integrate.install` (`TF_EXL3_APPLY`, default
  on), with its own load-time self-test.
- `ext.set_variant(id)` selects kernel variants for tests / A/B only (0 = shipped, 1 = the master kernels).
- the second review of branch `opt` (2026-09-27): the ORD 0 fallback fixed (above), E.4 on all three serving paths,
  bitwise tests of the gate/up SK 2 branch and its K3b discards, logs committed under `docs/logs/`, speed tables
  generated from them (`tests/doc_tables.py`); findings, re-checks and mutants in docs/OPTIMIZATION.md "Review fixes".
