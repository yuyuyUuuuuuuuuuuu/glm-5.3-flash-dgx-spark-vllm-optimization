# KDA_FLASHKDA3 — adversarial review of fkda2 / r16o, and the next cuts (branch `fkda3`)

Status 2026-10-02, branch `fkda3` (from `fkda2` ef3f88d). nodeC only; nodeA/nodeB untouched; no kit built or installed.
Evidence: `${HOME}/tf-exl3-assets/fkda3/` (raw) and `docs/logs/fkda3/` (copies).

## Part A — trying to refute fkda2 and r16o

### A.1 Root cause and accuracy: CONFIRMED, with two residual defects fkda2 missed

Independent rig `tests/fkda3/adv_accuracy.py`: my own fp64 recurrence (state kept as `h[K,V]`, different code and
layout from fkda2's reference), the state convention of every backend DETECTED on a random non-symmetric initial state
(all three read `[V,K]`), new seeds/inputs. 115 cases: edge lengths T = 1…4097 (T<16, T<64, non-multiples of 16/64),
zero vs random initial state, varlen batches with length-1 rows and mixed initial states, four gate regimes (real
A_log/dt_bias, gates pinned at the lower bound, long memory, mixed), v×16, saturated beta ±8, large state, the V-split
path (H=16), chained 6×2304 vs 1×13,824, 40 real mini-model calls (`adv_all3.log`):

| geomean ratio (fraction of cases better) | output | final state | worst token |
|---|---|---|---|
| fkda2 / stock (r16n) | 0.660 (100 %) | 0.665 (100 %) | 0.659 (97 %), worst 2.08 |
| fkda2 / Triton (R15) | 0.879 (96 %) | 0.992 (61 %), worst 7.9 | 0.889 (83 %), worst 3.53 |
| **fkda3 / Triton** | **0.874 (96 %)** | **0.758 (98 %), worst 1.04** | **0.871 (84 %), worst 1.21** |

Speed parity (`bench_ops_ab.py`, interleaved, production shapes): stock / fkda2 / fkda3 within ±0.05 ms per layer at
13,824 / 4,608 / 1,791 tokens. SASS: v_stock == the r16n-shipped c286213f build, the r16o .so == v_fix (identical
`cuobjdump -sass`), so "every difference is the patch" holds.

Defects found (both also in r16o):

1. **Subnormal flush in FKDA2_FP32_DECAY** (`tests/fkda2/patch_flashkda_precision.py:72-73`): under `--use_fast_math`
   the fp32 products `qf*ef*scale`, `kf*ef` flush subnormal results; the stock bf16-operator path kept bf16
   subnormals. In a 16-token tile whose gates all sit at the lower bound (in-tile log2 decay < ≈ −112) the tile's last
   row loses its small decayed-q/k elements: tile-position-15 error 1.10e-2 vs 3.8e-3 elsewhere (stock 5.4e-3, Triton
   4.0e-3) (`decay_sweep.py`, attribution `fast_decay_diag.py`: FP32_DECAY alone reproduces it). Real-input
   prevalence 6.3e-5 of (tile, head, channel) (max 2.6e-4 in one call; most negative tile −113.5).
2. **"fp32 recurrent state" is fp32 only inside a call.** FlashKDA 17a037d (stock and fkda2) converts the fp32 initial
   state to bf16 before seeding the fp32 registers (`csrc/smxx/fwd_kernel2.cuh:383`) and stores the final state via a
   bf16 narrowing (`:833`, `:956`): `state_identity_probe.py` — with no write and no decay, final == bf16(initial)
   exactly (1.66e-3 rel), every call. In production every 13,824-token chunk boundary and the prefill→decode hand-off
   carry a bf16-rounded state; the Triton chain carries it exactly. fkda2's doc lists this as "inherent"; it is not
   (fixed below, speed-neutral).

The KL-vs-R15 argument of fkda2 §2 stands (the probe measures distance from Triton's rounding; nodeC cannot rank the
real model).

### A.2 Extension-sha pin

A mismatching `.so` raises `RuntimeError` in `Glm5NextLinearAttention.__init__` (`overlay/glm53_flashkda.py:93`):
fail-closed, the engine does not boot, no fallback to Triton. It cannot trigger from a consistent kit (both files ship
together, MANIFEST-checked, fresh containers each start). Note: the pin binds the artifact, not the source — a rebuild
from the documented recipe gives a different sha (SASS and host code identical, ~416k bytes of non-code sections
differ: `v_fix_re`), so a rebuilt `.so` is refused by design.

### A.3 r16o kit

* Payload delta vs r16n: exactly `overlay/_flashkda_fp32_C.abi3.so` + `overlay/glm53_flashkda.py` (`diff -r`), MANIFEST
  ok, `prev-r16/MANIFEST.sha256` ⊂ r16n MANIFEST (45 lines), both start.sh variants byte-identical.
* **Doc defect, revert section** (`docs/DEPLOY_R16O.md:96-107`): step 2 installs the PLAIN start.sh `811728b2…` before
  `apply_r16.sh`, and `apply_r16.sh` backs up the live start.sh (`tools/apply_r16.sh:147`) — so the backup holds the
  plain one, and `revert_r16.sh` restores it. The doc's "expect f9574201 … the backed-up start.sh IS the r16n production
  one / no start.sh dance" is wrong: after a revert the operator must put production's start.sh variant back (as step 5).
* Compose-B (second bundle pass refuses): pre-existing r16n behaviour (r16o ships r16n's `patch_flashkda.py`), fixed by
  combo `79477a1` (in r16x/r16y; cherry-picked here). Not reachable in production: start.sh always `docker rm -f` +
  `docker run -d` with no `--restart` policy (start.sh), so the bundle runs on a
  pristine image filesystem; only a manual `docker restart`/`start` of an existing container hits it, and it fails
  closed (SystemExit, boot_checks MISS), never silently.
* Applicability: the procedure requires the live start.sh to be the r16n production variant (`f9574201`); production is
  currently running the r16x A/B (start.sh `ffa4d898`). If r16x is kept, the same two files ship in r16y (same sha
  `dd1788c2…` .so and `c6b4ee63…` wrapper, verified).

## Part B — further optimization

### B.0 Where the time goes (T = 13,824, H = 32/rank, `prof_kda_prefill.py`)

K1 (prepare) 3.35 ms + K2 (recurrence) 3.27 ms = 6.6 ms per layer, plus kda.py's merge copy of the output 1.06 ms.
Both kernels are DRAM-bound (~1.3 GB per layer per call at ~200 GB/s), not latency- or tile-config-bound:
K1 runs at full speed with only 16 of 48 SMs free (`k1_fewsm_probe.py`: 3.29 ms); K2 time scales with H at ~180 GB/s
(`k2_heads_probe.py`); running an independent K1 and K2 concurrently gives no gain (`overlap_probe.py`); K2 runs on 32
SMs (V-split cannot fit 64 blocks at 2/SM on sm_121). So tile configs do not help; only removing bytes does.

### B.1 Direct output (commit 02c596a) — −1.0 ms per layer, bit-identical

The patched kda.py passes `out=core_attn_out[:, :num_actual_tokens]` on steps without spec tokens; the wrapper writes
FlashKDA's output there (validated: bf16, `[1,T,H,D]`, contiguous, 16-B aligned, else the workspace as before), and
kda.py's merge statement becomes a same-storage copy that torch skips (no kernel). `direct_out_unit.py` 35/35:
bitwise equal to the workspace path for T = 1…13,824 and varlen; merge launches nothing; fallbacks work.
**Saving per 13,824-token chunk 34.2 ms** (1.005 ms × 34 layers; 4,608: 11.6 ms; 1,791: 4.1 ms).
Changes the patched `_forward` fingerprint to `eb0e8dedaee6deb6` (quickwins allowlist extended dynamically; kda_conv
transplant E2E ok: `check_kda_conv_install.sh`).

### B.2 fkda3 precision build (02c596a) — speed-neutral, fixes A.1 defects 1 and 2

`tests/fkda3/patch_flashkda_fkda3.py` on top of fkda2: `FKDA3_DECAY_NOFTZ` (decay products via `mul.rn.f32`, keeps
subnormals; normal values round identically) and `FKDA3_FP32_STATE_IO` (resident fp32 state seeded from the fp32 initial
state and written from the fp32 registers; the shared fp32 buffer is a union with the pipeline stages, ordered by named
barriers 1/2). Both 0 → SASS identical to fkda2. Production instantiation 226 → 250 registers, no spill, 1 block/SM as
before. Pinned in the wrapper as `d98adc4aec0046a2…` (`= fkda3 precision build (fp32 decay/u/out, no-ftz decay, fp32
state io)`). Results: carried state exact (identity probe 0), state vs Triton at T=1 4.3e-6 (fkda2 2.3e-3), chained
8×13,824 0.77× Triton's distance from fp64 (fkda2 0.86-0.88×), tile-end error 3.86e-3 at full saturation (fkda2 1.10e-2),
table in A.1. Existing rigs ALL OK on it (`wrapper_unit`, `wrapper_pin` (refuses the fkda2 .so), `wrapper_short_varlen`).

### B.3 Tried, not recommended

* K1→K2 pipeline (`patch_flashkda_fkda3_pipeline.py`, opt-in env, experiment build only): bit-identical (300-call race
  stress) but only −0.15…−0.18 ms per layer (≈ −6 ms per chunk): K1 runs ahead on the free SMs and K2's workspace reads
  still come from DRAM. Throttling K1 to K2 is deadlock-free only inside one co-resident (cooperative) launch. Found a
  real hazard on the way: under CUDA lazy loading, the first K1 launch while K2 spins waits for K2 → deadlock (trap
  after the spin limit); force-loading K1 first fixes it. Any future cross-kernel spin design must preload.
* Segmenting a call into 256-token pieces to keep the workspace in L2 (`seg_probe.py`): −0.57 ms per layer at best,
  eaten by the 4 MB fp32 state round trip per segment.

### B.4 What is left (estimates from the byte model, not implemented)

* One cooperative kernel with K1 throttled behind K2 (workspace L2-resident): ≈ −1.5…−2 ms per layer (−50…−70 ms per
  chunk).
* q/k short-conv fused into K1 (conv output never written/read): ≈ −2.3 ms per layer (−78 ms per chunk); entangled with
  the quickwins kda_conv transplant and conv_state semantics.

## Production A/B checklist for an fkda3 kit (r16o/r16y + 3 files: wrapper, .so, patch_flashkda.py)

* boot: `extension sha256 d98adc4aec0046a2 = fkda3 precision build (...)` on both ranks; `kda_conv VERIFIED +
  eb0e8dedaee6deb6`; never `is not the fkda3 precision build`; test fixtures carrying 9715f9b548cfa694
  (`tools/deploy16/test_boot_checks.sh:390`) must be updated in the kit branch.
* speed: real-text / random-24k prefill ≈ +0.5 % vs r16o (−34 ms per 13,824-token chunk); decode unchanged.
* quality: long-context KL and decode-vs-prefill KL (the state now crosses chunk boundaries and the prefill→decode
  hand-off exactly) — expect ≤ r16o; "quality short vs R15" ≈ r16o (single-call prompts are unaffected by the state fix).
* spec-decode mixed steps take the workspace path (`out=None`); pure prefill steps the direct path.

## Real-engine check (handoff mini model, real GLM-5.3-Flash KDA weights, r16o kit staging, QUICKWINS=all + MLA_PREFILL=1)

`tests/handoff/run.sh` runs `f3_on`, `f3_on_b` (repeat), `f3_on_nodirect` (`GLM53_FKDA3_DIRECT_OUT=0`, harness-only
knob), `f3_off` (Triton). Boot: `kda_conv VERIFIED + eb0e8dedaee6deb6`, the fkda3 .so installed, shadow control
IDENTICAL in every run. Two identical-config runs (`f3_on` vs `f3_on_b`) already differ in 16 indexer k_cache rows from
step 0 (engine run-to-run nondeterminism in the MLA indexer, not KDA), and `f3_on` vs `f3_on_nodirect` shows exactly the
same pattern — so B.1's bit-identity is proven at unit level (`direct_out_unit.py` D.1/D.4), the engine shows no extra
difference. Greedy tokens identical to the Triton arm (24/24). Decode-vs-fresh-prefill consistency KL (n = 8 positions,
noisy): fkda3 0.0012 / 0.0013 / 0.0014 (direct, repeat, no-direct), Triton arm 0.0152, fkda2 (fkda2 §3) 0.0153.
