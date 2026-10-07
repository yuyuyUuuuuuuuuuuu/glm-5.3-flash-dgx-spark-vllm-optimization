# MHC_SP: sequence-parallel prefill for the mHC bookkeeping over TP (r16msp, default OFF)

`GLM53_MHC_SP=1` installs `overlay/patch_mhc_sp.py` (operator switch, env.r16 `#switch`, the r16msp start.sh
forwards it to both ranks). Unset/empty/`0` = the patch does not run = production byte-identical. Decode is
byte-identical in every state.

## 1. The cost it removes

One 13,824-token prefill chunk spends **881.6 ms** of rank-0 GPU kernel time in the mHC family
(prefill3000/step0/ANATOMY.md 2.3: `mhc_post_tilelang` 389.8 + `mhc_pre_big_fuse_with_norm` 267.6 +
`sm120_tf32_hc_prenorm_gemm` 195.8 + `qw_post_mean` 28.4), 12.1 % of the 7.3 s step. Every mHC op is per-token
(per-token mixing matrices, sinkhorn, the 4-stream residual bookkeeping, a prenorm GEMM whose reduction is along
the hidden dim), so at TP=2 both ranks compute all of it twice on the same 13,824 tokens.

## 2. What the feature does (Kindling's per-forward switch, not the image's EP-style `is_sequence_parallel`)

For a forward of **>= 1024 tokens** (`MHC_SP_MIN_TOKENS`; decode and every CUDA-graph capture fail the gate, and
TP=1 / PP>1 / a non-mHC layer disable it) each rank keeps only its token shard of the residual stream:

```
embed all-reduce (unchanged) -> sp_shard -> per layer:
    hc_fused_post_pre (T/2) -> sp_all_gather -> attention (full T, o_proj.reduce_results off)
                            -> sp_reduce_scatter (reduces the rank partial AND shards)
    hc_fused_post_pre (T/2) -> sp_all_gather -> MLP/MoE (full T; the MoE runner's late all-reduce
                            (`moe_config.skip_final_all_reduce`) and the dense down_proj reduce are off)
                            -> sp_reduce_scatter
final layer: hc_post + hc_contract (T/2) -> sp_all_gather -> final norm / lm_head (unchanged)
aux (DFlash2 drafter) layers: sp_all_gather + clone of the contracted value (the drafter reads full T)
```

At 2 ranks reduce-scatter + all-gather move exactly the bytes of the all-reduce they replace
(2 x (N-1)/N x S = S vs 2(N-1)/N x S = S), so 91 x 113.2 MB all-reduces become 182 x 56.6 MB collectives and the
wire is neutral; the gain is the mHC compute that stops being replicated. The toggles are per-forward
(`Glm5NextModel._mhc_sp_begin`): `o_proj.reduce_results`, the dense `down_proj.reduce_results` and
`moe_config.skip_final_all_reduce` are read by their owners at every call, so flipping them between a prefill and
a decode step is safe; the flip decision is per-model.

The switch needs KDA/MLA to see the full sequence and it does: they run after the all-gather, on the full batch,
with the unchanged attention code (and the unchanged indexer, KV writes and drafter inputs).

## 3. Measured on nodeC (one GB10, 40 GiB cap; `tests/mhc_sp/bench_mhc_halving.py`, `tests/mhc_sp/test_mhc_sp_all.py`)

| check | result |
|---|---|
| halving (production op chain, T=13,824 vs 6,912) | fused 9,946 -> 5,016 us/call (**ratio 0.514**); per-chunk mHC 913 -> 470 ms, **-444 ms/chunk** (the anatomy's production 881.6 ms -> ~453 ms); kill gate "gross <= 0.44 s/chunk" met |
| shard numerics, T = 13,824 / 6,912 | **BITWISE identical** to the full-T run for residual_cur, post_mix, comb_mix, layer_input (both halves); `compute_num_split` = 1 on both sides |
| shard numerics, T = 1,536 | split_k = 2 -> fp32-order differences only: <= 4e-5 abs on the fp32 mixes, 1 bf16 ulp on rare layer_input elements (the TP path is unchanged) |
| 2-rank emulation of the patched model (4 layers, MoE + dense, quickwins mhc_aux/mhc_mean live) | every layer's inputs and (x, residual, post, comb), the aux hidden states and the final hidden states are **bitwise equal** to the TP path on both ranks; 10 all-gather / 8 reduce-scatter pairs per forward; T=8 decode runs the TP path with zero SP collectives and the reduction flags round-trip |

Communication estimate (no second GPU on nodeC: NOT RUN as a measurement): at 2 ranks RS and AG each move
(N-1)/N x S = 56.6 MB per rank; the measured production all-reduce of 113.2 MB is 5.40 ms (21 GB/s payload,
168 Gb/s), and a ring RS or AG of half the tensor does half the work of the ring all-reduce, so RS+AG ~= AR =
5.4 ms per pair of collectives: wire-neutral to first order, with 182 launch boundaries instead of 91 (+2-5 us
each, <= 1 ms/chunk). The 5 aux gathers add ~5 x 2.7 ms = ~14 ms/chunk. Net estimate **~-0.42 s per full
13,824-token chunk** (kill-test bound -0.25 s; the tail chunk of a request stays on the TP path: 3,925 tokens is
odd, and the switch requires T % tp == 0).

## 4. What the patch edits (all preflighted, idempotent, fail-closed)

1. `vllm/models/glm5next/nvidia/model.py` — the module-level switch and the SP branches in
   `Glm5NextDecoderLayer.forward` and `Glm5NextModel.forward` (see the patch header for the hunks).
2. `glm53_prefill_quickwins.py` — `VERIFIED` gains the fingerprints of the two SP-patched forwards (the mhc_aux /
   mhc_mean anchors are untouched, so the transplant still applies).
3. `glm53_moeglue.py` — `WARM_VERIFIED["Glm5NextDecoderLayer.forward"]` gains the SP-patched fingerprints (plain
   and quickwins-transplanted); without them the decode L2 warm silently disarms on its drift check. The warm
   itself is decode-only (<= 64 tokens) and never sees SP.

`tests/mhc_sp/test_mhc_sp_all.py` proves 2. and 3. against the live image's functions (the transplant runs on the
patched source) and runs the 2-rank emulation of item 3's table.

## 5. Rollout (r16msp kit)

1. `PREV_KIT=${HOME}/tf-exl3-deploy16.r16l tools/deploy16/make_kit.sh ${HOME}/tf-exl3-deploy16.r16msp`
   (the kit's `tools/off_equals_prev.sh` must show `GLM53_MHC_SP=0 == previous kit` and the =1 delta limited to
   the three files of section 4).
2. `tools/apply_r16.sh --apply` (reports `operator switch GLM53_MHC_SP: unset (= 0); not added by this script`),
   restart both ranks, `tools/boot_checks.sh`: unset -> both ranks print the bundle's
   `[glm53-tf-bundle] patch_mhc_sp.py: GLM53_MHC_SP unset -> skipped (stock)`.
3. Enable on an idle window: `tools/env_r16.sh on mhcsp` (refuses unless the installed start.sh forwards the knob
   to both ranks, the overlay file and the bundle entry are present) and restart both ranks together. Both ranks
   must print `[glm53-mhc-sp] vllm/models/glm5next/nvidia/model.py: patched;` — boot_checks B.15 checks exactly
   that plus the mhcsp summary line. Watch the first prefill's log for
   `glm53-mhc-sp: sequence-parallel mHC prefill ACTIVE (T=...)`.
4. Yardsticks on the owner's idle window (never contended): the gate 32k/128k x4 seeds and a real-text prompt,
   both arms, after memprep; plus the decode yardstick (the drafter is untouched, but the claim "decode unchanged"
   is checked by `boot_checks --after-traffic` and the FULL/PIECEWISE graph census at boot).
5. Revert: `tools/env_r16.sh off mhcsp` + restart both ranks; the switch line disappears, the compose is again
   byte-identical to the previous kit's (off_equals_prev S1). Full revert: ROLLOUT.md.

## 6. Risks / limits (read before enabling)

- **TP=2 engine run NOT RUN** (nodeC has one GPU; NCCL cannot put two ranks on one device; nodeA/nodeB are out of
  bounds). Proven instead: the real mHC kernels bitwise on shards, RS/AG == AR at N=2 by construction, and the
  patched wiring in a single-process 2-rank emulation of the patched forwards. The first production prefill after
  enabling must be watched for the ACTIVE line on both ranks and for NCCL errors.
- The MoE's `moe_config.skip_final_all_reduce` is read at every runner call, so the toggle cannot desynchronize a
  captured graph; but a **mixed prefill+decode step** (GLM53_MIXED_PREFILL_CHUNK) with >= 1,024 scheduled tokens
  takes the SP path for the whole step — correct by construction (attention is gathered; the mHC bookkeeping is
  per-token), and unmeasured.
- A step whose token count is not divisible by tp_size stays on the TP path (no padding: attention metadata is
  built for the exact batch). Odd tail chunks therefore get no saving.
- If `Glm5NextMoE`'s runner one day reduces its output early (`_fused_output_is_reduced`), the
  `skip_final_all_reduce` assertion fires loudly rather than double-reducing; that would be a boot-time error on
  the first prefill, not a silent wrong answer.
- The quickwins / moeglue fingerprint tables are extended by the patch. If either file drifts, the patch prints a
  `WARNING` and continues; the victim is a runtime WARNING + a disarmed quickwins item / warm, never a wrong
  output. `check_boot_strings.py` verifies the shipped code can print everything boot_checks greps for.
