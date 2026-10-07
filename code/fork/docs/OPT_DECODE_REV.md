# OPT_DECODE_REV — adversarial review of branch `opt-decode` (c2e1f3c, from r16z2rev 60b53bf)

Branch `opt-decode-rev` (this file). nodeC only, production image, every GPU job through `flock /tmp/tf-gpu-bench.lock`
(`tests/gpu_run.sh` / `tests/handoff/run.sh`). Production (nodeA/nodeB) was not touched or read. Logs:
`docs/logs/opt_decode_rev/`.

## 1. Verdicts

| claim (opt-decode) | verdict | evidence (this review) |
|---|---|---|
| KDA_LAZY verify output + committed state bitwise == production's strided kernel | **confirmed** (arithmetic) | T1-T9 re-run ALL OK (`test_kda_lazy.log`); engine runs below: 0 self-check mismatches |
| speed -1.39 / -1.54 / -2.28 ms (M=5/6/8, 1 req), -4.44 (M=8 x 2) | **confirmed on the nodeC step model** | A/A-controlled re-run (`bench_aa.log`, 4 graphs per process, random order per round): A/B **+1.19 / +1.62 / +2.49 ms** (M=5/6/8), **+4.22** (M=8 x 2); A/A medians within ±0.13 ms (±0.6 at 2 req) |
| ~-1.6 ms/step at production's K mix | **confirmed arithmetic** (0.5·1.19 + 0.3·1.62 + 0.2·2.49 = 1.58 ms); TP=2 / RoCE not measurable here |
| self-check = "the integration oracle" | **overstated** | it recomputes the state from the rows the lazy verify saved; it proves the commit ARITHMETIC, not that the saved rows / slot ids are the current step's. Integration evidence is the engine A/A below |
| repair mode "exact but no gain" | **refuted (understated)** | repair mode costs **~6.9-7.0 ms per commit** vs 0.65 ms fast (`commit_modes.log`): one mismatch turns the feature into a ~-5 ms/step regression for the life of the process |
| self-check cost | measured | a checked step costs ~19.3 ms (first 64 commits = ~1.2 s once after boot, then 1/1024 steps ≈ 0.02 ms/step average; both TP ranks check the same commit index) |
| hostloop compatibility | **static: compatible; engine: see §3** | hostloop does not reorder sampling / postprocess_state / drafter; its snapshot event is recorded after postprocess_sampled, i.e. after the commit (the event fires ~0.64 ms/request later; the drafter still covers it) |
| VTRIM_STATS: outputs untouched, 0.35-0.46 ms/step | **confirmed** | test re-run ALL OK; cost re-measured 0.351 / 0.344 / 0.458 ms (R = 1 / 4 / 8) (`vtrim.log`); q = softmax(draft_logits / T) matches what the image's rejection kernels use (no top-p / penalty on q) |
| trimming upper bound +8-30 % | **partially confirmed (upper end overstated)** | the 8 ms/row end is the 0922 sweep with old kernels; today's production (structured M=8 82.5 ms vs prose ~72.5 ms at mean M≈5.8) and the TF corr40 model (~2.5 ms/row) give ~2.5-4.6 ms/row -> oracle bound ≈ +7-14 %, a real rule a fraction of that |
| trimming exactness condition (decide row i from q_i and d_<i, not d_i) | **confirmed by argument** (stop-symbol drafter: q'_i = δ_STOP when the rule stops, p(STOP) = 0 -> block and token-wise verification of q' are exact); vLLM's -1 placeholder handling in the block kernel NOT tested here |

## 2. Static review of the wiring (image vLLM, `${HOME}/tf-exl3-assets/vllm-src`)

* Every reader of a KDA state column after a step is covered by the commit: the next verify's initial column
  (`num_accepted - 1` = A-1), the align pre-copy (`precopy_mamba_align_fused_kernel`: block_table[old state_idx +
  (num_accepted - 1)] = column A-1 of the previous step), the align post-copy (`postprocess_mamba_fused_kernel`:
  column `aligned - (nc - A + 1)` when `aligned >= nc - A + 1`, the commit kernel uses the same predicate and the
  same post-step `num_computed_tokens`), and the src==dest case (the copy into column 0 runs after the commit). In align
  mode `mamba_get_block_table_tensor` gives column t = block_table[(seq_len-1)//BS + t], the same base as
  `mamba_state_idx` in postprocess, so "column bias" is the copy source.
* Dummy / capture runs never call postprocess_state, but `get_dummy_block_tables` zeroes the block tables, so the
  verify writes flag = 0 (state_idx <= 0): no stale flag survives a dummy run.
* Mixed batches (prefill + spec) and plain decodes take production's function; the fallback clears the layer's flags.
* Production's launcher for KDA uses BV = 8, num_warps = 1, num_stages = 3 — the same as the lazy kernels.
* The DFlash2 drafter has no KDA modules (`not wired on DFlash2Qwen3ForCausalLM`), so the scratch is allocated once.

## 3. Engine runs (handoff rig, mini model, production decode knobs)

Config for all: R16 ship knobs + `GLM53_DEC_HOSTLOOP_WAKE=auto GLM53_REJECTION_METHOD=block GLM53_KDA_STRIDED_QKV=1
GLM53_KDA_FLASHKDA=1 HANDOFF_TEMPERATURE=1.0`, prompts 4590,3001 (batched) then 9200, 4600, 400 tokens each.

| run | KDA_LAZY | hostloop fast path | acceptance | self-checked commits / states (A>1) | mismatches |
|---|---|---|---|---|---|
| off, off2 | off | NOT installed (rig, see below) | real (mini drafter: ~1 token/step) | - | - |
| lazychk | on, every commit checked | NOT installed | real | 1207 / 8005 (0) | 0 |
| lazydef | on, first 64 then 1/16 | NOT installed | real | 136 of 1207 / 1065 (0) | 0 |
| hloff | off | **ON** + wake auto | real | - | - |
| hllazy | on, first 64 then 1/16 | **ON** + wake auto | real | 136 of 1207 / 1065 (0) | 0 |
| hlsyn | on, every commit checked | **ON** + wake auto | synthetic 0.9..0.5 (block kernel) | 232 / 1505 (**1345**) | 0 |

End-to-end (tokens and top-k logprobs at temperature 1.0, `handoff_compare.txt`): the rig is not deterministic across
processes, so the comparison needs the A/A control. First token divergence per request (4 requests, 400 tokens):
**A/A off vs off2: 282 / 57 / 6 / 256**; off vs lazychk: 192 / 22 / 350 / 240; off vs lazydef: 252 / 86 / 168 / none
(identical); hloff vs hllazy: 59 / 69 / 19 / 100; A/A hloff vs off: 59 / 57 / 6 / 13. Lazy is inside the A/A band;
max |dlogprob| before the first divergence is of the same order in A/A (0.06-0.27) and A/B (0.07-0.37; one 1.03 outlier
in lazychk, request 0, where the A/A of the same request reaches 0.27 — the rig's sampling noise near a near-tie; the
self-check confirmed every committed state of that run byte for byte).

Rig fix (this branch): `tests/handoff/run_engine.py` wraps `GPUModelRunner.execute_model`, so `glm53_hostloop`'s
source-fingerprint check saw the harness wrapper and refused to install ("unverified vLLM source:
GPUModelRunner.execute_model=029b0b50a9f13ce9") — the opt-decode engine runs therefore never had hostloop on.
`HANDOFF_HOSTLOOP_UNWRAP=1` (opt-in, default unchanged) exposes the original through `_glm53_orig`; with it hostloop
installs, plans from the post-sampling snapshot and stays ON for the whole run together with KDA_LAZY (no
"switched OFF", no self-check difference).

Not covered on nodeC: TP=2 (the lazy path has no collective; both ranks take the same decisions from identical
metadata and check the same commit indices), the full model's 34 layers in the engine (the mini has 5 KDA layers; the
34-layer kernels are covered by the unit tests and the benches), real high acceptance with the real drafter (the mini
drafter accepts ~nothing; hlsyn covers many-row commits through the block kernel's synthetic path).

## 4. Kit integration needed (in addition to docs/DEC_KDA_LAZY.md §5)

1. start.sh knob loop + `-e NAME="${NAME:-}"` for `GLM53_DEC_KDA_LAZY{,_VERIFY,_VERIFY_EVERY}` and
   `GLM53_DEC_VTRIM_STATS{,_CAP,_FILE}` on both ranks; env_r16.sh feature `kdalazy`.
2. The kit's `site/` stops being byte-identical to r16z2 with every switch unset: setup.py adds two py_modules and
   integrate.py imports them in every process (both are inert when unset; the r16k/r16n precedent shipped new modules
   through the bundle overlay instead). Either accept and regenerate MANIFEST.sha256, or ship them via the overlay.
3. Boot gate: treat any `glm53_kda_lazy: self-check: ... differ` as a rollback trigger (repair mode is ~5 ms/step
   slower than production, §1), not as a harmless fallback.
4. A/B order: KDA_LAZY changes no value, so acceptance and quality must be identical; the speed delta (~1.6 ms of
   ~75 ms) is near the run-to-run noise of one bench_round — use interleaved A/B/A/B restarts or the hostloop meter's
   step_ms on both ranks rather than a single A then B.
