# DEPLOY_R16Z8P - r16z8 + GLM53_MLA_PLAN_PIN + persistent ABLIT

> Operator record of how this kit was rolled out on the original two nodes (an update from the previous kit, with
> an A/B). It is kept for reference. A fresh installation follows `REPRODUCE.md` at the repository root instead.

Kit `tf-exl3-deploy16.r16z8p` (branch `r16z8p` of tf-exl3-fork = `r16z8` e232306 + this change; built by
`tools/deploy16/derive_kit_r16z8p.sh` from the reviewed r16z8 kit 7c0b2c7). It replaces r16z8 as the next kit after the
installed r16z7: same prev-r16 (r16z7 + its start.sh variant 86084ff9), same payload as r16z8 plus two changes.

| change | what ships | default | turned on by |
|---|---|---|---|
| GLM53_MLA_PLAN_PIN (docs/MLA_PLAN_PIN.md) | site/glm53_mla_planpin.py (pure Python), integrate.py imports it | unset: `glm53_mla_planpin plugin loaded ... -> off, production's pageable plan staging unchanged`, nothing installed | `tools/env_r16.sh on planpin` (=1) after the A/B below |
| persistent ABLIT | launcher/start.sh (AB1/AB2) | an .env without ABLIT (or `ABLIT=0`) boots ABLIT=0, exactly as before | `ABLIT=1` in production's .env (section 2) |

* PLAN_PIN: production's sparse-MLA plan stages its indptr / lens in page-locked memory (2-slot ring, per-slot CUDA event
  and page-locked int workspace) instead of pageable tensors whose 65540 B indptr copy at max-num-batched-tokens 16384
  blocks the host until the drafter graph drained. nodeC (real MRv2 path): -1.28 ms/step, kernel-visible plan state
  identical; production estimate -1.1 ms/step (-0.9 .. -1.6). Not a collective (each rank plans locally); forwarded to
  both ranks so both ranks run the same code and the A/B PROOF holds on both.
* ABLIT: today start.sh CLEARS an .env ABLIT right after reading .env and honours only a caller-exported ABLIT, which is
  why `~/tf-exl3-deploy/restart2.sh` (nodeA, sha 6f289f62) exports `ABLIT="${ABLIT:-1}"` and every other restart path
  must remember to. r16z8p's start.sh:
  * caller exported ABLIT (any value, also empty) -> the caller's value (unchanged precedence; restart2.sh, arm_env);
  * no caller export, `.env` `ABLIT=0|1` -> the .env value (NEW); any other .env value -> refused before anything runs,
    value not printed;
  * no caller export, no / empty .env ABLIT -> 0 (unchanged);
  * `GLM53_MODEL_PRESET=abliterated` still forces 0 (the preset checkpoint already carries the transplant);
  * one line per start: `ablit: effective ABLIT=<v> (source: caller export | .env | default (no ABLIT in .env)[; forced 0
    by GLM53_MODEL_PRESET=abliterated])`.
  boot_checks reports `container env ABLIT=` (head == worker, else MISS) and requires the container entry's
  `ablit: o_proj orthogonalization ON` line on both ranks iff ABLIT=1.

## 1. What changes against r16z8 (kit MANIFEST diff)

| file | change |
|---|---|
| site/glm53_mla_planpin.py | new |
| site/integrate.py | + the glm53_mla_planpin import in its own try (an import failure only logs a WARNING = production) |
| site/tf_exl3_moe-0.1.0.dist-info/{RECORD,top_level.txt} | RECORD regenerated as make_kit.sh does; top_level/METADATA/entry_points == `setup.py dist_info` of HEAD in the production image (minus tf_fp8_w8a8_ext) |
| launcher/start.sh | stage r16z8p = r16z8 + PP1-PP4 (validation `unset/empty, 0 or 1` + site-module check, head `-e`, worker serve_env_names, note) + AB1-AB2 (persistent ABLIT, above). The generator's r16z8 stage reproduces r16z8's 4973d635 byte for byte (checked by the derivation) |
| env.r16 | r16z8's file + the `#knob GLM53_MLA_PLAN_PIN unset\|0\|1` block (no `NAME=` line: the update writes nothing to .env) |
| tools/env_r16.sh | feature `planpin` (on refuses unless the installed start.sh forwards the knob to BOTH ranks and the site carries the module; off removes the line) |
| tools/boot_checks.sh | planpin rows (head == worker, off/on lines on every rank, never-rows, `--after-traffic` serving rows) + ABLIT rows (head == worker, the entry line iff ABLIT=1) |
| tools/make_start_sh.py | stage r16z8p (add_planpin, add_ablit_env) |
| docs/MLA_PLAN_PIN.md, docs/DEPLOY_R16Z8P.md (= ROLLOUT.md) | new |
| overlay/, launcher/overlay/, every other site/ file, apply_r16.sh, revert_r16.sh, check_config.sh, arm_env_ab.sh | == r16z8 (byte for byte) |
| prev-r16/ | r16z8's: the r16z7 kit AS PRODUCTION RUNS IT (69 site/overlay/launcher files, ./launcher/start.sh = a variant of r16z7's start.sh, 86084ff9); ENV_ADD empty; START_SH_NOTE rewritten |

## 2. ABLIT=1 in production's .env (do it at adoption; harmless before)

Production since its 2026-10-05 09:44 boot (serving from 09:49) runs ABLIT=1 on both ranks while nodeA's `.env` still says `ABLIT=0` (read-only check
10:55: one `ABLIT=0` line); only restart2.sh's export keeps ablit on. With r16z8p installed, the .env line is what decides
when nobody exports ABLIT. On nodeA, any time before or after the swap (r16z7's start.sh ignores the line, so editing it
first changes nothing until r16z8p's start.sh runs):

```bash
cd ~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks && cp -p .env .env.bak-ablit-$(date +%Y%m%d-%H%M%S) \
  && [ "$(grep -c '^ABLIT=' .env)" = 1 ] && sed -i 's/^ABLIT=0$/ABLIT=1/' .env && grep -x 'ABLIT=1' .env
```

After r16z8p's start.sh is installed (section 3), `env -u ABLIT ~/tf-exl3-deploy16.r16z8p/tools/check_config.sh` must say
OK, and the restart log must show `ablit: effective ABLIT=1 (source: caller export)` (restart2.sh) or `(source: .env)`
(any other start). restart2.sh's `export ABLIT="${ABLIT:-1}"` then becomes redundant (keep it or drop it: both give 1;
`ABLIT=0 restart2.sh` still forces a stock-weights start because a caller export wins). arm_env_ab.sh needs nothing new:
it passes the running containers' ABLIT to every restart (r16z8 review A1), which agrees with the .env value.

## 3. Operator steps from r16z7 (kit r16z8p)

On nodeA first: `sha256sum ~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/start.sh | cut -c1-8` must print `86084ff9` (r16z7 + its
start.sh variant; anything else: stop, apply_r16.sh refuses the kit on another tree). Wait until no other A/B runs
(`${HOME}/tf-exl3-assets/AB_RUNNING` absent, no `.arm_env.lock` in use).

### 3.1 Swap A/B (recommended: all three decode knobs in one arm, 4 arms, ABLIT=1 in every arm)

base = r16z8p with production's switches (incl. whatever r16z7 knobs production adopted), the three new knobs unset =
r16z7 behaviour; all = DLMH + AR1SHOT + PLAN_PIN; b2 = the three explicitly 0; all2 = all again. KEEP=base: the run
always ends on base; adopt by hand with 3.3. arm_env reads ABLIT from both running containers (must be 1 and equal),
passes it to every restart and requires it on both ranks after each one, so every arm runs ablit.

```bash
# nodeC
cd ${HOME}/tf-exl3-deploy16.r16z8p && sha256sum --quiet -c MANIFEST.sha256 && echo kit-ok
mkdir -p ${HOME}/tf-exl3-assets/env-ab
ABLIT=1 PREV_KIT_NAME=tf-exl3-deploy16.r16z7 PRE_START_SHA=86084ff9 \
KNOBS="GLM53_DEC_DLMH GLM53_DEC_AR1SHOT GLM53_MLA_PLAN_PIN" ARMS="base all b2 all2" KEEP=base \
ARM_all="GLM53_DEC_DLMH=1 GLM53_DEC_AR1SHOT=1 GLM53_MLA_PLAN_PIN=1" ARM_b2="GLM53_DEC_DLMH=0 GLM53_DEC_AR1SHOT=0 GLM53_MLA_PLAN_PIN=0" \
ARM_all2="GLM53_DEC_DLMH=1 GLM53_DEC_AR1SHOT=1 GLM53_MLA_PLAN_PIN=1" \
PROOF='serving confirmed (mode on): graph replays|graph-captured one-shot all-reduces|plans through the pinned ring' PROOF_b2=- \
NEVER='glm53_kda_lazy: self-check:|glm53_dlmh: setup failed|candidates differ from production|glm53_ar1shot: not installed|agreement: a rank is not ready|all-reduce differs from production|glm53_mla_planpin: NOT installed|pageable plan staging from now on' \
LMAX=0.065 TMIN=96.8 DMAX=0.025 BASE_GMIN=2900 OUT=${HOME}/tf-exl3-assets/env-ab/z8pall \
  tools/arm_env_ab.sh 2>&1 | tee ${HOME}/tf-exl3-assets/env-ab/z8pall.log
```

* `ABLIT=1` given to arm_env must equal the live value (arm_env aborts otherwise, nothing touched); drop it to let arm_env
  read it alone.
* PROOF: the three fixed strings are each feature's serving-confirmed line (DLMH `[glm53-dlmh] rank R serving confirmed
  (mode on): graph replays N ...`; AR1SHOT `[glm53-ar1shot] rank R serving confirmed (mode on): N graph-captured
  one-shot all-reduces`; PLAN_PIN `[glm53-mla-planpin] rank R serving confirmed (mode on): N plans through the pinned
  ring, ...`). ALL must be in BOTH ranks' logs or the arm is invalid.
* NEVER: the KDA_LAZY repair-mode line (production runs GLM53_DEC_KDA_LAZY=1) + each feature's failure lines (planpin:
  `NOT installed` = a source fingerprint mismatch; `pageable plan staging from now on` = a failed self-test or ring
  allocation).
* boot_checks (run by arm_env before each measurement) already requires per arm: the dlmh / ar1shot / planpin
  plugin-hook-self-test lines on both ranks, head == worker for all eight knob values and for ABLIT, and the ablit entry
  line on both ranks.
* Quality limits with ablit on: LMAX/TMIN/DMAX/BASE_GMIN above are DEPLOY_R16Z8's, calibrated on ABLIT=0 runs. Take the
  ABLIT=1 r16z7 numbers of the owner's tail A/B rerun (`${HOME}/tf-exl3-assets/env-ab/z7tail-on/run.log`; its
  base arm at 10:45: long5 0.05895, long5_top1 97.13, dvptf 0.01015) and set `BASE_LMAX/BASE_TMIN/BASE_DMAX` and
  `LMAX/TMIN/DMAX` to that run's worse of base/b2 + the usual margins (e.g. LMAX = max long5 + 0.006, TMIN = min top-1 -
  0.4, DMAX = max dvptf + 0.015) before starting. The decode-knob judgement (3.3) is relative (ON vs OFF arms of the same
  run, all with ABLIT=1) and holds regardless.
* If production adopted r16z7's tail fix, `.env` carries `GLM53_KPOOL_TAIL_POSITIONS=2`: every arm keeps it (base =
  production's own lines; KNOBS here are only the three decode names).
* Duration: about 4 x (restart + r16_measure + gate), as DEPLOY_R16Z8 3.1.

### 3.2 Per-knob arms (optional; .env only, r16z8p already installed)

When the all arm fails a rule (to find the knob that carries it) or to attribute the gain before adopting all three.
No PREV_KIT_NAME / PRE_START_SHA: production already runs r16z8p.

```bash
cd ${HOME}/tf-exl3-deploy16.r16z8p
KNOBS="GLM53_DEC_DLMH GLM53_DEC_AR1SHOT GLM53_MLA_PLAN_PIN" ARMS="base pp dl ar b2 pp2 dl2 ar2" KEEP=base \
ARM_pp=GLM53_MLA_PLAN_PIN=1 ARM_dl=GLM53_DEC_DLMH=1 ARM_ar=GLM53_DEC_AR1SHOT=1 \
ARM_b2="GLM53_DEC_DLMH=0 GLM53_DEC_AR1SHOT=0 GLM53_MLA_PLAN_PIN=0" \
ARM_pp2=GLM53_MLA_PLAN_PIN=1 ARM_dl2=GLM53_DEC_DLMH=1 ARM_ar2=GLM53_DEC_AR1SHOT=1 \
PROOF_pp='plans through the pinned ring' PROOF_pp2='plans through the pinned ring' \
PROOF_dl='serving confirmed (mode on): graph replays' PROOF_dl2='serving confirmed (mode on): graph replays' \
PROOF_ar='graph-captured one-shot all-reduces' PROOF_ar2='graph-captured one-shot all-reduces' PROOF_b2=- \
NEVER='glm53_kda_lazy: self-check:|glm53_dlmh: setup failed|candidates differ from production|glm53_ar1shot: not installed|agreement: a rank is not ready|all-reduce differs from production|glm53_mla_planpin: NOT installed|pageable plan staging from now on' \
LMAX=0.065 TMIN=96.8 DMAX=0.025 OUT=${HOME}/tf-exl3-assets/env-ab/z8pknob \
  tools/arm_env_ab.sh 2>&1 | tee ${HOME}/tf-exl3-assets/env-ab/z8pknob.log
```

Per-knob judgement: 3.3 with ON = {pp, pp2} (resp. {dl, dl2}, {ar, ar2}) and OFF = {base, b2}; per-knob expectation:
planpin -0.9 .. -1.6 ms/step (judge on every workload: it saves the same host stall once per step whatever K is),
dlmh ~-0.5 (structured first), ar1shot -0.2 .. -1.0. (Two-step variant: `ARMS="base pp b2 pp2"` with PROOF_pp only, to
decide planpin alone first.)

### 3.3 Adoption rule (all of it; all-arm run: ON = {all, all2}, OFF = {base, b2})

Inputs per arm: r16_measure's bench_round lines `[envab-<arm>-<ts>] <w> tok/s med ... acc/step A ms/step M` for the four
workloads w in **structured / prose / coding / ja**, the `KLH` / `KLH-CMP` lines (klh: `dvptf`, `dvptf_se`,
`long5_top1`), the RESULT / gate lines of arm_env, and on nodeA `python3 ~/tf-exl3-deploy/bench_check.py
envab-<arm>-<ts>` (sha 36edeeb9; structured exact, coding exec, prose/ja doubled-token + ja particle/kanji-doubled
heuristics, distinct counts).

1. Gates: both ON arms have a RESULT line (long KL <= LMAX, top-1 >= TMIN, dvp KL <= DMAX, gate32k > GMARGIN x base), every
   PROOF string found on both ranks, no NEVER string on either rank (arm_env invalidates the arm otherwise).
2. Speed (expected on EVERY workload: planpin -0.9 .. -1.6 + dlmh ~-0.5 at batch 1 + ar1shot -0.2 .. -1.0, each once per
   step whatever K is, i.e. about -1.6 .. -3.1 ms/step): structured AND coding max(M_ON) < min(M_OFF); prose AND ja
   mean(M_ON) < mean(M_OFF) and min(M_ON) < max(M_OFF); on no workload min(M_ON) > max(M_OFF) (worse everywhere =
   reject even if others win).
3. Acceptance per step (no target value changes; planpin plans identical kernel-visible state, dlmh's candidates are
   byte-identical when recalled, ar1shot's sum is bit-identical): per workload |mean(A_ON) - mean(A_OFF)| <=
   max(|A_base - A_b2|, 0.05 on structured/coding, 0.1 on prose/ja).
4. bench_check per ON arm: structured exact n/n, coding pass n/n (never below the worse OFF arm), prose/ja doubled-token
   runs and ja particle/kanji-doubled runs no more than the worse OFF arm, prose/ja distinct counts inside
   [min(OFF) - 1, n]. Do NOT gate on temp-0 text identity with base (production's temp-0 output varies run to run).
5. KL harness (klh): dvptf(all), dvptf(all2) <= max(dvptf base, b2) + 2 x dvptf_se; long5_top1 >= min(base, b2) - 0.3.
6. Memory, on BOTH nodes (dlmh +264 MiB reserved per rank; planpin +8.4 MiB pinned host per rank): from the measure log
   (`~/tf-exl3-deploy/measure/envab-<arm>-<ts>.log`, first line `mem nodeA: avail NG swap SM | nodeB: avail NG`, last
   line `done ... mem nodeA: avail NG swap SM`): in every ON arm min(nodeA avail, nodeB avail) >= 2G, and nodeA's swap
   growth over the measurement no more than the worse OFF arm's + 512M.

Decision:
* 1-6 pass -> adopt all three: on nodeA
  `~/tf-exl3-deploy16.r16z8p/tools/env_r16.sh on dlmh && ~/tf-exl3-deploy16.r16z8p/tools/env_r16.sh on ar1shot && ~/tf-exl3-deploy16.r16z8p/tools/env_r16.sh on planpin`
  (each refuses unless the installed start.sh forwards it to both ranks); section 2's `ABLIT=1` in .env if not done yet;
  `tools/check_config.sh` (OK; with ABLIT=1 in .env no export is needed), `tools/wait_idle.sh && ~/tf-exl3-deploy/restart2.sh z8p-decode`,
  `tools/boot_checks.sh` -> ALL OK (the known benign idx_gate BAD row aside) incl. `ok   [head=worker] container env ABLIT=1`,
  then after a few decode turns `tools/boot_checks.sh --after-traffic` -> the dlmh / ar1shot / planpin serving rows ok on
  head and worker.
* 2 fails only on prose/ja with 1, 3-6 passing -> neutral there (same arithmetic): may adopt on the structured/coding win.
* 2 fails on structured or coding, or any of 1, 3-6 fails -> section 3.2; adopt only a knob whose own arms pass 1-6
  (per-knob 2: its feature's expectation); never adopt a knob whose own arm fails 1, 3, 4 or 5.

## 4. Revert

* Knobs only: `tools/env_r16.sh off planpin` (and/or `off dlmh`, `off ar1shot`) + idle restart = r16z8 / r16z7 behaviour.
* ABLIT: `.env` back to `ABLIT=0` (or `ABLIT=0 restart2.sh` for one start); the .env backup of section 2.
* Full kit revert: `tools/revert_r16.sh <the apply backup ~/tf-exl3-deploy/backup-r16-<ts>> --apply` + idle restart:
  r16z7's files + its start.sh variant 86084ff9, every r16z8p knob line stripped from .env; a non-kit line such as
  `ABLIT=1` is kept (r16z7's start.sh then ignores it again: restart2.sh's export is what keeps ablit on under r16z7).

## 5. Verification (nodeC + read-only nodeA, 2026-10-05)

Logs: docs/logs/r16z8p/ (copies of ${HOME}/tf-exl3-assets/r16z8p/{gpu,engine,mock,mirror,realboot,verify}). The kit
verified on nodeC is the payload-identical draft built from this branch (site/ overlay/ launcher/ tools/ env.r16 byte for
byte the shipped kit's; the shipped kit differs in docs/ROLLOUT/BUILD/SOURCE_COMMIT/MANIFEST only and was re-verified,
section 5.1). No production request was sent (the owner's ABLIT=1 tail A/B ran on production throughout); nodeA/nodeB
were read only (sha256 of files, `docker logs` / `docker inspect` with credential values masked on the node).

| check | result |
|---|---|
| production sources the patch relies on (read-only sha256) | `flashinfer_mla_sparse_sm90.py.patched` 526399d8 and fi618 `mla/_core.py` 58d06e61 identical on nodeC, nodeA (head mount) and nodeB (`${WORKER_HOME}/...`, worker mount); nodeA start.sh 86084ff9, `.env` one `ABLIT=0` line, restart2.sh 6f289f62 exports `ABLIT="${ABLIT:-1}"`; live start.sh + overlay/ (115 files) == the r16z8 review's 09:48 mirror copy |
| tests/r16z8p/test_planpin.py (GPU, production mounts; twice, the second on the shipped module bytes) | ALL OK: P0 fingerprints 5c9836599374a0ec / bcadfb24d6959d4a; P4 40/40 + 40/40 plans with identical kernel-visible plan state (16384 and 7168); P5 ring 0/24 + 0/24 wrong snapshots under a back-to-back stress (ring waits 22), naive pinning 23/24 wrong (the hazard of section 2.1 of MLA_PLAN_PIN.md, so the stress is sensitive), production pageable 16384 0/24, production pageable 7168 23/24 wrong (production's latent schedule race below 64 KiB); P6 flashinfer's copies are ordered on torch's current stream; P7 capture RuntimeError / disabled fallback; P8 logs |
| review: tests/r16z8p/test_planpin_adv.py + test_planpin.py + bench_planpin.py on the fixed module (logs docs/logs/r16z8p/review_*) | fallback fix (MLA_PLAN_PIN.md 2): V1 a failed self-test after the ring served now returns production's PAGEABLE staging + the wrapper's own int workspace (stress 0/24 wrong; be04148 23/24); V2 CUDA-graph replay reader, 2 plans per step, mixed sizes: 0 wrong at 16384 / 7168; V3 60000 plans: no growth; test_planpin ALL OK; bench at 16384: gap 1.325 -> 0.049 ms, step **-1.27 ms**, copy_ host call 4.78 ms -> 0.003 ms, 8/8 identical |
| tests/r16z8p/bench_planpin.py (REAL MRv2 prologue + REAL `_kv_lens_host` with the hostloop fast path + REAL plan at max_tokens 16384; 2 runs) | drafter -> forward gap p50 1.332 / 1.394 ms -> 0.049 / 0.049 ms; step p50 **-1.28 / -1.35 ms**; plan() host 4.73 / 4.49 ms -> 0.15 / 0.19 ms; B == the 7168 no-stall reference within 0.02 ms; ring waits 0; A vs B kernel-visible plan state 8/8 identical; the indptr `copy_` host call with a drafter graph queued: pageable 4.79 / 4.72 ms, pinned 0.003 ms |
| engine smoke (tests/r16z7/run_kit.sh: the kit composed byte for byte + production's live non-secret env, real vLLM engine, DFlash2 drafter, EXL3-MoE mini target, max_num_batched_tokens 16384, FULL_AND_PIECEWISE, 4 prompts 4590/3001/1200/700, 2 concurrent, greedy 128 tokens; KDA_LAZY=1 + KPOOL_TAIL=2 in every arm) | arms off / pin (PLAN_PIN=1) / all (PLAN_PIN + DLMH + AR1SHOT) / off2, pin and all re-run on the shipped bytes: rc 0 each, 128/128 tokens, finite logprobs; pin vs off first divergence 2 / 111 / 34 / 70 and all vs off 6 / none / 34 / 70 against the off2-vs-off band 2 / 77 / 34 / 70; degeneration counts inside the rig's A/A spread; proofs on both on-arms: plugin `-> installing (mode on)`, `patched _SM90State.plan`, `self-test: 3/3 device plan buffers == pinned staging`, `[glm53-mla-planpin] rank 0 serving confirmed (mode on): 64 plans through the pinned ring, ring waits 0`; all: + dlmh self-test + serving, ar1shot hook, kda_lazy self-check, tail in-place; the planpin finder composes with the hostloop / quickwins / mla-prefill hooks of the same backend; no NOT installed / failed self-test / ring failure line; ENGINE-SANITY OK |
| kit_verify of the draft (manifest, S --s-prev r16z7, B, strings, offeq vs r16z8) | PASS (S 165 ok, B 292 ok, offeq 136 rows, 0 FAIL each): test_kit_scripts ALL OK incl. NEW S.34 (planpin: the previous kit's env_r16 does not know it, the kit's refuses under r16z7's start.sh, update from r16z7 + its start.sh variant `== the previous deploy-r16 kit`, .env byte-identical, the production start.sh variant forwards PLAN_PIN + every r16z8 knob + ABLIT once per rank, on/off exactly one line, refused without the module, check_config accepts 6 / refuses 5 values unprinted; ABLIT precedence 9 cases incl. preset and explicit-empty export, a bad .env value refused unprinted, the previous start.sh clearing an .env ABLIT=1, revert keeps the operator's ABLIT=1) and S.33 unchanged; test_boot_checks ALL OK incl. NEW B.33 (planpin 13 rows), B.34 (ABLIT 8 rows), B.35 (the all arm incl. tail=2 and --after-traffic 6 serving rows); strings 370 markers / 0 not printable (129 seen verbatim in the engine pin/all logs and the two production boots, every planpin need string and the ablit entry line included); offeq vs r16z8 ALL OK (PLAN_PIN=1 / =0 == unset byte for byte in S1; PRODUCTION PARITY + tail=2 + DLMH + AR1SHOT + PLAN_PIN == the tail arm; PRODUCTION PARITY + PLAN_PIN == PRODUCTION PARITY; the site delta vs r16z8 is exactly glm53_mla_planpin.py + integrate.py + RECORD + top_level.txt; r16z8's ar1shot / dlmh / tf_dlmh_ext installed byte-identically in both kits) |
| boot_checks on production's REAL boot logs + container env (tests/r16z8p/realboot_check.sh, read-only) | two boots, both ABLIT=1: production's 09:44 boot (r16z7, fetched 09:51) and the running A/B's b2 arm (started 11:22, fetched ~11:35): K (as they run): only the three not-yet-installed plugin rows report, `ok [head=worker] container env ABLIT=1` + the entry line on both ranks; U (r16z8p, knobs unset): ALL OK (benign idx_gate aside); T (the all arm + tail=2): ALL OK, T --after-traffic 6 serving rows ok; H1 planpin head-only, H2 no worker self-test, H3 no worker serving line, H4 worker ABLIT=0, H5 worker without the ablit entry line: MISS [worker] each |
| arm_env (the kit's tools/arm_env_ab.sh == r16z8's; mock harness + r16z8p scenarios) | z8p_all (section 3.1 verbatim, production ABLIT=1, swap from r16z7): 4 arms, ABLIT 1 on every restart, ends on base 1|1, .env identical; z8p_pphalf (the worker's planpin serving line missing) -> arm invalid, back to base; z8p_badval (PLAN_PIN=2) refused before anything is touched; z8p_knob (3.2, .env only); z8_ablit / z8_ablit_basebad on the r16z8p kit; the 30 legacy scenarios: normalized output identical to the r16z8 harness (0 diff lines). The mock restart2 now honours an .env ABLIT when the installed start.sh carries AB1 and the caller exported none |
| apply / revert round trip on a byte copy of the live tree (scripts mirror_roundtrip.sh) | dry run `== the previous deploy-r16 kit a0ce5dc4`; --apply == kit, .env untouched; env_r16 on planpin + dlmh + ar1shot (3 lines); check_config OK with tail=2 on production's .env, refuses 4 bad values unprinted; ABLIT on production's own .env: as it is (ABLIT=0) -> 0 from .env, + restart2's export -> 1, after section 2's sed -> 1 from .env, + `ABLIT=0` export -> 0; check_config OK with ABLIT=1 in .env; off x3 -> .env == before; revert -> tree byte-identical to the live copy. ALL OK |

### 5.1 Shipped kit

| check | result |
|---|---|
| kit_verify of ${HOME}/tf-exl3-deploy16.r16z8p (payload == the verified draft) | see docs/logs/r16z8p/final_kit_verify.txt |

## 6. Not run / risks

* Nothing ran on production or on a 2-node pair (production A/B running, no production requests). PLAN_PIN's gain on
  production is the profile analysis' estimate + nodeC's MRv2-path bench; the A/B decides.
* AR1SHOT / DLMH: as DEPLOY_R16Z8 7.
* ABLIT persistence changes behaviour only for an .env that carries `ABLIT=1` (today production's says `ABLIT=0`): no
  restart changes until the operator edits the line.
