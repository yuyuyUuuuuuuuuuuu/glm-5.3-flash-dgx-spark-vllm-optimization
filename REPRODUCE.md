# Reproducing the production setup

This is how to rebuild, on your own two GB10 machines, the serving configuration that produced the "current
production" numbers in `README.md` (2026-10-05 17:30). It uses the code under `code/`, the public launcher at a pinned
commit, and public weights at pinned revisions. Section 11 lists what could not be pinned or shipped.

Before you start, read the licence notes in section 2. The DFlash2 drafter is **CC BY-NC-ND 4.0**, which rules out
commercial use of the stack as configured here.

Contents:
1. Hardware and OS
2. Downloads (pinned)
3. Prepare both nodes
4. Get the launcher
5. Build the extensions
6. Weights and host-side files
7. Configure `.env`
8. Install the kit
9. Start and check the boot
10. Measure: the four-workload bench and the quality probes
11. Known gaps
12. Configuration reference
13. Running the fork's test suite

The examples call the head node **node A** and the worker **node B**. `code/operator/` scripts read `WORKER_SSH`
(`<user>@<worker address>`), `LAUNCHER_DIR` (default `~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks`) and `DEPLOY_DIR` (default
`~/tf-exl3-deploy`). Code under `code/fork/tests/` was written for a separate single-GB10 test box ("node C"); it finds
model copies and test assets through `TF_EXL3_MODELS`, `TF_EXL3_ASSETS` and `TF_EXL3_KITS` (section 13).

## 1. Hardware and OS

What the numbers were measured on. Other GB10 machines should behave the same; nothing else was tested.

| item | value |
|---|---|
| machines | 2 x NVIDIA GB10 (DGX Spark class; ours are Lenovo ThinkStation PGX), 121 GiB of unified memory visible to the OS, 20 CPU cores each |
| memory bandwidth | spec 273 GB/s; measured read ceiling 250-265 GB/s, which decays with uptime unless memory is compacted before the engine starts (section 3) |
| OS | Ubuntu 24.04.4 LTS, kernel `6.17.0-1029-nvidia`, aarch64 |
| GPU driver | NVIDIA open kernel module 580.173.02 (both nodes) |
| container runtime | Docker 29.2.1 with the NVIDIA container toolkit |
| CUDA, Python | inside the image: CUDA 13.0.1, `TORCH_CUDA_ARCH_LIST=12.1a` (sm_121a), Python 3.12 |
| network | ConnectX-7 (firmware 28.45.4028), RoCE. Two direct cables, port f0 to f0 and f1 to f1, MTU 9000 |
| NCCL | the image's own NCCL, `nvidia-nccl-cu13` 2.30.7 (`USE_HOST_NCCL=0`) |
| host tools | `bash`, `python3` (3.12), `sha256sum`, `rsync`, `curl`, `git`, `ethtool`; passwordless ssh from node A to node B |
| disk per node | about 200 GiB free: weights 163.6 GiB, drafter 2.2 GiB, o_proj transplant 2.5 GiB, flashinfer overlay 6.9 GiB, image 20.9 GB |

**Two RoCE rails.** The CX7 on GB10 has two PCIe x4 domains (`enp1s0f*` / `rocep1s0f*` and `enP2p1s0f*` /
`roceP2p1s0f*`). One rail carries about 112 Gb/s; NCCL over one function in each domain, on two cables, reached
223.8 Gb/s (ib_write_bw, `code/fork/tools/rail2/README.md`). Production uses `rocep1s0f1` + `roceP2p1s0f0` on both
nodes, `NCCL_IB_MERGE_NICS=0`, RoCE v2 IPv4 GID index 3.

## 2. Downloads (pinned)

| what | where | revision to use | licence | notes |
|---|---|---|---|---|
| serving image | `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor` | `sha256:447114ee77d14c9b4732ee23978ada2a0ee9027868a231d6fd42700a8b25be1d` | mixed (see `THIRD_PARTY_NOTICES.md`) | arm64, 20.9 GB, created 2026-09-16, anonymous pull. Contains vLLM `0.1.dev20051+g487ecf187`, torch `2.13.0+cu130` (git cf30153c), flashinfer 0.6.17 (a0a6b019), exllamav3 0.0.43 (commit c5d9c657, built for sm_121a with the launcher's patches), InstantTensor 0.2.0, NCCL 2.30.7. Built `FROM vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c0293...`; that tag now points elsewhere, so pull the GHCR image by digest |
| launcher | https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks | commit `0f49cfdbaa131286eb592cd6ebfa048f3aa85c4e` (2026-09-24) | AGPL-3.0 (MIT before 2026-09-07) | production ran a local merge of this commit; `code/kit/launcher/` and `code/launcher-patches/` carry every difference |
| target weights | `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold` | `6c5b28260ab9e80c6608de8b419e624cbe71b7cf` | Apache-2.0 (base model MIT) | 83 safetensors shards, 163.6 GiB. The repository head has moved on (93 files): pin the revision. Made with TensorFold 0.6.0 and an unpublished calibration recipe, so it cannot be re-made, only downloaded |
| drafter | `incoai/GLM-5.3-Flash-DFlash2` | `dc77ff1c99eeb2df044ee3d4f0094eb033fee410` | **CC BY-NC-ND 4.0** | 2.2 GiB. The kit's `start.sh` pins this revision (`DFLASH_REVISION`). Its FP8 conversion happens in memory at load; never write converted weights out |
| o_proj transplant donor | `dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4` | head was `745aac2ff0f10acf961f396df3f9418598aa7327` on 2026-10-06; the revision production fetched was not recorded | MIT | only 31 bf16 tensors (2.5 GiB) are range-fetched; check them against `data/ablit_transplant_sha256.csv` |
| FlashKDA | https://github.com/vllm-project/FlashKDA | `17a037d98da546deb4591e967cf961a43c034d8b` | MIT | with CUTLASS `5c149f52a436782210263fb2f19b354443a61c6a` (BSD-3-Clause) and two files of vLLM `ddd6fbca148a867aad1fcab7ec72f582b9977db4`; `code/build/fetch_flashkda_sources.sh` stages all three |
| flashinfer overlay | flashinfer-python + flashinfer-cubin `0.6.18.dev20260819` (git `61a6c651872a7d3f2f6dcc1ced61633d8f8ba3dd`) | see section 11 | Apache-2.0 | mounted over the image's 0.6.17 for the SM90 sparse-MLA KV layout |
| 32k prefill yardstick | https://github.com/kindlingai/glm-5.3-flash-gx10 | commit `45b438be5af02f922403d56272a15d04922cbf08` | none stated | only its `gate/prefill.py` is used, from a local checkout |
| base model (reference only) | `zai-org/GLM-5.3-Flash` | not served directly | MIT | |

Not needed: TensorFold itself (only its checkpoint is used), the older TR3 checkpoint, and NCCL 2.30.7 sources (the
tuner plugin built from them was dropped on 09-28).

## 3. Prepare both nodes

Do every step on node A and on node B.

1. **Pull the image by digest** and give it the tag the launcher expects:
   ```bash
   docker pull ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:447114ee77d14c9b4732ee23978ada2a0ee9027868a231d6fd42700a8b25be1d
   docker tag ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:447114ee77d14c9b4732ee23978ada2a0ee9027868a231d6fd42700a8b25be1d \
     ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor
   ```
2. **Memory compaction off**, persistently (bandwidth decays with free-memory fragmentation; `docs/methodology.md`):
   ```bash
   echo 'vm.compaction_proactiveness = 0' | sudo tee /etc/sysctl.d/99-glm53-compaction.conf
   sudo sysctl -w vm.compaction_proactiveness=0
   ```
3. **First rail**: give `enp1s0f1np1` an address on a private /24 shared by both nodes, MTU 9000 (netplan or
   NetworkManager, as you normally do). These addresses are `HEAD_IP` / `WORKER_IP` in `.env`.
4. **Second rail**: edit `RAIL2_SUBNET` in `code/operator/glm53-rail2` (a second free /24), then install it once per
   node with `sudo bash code/operator/install-glm53-rail2.sh`. It installs `/usr/local/sbin/glm53-rail2`, a
   `glm53-rail2.service` that brings up `enP2p1s0f0np0` with trimmed RX rings before docker at every boot, and a
   NOPASSWD rule for exactly that command. Check with `glm53-rail2 status`.
5. **GID index**: the RoCE v2 IPv4 GID index can change after a reboot. `code/fork/tools/prodcheck/fix_gids.sh`
   detects it on both nodes and rewrites `HEAD_GID` / `WORKER_GID` / `NCCL_IB_GID_INDEX` in `.env`
   (`code/operator/restart2.sh` runs it on every restart).
6. **ssh**: node A must reach node B without a password (`ssh -o BatchMode=yes <user>@<worker> true`).

## 4. Get the launcher

On node A:
```bash
git clone https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks ~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks
cd ~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks
git checkout 0f49cfdbaa131286eb592cd6ebfa048f3aa85c4e
git apply <this repo>/code/launcher-patches/chat_template-reasoning-effort-placement.patch
```
The patch adds the `reasoning_effort_placement` template kwarg that production's `EXTRA_ARGS` sets (it keeps the
effort line out of the cached prompt head). The benches run with thinking off, so it does not change the bench
numbers. The kit (section 8) replaces `start.sh` and four files of `overlay/`.

## 5. Build the extensions

The kit ships sources only. Twelve `.so` files must be built inside the serving image (CPU-only containers; no GPU
is used). Production's bytes are listed in `code/kit/PRODUCTION_BINARIES.sha256`; your build will not match them
byte for byte (nvcc embeds temporary names), but the code is the same.

```bash
code/build/fetch_flashkda_sources.sh ~/fkda-src          # git + network; verifies the two vLLM shim files
FKDA_SRC=~/fkda-src MAX_JOBS=8 code/build/build_all.sh   # builds, stages into code/kit/{site,overlay}, rewrites the MANIFEST
```

What it builds:

| extension | from | goes to | used by |
|---|---|---|---|
| `tf_exl3_moe_ext` | `code/fork/kernels/exl3.cu` (TensorFold-derived grouped EXL3 decode MoE, `TF_PARITY=1`) | `site/` | `TF_EXL3_MOE=1` |
| `tf_fp8_gemv_ext`, `tf_fp8_large_m_ext`, `tf_fp8_roof_ext` | `kernels/fp8_gemv.cu`, `fp8_large_m.cu`, `fp8_roof.cu` | `site/` | `GLM53_FP8_GEMV`, `GLM53_FP8_LARGE_M`, (`GLM53_DEC_FP8ROOF`, off in production) |
| `glm53_gemv_ext` | `kernels/gemv_bf16.cu` | `site/` | `GLM53_BF16_GEMV` |
| `glm53_mla_prefill_ext` | `kernels/mla_prefill/` | `site/` | `GLM53_MLA_PREFILL` |
| `glm53_smallops_ext` | `kernels/smallops.cu` | `site/` | `GLM53_DEC_SMALLOPS` |
| `tf_dlmh_ext` | `kernels/dlmh_gemv.cu` | `site/` | `GLM53_DEC_DLMH` |
| `tf_fp8_w8a8_ext` | `kernels/fp8_w8a8.cu` (CUTLASS headers from the image's flashinfer) | `overlay/` | `GLM53_DENSE_W8A8` |
| `glm53_moe_e4m3_ext` | `kernels/moe_e4m3.cu` (`tools/moee4m3/build.py`) | `overlay/` | `GLM53_MOE_E4M3` |
| `glm53_mhc_fused_ext` | `kernels/mhc_post_prenorm.cu` (`tools/mhcfused/build.py`) | `overlay/` | `GLM53_MHC_FUSED` |
| `_flashkda_fp32_C.abi3.so` | FlashKDA 17a037d, fp32 recurrent state, renamed namespace (`tests/fkda/build_flashkda_fp32.sh`) | `overlay/` | `GLM53_KDA_FLASHKDA` |

The two other FlashKDA variants (`GLM53_KDA_FLASHKDA_V=2|3`) are not used in production. Their wrappers refuse any
build whose sha256 differs from the one they were validated with, so a rebuild of those needs the constant in
`overlay/glm53_flashkda{2,3}.py` updated by hand.

## 6. Weights and host-side files

1. **Target weights and drafter**: `cd ~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks && ./start.sh download` (after section 7,
   which sets `MODEL`, `MODEL_REVISION`, `EXPECTED_SHARDS=83`). The first `start` rsyncs the cache to node B
   (`NFS_SHARE=1` shares it instead).
2. **o_proj transplant (ABLIT)** on node A, inside the launcher checkout:
   ```bash
   ABLIT_DONOR=dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4 python3 ablit/fetch_transplant.py
   python3 - <<'EOF'
   import csv, hashlib
   bad = [r["layer"] for r in csv.DictReader(open("<this repo>/data/ablit_transplant_sha256.csv"))
          if hashlib.sha256(open(f"ablit/transplant/L{r['layer']}.bin", "rb").read()).hexdigest() != r["sha256"]]
   print("all 31 tensors match production" if not bad else f"MISMATCH in layers {bad}")
   EOF
   ```
   `start.sh` copies `ablit/` to node B on every start. If the donor's head no longer matches, find the donor
   revision whose bytes do, or run with `ABLIT=0` (stock o_proj; section 12 says what changes).
3. **Patched vLLM files**: apply `code/vllm-patches/*.patch` to copies of the image's own files and place the
   results on both nodes at the paths `start.sh` mounts (or point the variables elsewhere):

   | patch | image file | node A path (`$HOME/vllm-patches/`) | node B path (`$WORKER_HOME/`) | sha256 of the result |
   |---|---|---|---|---|
   | `cuda.py.patch` | `vllm/platforms/cuda.py` | `cuda.py.exl3.patched` (`CUDA_PATCH_HOST`) | `cuda.py.exl3.patched` (`WORKER_CUDA_PATCH`) | `2bcbe6be5cc0f7da...` |
   | `flashinfer_mla_sparse_sm90.patch` | `vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py` | `flashinfer_mla_sparse_sm90.py.patched` (`SM90_PATCH_HOST`) | same name (`WORKER_SM90_PATCH`) | `526399d812d5579c...` |
   | `responses_utils.patch` | `vllm/entrypoints/openai/responses/utils.py` | `responses_utils.py.patched` (`RESP_UTILS_PATCH_HOST`, head only) | - | (comment text differs from production) |

   To get the originals without running anything: `cid=$(docker create <image> true)`, `docker cp
   $cid:/usr/local/lib/python3.12/dist-packages/<file> .`, `docker rm $cid`; then `patch -o <out> <orig> <patch>`.
   The first two let vLLM pick the SM90 sparse-MLA backend on sm_121 (FA2 path), which halves the KV bytes per token
   (`docs/timeline.md`, 09-04). They take effect only together with the flashinfer overlay below.
4. **flashinfer 0.6.18 overlay** (`SM90_KV=1`): a directory holding the `flashinfer/` and `flashinfer_cubin/`
   packages of `0.6.18.dev20260819` (git 61a6c651), at `$HOME/fi618` on node A and `$WORKER_HOME/fi618` on node B
   (`FI618_DIR` / `WORKER_FI618_DIR`). Get it with `code/fork/tests/derive_assets.sh fi618`, which copies the four
   package directories out of the public image `ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e98...`
   (never started) and checks them against production's overlay (95,185 files, aggregate sha256 `c60e3671...`);
   copy the result to both nodes. Without it `start.sh` silently stays on the SM120 layout: 15,591 instead of
   7,227 bytes per token, so the KV pool and `MAX_MODEL_LEN=1000000` no longer fit (section 11).
5. **memprep**: copy `code/fork/tools/memprep/memprep.py` to `<launcher>/overlay/tf/tools/memprep.py` after
   section 8. With `GLM53_MEMPREP=1`, `start.sh` runs it on both nodes right before the engines allocate.

## 7. Configure `.env`

```bash
cp <this repo>/code/kit/env.production.example ~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/.env
```
Edit the lines marked `EDIT` (`HEAD_IP`, `WORKER_IP`, `WORKER_USER`, `VLLM_API_KEY`) and, if your interfaces differ,
`HEAD_CX7_IF` / `WORKER_CX7_IF` / `HEAD_CX7_IB` / `WORKER_CX7_IB`. Section 12 explains every line. `KV_CACHE_BYTES`,
`GPU_MEM_UTIL=0.87` and `MAX_MODEL_LEN=1000000` leave about 3 GiB of host memory free on our nodes; keep other
workloads off the two machines.

## 8. Install the kit

The kit (`code/kit/`) is what production runs: `launcher/start.sh` (the launcher's start script plus every knob
below, each validated and forwarded to both ranks), four launcher overlays, the bundle `site/` (Python modules and
`.so` files that go into site-packages) and `overlay/` (opt-in patches applied at container start), and `tools/`.

```bash
cd <this repo>/code/kit
sha256sum --quiet -c MANIFEST.sha256 && echo kit-ok
FRESH_INSTALL=1 LAUNCHER_DIR=~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks tools/apply_r16.sh            # dry run
FRESH_INSTALL=1 LAUNCHER_DIR=~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks tools/apply_r16.sh --apply    # backup, install, .env
tools/env_r16.sh off fp8roof && tools/env_r16.sh off moeglue    # production has these two R16 features off
tools/check_config.sh                                           # the kit start.sh's own validation on this .env
```
`apply_r16.sh` copies `site/` and `overlay/` to `<launcher>/overlay/tf/`, the kit `start.sh` and launcher overlays
into the launcher, appends the `env.r16` lines that `.env` lacks, and verifies the result. It keeps a backup under
`~/tf-exl3-deploy/backup-r16-<time>/`; `tools/revert_r16.sh <backup dir> --apply` puts it back.

Always start with `SKIP_BUILD=1`: the kit changes files in `overlay/`, which makes `start.sh` want to rebuild the
image, and the prebuilt image is the one that was measured.

## 9. Start and check the boot

```bash
cd ~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks && SKIP_BUILD=1 ./start.sh start
# or, for every later restart: waits for an idle server, compacts memory, re-applies rail 2, fixes GIDs, starts
mkdir -p ~/tf-exl3-deploy && cp -r <this repo>/code/operator/. ~/tf-exl3-deploy/   # the operator workspace (node A)
WORKER_SSH=<user>@<worker> ~/tf-exl3-deploy/restart2.sh <label>
```
Then:
```bash
WORKER_SSH=<user>@<worker> <this repo>/code/kit/tools/boot_checks.sh                    # right after /health
WORKER_SSH=<user>@<worker> <this repo>/code/kit/tools/boot_checks.sh --after-traffic    # after one >= 4k prompt + a few turns
```
`boot_checks.sh` reads both ranks' boot logs and container environments and prints `ok` / `MISS` per expected line
(each feature's install, self-test and serving lines, the head == worker env check, `ABLIT` on both ranks). It must
end with `boot_checks: ALL OK`. The one known benign row is a `BAD ... idx_gate result differs` line.

Boot facts to compare: the engine reports a KV pool of 2,003,436 tokens with `KV_CACHE_BYTES=16106127360`
(`data/kv_capacity.csv`), and each rank loads about 82.4 GiB of weights.

## 10. Measure: the four-workload bench and the quality probes

Run only on an idle server (`vllm:num_requests_running` 0); every probe checks this and refuses or retries otherwise.

| what | command (in `~/tf-exl3-deploy`) | expected (production, 10-05) | data |
|---|---|---|---|
| decode, 4 workloads x 10 runs, temperature 0, thinking off | `./bench_round.sh <label>` (the launcher's `tests/bench_decode.py`; `bench_decode_ja.py` for Japanese) | structured 99.44 tok/s (80.05 ms/step), prose 46.58 (67.82), coding 62.61 (77.91), Japanese 45.47 (62.54) | `data/decode_bench.csv`, `raw/bench/vtrim-prod_*.json` |
| output correctness of that bench | `python3 bench_check.py <label>` | structured exact 10/10, coding pass 10/10, 0/10 doubled-token runs (the shipped raw files give exactly this) | |
| full probe set (decode, stream gaps, prefill, APC, decode-vs-prefill, long/short KL, teacher-forced KL) | `WORKER_SSH=... ./r16_measure.sh <label>` | real-text prefill 8.5k ~2,600 / 14.8k ~2,960 tok/s; 97k cold TTFT ~32 s, warm 0.73-0.79 s with 96,768 cached | `data/production_ab_runs.csv`, `raw/measure/` |
| 32k cold prefill (kindling yardstick) | `KINDLING_GATE=<checkout>/gate/prefill.py VLLM_API_KEY=... python3 gate_prefill_32k.py 32k <seed>` for seeds 71-76, median | 3,016.5-3,065.5 tok/s | `data/production_ab_runs.csv`, `raw/env-ab/` |
| concurrency | the launcher's `python3 tests/bench_decode.py --phase <label> --out <file> --runs 10 --concurrency N` (add `--coding` for coding) | 8 streams: 131.3 tok/s prose, 192.3 coding in total (10-04) | `data/concurrency.csv` |
| concurrent-decode quality | `python3 <kit>/tools/conc_quality_probe.py --out <file>` (4 streams; `--compare` summarises files) | ratio about 1.0 | |

The KL probes compare against **references you make yourself**: run `quality_long.py`, `quality_probe.py` and
`kpool_decode_consistency.py` once on your base configuration and keep the JSON files; `r16_measure.sh` then compares
every later run with them (it expects the names it uses for production's R15 reference; rename or edit). The
teacher-forced harness (`klh/`) needs fixtures built with `klh/make_fixtures.py` (it reads a public-domain Japanese
novel, Japanese man pages, `/usr/include` and Python's standard library, so its token ids depend on the host) and a
reference run (`klh.py run ref_v1 --reps 3`). Absolute KL levels are therefore comparable only within your own
reference, as they were for us (`docs/methodology.md`, caveats).

A/B tests of switches: `code/kit/tools/arm_env_ab.sh` (run from a third machine with `HEAD_SSH=<alias of node A>`,
`WORKER_SSH`, `KIT_NAME`, `ARMS`, `KNOBS`; header comment). It flips `.env` lines, restarts at idle, measures, judges
against the base arm and ends on the best passing arm.

## 11. Known gaps

What only existed on our machines, or could not be pinned:

- **Donor revision for the o_proj transplant** was not recorded. The sha256 list in
  `data/ablit_transplant_sha256.csv` is the ground truth for what production loads.
- **Binary identity.** Rebuilt `.so` files differ from production's bytes (`code/kit/PRODUCTION_BINARIES.sha256`).
- **The image's base layer** (`vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c0293...`) is no longer what that
  tag points to; only the GHCR image by digest reproduces production.
- **Regenerating `start.sh`.** `code/kit/tools/make_start_sh.py` documents how the kit `start.sh` was derived, but it
  needs the operator's earlier production `start.sh` and a local launcher fork as inputs, which are not published. Use
  `code/kit/launcher/start.sh` as shipped.
- **Rollout internals not shipped**: the previous kits, the backup copies of production files that
  `apply_r16.sh` compares against when updating from an earlier kit (hence `FRESH_INSTALL=1`), and the kit-building
  chain scripts. `revert_r16.sh --from-kit` has nothing to restore from; use the backup directory `apply_r16.sh` writes.
- **Test fixtures not shipped**: the klh fixture file and frozen trajectories, the drafter hidden-state features some
  `tests/dlmh` benches load, the bf16 samples of the TR3 checkpoint, the deploy-r15 bundle, and the 529 MB of raw logs
  of the fork (`docs/logs/`). `tests/run_all.sh` reports a step whose inputs are missing as SKIP; section 13 lists
  which inputs can be rebuilt (`tests/derive_assets.sh`) and which cannot.
- **Numbers from operator notes.** Everything before 09-27, the 09-24 24-hour evaluation and a few others have no raw
  log; they are marked `operator notes` in `data/` and the docs.
- **Mixed launcher licence history**: launcher contributions before 2026-09-07 were MIT; the launcher states the rest
  is AGPL-3.0. The fork's statement that no AGPL source was copied into its kernels was checked by line matching only.

## 12. Configuration reference

Every non-secret line of production's `.env` (`code/kit/env.production.example`). "kit" marks switches added by this
repository's kit; the rest are the public launcher's. Docs are under `code/fork/docs/` unless noted.

### Cluster, model, image

| line | production value | what it does |
|---|---|---|
| `HEAD_IP`, `WORKER_IP`, `WORKER_USER` | yours | NCCL bootstrap addresses on the first rail; the worker login |
| `HEAD_CX7_IF`, `WORKER_CX7_IF` | `enp1s0f1np1` | socket interface for NCCL bootstrap |
| `HEAD_CX7_IB`, `WORKER_CX7_IB` | `rocep1s0f1,roceP2p1s0f0` | the two RDMA devices NCCL uses (one per PCIe domain) |
| `NCCL_IB_GID_INDEX`, `HEAD_GID`, `WORKER_GID` | 3 | RoCE v2 IPv4 GID index (re-detected by `fix_gids.sh`) |
| `NCCL_IB_MERGE_NICS` | 0 | keep the two rails as two separate NICs (the setting measured in `code/fork/tools/rail2/README.md`) |
| `NCCL_NCHANNELS` | 8 | pinned host memory of NCCL 3.6 -> 0.53 GiB (R7, 09-28); 4 channels cost 3% of 24k prefill |
| `USE_HOST_NCCL`, `NCCL_SO_NAME` | 0, `libnccl.so.2.30.7` | use the image's NCCL; the name is used only with a host NCCL |
| `NCCL_TUNER_PLUGIN`, `NCCL_TUNER_CONFIG_FILE` | empty | the tuner plugin was dropped on 09-28 |
| `NCCL_DEBUG` | WARN | |
| `MODEL`, `MODEL_FALLBACK`, `MODEL_REVISION`, `MODEL_CACHE_NAME`, `MODEL_FALLBACK_CACHE_NAME`, `EXPECTED_SHARDS` | TensorFold checkpoint @ 6c5b2826, 83 shards | target weights |
| `IMAGE` | `...:exl3-instanttensor` | the serving image (tag; pull by digest, section 3) |
| `PORT`, `SERVED_MODEL_NAME` | 8888, `GLM-5.3-Flash-EXL3` | API port and model name |
| `TP`, `NNODES`, `MASTER_PORT` | 2, 2, 29521 | tensor parallel over the two nodes |
| `QUANTIZATION` | exl3 | |
| `VLLM_API_KEY` | yours | Bearer key of the API |

### Engine and memory

| line | production value | what it does |
|---|---|---|
| `MAX_MODEL_LEN` | 1000000 | context window |
| `MAX_NUM_SEQS` | 8 | concurrent sequences |
| `MAX_NUM_BATCHED_TOKENS` | 16384 | prefill chunk (`data/prefill_scheduling.csv`) |
| `GPU_MEM_UTIL` | 0.87 | |
| `KV_CACHE_BYTES`, `KV_CACHE_DTYPE` | 16106127360 (15 GiB), fp8 | fixed KV pool: 2,003,436 tokens |
| `SM90_KV` | 1 | SM90 sparse-MLA KV layout (7,227 B/token); needs the vLLM patches and the flashinfer overlay (section 6) |
| `CG_ESTIMATE` | 0 | keep CUDA graphs, drop the upstream ~2.6 GiB graph-memory deduction from the KV budget |
| `ENFORCE_EAGER` | 0 | CUDA graphs on |
| `EXTRA_ARGS` | prompt token details, generation defaults (temperature 1.0, top_p 0.95, repetition_penalty 1.05, three EOS ids), `reasoning_effort_placement=before_first_user` | passed to `vllm serve` |
| `LANGUAGE_MODEL_ONLY`, `SKIP_MM_PROFILING`, `LIMIT_MM` | 0, 1, 64 images / 1 video | vision stays enabled |
| `GLM53_BOOT_SHAPE_WARMUP` | 1 | after /health, run the drafter / sampler / kpool shapes once |
| `GLM53_SUPPRESS_STOPS_IN_REASONING` | 1 | client stop strings apply only after `</think>` |
| `GLM53_MIXED_PREFILL_CHUNK` | 0 | prefill chunks join decode steps with no cap, so a newcomer does not wait behind running decodes (`docs/timeline.md`, 09-07 and 09-24) |
| `GLM53_INDEXER_WORKSPACE` | rightsize | sparse-indexer workspace at the legal per-step maximum instead of a larger default (about +26% KV) |
| `GLM53_SPINWAIT_MS` | 16 | busy-wait window of the engine's condition reader (swept on TP=2) |
| `GLM53_APC_RETENTION_INTERVAL`, `GLM53_APC_RETENTION_INTERVAL_SWA` | 32256, 0 | prefix-cache retention grid (LCM of 3584 and 4608) and drafter SWA group boundaries only (09-24) |
| `GLM53_MEM_HYGIENE` (kit) | 1 | after model load and warm-up: garbage-collect, release cached pinned host blocks, `malloc_trim`; nothing GPU-side, no numerics change |
| `GLM53_TF_PROFILE` (kit) | `/tmp/glm53-prof` | installs a SIGUSR2 handler: `kill -USR2 <worker pid>` profiles the next 40 steps into this directory; nothing runs until the signal |
| `GLM53_MEMPREP` (kit) | 1 | compact free memory on both nodes before the engines allocate (`tools/memprep/memprep.py`) |

### Speculative decoding

| line | production value | what it does |
|---|---|---|
| `SPEC_METHOD`, `DFLASH_MODEL`, `DFLASH_TOKENS`, `DFLASH_DRAFT_TP` | dflash, `incoai/GLM-5.3-Flash-DFlash2`, 7, 2 | DFlash2 drafter, block of 8 (K at most 7), drafter TP=2 |
| `MTP_TOKENS` | 2 | used only with `SPEC_METHOD=mtp` |
| `GLM53_ADAPTIVE_K`, `_SET`, `_MARGIN`, `_ALPHA` | ema, 4,5,7, 1.0, 0.25 | adaptive verify length from an EMA of acceptance (09-22) |
| `GLM53_SPEC_RESAMPLE_INDEPENDENT` (kit) | 1 | after a rejected draft, resample with Gumbel noise independent of the draft's (the stock runner reused it, which biases sampled output); required by block verification |
| `GLM53_REJECTION_METHOD` (kit) | block | block verification for sampled decoding (`BLOCK_VERIFY.md`) |
| `GLM53_DRAFT_FP8` (kit) | layers,fc | drafter layers and fc in FP8, converted in memory (`DRAFTER_IMPL.md`) |
| `GLM53_DRAFT_LMHEAD_FP8` (kit) | empty | separate FP8 copy of the drafter's lm_head: off (the drafter shares the target's FP8 head) |
| `GLM53_DRAFT_KV_COMPACT` (launcher-apc) | 1 | larger drafter KV pages; no weight or cache precision change |
| `GLM53_APC_DRAFTER_LOW_PRIORITY` (launcher-apc) | 0 | the retained drafter window stays in the ordinary prefix-cache LRU instead of being evicted first |
| `GLM53_SPEC_VTRIM`, `GLM53_SPEC_VTRIM_TAU` (kit) | on, 0.3 | skip routed experts of verify rows the drafter's confidence says will be rejected (`SPEC_VTRIM.md`) |
| `GLM53_DEC_DLMH` (kit) | 1 | drafter candidate head without reading the whole FP8 lm_head, exact rescoring (`DEC_DLMH.md`) |

### Weight formats and kernels

| line | production value | what it does |
|---|---|---|
| `EXL3_FUSED_MOE`, `EXL3_FAT_KERNEL`, `EXL3_FAT_GROUPED`, `EXL3_TEMP_ROWS_FUSED` | 1, 1, 1, 256 | the launcher's fused EXL3 MoE paths; grouped prefill kernels for fat experts only |
| `TF_EXL3_MOE` (kit) | 1 | our grouped EXL3 decode MoE kernel replaces `exllamav3_ext.exl3_moe` (`DESIGN.md`) |
| `GLM53_DENSE_FP8` | dense,kda,mla,shared | dense and shared-expert BF16 weights to FP8 weight-only (Marlin layout) |
| `GLM53_LMHEAD_FP8` (kit) | 1 | target lm_head in FP8 |
| `GLM53_FP8_GEMV`, `GLM53_FP8_GEMV_MAX_M` (kit) | 1, 16 | our small-M FP8 GEMV for decode shapes up to M=16 |
| `GLM53_FP8_LARGE_M` (kit) | 1 | exact fast path for large-M (prefill) FP8 linears (`FP8_LARGE_M.md`) |
| `GLM53_BF16_GEMV`, `GLM53_BF16_GEMV_DEDUP_ROUTER` (kit) | 1, 1 | BF16 GEMV for small dense decode GEMMs; drop the duplicate router GEMM (`BF16_GEMV.md`) |
| `GLM53_PREFILL_FUSED_CAP` (kit) | 1 | smaller thin/fat split for the E3 prefill branch (`PREFILL_CAP.md`) |
| `GLM53_PREFILL_QUICKWINS` (kit, env.r16) | all | exact prefill quick wins (`PREFILL_QUICKWINS.md`) |
| `GLM53_MLA_PREFILL` (kit, env.r16) | 1 | exact sparse-MLA prefill kernel (`MLA_PREFILL.md`) |
| `GLM53_DENSE_W8A8`, `GLM53_DENSE_W8A8_ONLY`, `GLM53_DENSE_W8A8_FP8AG` (kit) | 1, `kda.in_proj_qkvbfg_a,mla.o_proj`, 1 | prefill W8A8 GEMMs on two projection types only, with an FP8 sequence-parallel all-gather (`DENSE_W8A8.md`) |
| `GLM53_MOE_E4M3`, `GLM53_MOE_E4M3_MAINLOOP`, `GLM53_MOE_E4M3_ACC`, `GLM53_MOE_E4M3_FOLD_SHARED` (kit) | 1, 1, bf16, 1 | routed-MoE prefill on e4m3 tensor cores, lean fused mainloop, bf16 accumulation folded into the shared expert (`MOE_E4M3.md`, `OPT_MOE*.md`). This one changes numerics (long KL 0.0049 -> 0.0132), accepted on 10-02 |
| `GLM53_KDA_FLASHKDA` (kit) | 1 | FlashKDA chunked prefill for the KDA layers (`KDA_FLASHKDA.md`) |
| `GLM53_KDA_STRIDED_QKV` (kit) | 1 | strided q/k/v/beta in the KDA recurrent decode, vLLM #55736 (`KDA_STRIDED_QKV.md`) |
| `GLM53_MHC_SP`, `GLM53_MHC_FUSED` (kit) | 1, 1 | sequence-parallel mHC prefill over TP; fused mHC post+prenorm (`MHC_SP.md`, `OPT_KDAMHC.md`) |
| `GLM53_DEC_SMALLOPS`, `GLM53_DEC_SMALLOPS_KINDS` (kit, env.r16) | 1, dconv | small decode kernels outside the MoE (`DEC_SMALLOPS.md`) |
| `GLM53_DEC_HOSTLOOP`, `GLM53_DEC_HOSTLOOP_WAKE` (kit) | 1, auto | decode host-loop changes and rank-skew wake-up (`DEC_HOSTLOOP.md`) |
| `GLM53_DEC_KDA_LAZY` (kit) | 1 | one KDA recurrent-state write per verify step instead of per row (`DEC_KDA_LAZY.md`) |
| `GLM53_DEC_AR1SHOT` (kit) | 1 | 2-rank decode all-reduce as one all-gather plus a local add, bit-identical (`DEC_AR1SHOT.md`) |
| `GLM53_MLA_PLAN_PIN` (kit) | 1 | page-locked staging of the sparse-MLA plan (`MLA_PLAN_PIN.md`) |
| `GLM53_DEC_FP8ROOF`, `GLM53_DEC_MOEGLUE_WARM` (kit, env.r16) | absent (off) | L2 pre-read of the next weights during decode: -5.8 ms in isolation, slower all-reduces in production (`docs/what-did-not-work.md`) |

### Prefix cache and the sparse indexer

| line | production value | what it does |
|---|---|---|
| `GLM53_KPOOL_SEED_STRIDE` (kit) | 1 | backport of vLLM #57477: the kpool prefill tail-seed kernel uses the real tail stride (without it every prefill seeds the wrong indexer block); prerequisite of the ring |
| `GLM53_KPOOL_RING` (kit, env.r16) | 1 | kpool tail ring sized for speculative decoding (`KPOOL_RING.md`) |
| `GLM53_KPOOL_TAIL_POSITIONS` (kit) | 2 | per-request tail rings in the persistent slot buffer; fixes concurrent decodes sharing tail block 0 (`PREFIX_HIT_TAIL.md`) |

### Weight edit

| line | production value | what it does |
|---|---|---|
| `ABLIT`, `ABLIT_METHOD`, `ABLIT_LAYERS`, `ABLIT_INCLUDE_MTP` | 1, auto, 15-45, 1 | replace o_proj of layers 15-45 with the uncensored donor's tensors at load (`docs/timeline.md`, 10-05). `auto` resolves to the transplant when `ablit/transplant/` is present |
| `ABLIT_DIRECTION`, `ABLIT_ALPHA` | dealign, 3.0 | used only by `ABLIT_METHOD=proj`, which production does not use |

Turning `ABLIT` off (`ABLIT=0`) gives the stock o_proj: same speed, long KL about 0.055 instead of 0.058, NLL 0.0035
lower, and the refusal behaviour of the stock model.

## 13. Running the fork's test suite

`code/fork/tests/run_all.sh` runs the fork's unit, integration, adversarial and handoff tests on a single GB10, inside
the serving image: one GPU container at a time under `flock /tmp/tf-gpu-bench.lock`, `--rm --network none`, refused
while less than 40 GB of host memory is available, each PyTorch process capped at `GPU_MEM_CAP_GB` (default 40) by
`tests/gpu_guard/` when it first initializes CUDA. After section 5 has built the extensions (they are left in `code/fork/` and `code/fork/overlay/`):

```bash
cd code/fork
tests/derive_assets.sh                     # once: the inputs that can be rebuilt (below), about 45 GB
SKIP_BUILD=1 R16=on LOG_DIR=/tmp/glm53-tests tests/run_all.sh
```

**Paths.** Tests find host directories through three variables, defaulted by `tests/paths.sh` and passed into every
test container (where `HOME=/tmp`), so Python code resolves the same paths inside and outside containers:

| variable | default | holds |
|---|---|---|
| `TF_EXL3_MODELS` | `~/models` | checkpoints (`GLM-5.3-Flash-Uncensored-NVFP4`, `GLM-5.3-Flash-DFlash2-dc77ff1c`, `GLM-OCR`) and the handoff mini checkpoints |
| `TF_EXL3_ASSETS` | `~/tf-exl3-assets` | `vllm-src/vllm`, `vllm-patches/*.patched`, `fi618/` (flashinfer 0.6.18), `prod-launcher/overlay` |
| `TF_EXL3_KITS` | `~` | deploy kits (`tf-exl3-deploy16*`) and branch worktrees some one-off review scripts compare against |

**Inputs.** `run_all.sh` checks each step's inputs first; a step whose inputs are missing prints
`--- <step>: SKIP (needs <what>: <path> missing ...)`, is counted apart from PASS and FAIL, and is listed again at the
end (`SUMMARY: n PASS, n FAIL, n SKIP`). A skipped step is not a passed step.

**Expected results.** `docs/test-status.md` lists every step with our result, the result to expect on your box
(which steps SKIP without the unpublished inputs) and the class and cause of every non-PASS result we saw.

| input | how to get it | steps that need it |
|---|---|---|
| serving image | section 2 | every GPU step |
| `GLM-5.3-Flash-Uncensored-NVFP4`: shards 1-8 of 62 + `config.json`, `generation_config.json` of `thebriangao/GLM-5.3-Flash-Uncensored-NVFP4` @ `59a99c95e6ea1142be39af0e617bdfcec0766052` (MIT; real bf16 / fp32 attention, indexer, mHC and dense-MLP tensors of layers 0, 1, 10-13, 45) | `derive_assets.sh models` | `test_gemv_install`, `test_smallops_*`, `review_smallops_adversarial`, the handoff mini |
| DFlash2 drafter (`incoai/GLM-5.3-Flash-DFlash2` @ `dc77ff1c`) | `derive_assets.sh models` | `test_fp8_roof`, `test_gemv_install`, `test_smallops_*`, `handoff`, the real drafter-fc checks of `test_fp8_large_m` (only together with the TR3 samples below) |
| `GLM-OCR/tokenizer.json` (`zai-org/GLM-OCR` @ `2e85a628`) | `derive_assets.sh models` | `handoff` (prompt tokenizer) |
| the image's vLLM sources | `derive_assets.sh vllm-src` (`docker create` + `docker cp`; the same files as ours) | `test_patch_kpool_tail_ring` |
| patched `cuda.py` / `flashinfer_mla_sparse_sm90.py` | `derive_assets.sh vllm-patches` (`code/vllm-patches/` applied; sha256 checked) | section C steps, `handoff` |
| launcher overlay | `derive_assets.sh launcher` (the launcher at `0f49cfd`; identical to the copy we tested with) | `handoff` |
| handoff mini checkpoint | `derive_assets.sh mini` (`tests/handoff/build_mini.py`, about 8 GB RAM, 20 s; `model.safetensors` byte-identical to ours. Its `config.json` records the source path, so its hash depends on `TF_EXL3_MODELS`) | `handoff` |
| flashinfer 0.6.18 (`fi618/`) | `tests/derive_assets.sh fi618` (public image by digest, byte-identical to production) | section C steps (`test_quickwins` ... `r16_loo_census`), `handoff` |
| deploy-r15 bundle (`prod-launcher/overlay/tf/site`) | not published | `r16_plugins` |
| bf16 samples and `lm_head` of the TR3 checkpoint (`GLM-5.3-Flash-EXL3-TR3-4bpw-partial`) | not published | none as a whole step: without them `test_fp8_large_m` runs its synthetic checks only (160 of 285 checks; the real drafter-fc checks belong to the real-sample set and are left out too, drafter mounted or not) and prints `(real samples not mounted: ...)`; `run_all.sh` prints a NOTE |

`tests/assets/models.sha256` and `tests/assets/assets.sha256` hold the sha256 of every derived file;
`derive_assets.sh` checks them. `tests/handoff/env_nonsecret.txt` is production's non-secret `.env` of 09-28 that
the handoff containers start from (addresses replaced by placeholders, which a single-container run does not use).

**`R16=on` and four precondition clashes.** `R16=on` puts the 8 ship flags of `tests/r16/flags.sh` into every
container, on top of what each test sets. Four tests assert a precondition that one of these flags overrides on purpose,
so under the full flag set they fail by design; `run_all.sh` runs each of them with only that knob left out (all other
flags on) and says so in the step header (`[R16=on without <knob>: precondition clash]`). `R16_STRICT=1` keeps the knob
and reproduces the four failures.

| step | knob left out | why the full set fails it |
|---|---|---|
| `test_smallops_install`, `review_smallops_adversarial` | `GLM53_DEC_SMALLOPS_KINDS` | they test the `mhc` kind, which the ship value `KINDS=dconv` excludes |
| `test_quickwins` | `GLM53_MLA_PREFILL` | W.0 expects its own function as the outermost `forward_mqa`; the MLA prefill wrapper deliberately is (their layering is checked by `r16_prefill_combo` X.1 and `r16_loo_census` C) |
| `test_hostloop_plugin` | `GLM53_DEC_MOEGLUE_WARM` | P1 / P2 / P5 assert that loading the plugins creates no CUDA context; `GLM53_DEC_MOEGLUE_WARM` creates one, as production's R15 flags already do. (The guard of our test box set its cap at `import torch`, which also created a context, so this test failed there in every configuration; the shipped guard sets the cap at the first CUDA initialization instead, and still turns a 3 GB allocation under `GPU_MEM_CAP_GB=2` into an OOM whether CUDA is first touched by a tensor, a stream or `mem_get_info`) |

On the original fork at `393c2a5` with production's `.so` files, `R16=on` without these exclusions gave exactly these
four failures and 49 passing steps (53 in all, including the old `check_docs`, which checked unpublished docs and logs and is
replaced by `tests/check_paths.py`).

**Build reproducibility.** A fresh `code/build/build_all.sh` does not reproduce production's bytes
(`code/kit/PRODUCTION_BINARIES.sha256`), but the differences are build metadata only. Of the 12 extensions,
9 differ in 1-3 bytes, all inside the nvcc temporary name `tmpxft_<pid>_...`; `glm53_mhc_fused_ext` and
`glm53_moe_e4m3_ext` also differ in the GNU build-id and the source mtime recorded in the fatbin's line table; and
`_flashkda_fp32_C` in the build-id, in nvcc's per-file hash of its anonymous namespace (`_INTERNAL_..._fwd_launch_cu_<hash>`
symbols in `.rodata`) and in its compressed fatbin. `cuobjdump -sass` of all three is identical line for line. The test
outcomes of a fresh build and of production's `.so` files were the same step for step.
