# code/

Everything needed to rebuild the production configuration. `REPRODUCE.md` at the repository root is the procedure;
this file says what each directory is and what was left out.

| path | contents |
|---|---|
| `fork/` | the kernel fork ("tf-exl3-fork") at commit `393c2a5e13a6351393dfa4c16e9e927ede391603`, the commit the production kit was built from: CUDA kernels (`kernels/`), the Python modules that load them into vLLM (top level), the opt-in overlay patches (`overlay/`), build scripts (`setup.py`, `tools/moee4m3/build.py`, `tools/mhcfused/build.py`, `tests/fkda/build_flashkda_fp32.sh`), unit and integration tests (`tests/`), per-feature design and measurement notes (`docs/`), and the production check tools (`tools/prodcheck/`, `tools/rail2/`, `tools/memprep/`) |
| `kit/` | the deploy kit `r16z8p` as production installs it: the launcher `start.sh` and four launcher overlays (`launcher/`), the bundle `site/` and `overlay/` (sources; the `.so` files are built by `build/build_all.sh`), `env.r16`, the install / switch / check tools (`tools/`), `ROLLOUT.md` (the operator's rollout record), `BUILD.txt` (the build record), `env.production.example` (production's `.env` without secrets) and `PRODUCTION_BINARIES.sha256` (production's `.so` bytes) |
| `operator/` | the operator workspace (`~/tf-exl3-deploy` on the head node): the bench and probe scripts behind every published number, the idle-gated restart, the second-rail helper, the 32k prefill yardstick wrapper and the teacher-forced KL harness (`klh/`) |
| `build/` | `fetch_flashkda_sources.sh` (pinned FlashKDA, CUTLASS and vLLM shim sources) and `build_all.sh` (builds every extension inside the serving image and stages it into the kit) |
| `vllm-patches/` | three patches against files of the serving image's vLLM, mounted by `start.sh` (SM90 sparse-MLA backend on sm_121, and a Responses API input fix) |
| `launcher-patches/` | the one launcher change outside `start.sh` and `overlay/`: the chat-template `reasoning_effort_placement` kwarg |

Some files exist in two places on purpose: `kit/` holds the installed copies and `fork/` the sources they were taken
from (`fork/overlay/*` = `kit/overlay/*`, the top-level `fork/*.py` modules = `kit/site/*.py`,
`fork/tools/deploy16/*` = `kit/tools/*`, `fork/tools/prodcheck/*` and `fork/tools/memprep/memprep.py` =
`operator/*`). `kit/docs/` of the original kit was a copy of `fork/docs/` and is not repeated here.

## Changes against the original trees

- Host names, addresses, user names and home paths were replaced: the head is `nodeA`, the worker `nodeB`, the
  single test GPU box `nodeC`; addresses are `${HEAD_IP}` / `${WORKER_IP}` or example `10.0.x.y` subnets; home
  directories are `${HOME}` / `${WORKER_HOME}`. Scripts that run on the cluster read `WORKER_SSH`, `HEAD_SSH`,
  `LAUNCHER_DIR` and `DEPLOY_DIR` instead of fixed values. API keys are read from the launcher `.env` or from
  `VLLM_API_KEY`, never stored.
- `kit/tools/apply_r16.sh` gained `FRESH_INSTALL=1` (a first install into the public launcher); `kit/tools/make_manifest.sh`
  is new. `kit/MANIFEST.sha256` was regenerated over the shipped files.
- Not included: the build outputs (`.so` files and a 1 MB test binary), the fork's raw logs (`docs/logs/`, 529 MB),
  the rollout internals of the kit (`prev-r16/`, `revert-r15/`) and of the fork (the kit-derivation chain scripts
  under `tools/deploy16/`, which need the previous kits on the operator's disk), internal status and planning notes (comments and docstrings that cite `docs/STATUS.md`, `docs/PRODUCTION_PLAN.md` or `docs/logs/` point at these unpublished files; `kit/site/` keeps production's exact bytes, so they were not edited),
  the per-kit rollout notes (`docs/DEPLOY_R16*.md`; the last one is `kit/ROLLOUT.md`), a helper that granted
  temporary sudo on the operator's machines, a TR3 model config file (model licence), and test fixtures that hold
  model weights or recorded model states.
- Code under `fork/tests/` was run on the operator's test box against local model copies. Its paths now go through
  `TF_EXL3_MODELS` (default `~/models`), `TF_EXL3_ASSETS` (default `~/tf-exl3-assets`) and `TF_EXL3_KITS` (default
  `~`), set in `fork/tests/paths.sh` and passed into the test containers; `fork/tests/check_paths.py` fails on any
  machine-specific path left in code. `fork/tests/gpu_guard/` (the per-process CUDA memory cap the test containers
  load; it now sets the cap at the first CUDA initialization instead of at `import torch`, which created a CUDA
  context and broke `tests/test_hostloop_plugin.py`), `fork/tests/derive_assets.sh`, `fork/tests/assets/*.sha256` and `fork/tests/handoff/env_nonsecret.txt` are
  additions; `tests/check_docs.py` (it checked docs and logs that are not published) was removed from `run_all.sh`.
