# Third-party notices

This repository is licensed under the GNU Affero General Public License v3.0 (`LICENSE`), the heaviest licence of the
code it builds on (the Mia-AI-Lab launcher). It contains code derived from, or copied from, the projects below. Their
notices are kept here and, where a file carries its own header, in that file.

## Code in this repository

| component | where it appears here | licence |
|---|---|---|
| Mia-AI-Lab "GLM-5.3-Flash-EXL3-2x-DGX-Sparks" launcher (github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) | `code/kit/launcher/start.sh` and `code/kit/launcher/overlay/*` are modified versions of its `start.sh` and overlay scripts; `code/fork/launcher/`, `code/fork/overlay/patch_tf_bundle.py`, `code/fork/docs/ref/mia_*`, `code/fork/docs/ref/prod_live/*`, `code/fork/docs/ref/launcher_0924/*` and `code/fork/docs/prod_exl3_reference.py` are copies or modified copies of its files; `code/launcher-patches/` is a patch against it | AGPL-3.0; contributions made before 2026-09-07 were MIT (notice below) |
| TensorFold (github.com/ashhart/TensorFold) | `code/fork/kernels/exl3.cu`, `exl3.cpp` (modified), `exl3_format_ref.py`, `exl3_mm_ref.py` (unmodified), `code/fork/docs/ref/tf_*` | MIT for the code copied here (copied 2026-09-27; TensorFold moved to Apache-2.0 from its 0.6.0 release, commit e3ac0eaa of 2026-09-30). Notice in `code/fork/THIRD_PARTY_NOTICES.md` |
| ExLlamaV3 (github.com/turboderp-org/exllamav3) | `code/fork/docs/ref/xl_*` (reference copies); the EXL3 format and the numerics that `TF_PARITY=1` reproduces | MIT, Copyright (c) 2025 Turboderp. Notice in `code/fork/THIRD_PARTY_NOTICES.md` |
| vLLM (github.com/vllm-project/vllm) | `code/vllm-patches/*.patch` (diffs of three vLLM files); the runtime overlays in `code/kit/overlay/` and `code/kit/site/` patch vLLM source text and carry fragments of it; `code/fork/tests/kpoolring/ref/` (vLLM PR #58454) | Apache-2.0 |
| FlashKDA (github.com/vllm-project/FlashKDA) | not vendored. `code/build/fetch_flashkda_sources.sh` downloads it at 17a037d; `code/fork/tests/fkda*/` patch and build it | MIT, Copyright (c) 2026 MoonshotAI (the launcher's `docs/licenses/Apache-2.0-FlashKDA.txt` says Apache-2.0; the repository's own LICENSE says MIT) |
| CUTLASS (github.com/NVIDIA/cutlass) | not vendored; fetched at 5c149f52 for the FlashKDA build; the W8A8 kernel includes the CUTLASS headers shipped in the image's flashinfer | BSD-3-Clause, Copyright (c) 2017-2025 NVIDIA CORPORATION & AFFILIATES |
| FlashInfer (github.com/flashinfer-ai/flashinfer) | not vendored; production mounts a flashinfer 0.6.18 dev build (0.6.18.dev20260819, git 61a6c651) over the image's 0.6.17, copied by `tests/derive_assets.sh fi618` out of the public image ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef6... | Apache-2.0 |
| kindling GLM-5.3-Flash GX10 notes (github.com/kindlingai/glm-5.3-flash-gx10) | not vendored; `code/operator/gate_prefill_32k.py` loads its `gate/prefill.py` from a local checkout | no licence file found at commit 45b438be, so nothing of it is copied here |

## Run-time components (not distributed here)

| component | used as | licence |
|---|---|---|
| Docker image `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor` (sha256:447114ee...) | the serving image | a mix: vLLM (Apache-2.0), PyTorch (BSD-3-Clause), exllamav3 (MIT), FlashInfer (Apache-2.0), InstantTensor 0.2.0 (Apache-2.0), NCCL (nvidia-nccl-cu13 2.30.7, NVIDIA licence), CUDA 13.0.1 (NVIDIA EULA), the launcher's own layers (AGPL-3.0) |
| `zai-org/GLM-5.3-Flash` | base model | MIT |
| `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` | production weights until 2026-10-04 (the older numbers in `docs/`); not needed to reproduce the current setup, and none of its files are shipped here | its own repository's terms (no longer publicly listed on 2026-10-07; the mirror `brandonmusic/GLM-5.3-Flash-tr3-4bpw`, whose `config.json` is byte-identical, declares `other`) |
| `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold` @ 6c5b2826 | production weights | Apache-2.0 (the underlying model stays MIT) |
| `incoai/GLM-5.3-Flash-DFlash2` @ dc77ff1c | speculative-decoding drafter | **CC BY-NC-ND 4.0**: non-commercial, no derivatives. This repository ships none of its weights; the FP8 conversion of the drafter happens in memory at load and is never written out |
| `thebriangao/GLM-5.3-Flash-Uncensored-NVFP4` @ 59a99c95 | test input only: 8 of its 62 shards give real bf16 / fp32 tensors to `code/fork/tests/` (`tests/derive_assets.sh models`) | MIT |
| `zai-org/GLM-OCR` @ 2e85a628 | test input only: its `tokenizer.json` is the prompt tokenizer of `code/fork/tests/handoff` | MIT |
| `dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4` | donor of the o_proj tensors of the ABLIT transplant (fetched at install time, never committed) | MIT |

Weights are never redistributed in this repository. Each model keeps the licence of its own repository.

## AGPL-3.0 section 13

The patched server is a network service. If you run a modified version of it for other users, section 13 requires you
to offer those users the corresponding source. This repository, together with the public launcher commit named in
`REPRODUCE.md`, is that source for the configuration described here.

## Notice of the launcher's pre-2026-09-07 MIT licence

```
This project is licensed under the GNU Affero General Public License v3.0
(see LICENSE). Before 2026-09-07 it was distributed under the MIT license
reproduced below, which is retained for the contributions made under it.

----------------------------------------------------------------------

MIT License

Copyright (c) 2026 Mia's AI Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
