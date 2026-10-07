# Third-party notices

This repository (tf-exl3-fork) contains code derived from, and reference copies of, third-party software.

## 1. Code in this repository derived from third-party sources

### TensorFold — MIT (TensorFold relicensed to Apache-2.0 from its 0.6.0 release on 2026-09-30; this code was copied on 2026-09-27, under MIT)
`kernels/exl3.cu` and `kernels/exl3.cpp` are modified versions of
`src/tensorfold/families/glm5_next/cuda/exl3.cu` / `exl3.cpp` from https://github.com/ashhart/TensorFold.
`kernels/exl3_format_ref.py` and `kernels/exl3_mm_ref.py` are unmodified copies of `exl3.py` / `exl3_mm.py` from the same directory.

```
MIT License

Copyright (c) 2026 TensorFold contributors

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

### ExLlamaV3 — MIT
The EXL3 format and the numerical reference that `TF_PARITY=1` reproduces are ExLlamaV3's
(https://github.com/turboderp-org/exllamav3, v0.0.43). No ExLlamaV3 source is compiled into this repository's
extension; the production `exllamav3_ext` binary is called at run time.

```
MIT License

Copyright (c) 2025 Turboderp

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
SOFTWARE.```

## 2. Reference copies under `docs/`

These files were copied read-only for analysis and keep their original licenses (the table). The published
repository as a whole is AGPL-3.0 (see `../../LICENSE` and `../../THIRD_PARTY_NOTICES.md`), which is compatible with
every licence below.

| Path | Origin | License |
|---|---|---|
| `docs/prod_exl3_reference.py` | `vllm/model_executor/layers/quantization/exl3.py` inside the image `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor` (MiaAI-Lab's EXL3 integration for vLLM) | treat as **AGPL-3.0** (MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks, per GitHub) unless shown to be upstream vLLM (Apache-2.0) |
| `docs/ref/mia_*` | Mia's fat-kernel sources and patch script from the same image | **AGPL-3.0** (MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) |
| `docs/ref/prod_live/overlay_exl3.py`, `docs/ref/prod_live/inner_start.masked.sh` | the launcher's `overlay/exl3.py` (HEAD df864b5) that production installs over the image's `quantization/exl3.py`, and the production inner start script (secrets masked) | **AGPL-3.0** (MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) |
| `docs/ref/launcher_0924/*.py` | three `overlay/` patch scripts of the launcher MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks (09-24 snapshot): `patch_exl3_decode_pipeline.py` and `patch_exl3_ext_aarch64.py` (inputs of the thin-decode `exllamav3_ext` build nodeC measured against), `patch_dense_fp8.py` (installs the overlay exl3.py over the image's) | **AGPL-3.0** (MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) |
| `docs/ref/xl_*` | ExLlamaV3 extension sources from the same image | MIT (above) |
| `docs/ref/tf_*` | TensorFold python sources | MIT (above) |

## 3. Run-time interaction with AGPL-3.0 code

`integrate.py` replaces `exllamav3_ext.exl3_moe` and wraps `build_exl3_fused_state` of the production module at run
time; `tf_exl3_moe.production_routing` calls the production module's `map_topk_to_local`. No AGPL source is copied into
`kernels/`, `tf_exl3_moe.py`, `integrate.py` or `tests/` (checked 2026-09-27: no line of >= 45 characters in those files
matches a line of the AGPL reference copies, other than production attribute names read for interoperability:
`_exl3_bits`, `_exl3_k_words`, `_exl3_fused_temps`). If a patched production server is offered to users over a network,
review AGPL-3.0 section 13 obligations first.
