#!/usr/bin/env bash
# CPU-only python in the production image (no GPU, no network): tools/cpu.sh script.py args
exec docker run --rm --network none --memory 16g -e CUDA_VISIBLE_DEVICES= -v ${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}/prefixhit:${TF_EXL3_ASSETS:-$HOME/tf-exl3-assets}/prefixhit -w "$PWD" --entrypoint python3 ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor "$@"
