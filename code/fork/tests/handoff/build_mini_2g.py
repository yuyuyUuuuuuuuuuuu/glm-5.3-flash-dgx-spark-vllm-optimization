#!/usr/bin/env python3
"""decode4-kdalazyfix: the handoff mini model with TWO mamba groups (15 layers: 10 KDA + 5 DSA), host side, numpy only.

Why: GLM-5-Next's hybrid KV layout (vllm/v1/core/kv_cache_utils.py `_glm5_next_tensor_layout`) gives one KV tensor per
MLA layer, co-owned by that MLA layer and ONE KDA layer of EACH mamba group (group size = the MLA layer count). The
10-layer mini (5 KDA + 5 DSA) has a single mamba group, so no two KDA layers ever share a recurrent-state tensor;
production (34 KDA + 11 MLA) shares each tensor between up to four KDA layers. This mini (10 KDA + 5 DSA) shares each
tensor between two KDA layers, the condition under which GLM53_DEC_KDA_LAZY v1 corrupted decode.
Same sources / same per-rank shapes as build_mini.py; layer i's weights repeat the five real KDA / DSA sources.
Usage: build_mini_2g.py [out dir, default $TF_EXL3_MODELS/GLM-5.3-Flash-handoff-mini2g]
"""
import os
import sys
from pathlib import Path

sys.argv = [sys.argv[0], sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("TF_EXL3_MODELS") or os.path.expanduser("~/models"), "GLM-5.3-Flash-handoff-mini2g")]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_mini as B  # noqa: E402

K = [(0, 0, 0), (1, 1, 1), (10, 10, 0), (12, 12, 1), (13, 13, 0)]
D = [(11, 11, 1), (45, 13, 0), (11, 10, 1), (45, 12, 0), (11, 11, 1)]
L = []
for i in range(5):            # K K D  K K D ...  (production's 3:1 is not needed; group size = 5 MLA layers)
    a, b = K[(2 * i) % 5], K[(2 * i + 1) % 5]
    L.append(("linear_attention",) + a)
    L.append(("linear_attention",) + b)
    L.append(("deepseek_sparse_attention",) + D[i])
B.LAYERS[:] = L
B.OUT = Path(sys.argv[1])
B.main()
