#!/usr/bin/env python3
"""Print the exact regions of the image's vllm glm5next model.py that patch_mhc_sp.py anchors on.

Run inside the production image (tests/gpu_run.sh python3 tests/mhc_sp/probe_anchor.py). Read-only.
"""
import hashlib
import os
import sys

SITE = os.environ.get("GLM53_SITEPKG", "/usr/local/lib/python3.12/dist-packages")
P = os.path.join(SITE, "vllm/models/glm5next/nvidia/model.py")
src = open(P).read()
print(f"file={P} sha256={hashlib.sha256(src.encode()).hexdigest()[:16]} lines={src.count(chr(10))}")
lines = src.split("\n")

MARKS = [
    "def sp_all_gather",
    "import os\n",
    "from vllm.models.common.ops.sequence_parallel import",
    "# In SP, the attention output projection leaves a partial sum",
    "        x = self.self_attn(",
    "        # Fully Connected",
    "    def forward(\n        self,\n        positions: torch.Tensor",
    "        full_num_tokens = positions.shape[0]",
    "            if post is not None and hasattr(layer, \"hc_post\"):",
    "            if self.is_sequence_parallel:\n                value = sp_all_gather(value)[:full_num_tokens]",
    "        if self.is_sequence_parallel:\n            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]",
    "        if self.layer_idx == self.num_hidden_layers - 1:",
    "        self._active_layers = self.layers[self.start_layer : self.end_layer]",
]
for m in MARKS:
    idxs = [i for i, l in enumerate(lines) if m.replace("\n", "\n") in l or m in "\n".join(lines[i:i + 2])]
    print(f"\n===== anchor {m.splitlines()[0][:60]!r}: {len(idxs)} hit(s) {idxs[:6]}")
    for i in idxs[:2]:
        lo, hi = max(0, i - 3), min(len(lines), i + 14)
        for n in range(lo, hi):
            print(f"{n + 1:5d}|{lines[n]}")
sys.exit(0)
