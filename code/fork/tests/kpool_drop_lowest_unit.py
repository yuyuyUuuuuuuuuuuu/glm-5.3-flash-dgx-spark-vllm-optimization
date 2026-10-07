#!/usr/bin/env python3
"""[glm53-kpool-drop-lowest] Unit test of the overlay's _kpool_keep_highest_pools (nodeC, tests/gpu_run.sh).

Proves, for the helper AS INSTALLED by overlay/patch_kpool_drop_lowest.py into a copy of the image's
vllm/model_executor/layers/sparse_attn_indexer_kpool.py (pristine source = the assets tree, overlaid here into
/tmp -- never the repo, never the image):
  U1 the kept SET is exactly the top ``keep`` pools by score, and the ORDER is score desc then pool id asc,
     against an independent numpy float64 reference (lexicographic (-score, id));
  U2 with exact score ties (including an all-equal row) the tie-break is the LOWER pool id, in both set and order;
  U3 -1 fills: a valid pool is never dropped while a -1 is kept; a row with fewer valid pools than ``keep`` keeps
     all of them plus its -1s (the expand kernel maps pid < 0 to -1, so this is semantics-preserving);
  U4 extreme magnitudes (0.0 vs -0.0, +-inf, tiny/huge) order like the reference; NaN never appears (a selected
     pool's logit is a real number) - one all-NaN row is asserted to be passed through without a host error and
     with a shape/dtype-identical result instead of relying on NaN ordering;
  U5 output dtype is int64 and the shape is exactly [rows, keep] (the FULL-graph contract: the patched statement
     must keep the truncated slice's shape);
  U6 no CUDA -> host sync and CUDA-graph capture/replay safety (tests/kpool_drop_lowest_det.py covers the sync +
     graph part with the real top-k ops; this file only needs the helper).
Run: tests/gpu_run.sh python3 tests/kpool_drop_lowest_unit.py          (~2 GB GPU, no allocation beyond that)
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRISTINE = os.path.join(os.environ.get("TF_EXL3_ASSETS") or os.path.expanduser("~/tf-exl3-assets"), "vllm-src/vllm/model_executor/layers/sparse_attn_indexer_kpool.py")

fail = 0
def ck(cond, label):
    global fail
    print(("ok   " if cond else "FAIL ") + label)
    if not cond:
        fail = 1


import torch  # noqa: E402  (needed by the exec namespace; CUDA is initialised lazily)


def load_patched_helper(device: str):
    """Apply the overlay to a pristine copy, then exec the patched file's _kpool_keep_highest_pools AS WRITTEN.

    Importing the whole patched module would re-register the image's ``vllm::sparse_attn_indexer_kpool`` custom op
    (a second definition of the same schema raises), so the shipped statements are extracted from the patched file
    with ast and run in a bare namespace: exactly the code the overlay installs, nothing else.
    """
    tmp = tempfile.mkdtemp(prefix="kpooldown-")
    dst = os.path.join(tmp, "sparse_attn_indexer_kpool.py")
    shutil.copyfile(PRISTINE, dst)
    env = dict(os.environ, GLM53_KPOOL_DROP_LOWEST="1", GLM53_SPARSE_INDEXER_KPOOL_PY=dst)
    r = subprocess.run([sys.executable, os.path.join(REPO, "overlay", "patch_kpool_drop_lowest.py")],
                       env=env, capture_output=True, text=True)
    ck(r.returncode == 0, f"overlay applied to a pristine copy (rc={r.returncode}: {r.stdout.strip()[-60:]}{r.stderr.strip()[-120:]})")
    patched = open(dst).read()
    ck(patched.count("[glm53-kpool-drop-lowest]") == 3 and "pool_ids[:, : select_k - 1]" not in patched,
       "the patched file carries the overlay's 3 markers and no last-column truncation")
    import ast
    tree = ast.parse(patched)
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_kpool_keep_highest_pools"]
    ck(len(fns) == 1, "the patched module defines _kpool_keep_highest_pools exactly once")
    src = ast.get_source_segment(patched, fns[0])
    ns: dict = {"torch": torch,   # the module-level name the shipped statements use
                # [opt-decodekit] the module-level mode flags: this test checks the torch helper (ORDER=sorted, the
                # r16l reference); the default fused Triton kernel is checked BITWISE against it by
                # tests/kpool_drop_lowest_bench.py (which imports the helper from a real file: triton.jit needs one)
                "_GLM53_KPOOL_DL_ORDER": "sorted", "_GLM53_KPOOL_DL_FUSED": False, "_GLM53_KPOOL_DL_STOCK_ORDER": False}
    exec(compile(src, dst, "exec"), ns)   # noqa: S102 - the extracted statements of the shipped patch
    fn = ns["_kpool_keep_highest_pools"]
    ck(callable(fn), "_kpool_keep_highest_pools callable (the shipped statements)")
    return fn, tmp


def reference(scores: "np.ndarray", ids: "np.ndarray", keep: int):
    """Independent reference: top keep by (-score, id) over the selected, valid columns; -1 columns are -inf.

    ``scores`` is the full [rows, num_pools] logit matrix, ``ids`` the [rows, K] selection to cut down.
    """
    import numpy as np
    sc = np.take_along_axis(scores.astype(np.float64), ids, axis=1)
    sc = np.where(ids >= 0, sc, -np.inf)
    # exact ties -> lower id first: sort by (-score, id) lexicographically, per row
    order = np.lexsort((ids, -sc), axis=1)
    picked = order[:, :keep]
    return np.take_along_axis(ids, picked, axis=1)


def main() -> int:
    import numpy as np

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")
    fn, tmp = load_patched_helper(dev)
    try:
        g = np.random.default_rng(20260930)
        cases = []  # (label, scores fp32 [R,P], ids int32 [R,K], keep)
        R, P, K = 6, 64, 16
        keep = K - 1
        # U1 random continuous scores, every row fully valid
        sc = g.standard_normal((R, P)).astype(np.float32) * 8
        ids = np.stack([g.choice(P, K, replace=False) for _ in range(R)]).astype(np.int32)
        cases.append(("random continuous", sc, ids, keep))
        # U2 exact ties: quantized scores, and one all-equal row (every score == 0.25)
        sc = (g.standard_normal((R, P)) * 4).astype(np.float32)
        sc = np.round(sc * 4) / 4  # many exact fp32 ties
        cases.append(("quantized ties", sc, ids, keep))
        cases.append(("all scores equal", np.full((R, P), 0.25, np.float32), ids, keep))
        # U3 -1 fills: per-row valid counts < K; one row with a single valid pool
        sc = g.standard_normal((R, P)).astype(np.float32) * 3
        ids3 = ids.copy()
        for r, nval in zip(range(R), (K - 4, K - 1, 3, 1, K, 2)):
            cols = g.permutation(K)[: K - nval]
            ids3[r, cols] = -1
        cases.append(("-1 fills (valid < K)", sc, ids3, keep))
        sc = g.standard_normal((R, P)).astype(np.float32) * 3
        ids4 = ids.copy()
        ids4[:, K - 1:] = -1  # exactly keep valid pools: every -1 must be dropped, no valid lost
        cases.append(("exactly keep valid", sc, ids4, keep))
        # U4 extremes: +-0, +-inf, tiny/huge magnitudes; one row that is all-NaN (a logit of a selected pool is
        # a real number, so this only proves the helper degrades gracefully: no host error, shape kept)
        sc = g.standard_normal((R, P)).astype(np.float32) * 3
        sc[0, 0], sc[1, 1], sc[2, 2], sc[3, 3] = 0.0, -0.0, np.inf, -np.inf
        sc[4, :8] = 1e-38
        sc[5, :8] = 3e38
        cases.append(("extremes", sc, ids, keep))
        sc = g.standard_normal((R, P)).astype(np.float32) * 3
        sc[2, :] = np.nan
        cases.append(("one all-NaN row (graceful)", sc, ids, keep))

        for label, sc_np, ids_np, kp in cases:
            sc_t = torch.from_numpy(sc_np).to(dev)
            ids_t = torch.from_numpy(ids_np).to(dev)
            out = fn(sc_t, ids_t, kp)
            ref = reference(sc_np, ids_np, kp)
            got = out.cpu().numpy()
            ck(out.dtype == torch.int64 and tuple(out.shape) == (R, kp),
               f"{label}: dtype int64, shape [R,{kp}] (got {out.dtype}, {tuple(out.shape)})")
            # a row of NaN scores has no defined order (the reference's lexsort would be arbitrary too): only
            # its shape/dtype/no-host-error is asserted above, the per-row checks skip it
            rows = [r for r in range(R) if not np.isnan(sc_np[r]).any()]
            same_rows = [
                (sorted(got[r].tolist()) == sorted(ref[r].tolist()), got[r].tolist() == ref[r].tolist())
                for r in rows
            ]
            ck(all(s for s, _ in same_rows),
               f"{label}: kept SET == top {kp} by score on every row")
            ck(all(o for _, o in same_rows),
               f"{label}: order == score desc, pool id asc on every row (bitwise-stable order)")
            # the dropped pool (when all K are valid) is the lowest by (-score, id)
            if ids_np.min() >= 0 and not np.isnan(sc_np).any():
                sel_sc = np.take_along_axis(sc_np.astype(np.float64), ids_np, axis=1)
                sel_order = np.lexsort((ids_np, -sel_sc))
                want = np.take_along_axis(ids_np, sel_order[:, K - 1:], axis=1)  # the lowest, per row
                ok_drop = all(set(ids_np[r].tolist()) - set(got[r].tolist()) == {int(want[r][0])}
                              for r in range(R))
                ck(ok_drop, f"{label}: the dropped pool is the lowest-scored selected one (ties: higher id)")
        # U3 explicit: with a valid pool and a -1 both at risk of the cut, the -1 goes and the valid stays
        sc_t = torch.zeros(1, 8, device=dev)
        sc_t[0, 3] = -5.0
        ids_t = torch.tensor([[0, 1, 2, 3, 4, 5, 6, -1]], dtype=torch.int32, device=dev)
        out = fn(sc_t, ids_t, 3)
        ck(-1 not in out.tolist()[0] and {0, 1, 2} <= set(out.tolist()[0]),
           "the -1 fill is dropped first, no valid pool lost (explicit 8-pool row)")
        print(f"unit: {'ALL OK' if not fail else 'FAILURES'}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return fail


if __name__ == "__main__":
    sys.exit(main())
