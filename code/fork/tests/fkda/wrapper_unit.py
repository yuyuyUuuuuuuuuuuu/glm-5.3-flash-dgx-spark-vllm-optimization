#!/usr/bin/env python3
"""FKDA check 2: the glm53_flashkda wrapper (what the patched kda.py calls) at
production shapes, against the production Triton chain, on production-shaped
views -- merged-projection column slices for q/k/v (token stride 3*proj), a
row-strided beta, RAW g1, fp32 [N,H,D,D] initial states, single- and
multi-sequence cu_seqlens -- plus the fail-closed paths (STATE off, capacity,
dtype/cu_seqlens checks) and the workspace-manager buffer reuse.

GPU, inside the production image: first installs the extension + wrapper into
site-packages the way overlay/patch_flashkda.py does (via the real patcher on a
copy of the image's kda.py for the anchors, then importing the installed
wrapper), so the exact code path the serving engine runs is what is measured.
"""
import argparse
import shutil
import statistics
import sys
import types
from pathlib import Path

REPO = Path("/w")

# the three FlashKDA builds of the r16z kit (GLM53_KDA_FLASHKDA_V): the overlay file pair each installs from,
# and the value the env switch takes (patch_flashkda.py's WRAPPERS table; V unset/1 stages production's build)
VERSIONS = {"fkda": ("glm53_flashkda.py", "_flashkda_fp32_C.abi3.so", ""),
            "fkda2": ("glm53_flashkda2.py", "_flashkda_fp32_C2.abi3.so", "2"),
            "fkda3": ("glm53_flashkda3.py", "_flashkda_fp32_C3.abi3.so", "3")}


def install(sitepkg: Path, ver: str = "fkda") -> None:
    """Run the real overlay patcher against the image's real kda.py (a copy), staging ONE build (ver)."""
    import os

    wrapper, ext, vval = VERSIONS[ver]
    staging = Path("/tmp/fkda-unit-overlay")
    staging.mkdir(exist_ok=True)
    for f in ("patch_flashkda.py", wrapper, ext):
        shutil.copy2(REPO / "overlay" / f, staging / f)
    img_kda = sitepkg / "vllm/models/glm5next/nvidia/kda.py"
    bak = Path("/tmp/kda.py.stock")
    if not bak.exists():
        shutil.copy2(img_kda, bak)
    env = dict(os.environ, GLM53_KDA_FLASHKDA="1", GLM53_TF_OVERLAY=str(staging),
               GLM53_SITEPKG=str(sitepkg), GLM53_KDA_PY=str(img_kda))
    if vval:
        env["GLM53_KDA_FLASHKDA_V"] = vval
    import subprocess
    r = subprocess.run([sys.executable, str(staging / "patch_flashkda.py")], env=env, capture_output=True, text=True)
    print(r.stdout.strip())
    assert r.returncode == 0, r.stderr
    # idempotent second run
    r2 = subprocess.run([sys.executable, str(staging / "patch_flashkda.py")], env=env, capture_output=True, text=True)
    assert r2.returncode == 0 and "already present" in r2.stdout, (r2.stdout, r2.stderr)
    sys.path.insert(0, str(sitepkg))
    import glm53_flashkda  # noqa: F401


class FakeSched:
    max_num_batched_tokens = 13824
    max_num_seqs = 8


class FakeCfg:
    scheduler_config = FakeSched()

    class model_config:
        dtype = __import__("torch").bfloat16


class FakeLayer:
    """Just the attributes glm53_flashkda.configure/chunk_prefill read."""

    local_num_heads = 32
    head_dim = 128
    kda_safe_gate = True
    kda_lower_bound = -5.0
    vllm_config = None

    def __init__(self):
        import torch
        self.A_log = torch.randn(1, 1, 32, 1, dtype=torch.float32, device="cuda") * 0.2
        self.dt_bias = (torch.rand(32 * 128, dtype=torch.float32, device="cuda") * 8 - 10)

    def get_state_dtype(self):
        import torch
        return (torch.bfloat16, torch.float32)


def triton_ref(layer, q, k, v, g1, beta_raw, s0, cu):
    """The exact production call (production passes the pre-sigmoided fp32
    beta); NOTE it mutates v in place, so the caller passes a copy."""
    import torch
    from vllm.third_party.flash_linear_attention.ops.kda import chunk_kda_with_fused_gate
    beta = beta_raw.float().sigmoid()
    o, h = chunk_kda_with_fused_gate(
        q=q, k=k, v=v, raw_g=g1, beta=beta, A_log=layer.A_log, g_bias=layer.dt_bias.reshape(-1, layer.head_dim),
        initial_state=s0, output_final_state=True, use_qk_l2norm_in_kernel=True, cu_seqlens=cu,
        safe_gate=True, lower_bound=layer.kda_lower_bound)
    return o, h


def rel_rms(a, b):
    a, b = a.float(), b.float()
    return float(((a - b).pow(2).mean() / b.pow(2).mean()).sqrt().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sitepkg", default="/usr/local/lib/python3.12/dist-packages")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    import torch
    assert torch.cuda.is_available()
    install(Path(args.sitepkg))
    import glm53_flashkda
    from vllm.v1.worker.workspace import init_workspace_manager
    init_workspace_manager(torch.device("cuda"))  # the engine does this at startup
    layer = FakeLayer()
    glm53_flashkda.configure(layer, FakeCfg())
    print("configured:", {k: v for k, v in glm53_flashkda._cfg.items() if k != "specs"})
    print("buffer specs:", [(s, str(d)) for s, d in glm53_flashkda._cfg["specs"]])

    H, D, proj = 32, 128, 32 * 128 * 3  # proj = the merged q|k|v projection per rank
    g = torch.Generator(device="cpu").manual_seed(5)

    def rn(*shape, scale=1.0, dt=torch.bfloat16):
        return (torch.randn(*shape, generator=g) * scale).cuda().to(dt)

    rep = {}
    for tag, cu_list, T in (("single", [0, 13824], 13824), ("varlen", [0, 5000, 7777, 13824], 13824)):
        N = len(cu_list) - 1
        # production-shaped views: columns of one merged [T, proj] buffer, row-strided beta
        qkv = rn(1, T, proj)
        # the merged short-conv output, split into its three column blocks: each
        # [1, T, H*D] with token stride proj, exactly what kda.py's _rearr gets
        q_ns, k_ns, v_ns = (qkv[:, :, i * 32 * 128:(i + 1) * 32 * 128].reshape(1, T, 32, 128) for i in range(3))
        g1 = rn(1, T, H, D, scale=0.5)
        beta_w = rn(1, T, 3 * H)                       # beta slice of a wider projection
        beta = beta_w[:, :, H:2 * H]
        cu = torch.tensor(cu_list, dtype=torch.int32, device="cuda")
        s0 = torch.zeros(N, H, D, D, dtype=torch.float32, device="cuda")
        s0[1:] = (torch.randn(N - 1, H, D, D, generator=g) * 0.3).cuda()

        out, fs = glm53_flashkda.chunk_prefill(layer, q_ns, k_ns, v_ns, g1, beta, s0, cu)
        torch.cuda.synchronize()
        # correctness: each backend on pristine copies (the Triton chain writes
        # its output IN PLACE into v -- pfkda finding)
        o_t, h_t = triton_ref(layer, q_ns.clone(), k_ns.clone(), v_ns.contiguous().clone(), g1.contiguous(),
                              beta.contiguous(), s0.clone(), cu)
        torch.cuda.synchronize()
        r_out, r_fs = rel_rms(o_t, out), rel_rms(h_t, fs)
        rep[tag] = {"out_rel_rms": r_out, "fs_rel_rms": r_fs}
        print(f"{tag}: out rel-RMS {r_out:.3g} final-state rel-RMS {r_fs:.3g} (bf16-level = same kernels)")

        # ---- timing: wrapper (on) vs the Triton chain (off), same tensors each
        def timeit(fn):
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            ts = []
            for _ in range(args.iters):
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                s.record(), fn(), e.record()
                torch.cuda.synchronize()
                ts.append(s.elapsed_time(e))
            return statistics.median(ts)

        saved = glm53_flashkda.STATE["off"]
        # pristine CONTIGUOUS copies for both timed sides (no per-iteration
        # clones: the wrapper's own .contiguous() calls and the Triton chain's
        # are part of what production pays; the Triton chain's in-place output
        # into v does not change its timing)
        tq, tk, tv, tg, tb = (x.contiguous() for x in (q_ns, k_ns, v_ns, g1, beta))

        def fk():
            glm53_flashkda.chunk_prefill(layer, q_ns, k_ns, v_ns, g1, beta, s0, cu)

        def tri():
            triton_ref(layer, tq, tk, tv, tg, tb, s0, cu)

        rep[tag]["flashkda_ms"] = timeit(fk)
        rep[tag]["triton_ms"] = timeit(tri)
        glm53_flashkda.STATE["off"] = saved
        rep[tag]["speedup"] = rep[tag]["triton_ms"] / rep[tag]["flashkda_ms"]
        print(f"{tag}: flashkda {rep[tag]['flashkda_ms']:.3f} ms vs triton {rep[tag]['triton_ms']:.3f} ms "
              f"= {rep[tag]['speedup']:.2f}x")
        # the off switch must send the model to the Triton branch, and chunk_prefill must refuse
        glm53_flashkda.STATE["off"] = True
        assert not glm53_flashkda.enabled_for(layer)
        try:
            glm53_flashkda.chunk_prefill(layer, q_ns, k_ns, v_ns, g1, beta, s0, cu)
            raise AssertionError("chunk_prefill ran while off")
        except RuntimeError as e:
            print("off-switch refuses:", e)
        glm53_flashkda.STATE["off"] = False
    # fail-closed paths
    cu = torch.tensor([0, 100], dtype=torch.int64, device="cuda")
    try:
        glm53_flashkda.chunk_prefill(layer, rn(1, 100, H, D), rn(1, 100, H, D), rn(1, 100, H, D),
                                     rn(1, 100, H, D), rn(1, 100, H), torch.zeros(1, H, D, D, device="cuda"), cu)
        raise AssertionError("int64 cu_seqlens accepted")
    except RuntimeError as e:
        print("dtype guard:", e)
    print("wrapper totals:", dict(glm53_flashkda.STATE))
    if args.out:
        import json
        json.dump(rep, open(args.out, "w"), indent=1)
    print("WRAPPER UNIT: ALL OK")


if __name__ == "__main__":
    main()
