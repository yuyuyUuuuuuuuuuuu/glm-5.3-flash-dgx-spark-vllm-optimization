#!/usr/bin/env python3
"""FKDA review check: short / degenerate varlen rows through the glm53_flashkda
wrapper vs the production Triton chain, compared PER SEQUENCE.

wrapper_unit.py only covers long rows (13,824 and 5000/2777/6047), and its
rel-RMS is over the whole batch, where a broken short row is invisible. The
serving engine can hand the chunked-prefill path:
  * a 1-token prefill with an initial state (prefix-cache hit covering all
    but the last prompt token),
  * rows shorter than FlashKDA's 16-token CHUNK and rows not a multiple of it,
  * a mix of rows with and without initial state (has_initial_state=False
    rows arrive as zero states from gather_initial_states),
  * up to max_num_seqs rows in one call.
Each case is checked for per-row output and per-row final state against the
Triton chain (fp32 state). GPU, production image, via tests/fkda/gpu_run.sh
with FKDA_USER=root (installs the overlay into the throwaway container).
"""
import sys
from pathlib import Path

sys.path.insert(0, "/w/tests/fkda")
import wrapper_unit as wu  # noqa: E402


def main():
    import numpy as np
    import torch

    wu.install(Path("/usr/local/lib/python3.12/dist-packages"))
    import glm53_flashkda
    from vllm.v1.worker.workspace import init_workspace_manager

    init_workspace_manager(torch.device("cuda"))
    layer = wu.FakeLayer()
    # the real-checkpoint gate parameters (pfkda staging), when present
    try:
        a = torch.from_numpy(np.load("/pf3000/real_A_log.npy")).float().cuda()
        d = torch.from_numpy(np.load("/pf3000/real_dt_bias.npy")).float().cuda()
        layer.A_log = a.reshape(layer.A_log.shape) if a.numel() == layer.A_log.numel() else layer.A_log
        layer.dt_bias = d.reshape(-1)[: 32 * 128].contiguous() if d.numel() >= 32 * 128 else layer.dt_bias
        print("using real A_log/dt_bias", tuple(a.shape), tuple(d.shape))
    except Exception as e:  # noqa: BLE001
        print("real gate params unavailable, random:", e)
    glm53_flashkda.configure(layer, wu.FakeCfg())

    H, D = 32, 128
    proj = H * D * 3
    g = torch.Generator(device="cpu").manual_seed(11)

    def rn(*shape, scale=1.0, dt=torch.bfloat16):
        return (torch.randn(*shape, generator=g) * scale).cuda().to(dt)

    cases = {
        "T1_with_state": ([0, 1], [True]),
        "T1_zero_state": ([0, 1], [False]),
        "T15_with_state": ([0, 15], [True]),
        "decodes_plus_prefill": ([0, 1, 2, 3, 4611], [True, True, True, False]),
        "mixed_short_8rows": ([0, 1, 17, 33, 34, 50, 4658, 4659, 9267], [True, False, True, True, False, True, True, False]),
        "apc_tail_odd": ([0, 2352], [True]),
        "max_rows_13824": ([0, 1, 2, 3, 4, 5, 6, 7, 13824], [True] * 7 + [False]),
    }
    worst = {}
    fails = []
    for tag, (cu_list, has_state) in cases.items():
        T = cu_list[-1]
        N = len(cu_list) - 1
        qkv = rn(1, T, proj)
        q_ns, k_ns, v_ns = (qkv[:, :, i * H * D:(i + 1) * H * D].reshape(1, T, H, D) for i in range(3))
        g1 = rn(1, T, H, D, scale=0.5)
        beta_w = rn(1, T, 3 * H)
        beta = beta_w[:, :, H:2 * H]
        cu = torch.tensor(cu_list, dtype=torch.int32, device="cuda")
        s0 = torch.zeros(N, H, D, D, dtype=torch.float32, device="cuda")
        for i, hs in enumerate(has_state):
            if hs:
                s0[i] = (torch.randn(H, D, D, generator=g) * 0.3).cuda()
        v_before = v_ns.clone()
        out, fs = glm53_flashkda.chunk_prefill(layer, q_ns, k_ns, v_ns, g1, beta, s0.clone(), cu)
        out, fs = out.clone(), fs.clone()
        torch.cuda.synchronize()
        assert torch.equal(v_ns, v_before), "wrapper mutated v"
        o_t, h_t = wu.triton_ref(layer, q_ns.clone(), k_ns.clone(), v_ns.contiguous().clone(), g1.contiguous(),
                                 beta.contiguous(), s0.clone(), cu)
        torch.cuda.synchronize()
        rows = []
        for i in range(N):
            a, b = cu_list[i], cu_list[i + 1]
            ro = wu.rel_rms(o_t[:, a:b], out[:, a:b])
            rs = wu.rel_rms(h_t[i], fs[i])
            fin = bool(torch.isfinite(out[:, a:b]).all() and torch.isfinite(fs[i]).all())
            rows.append((b - a, has_state[i], ro, rs, fin))
            if not fin or ro > 3e-2 or rs > 3e-2:
                fails.append((tag, i, b - a, ro, rs, fin))
        worst[tag] = (max(r[2] for r in rows), max(r[3] for r in rows))
        print(f"{tag}: N={N} T={T} worst out rel-RMS {worst[tag][0]:.3g} worst state rel-RMS {worst[tag][1]:.3g}")
        for r in rows:
            print(f"    len={r[0]:5d} state={'y' if r[1] else 'n'} out {r[2]:.3g} state {r[3]:.3g} finite={r[4]}")
    if fails:
        print("SHORT VARLEN: FAIL", fails)
        sys.exit(1)
    print("SHORT VARLEN: ALL OK (per-row out/state rel-RMS <= 3e-2, finite, v untouched)")


if __name__ == "__main__":
    main()
