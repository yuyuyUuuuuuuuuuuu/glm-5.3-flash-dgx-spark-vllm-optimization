#!/usr/bin/env python3
"""FKDA speed bench: the KDA chunked prefill per layer, on vs off, at the
production shapes (H = 32 per rank, D = 128, bf16, safe_gate=True,
lower_bound=-5.0), with the wrapper in the loop (the .contiguous() copies the
production call site pays are on the FlashKDA side; the Triton chain pays its
own copies/l2norm internally) and the bare op for the decomposition.

Arms: single sequence (production's chunk; GLM53_MIXED_PREFILL_CHUNK=0), all
zero and nonzero initial states, T = 13,824 / 4,608 / 1,791 (one full chunk /
one mamba-align piece / the 32k gate's last chunk). Per-chunk number =
median_per_layer x 34 KDA layers (34 of 45 layers are KDA), the same
projection the pfkda kill-test used.
"""
import argparse
import json
import statistics
import sys

import torch

D = 128
LOWER_BOUND = -5.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--lengths", default="13824,4608,1791")
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--layers", type=int, default=34)
    ap.add_argument("--out", default="")
    ap.add_argument("--ext-dir", default="", help="fkda2: a _flashkda_fp32_C build dir to load instead of /w/overlay's")
    args = ap.parse_args()

    assert torch.cuda.is_available()
    props = torch.cuda.get_device_properties(0)
    print(f"device {props.name} sms {props.multi_processor_count} torch {torch.__version__}")

    sys_pkg = "/usr/local/lib/python3.12/dist-packages"
    sys.path.insert(0, "/w/overlay")           # the wrapper the patched kda.py imports
    if args.ext_dir:
        sys.path.insert(0, args.ext_dir)       # fkda2 A/B: this build's extension, the same wrapper
    import _flashkda_fp32_C  # noqa: F401
    import glm53_flashkda
    if args.ext_dir:
        glm53_flashkda.STATE["allow_any_ext"] = True   # fkda2 A/B of a build other than the pinned one
    print("extension:", _flashkda_fp32_C.__file__)
    from vllm.v1.worker.workspace import init_workspace_manager
    from vllm.third_party.flash_linear_attention.ops.kda import chunk_kda_with_fused_gate

    init_workspace_manager(torch.device("cuda"))
    layer = type("L", (), {})()
    layer.local_num_heads, layer.head_dim = args.heads, D
    layer.kda_safe_gate, layer.kda_lower_bound = True, LOWER_BOUND
    layer.A_log = (torch.randn(1, 1, args.heads, 1) * 0.2).cuda().float()
    layer.dt_bias = (torch.rand(args.heads * D) * 8 - 10).cuda().float()
    layer.get_state_dtype = lambda: (torch.bfloat16, torch.float32)

    class Cfg:
        class scheduler_config:
            max_num_batched_tokens = max(int(x) for x in args.lengths.split(","))
            max_num_seqs = 8
        class model_config:
            dtype = torch.bfloat16
    glm53_flashkda.configure(layer, Cfg())
    glm53_flashkda.STATE["logged"] = True   # keep the bench log clean

    H = args.heads
    g = torch.Generator(device="cpu").manual_seed(3)

    def rn(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g) * scale).cuda().to(torch.bfloat16)

    res = {}
    for T in [int(x) for x in args.lengths.split(",")]:
        proj = 3 * H * D
        qkv = rn(1, T, proj)
        q, k, v = (qkv[:, :, i * H * D:(i + 1) * H * D].reshape(1, T, H, D) for i in range(3))
        g1 = rn(1, T, H, D, scale=0.5)
        beta = rn(1, T, 3 * H)[:, :, H:2 * H]   # row-strided, like production's beta_ns
        cu = torch.tensor([0, T], dtype=torch.int32, device="cuda")
        for init in ("zeros", "state"):
            s0 = torch.zeros(1, H, D, D, dtype=torch.float32, device="cuda")
            if init == "state":
                s0 = (torch.randn(1, H, D, D, generator=g) * 0.3).cuda().float()
            tq, tk, tv, tg, tb = (x.contiguous() for x in (q, k, v, g1, beta))

            def tri():
                # the exact production call (the pre-sigmoided fp32 beta); the
                # chain mutates v in place, which does not change its timing
                chunk_kda_with_fused_gate(
                    q=tq, k=tk, v=tv, raw_g=tg, beta=tb.float().sigmoid(), A_log=layer.A_log,
                    g_bias=layer.dt_bias.reshape(-1, D), initial_state=s0, output_final_state=True,
                    use_qk_l2norm_in_kernel=True, cu_seqlens=cu, safe_gate=True, lower_bound=LOWER_BOUND)

            def wrapped():
                glm53_flashkda.chunk_prefill(layer, q, k, v, g1, beta, s0, cu)

            def wrapped_contig():
                # with the quickwins kda_conv item installed (production runs
                # GLM53_PREFILL_QUICKWINS=all) q/k/v arrive contiguous, so the
                # wrapper's .contiguous() calls are no-ops
                glm53_flashkda.chunk_prefill(layer, tq, tk, tv, g1, beta, s0, cu)

            def bare():
                ws = torch.empty(torch.ops._flashkda_fp32_C.get_workspace_size(T, H, 1), dtype=torch.uint8,
                                 device="cuda")
                out = torch.empty(1, T, H, D, dtype=torch.bfloat16, device="cuda")
                fs = torch.empty(1, H, D, D, dtype=torch.float32, device="cuda")
                torch.ops._flashkda_fp32_C.fwd(tq, tk, tv, tg, tb, D ** -0.5, out, ws, layer.A_log.reshape(-1),
                                               layer.dt_bias.reshape(-1, D), LOWER_BOUND, s0, fs, cu, None, None)

            def med(fn):
                for _ in range(args.warmup):
                    fn()
                torch.cuda.synchronize()
                ts = []
                for _ in range(args.iters):
                    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                    s.record()
                    fn()
                    e.record()
                    torch.cuda.synchronize()
                    ts.append(s.elapsed_time(e))
                return statistics.median(ts)

            r = {"triton_ms": med(tri), "wrapper_ms": med(wrapped), "wrapper_contig_ms": med(wrapped_contig),
                 "op_ms": med(bare)}
            r["wrapper_speedup"] = r["triton_ms"] / r["wrapper_ms"]
            r["wrapper_contig_speedup"] = r["triton_ms"] / r["wrapper_contig_ms"]
            r["op_speedup"] = r["triton_ms"] / r["op_ms"]
            r["chunk_saving_wrapper_s"] = (r["triton_ms"] - r["wrapper_ms"]) * args.layers / 1000
            r["chunk_saving_contig_s"] = (r["triton_ms"] - r["wrapper_contig_ms"]) * args.layers / 1000
            res[f"T{T}_{init}"] = r
            print(f"T={T} {init}: triton {r['triton_ms']:.3f} ms | wrapper {r['wrapper_ms']:.3f} ms "
                  f"({r['wrapper_speedup']:.2f}x) | wrapper+kda_conv(contig) {r['wrapper_contig_ms']:.3f} ms "
                  f"({r['wrapper_contig_speedup']:.2f}x) | op alone {r['op_ms']:.3f} ms ({r['op_speedup']:.2f}x) | "
                  f"saving/chunk {r['chunk_saving_wrapper_s']*1000:.0f} / contig {r['chunk_saving_contig_s']*1000:.0f} ms "
                  f"({args.layers} layers)")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
