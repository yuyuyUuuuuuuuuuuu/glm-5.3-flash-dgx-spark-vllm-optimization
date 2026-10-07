#!/usr/bin/env python3
"""FKDA2: precision edits to a FlashKDA 17a037d source tree (csrc/smxx/*.cuh), each behind a compile-time switch
so the A/B builds can isolate them. Defaults: FP32_DECAY / FP32_U / FP32_OUT ON (the fkda2 build), EXACT_SIGMOID
OFF (measured: no effect on the output or state error vs fp64, so the stock tanh.approx activation stays). Exact-string edits, every anchor must match exactly once.

  FKDA2_EXACT_SIGMOID  gate and beta sigmoid via 1/(1+exp(-x)) (fp32) instead of tanh.approx.f32
                       (MUFU.TANH, ~2^-11 relative error) -- K1 gate activation, K1 beta_act, K2 beta0/beta1
  FKDA2_FP32_DECAY     K1: q_decayed / k_decayed / k_inv / k_restored computed in fp32 and rounded to bf16 ONCE
                       (stock: normalized q/k rounded to bf16, exp(cumsum) rounded to bf16, then bf16*bf16(*bf16)
                       products, each rounded: 3-4 roundings per operand)
  FKDA2_FP32_U         K2: u = (v - k@S) * beta in fp32 with the fp32 beta, rounded to bf16 once for the MMA
                       (stock: k@S rounded to bf16, the subtraction rounded, beta rounded to bf16, product rounded)
  FKDA2_FP32_OUT       K2: out = q@S + Mqk@U accumulated in ONE fp32 accumulator, rounded once
                       (stock: bf16(q@S) + bf16(Mqk@U) added in bf16: three roundings of the output)
Usage: patch_flashkda_precision.py <flashkda checkout>   (idempotent: refuses a tree that is already patched)
"""
import sys
from pathlib import Path

root = Path(sys.argv[1])
U = root / "csrc/smxx/utils.cuh"
K1 = root / "csrc/smxx/fwd_kernel1.cuh"
K2 = root / "csrc/smxx/fwd_kernel2.cuh"

EDITS = [
    (U, """__device__ __forceinline__ float bf16_to_f32(cutlass::bfloat16_t x) {""",
     """// [fkda2] precision switches (patch_flashkda_precision.py); EXACT_SIGMOID default OFF, the rest ON
#ifndef FKDA2_EXACT_SIGMOID
#define FKDA2_EXACT_SIGMOID 0
#endif
#ifndef FKDA2_FP32_DECAY
#define FKDA2_FP32_DECAY 1
#endif
#ifndef FKDA2_FP32_U
#define FKDA2_FP32_U 1
#endif
#ifndef FKDA2_FP32_OUT
#define FKDA2_FP32_OUT 1
#endif

// [fkda2] the activation used for the gate and beta
__device__ __forceinline__ float fkda2_sigmoid_f32(float x) {
#if FKDA2_EXACT_SIGMOID
    return 1.0f / (1.0f + __expf(-x));
#else
    return sigmoid_tanh_approx_f32(x);
#endif
}

__device__ __forceinline__ float bf16_to_f32(cutlass::bfloat16_t x) {"""),
    # ---- K1: beta activation
    (K1, """        shared_storage.beta_act.begin()[compute_tid] = sigmoid_tanh_approx_f32(""",
     """        shared_storage.beta_act.begin()[compute_tid] = fkda2_sigmoid_f32("""),
    # ---- K1: gate activation
    (K1, """                g_val = gate_scale * sigmoid_tanh_approx_f32(g_val);""",
     """                g_val = gate_scale * fkda2_sigmoid_f32(g_val);"""),
    # ---- K1: decayed q / k
    (K1, """                for (int v = 0; v < 2; ++v) {
                    float g = reg_g[pass][tile_idx][v];
                    BF16 q = BF16(bf16_to_f32(reg_q[pass][tile_idx][v]) * q_inv);
                    BF16 k = row < actual_len ? BF16(bf16_to_f32(reg_k[pass][tile_idx][v]) * k_inv_scale) : BF16(0);
                    BF16 exp_cumsum = BF16(ex2_approx_ftz_f32(g));
                    r_qd(0, v) = q * exp_cumsum * BF16(scale);
                    r_kd(0, v) = k * exp_cumsum;
                }""",
     """                for (int v = 0; v < 2; ++v) {
                    float g = reg_g[pass][tile_idx][v];
#if FKDA2_FP32_DECAY
                    float qf = bf16_to_f32(reg_q[pass][tile_idx][v]) * q_inv;
                    float kf = row < actual_len ? bf16_to_f32(reg_k[pass][tile_idx][v]) * k_inv_scale : 0.0f;
                    float ef = ex2_approx_ftz_f32(g);
                    r_qd(0, v) = BF16(qf * ef * scale);
                    r_kd(0, v) = BF16(kf * ef);
#else
                    BF16 q = BF16(bf16_to_f32(reg_q[pass][tile_idx][v]) * q_inv);
                    BF16 k = row < actual_len ? BF16(bf16_to_f32(reg_k[pass][tile_idx][v]) * k_inv_scale) : BF16(0);
                    BF16 exp_cumsum = BF16(ex2_approx_ftz_f32(g));
                    r_qd(0, v) = q * exp_cumsum * BF16(scale);
                    r_kd(0, v) = k * exp_cumsum;
#endif
                }"""),
    (K1, """                for (int v = 0; v < 2; ++v) {
                    float g = reg_g[pass][tile_idx][v];
                    BF16 k = row < actual_len ? BF16(bf16_to_f32(reg_k[pass][tile_idx][v]) * k_inv_scale) : BF16(0);
                    BF16 inv_cumsum = BF16(ex2_approx_ftz_f32(-g));
                    r_ki(0, v) = k * inv_cumsum;
                    r_kr(0, v) = k * inv_cumsum * BF16(reg_gt[pass][tile_idx][v]);
                }""",
     """                for (int v = 0; v < 2; ++v) {
                    float g = reg_g[pass][tile_idx][v];
#if FKDA2_FP32_DECAY
                    float kf = row < actual_len ? bf16_to_f32(reg_k[pass][tile_idx][v]) * k_inv_scale : 0.0f;
                    float ei = ex2_approx_ftz_f32(-g);
                    r_ki(0, v) = BF16(kf * ei);
                    r_kr(0, v) = BF16(kf * ei * reg_gt[pass][tile_idx][v]);
#else
                    BF16 k = row < actual_len ? BF16(bf16_to_f32(reg_k[pass][tile_idx][v]) * k_inv_scale) : BF16(0);
                    BF16 inv_cumsum = BF16(ex2_approx_ftz_f32(-g));
                    r_ki(0, v) = k * inv_cumsum;
                    r_kr(0, v) = k * inv_cumsum * BF16(reg_gt[pass][tile_idx][v]);
#endif
                }"""),
    # ---- K2: keep out_acc fp32 through phase 3
    (K2, """            SFragT out_bf16[kValueBlocksPerWarp];
            #pragma unroll
            for (int i = 0; i < kValueBlocksPerWarp; ++i)
                cute::transform(out_acc[i], out_bf16[i], [] __device__ (float x) { return BF16(x); });
""",
     """            SFragT out_bf16[kValueBlocksPerWarp];
#if !FKDA2_FP32_OUT
            #pragma unroll
            for (int i = 0; i < kValueBlocksPerWarp; ++i)
                cute::transform(out_acc[i], out_bf16[i], [] __device__ (float x) { return BF16(x); });
#endif
"""),
    # ---- K2: beta
    (K2, """            BF16 beta0 = BF16(sigmoid_tanh_approx_f32(float(beta_tile(beta_smem_offset + group_id))));
            BF16 beta1 = BF16(sigmoid_tanh_approx_f32(float(beta_tile(beta_smem_offset + group_id + 8))));""",
     """#if FKDA2_FP32_U
            float beta0 = fkda2_sigmoid_f32(float(beta_tile(beta_smem_offset + group_id)));
            float beta1 = fkda2_sigmoid_f32(float(beta_tile(beta_smem_offset + group_id + 8)));
#else
            BF16 beta0 = BF16(fkda2_sigmoid_f32(float(beta_tile(beta_smem_offset + group_id))));
            BF16 beta1 = BF16(fkda2_sigmoid_f32(float(beta_tile(beta_smem_offset + group_id + 8))));
#endif"""),
    # ---- K2: u = (v - kS) * beta
    (K2, """                cute::transform(u_acc[i], u_bf16[i], [] __device__ (float x) { return BF16(x); });

                #pragma unroll
                for (int a = 0; a < 2; ++a) {
                    #pragma unroll
                    for (int d = 0; d < 2; ++d) {
                        auto c0 = make_coord(make_coord(a, 0), 0, d);
                        auto c1 = make_coord(make_coord(a, 1), 0, d);
                        u_bf16[i](c0) = (v_bf16[i](c0) - u_bf16[i](c0)) * beta0;
                        u_bf16[i](c1) = (v_bf16[i](c1) - u_bf16[i](c1)) * beta1;
                    }
                }""",
     """#if FKDA2_FP32_U
                #pragma unroll
                for (int a = 0; a < 2; ++a) {
                    #pragma unroll
                    for (int d = 0; d < 2; ++d) {
                        auto c0 = make_coord(make_coord(a, 0), 0, d);
                        auto c1 = make_coord(make_coord(a, 1), 0, d);
                        u_bf16[i](c0) = BF16((bf16_to_f32(v_bf16[i](c0)) - u_acc[i](c0)) * beta0);
                        u_bf16[i](c1) = BF16((bf16_to_f32(v_bf16[i](c1)) - u_acc[i](c1)) * beta1);
                    }
                }
#else
                cute::transform(u_acc[i], u_bf16[i], [] __device__ (float x) { return BF16(x); });

                #pragma unroll
                for (int a = 0; a < 2; ++a) {
                    #pragma unroll
                    for (int d = 0; d < 2; ++d) {
                        auto c0 = make_coord(make_coord(a, 0), 0, d);
                        auto c1 = make_coord(make_coord(a, 1), 0, d);
                        u_bf16[i](c0) = (v_bf16[i](c0) - u_bf16[i](c0)) * beta0;
                        u_bf16[i](c1) = (v_bf16[i](c1) - u_bf16[i](c1)) * beta1;
                    }
                }
#endif"""),
    # ---- K2: out = q@S + Mqk@U in one fp32 accumulator
    (K2, """                clear(out_acc[i]);
                gemm(thr_mma, tCrA_k(_,_,Int<0>{}), tCrB_u_arr[i](_,_,Int<0>{}), out_acc[i]);

                SFragT gemm_bf16;
                cute::transform(out_acc[i], gemm_bf16, [] __device__ (float x) { return BF16(x); });
                cute::transform(out_bf16[i], gemm_bf16, out_bf16[i], [] __device__ (BF16 c, BF16 a) { return c + a; });""",
     """#if FKDA2_FP32_OUT
                gemm(thr_mma, tCrA_k(_,_,Int<0>{}), tCrB_u_arr[i](_,_,Int<0>{}), out_acc[i]);
                cute::transform(out_acc[i], out_bf16[i], [] __device__ (float x) { return BF16(x); });
#else
                clear(out_acc[i]);
                gemm(thr_mma, tCrA_k(_,_,Int<0>{}), tCrB_u_arr[i](_,_,Int<0>{}), out_acc[i]);

                SFragT gemm_bf16;
                cute::transform(out_acc[i], gemm_bf16, [] __device__ (float x) { return BF16(x); });
                cute::transform(out_bf16[i], gemm_bf16, out_bf16[i], [] __device__ (BF16 c, BF16 a) { return c + a; });
#endif"""),
]


def main():
    texts = {p: p.read_text() for p in (U, K1, K2)}
    if any("[fkda2]" in t for t in texts.values()):
        sys.exit("already patched ([fkda2] marker present)")
    for p, old, new in EDITS:
        n = texts[p].count(old)
        if n != 1:
            sys.exit(f"anchor count {n} != 1 in {p.name}: {old.splitlines()[0]!r}")
        texts[p] = texts[p].replace(old, new)
    for p, t in texts.items():
        p.write_text(t)
    print(f"patched {len(EDITS)} anchors in {root}")


if __name__ == "__main__":
    main()
