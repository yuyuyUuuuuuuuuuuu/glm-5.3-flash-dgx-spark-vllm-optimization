"""TileLang kernel of fp8_gemv's large-M path (GLM53_FP8_LARGE_M, backend "tilelang"; docs/FP8_LARGE_M.md).

    C[m, n] = bf16( (sum_k A[m, k] * B[n, k])_fp32 * S[n] )

A = activations [M, K] bf16 (dense rows), B = the exactly dequantized e4m3 weight chunk [N, K] bf16, S = the fp32
per-output-channel scale (read up to the next multiple of the N tile: the caller pads it with zeros), C = a [M, N]
view with any row stride (a column slice of the layer output). fp32 accumulation, the scale applied to the fp32 sum
and one rounding to bf16 -- the value class of Marlin's bf16(sum * s). M, N and C's row stride are symbolic: one
compile per K. Kept in its own module without `from __future__ import annotations`: TileLang evaluates the
parameter annotations, which refer to the local symbolic dims.
"""
import tilelang
import tilelang.language as T


def build(k: int, bM: int, bN: int, bK: int, stages: int, threads: int, swz: int):
    M, N, LDC = T.symbolic("m"), T.symbolic("n"), T.symbolic("ldc")

    @T.prim_func
    def tf_fp8_large_m_gemm(A: T.Tensor((M, k), "bfloat16"), B: T.Tensor((N, k), "bfloat16"),
                            S: T.Tensor((N,), "float32"), C: T.StridedTensor((M, N), (LDC, 1), "bfloat16")):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=threads) as (bx, by):
            T.use_swizzle(panel_size=swz)
            As = T.alloc_shared((bM, bK), "bfloat16")
            Bs = T.alloc_shared((bN, bK), "bfloat16")
            Cl = T.alloc_fragment((bM, bN), "float32")
            T.clear(Cl)
            for kk in T.Pipelined(T.ceildiv(k, bK), num_stages=stages):
                T.copy(A[by * bM, kk * bK], As)
                T.copy(B[bx * bN, kk * bK], Bs)
                T.gemm(As, Bs, Cl, transpose_B=True)
            for i, j in T.Parallel(bM, bN):
                Cl[i, j] = Cl[i, j] * S[bx * bN + j]
            T.copy(Cl, C[by * bM, bx * bN])

    return tilelang.compile(tf_fp8_large_m_gemm, out_idx=None, target="cuda")
