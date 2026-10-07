"""Probe: a TileLang GEMM with the per-channel scale fused in its epilogue (the fused alternative to torch.mm fp32 +
scale_cast), bf16 in, fp32 accumulate, bf16 out, dynamic M, at the in_proj / gate_up shapes. Speed vs cuBLAS
(torch.mm bf16 out, fp32 out) and bitwise agreement with bf16(torch.mm(fp32 out) * scale)."""
import time
import torch
import tilelang
import tilelang.language as T

print("tilelang", tilelang.__version__, torch.cuda.get_device_name())


def make(N, K, bM, bN, bK, stages, threads, swz):
    M = T.symbolic("m")

    @T.prim_func
    def main(A: T.Tensor((M, K), "bfloat16"), B: T.Tensor((N, K), "bfloat16"), S: T.Tensor((N,), "float32"),
             C: T.Tensor((M, N), "bfloat16")):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=threads) as (bx, by):
            if swz:
                T.use_swizzle(panel_size=swz)
            As = T.alloc_shared((bM, bK), "bfloat16")
            Bs = T.alloc_shared((bN, bK), "bfloat16")
            Cl = T.alloc_fragment((bM, bN), "float32")
            T.clear(Cl)
            for kk in T.Pipelined(T.ceildiv(K, bK), num_stages=stages):
                T.copy(A[by * bM, kk * bK], As)
                T.copy(B[bx * bN, kk * bK], Bs)
                T.gemm(As, Bs, Cl, transpose_B=True)
            for i, j in T.Parallel(bM, bN):
                Cl[i, j] = Cl[i, j] * S[bx * bN + j]
            T.copy(Cl, C[by * bM, bx * bN])
    return main


def timeit(f, n=5):
    f()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(n):
        f()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / n


def main():
  for N, K in [(12576, 4096), (12288, 4096)]:
      b = torch.randn(N, K, device="cuda").to(torch.bfloat16)
      s = torch.rand(N, device="cuda") * 1e-3
      for cfg in [(256, 128, 32, 4, 256, 10), (256, 128, 32, 4, 256, 0), (128, 128, 32, 3, 128, 0),
                (256, 128, 32, 4, 256, 16), (256, 128, 32, 4, 256, 6),
                  (128, 256, 32, 4, 256, 10), (128, 128, 32, 5, 128, 10), (128, 128, 64, 3, 256, 10)]:
          if (cfg[0] + cfg[1]) * cfg[2] * 2 * cfg[3] > 101376:
              print(N, K, cfg, "skipped: shared memory > 99 KiB")
              continue
          try:
              t0 = time.time()
              k = tilelang.compile(make(N, K, *cfg), out_idx=None, target="cuda")
              ct = time.time() - t0
          except Exception as e:  # noqa: BLE001
              print(N, K, cfg, "failed:", repr(e)[:200], flush=True)
              continue
          line = f"N={N} K={K} cfg={cfg} compile {ct:.1f}s:"
          for M in (1791, 4608, 13824):
              a = torch.randn(M, K, device="cuda").to(torch.bfloat16)
              c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
              ref = (torch.mm(a, b.t(), out_dtype=torch.float32) * s).to(torch.bfloat16)
              k(a, b, s, c)
              torch.cuda.synchronize()
              same = (c == ref).float().mean().item()
              ms = timeit(lambda: k(a, b, s, c))
              fl = 2 * M * N * K
              if cfg == (256, 128, 32, 4, 256, 10):
                  t16 = timeit(lambda: torch.mm(a, b.t()))
                  t32 = timeit(lambda: torch.mm(a, b.t(), out_dtype=torch.float32))
                  line += f" [cuBLAS M={M}: bf16 {t16:.2f} ms {fl/t16/1e9:.0f} TF, fp32 {t32:.2f} ms]"
              line += f" M={M}: {ms:.2f} ms {fl/ms/1e9:.1f} TF =ref {same*100:.3f}%;"
              del a, c, ref
          print(line, flush=True)


if __name__ == "__main__":
    main()
