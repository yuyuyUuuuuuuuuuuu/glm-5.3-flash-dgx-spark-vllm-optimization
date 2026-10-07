"""Probe: TileLang fused-scale GEMM with symbolic M, N and a strided C (a column slice of the full output): correctness
vs bf16(torch.mm(fp32) * scale) incl. partial tiles, and speed for in_proj / gate_up (whole weight) and fc (N chunks)."""
import time
import torch
import tilelang
import tilelang.language as T

print("tilelang", tilelang.__version__, torch.cuda.get_device_name())


def make(K, bM=128, bN=256, bK=32, stages=4, threads=256, swz=10):
    M, N, LDC = T.symbolic("m"), T.symbolic("n"), T.symbolic("ldc")

    @T.prim_func
    def main(A: T.Tensor((M, K), "bfloat16"), B: T.Tensor((N, K), "bfloat16"), S: T.Tensor((N,), "float32"),
             C: T.StridedTensor((M, N), (LDC, 1), "bfloat16")):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=threads) as (bx, by):
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
    for N, K, nc in [(12576, 4096, 12576), (12288, 4096, 12288), (4096, 20480, 2048), (12576, 4096, 4224)]:
        t0 = time.time()
        k = tilelang.compile(make(K), out_idx=None, target="cuda")
        ct = time.time() - t0
        b = torch.randn(N, K, device="cuda").to(torch.bfloat16)
        s = torch.rand(N, device="cuda") * 1e-3
        sp = torch.zeros(N + 256, device="cuda")
        sp[:N] = s
        line = f"N={N} K={K} nc={nc} compile {ct:.1f}s:"
        for M in (777, 1791, 4608, 13824):
            a = torch.randn(M, K, device="cuda").to(torch.bfloat16)
            ref = (torch.mm(a, b.t(), out_dtype=torch.float32) * s).to(torch.bfloat16)
            big = torch.full((M, N + 16), float("nan"), device="cuda", dtype=torch.bfloat16)
            out = big[:, 8:8 + N]

            def run():
                for n0 in range(0, N, nc):
                    c = min(nc, N - n0)
                    k(a, b[n0:n0 + c], sp[n0:n0 + c], out[:, n0:n0 + c])
            run()
            torch.cuda.synchronize()
            same = torch.equal(out, ref)
            clean = torch.isnan(big[:, :8]).all().item() and torch.isnan(big[:, 8 + N:]).all().item()
            ms = timeit(run)
            line += f" M={M}: {ms:.2f} ms {2*M*N*K/ms/1e9:.1f} TF bitwise=ref {same} clean {clean};"
            del a, ref, big
        print(line, flush=True)


if __name__ == "__main__":
    main()
