"""Is production's indexer head gate (torch.mm(x.float(), w32), M x 4096 x 32) run-to-run deterministic at the M where
cuBLAS splits K? And how far is the Triton IEEE kernel (quickwins qw_gate_kernel) from it vs float64?"""
import sys, torch
sys.path.insert(0, "/w")
import glm53_prefill_quickwins as Q
DEV = "cuda"
for T in [int(v) for v in sys.argv[1:]] or [1791, 4289, 6912, 9216, 13824]:
    g = torch.Generator(device=DEV).manual_seed(T)
    x = (torch.randn(T, 4096, device=DEV, generator=g) * 0.5).to(torch.bfloat16)
    wb = (torch.randn(32, 4096, device=DEV, generator=g) * 0.02).to(torch.bfloat16)
    w32 = wb.t().contiguous().float()
    refs = [torch.mm(x.float(), w32) for _ in range(20)]
    det = all(torch.equal(refs[0], r) for r in refs[1:])
    ndiff = max(int((refs[0] != r).sum()) for r in refs[1:])
    got = Q.qw_head_gate(x, w32)
    r64 = torch.mm(x.double(), w32.double())
    bound = (x.double().abs() @ w32.double().abs())
    e_prod = ((refs[0].double() - r64).abs() / bound).max().item()
    e_tri = ((got.double() - r64).abs() / bound).max().item()
    e_pp = max(((refs[0].double() - r.double()).abs() / bound).max().item() for r in refs[1:])
    print(f"M={T:6d} prod deterministic over 20 runs: {det} (max {ndiff} elements differ); "
          f"max |err|/sum|xw| vs fp64: prod {e_prod:.2e} triton {e_tri:.2e}; prod-vs-prod {e_pp:.2e}; "
          f"triton==prod {torch.equal(got, refs[0])}", flush=True)
