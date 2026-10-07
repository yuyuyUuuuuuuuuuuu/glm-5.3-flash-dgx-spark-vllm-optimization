"""Does causal_conv1d_update's BLOCK_N change results (bitwise) or speed? KDA decode shape: dim 12288 per rank,
width 4, spec decoding (T = 5 / 6 / 8 per request, state_len 3 + 7), bf16. Variants = production's wrapper source with
BLOCK_N=256 replaced (+ num_warps). Paired A/B in CUDA graphs over 34 distinct layers' states (one decode step)."""
import inspect, os, sys, textwrap
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_smallops import ab, graph_of, report
from vllm.model_executor.layers.mamba.ops import causal_conv1d as C

src = textwrap.dedent(inspect.getsource(C.causal_conv1d_update))
assert src.count("BLOCK_N=256,") == 1
variants = {}
for bn, nw in ((256, None), (128, None), (64, None), (32, None), (64, 2), (128, 2)):
    s = src.replace("BLOCK_N=256,", f"BLOCK_N={bn}," + (f" num_warps={nw}," if nw else ""))
    s = s.replace("def causal_conv1d_update(", f"def causal_conv1d_update_{bn}_{nw}(")
    ns = dict(C.__dict__)
    exec(compile(s, C.__file__, "exec"), ns)
    variants[(bn, nw)] = ns[f"causal_conv1d_update_{bn}_{nw}"]
dev = torch.device("cuda")
dim, width, nspec = 12288, 4, 7
L = 34
w = [(torch.randn(dim, width, device=dev) * 0.3).bfloat16() for _ in range(L)]
for T in (5, 6, 8):
    for layout in ("dim_first", "tok_first"):
        nslots = 4
        if layout == "dim_first":
            states = [torch.randn(nslots, dim, width - 1 + nspec, device=dev).bfloat16() for _ in range(L)]
            views = states
        else:
            states = [torch.randn(nslots, width - 1 + nspec, dim, device=dev).bfloat16() for _ in range(L)]
            views = [s.transpose(-1, -2) for s in states]
        x = torch.randn(T, dim, device=dev).bfloat16()
        idx = torch.tensor([2], dtype=torch.int32, device=dev)
        nacc = torch.tensor([3], dtype=torch.int32, device=dev)
        qsl = torch.tensor([0, T], dtype=torch.int32, device=dev)

        def call(fn, i, st):
            return fn(x, st, w[i], None, activation="silu", conv_state_indices=idx, num_accepted_tokens=nacc,
                      query_start_loc=qsl, max_query_len=nspec + 1)
        # bitwise: out and resulting state, from identical starting states
        x0 = x.clone()                      # the wrapper writes its output into x (in place): fresh copy per call
        ref_state = views[0].clone()
        ref = variants[(256, None)](x0.clone(), ref_state, w[0], None, activation="silu", conv_state_indices=idx,
                                    num_accepted_tokens=nacc, query_start_loc=qsl, max_query_len=nspec + 1).clone()
        for k, fn in variants.items():
            st = views[0].clone()
            out = fn(x0.clone(), st, w[0], None, activation="silu", conv_state_indices=idx, num_accepted_tokens=nacc,
                     query_start_loc=qsl, max_query_len=nspec + 1)
            ok = torch.equal(out.view(torch.int16), ref.view(torch.int16)) and \
                torch.equal(st.contiguous().view(torch.int16), ref_state.contiguous().view(torch.int16))
            print(f"T={T} {layout} BLOCK_N={k[0]} warps={k[1]}: bitwise {'OK' if ok else 'MISMATCH'}", flush=True)
        g = {}
        for k, fn in variants.items():
            def f(fn=fn):
                for i in range(L):
                    call(fn, i, views[i])
            g[f"BLOCK_N={k[0]} warps={k[1] or 4}" + (" (production)" if k == (256, None) else "")] = graph_of(f)
        report(f"causal_conv1d_update T={T} {layout}, {L} layers per graph", ab(g, L),
               "BLOCK_N=256 warps=4 (production)")
