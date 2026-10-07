#!/usr/bin/env python3
"""FKDA3 B1 unit + bench: glm53_flashkda.chunk_prefill(..., out=core_attn_out[:, :n]) (what the fkda3-patched kda.py
passes on a step without spec tokens) writes the layer output directly, and kda.py's merge statement
`core_attn_out[0, :n] = core_attn_out_non_spec[0, :n]` then launches nothing.

  D.1 bit-equality direct vs workspace path (out and final state), single sequence and varlen, T from 1 to 13,824
  D.2 the returned output IS the caller's buffer; direct_calls counts it
  D.3 fallbacks to the workspace path: wrong row count, non-contiguous view, fp16, STATE["direct_out"] False, None
  D.4 the merge statement is a no-op (no CUDA kernel in the profiler) and the layer buffer holds the result;
      with the workspace path the same statement is one copy kernel
  D.5 timing at production shapes: (workspace + merge copy) vs (direct + no-op merge), per layer and per chunk
  D.6 the pinned extension configures and the boot line names the fkda3 build
Run: FKDA_SCRATCH=... flock /tmp/tf-gpu-bench.lock tests/fkda/gpu_run.sh -c "python3 /w/tests/fkda3/direct_out_unit.py"
"""
import shutil
import statistics
import sys
import tempfile

import torch

# stage the fkda3 build under the site-packages names, like patch_flashkda.py does at GLM53_KDA_FLASHKDA_V=3
# (the r16z overlay's base glm53_flashkda.py / _flashkda_fp32_C.abi3.so are the shipped r16x build)
_stage = tempfile.mkdtemp(prefix="fkda3-stage-")
shutil.copy2("/w/overlay/glm53_flashkda3.py", _stage + "/glm53_flashkda.py")
shutil.copy2("/w/overlay/_flashkda_fp32_C3.abi3.so", _stage + "/_flashkda_fp32_C.abi3.so")
sys.path.insert(0, _stage)
import _flashkda_fp32_C  # noqa: E402,F401  (the fkda3 extension, d98adc4a)
import glm53_flashkda as W  # noqa: E402

D, H = 128, 32
FAIL = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAIL.append(msg)


def main():
    from vllm.v1.worker.workspace import init_workspace_manager
    init_workspace_manager(torch.device("cuda"))
    layer = type("L", (), {})()
    layer.local_num_heads, layer.head_dim = H, D
    layer.kda_safe_gate, layer.kda_lower_bound = True, -5.0
    g = torch.Generator(device="cpu").manual_seed(11)
    layer.A_log = (torch.randn(1, 1, H, 1, generator=g) * 0.2).cuda().float()
    layer.dt_bias = (torch.rand(H * D, generator=g) * 8 - 10).cuda().float()
    layer.get_state_dtype = lambda: (torch.bfloat16, torch.float32)

    class Cfg:
        class scheduler_config:
            max_num_batched_tokens = 13824
            max_num_seqs = 8

        class model_config:
            dtype = torch.bfloat16

    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        W.configure(layer, Cfg())
    line = buf.getvalue().strip()
    print(line)
    check(W.EXT_BUILD in line and W.EXT_SHA256[:16] in line and "harness override" not in line,
          "D.6 pinned fkda3 extension configures; boot line names the build")

    def rn(*s, sc=1.0):
        return (torch.randn(*s, generator=g) * sc).cuda().to(torch.bfloat16)

    def inputs(lens):
        T = sum(lens)
        qkv = rn(1, T, 3 * H * D)
        q, k, v = (qkv[:, :, i * H * D:(i + 1) * H * D].reshape(1, T, H, D).contiguous() for i in range(3))
        beta = rn(1, T, 3 * H)[:, :, H:2 * H]
        cu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32, device="cuda")
        s0 = (torch.randn(len(lens), H, D, D, generator=g) * 0.05).cuda()
        return dict(q=q, k=k, v=v, g=rn(1, T, H, D, sc=0.5), beta=beta, initial_state=s0, cu_seqlens=cu), T

    for lens in ([1], [15], [17], [1791], [4608], [13824], [5, 300, 1], [1, 1, 4000, 1]):
        x, T = inputs(lens)
        ref_o, ref_s = W.chunk_prefill(layer, **x)
        ref_o, ref_s = ref_o.clone(), ref_s.clone()
        core = torch.full((1, T + 7, H, D), float("nan"), dtype=torch.bfloat16, device="cuda")   # padded rows
        n0 = W.STATE["direct_calls"]
        o, s = W.chunk_prefill(layer, **x, out=core[:, :T])
        torch.cuda.synchronize()
        check(torch.equal(o, ref_o) and torch.equal(s, ref_s), f"D.1 lens={lens}: direct == workspace (out + state)")
        check(o.data_ptr() == core.data_ptr() and W.STATE["direct_calls"] == n0 + 1,
              f"D.2 lens={lens}: returned output is the caller's buffer")
        check(bool(torch.isnan(core[:, T:].float()).all()), f"D.2 lens={lens}: rows past T untouched")

    x, T = inputs([300])
    ref_o, _ = W.chunk_prefill(layer, **x)
    ws_ptr = ref_o.data_ptr()          # the workspace output view (before cloning)
    ref_o = ref_o.clone()
    bad = {
        "rows T+1": torch.empty(1, T + 1, H, D, dtype=torch.bfloat16, device="cuda"),
        "non-contiguous": torch.empty(1, T, H, 2 * D, dtype=torch.bfloat16, device="cuda")[..., :D],
        "fp16": torch.empty(1, T, H, D, dtype=torch.float16, device="cuda"),
        "None": None,
    }
    for name, o_bad in bad.items():
        n0 = W.STATE["direct_calls"]
        o, _ = W.chunk_prefill(layer, **x, out=o_bad)
        check(W.STATE["direct_calls"] == n0 and torch.equal(o, ref_o) and (o_bad is None or o.data_ptr() != o_bad.data_ptr()),
              f"D.3 fallback to the workspace ({name})")
    W.STATE["direct_out"] = False
    good = torch.empty(1, T, H, D, dtype=torch.bfloat16, device="cuda")
    o, _ = W.chunk_prefill(layer, **x, out=good)
    check(o.data_ptr() == ws_ptr and torch.equal(o, ref_o), "D.3 STATE['direct_out']=False uses the workspace")
    W.STATE["direct_out"] = True

    # D.4 kda.py's merge statement
    from torch.profiler import profile, ProfilerActivity
    x, T = inputs([4608])
    n = T
    core = torch.empty(1, n + 3, H, D, dtype=torch.bfloat16, device="cuda")
    o, _ = W.chunk_prefill(layer, **x, out=core[:, :n])
    expect = o.clone()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        core[0, :n] = o[0, :n]
        torch.cuda.synchronize()
    kn = [e.name for e in p.events() if e.device_type.name == "CUDA"]
    check(kn == [] and torch.equal(core[:, :n], expect), f"D.4 merge statement after direct out launches nothing ({kn})")
    o2, _ = W.chunk_prefill(layer, **x)       # workspace path
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        core[0, :n] = o2[0, :n]
        torch.cuda.synchronize()
    kn = [e.name for e in p.events() if e.device_type.name == "CUDA"]
    check(len(kn) == 1 and torch.equal(core[:, :n], expect), f"D.4 workspace path: the merge is one copy ({len(kn)} kernel)")

    # D.5 timing
    for T in (13824, 4608, 1791):
        x, T = inputs([T])
        core = torch.empty(1, T, H, D, dtype=torch.bfloat16, device="cuda")

        def ws_path():
            o, _ = W.chunk_prefill(layer, **x)
            core[0, :T] = o[0, :T]

        def direct_path():
            o, _ = W.chunk_prefill(layer, **x, out=core[:, :T])
            core[0, :T] = o[0, :T]

        ts = {"workspace+copy": [], "direct": []}
        for r in range(8):
            for name, fn in ((("workspace+copy", ws_path), ("direct", direct_path)) if r % 2 == 0
                             else (("direct", direct_path), ("workspace+copy", ws_path))):
                for _ in range(2):
                    fn()
                torch.cuda.synchronize()
                for _ in range(8):
                    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                    s.record(); fn(); e.record(); torch.cuda.synchronize()
                    ts[name].append(s.elapsed_time(e))
        m = {k: statistics.median(v) for k, v in ts.items()}
        print(f"D.5 T={T}: workspace+copy {m['workspace+copy']:.3f} ms | direct {m['direct']:.3f} ms | saving "
              f"{m['workspace+copy'] - m['direct']:.3f} ms/layer = {(m['workspace+copy'] - m['direct']) * 34:.1f} ms per chunk "
              f"(34 KDA layers)", flush=True)
        check(m["direct"] < m["workspace+copy"], f"D.5 T={T}: direct is faster")
    print("DIRECT_OUT UNIT:", "ALL OK" if not FAIL else f"{len(FAIL)} FAILED")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
