"""Node1 tests for overlay/patch_kda_strided_qkv.py (vLLM #55736 KDA backport, GLM53_KDA_STRIDED_QKV).

Run:  flock /tmp/tf-gpu-bench.lock tests/gpu_run.sh python3 tests/test_kda_strided_qkv.py

What it proves, against the RUNNING IMAGE'S own site-packages text (the patched side is the shipped
overlay's own prepare() output, not a hand copy):
  1. patched (strided q/k/v/beta in place) == stock (production's .contiguous() copies), BITWISE:
     outputs (bf16) and final recurrent states (fp32), over the decode/spec shapes including the
     production verify shape (batch 1, 8 tokens) and a mixed batch with num_accepted_tokens.
  2. patched with strided inputs == patched with the same inputs made contiguous, BITWISE (the
     kernel's token-stride addressing reads exactly the bytes the copies would).
  3. both match a pure-fp32 recurrence (ported from upstream #55736's test) at upstream's tolerance.
  4. layouts the token-stride addressing cannot express (batch slice of a wider buffer, overlapping
     tokens, head-strided block) raise loudly instead of reading the wrong tokens.
  5. FULL CUDA-graph capture of the patched call replays bitwise-identically (production decode is
     captured/replayed, so this must keep working), the out= write-through path still aliases, and
     the patched launch really is one kernel where stock launches copies + kernel.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kda_strided_common as KC  # noqa: E402

import torch  # noqa: E402

from harness import Checks, run_main  # noqa: E402

DEV = "cuda"
H, K = 64, 128  # production GLM-5.3-Flash: linear_attn_config num_heads=64, head_dim=128
LOWER_BOUND = -5.0


def snap(src):
    return {n: (v.clone() if isinstance(v, torch.Tensor) else v) for n, v in src.items()}


def naive_reference(inp: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-fp32 recurrence for every (seq, token); ports upstream #55736's test reference."""
    q, k, v, g, beta = (x.float() for x in (inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"]))
    HH, D = q.shape[2], q.shape[3]
    nseq = inp["cu_seqlens"].numel() - 1
    idx = inp["ssm_state_indices"].view(nseq, q.shape[1] // nseq)
    q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6) * D**-0.5
    k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    beta = torch.sigmoid(beta)
    acc = inp.get("num_accepted_tokens")
    state = inp["initial_state"].clone()
    out = torch.empty_like(v, dtype=torch.float32)
    for n in range(nseq):
        first = idx[n, 0 if acc is None else int(acc[n]) - 1]
        s = state[int(first)]
        for t in range(idx.shape[1]):
            tok = slice(n * idx.shape[1] + t, n * idx.shape[1] + t + 1)
            qq, kk, vv = q[0, tok], k[0, tok], v[0, tok]
            gate = LOWER_BOUND * torch.sigmoid(
                inp["a_log"].exp()[:, None] * (g[0, tok] + inp["g_bias"].view(HH, D))
            )
            bb = beta[0, tok]
            o = torch.empty(1, HH, D, dtype=torch.float32, device=q.device)
            for j in range(1):
                st = s * gate[0].exp()[:, None, :]
                u = bb[j][:, None] * (vv[j] - torch.einsum("hvk,hk->hv", st, kk[j]))
                st = st + u[:, :, None] * kk[j][:, None, :]
                o[j] = torch.einsum("hvk,hk->hv", st, qq[j])
                s = st  # the recurrence carries the state across the sequence's tokens
            out[0, tok] = o
            state[idx[n, t]] = st
    return out, state


def run_case(checks: Checks, stock, patched, num_seqs: int, query_len: int, seed: int = 0) -> None:
    tag = f"{num_seqs}x{query_len}"
    inp = KC.make_decode_inputs(num_seqs, query_len, H, K, DEV, seed=seed)
    slots = inp["ssm_state_indices"].reshape(-1).long()

    s_in = dict(inp, initial_state=inp["initial_state"].clone())
    out_s, st_s = KC.run_kda(stock, s_in)
    p_in = dict(inp, initial_state=inp["initial_state"].clone())
    out_p, st_p = KC.run_kda(patched, p_in)
    checks(bool(out_s.shape == out_p.shape and out_s.dtype == out_p.dtype),
           f"[{tag}] output shape/dtype {out_s.shape}/{out_s.dtype} vs {out_p.shape}/{out_p.dtype}")
    checks(torch.equal(out_p, out_s), f"[{tag}] output not bitwise-equal to stock")
    checks(torch.equal(st_p[slots], st_s[slots]), f"[{tag}] final states not bitwise-equal to stock")

    # patched must read the SAME bytes in place that the copies would: contiguous inputs, same module
    cin = {n: (x.contiguous() if n in ("q", "k", "v", "beta") else x) for n, x in inp.items()}
    cin["initial_state"] = inp["initial_state"].clone()
    out_c, st_c = KC.run_kda(patched, cin)
    checks(torch.equal(out_c, out_p), f"[{tag}] strided != contiguous inside the patched kernel")
    checks(torch.equal(st_c[slots], st_p[slots]), f"[{tag}] strided vs contiguous final states differ")

    # pure-fp32 recurrence (upstream's reference): loose tolerance, the patched path must sit inside it
    ref_out, ref_state = naive_reference(inp)
    diff = (out_p.float() - ref_out[0]).abs().max().item()
    checks(diff <= 1e-2, f"[{tag}] patched output vs fp32 reference max abs diff {diff:.3e} > 1e-2")
    sdiff = (st_p[slots] - ref_state[slots]).abs().max().item()
    checks(sdiff <= 1e-4, f"[{tag}] patched final state vs fp32 reference max abs diff {sdiff:.3e}")
    print(f"  ok   {tag}: bitwise vs stock; fp32-ref diff out {diff:.2e}, state {sdiff:.2e}")


def reject_bad_layouts(checks: Checks, patched) -> None:
    """Upstream's third test: layouts the token stride cannot express must fail loudly."""
    inp = KC.make_decode_inputs(1, 1, H, K, DEV, seed=3)
    T = 4
    base = torch.randn(4, T, 2 * H * K, dtype=torch.bfloat16, device=DEV)
    bad_q = {
        "batch slice": base[::2, :, : H * K].view(2, T, H, K),
        "overlapping tokens": base[:1, :, : H * K].as_strided((1, T, H, K), (0, K, K, 1)),
        "head-strided": base[:1, :, : H * K].view(1, T, K, H).transpose(2, 3),
    }
    for name, q in bad_q.items():
        broken = dict(inp, q=q, k=q, v=q)
        if q.shape[0] > 1:
            broken["cu_seqlens"] = None
        try:
            KC.run_kda(patched, broken)
        except AssertionError as exc:
            checks("torch.Size" in str(exc), f"rejected layout {name}: wrong assert text: {exc}")
            print(f"  ok   rejected layout '{name}' raised: {str(exc).strip()[:110]}")
        else:
            checks(False, f"layout '{name}' was NOT rejected (silent wrong reads)")


def graph_capture(checks: Checks, stock, patched) -> None:
    """FULL decode replay: capture the patched call on static strided buffers, replay, compare."""
    inp = KC.make_decode_inputs(1, 8, H, K, DEV, seed=4)  # production verify shape: 1 req, 8 tokens
    T, HK = inp["q"].shape[1], inp["q"].shape[2] * inp["q"].shape[3]

    def snap(src):
        return {n: (v.clone() if isinstance(v, torch.Tensor) else v) for n, v in src.items()}

    # eager on live static buffers (patched). The call mutates initial_state in place, so every
    # comparison below starts from the same fresh state (snap(inp)).
    out_buf = torch.empty(1, T, H, K, dtype=torch.bfloat16, device=DEV)
    o_eager, s_eager = KC.run_kda(patched, snap(inp))
    o_eager_out, s_eager_out = KC.run_kda(patched, snap(inp), out=out_buf)
    checks(torch.equal(o_eager, o_eager_out), "out= path differs from the fresh allocation")
    checks(torch.equal(s_eager, s_eager_out), "out= path final state differs")
    checks(o_eager_out.data_ptr() == out_buf.data_ptr(), "out= was not written in place")

    # FULL capture (torch.cuda.graph, the FULL replay mode) of the patched call on static buffers
    o_c = torch.empty(1, T, H, K, dtype=torch.bfloat16, device=DEV)
    sb = snap(inp)
    KC.run_kda(patched, sb, out=o_c)  # JIT/warm-up before capture
    torch.cuda.synchronize()
    sb = snap(inp)  # the warm-up mutated the state; capture from the same fresh state as eager
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        # stream capture records the launches without executing them: the buffers only get the
        # values on replay, from whatever the static inputs hold then
        o_c, s_c = KC.run_kda(patched, sb, out=o_c)
    o_c.fill_(float("nan"))
    g.replay()
    torch.cuda.synchronize()
    checks(torch.equal(o_c, o_eager), "replayed output != eager (capture baked the wrong strides)")
    checks(torch.equal(sb["initial_state"], s_eager), "replayed final state != eager")
    # a replay mutates the paged state in place (production reloads it per step): reset it first,
    # then a second replay must again produce the same output — the launch is replay-deterministic
    sb["initial_state"].copy_(inp["initial_state"])
    o_c.fill_(float("nan"))
    g.replay()
    torch.cuda.synchronize()
    checks(torch.equal(o_c, o_eager), "second replay (state reset first) diverged")
    print("  ok   patched: FULL capture + 2 replays bitwise-equal to eager")

    # the stock path captures and replays identically too (both sides keep working)
    KC.run_kda(stock, snap(inp))  # warm-up on throwaway buffers
    torch.cuda.synchronize()
    sb_old = snap(inp)  # capture from a fresh state, like the patched graph
    go = torch.cuda.CUDAGraph()
    with torch.cuda.graph(go):
        o_old, _ = KC.run_kda(stock, sb_old)
    o_old.fill_(float("nan"))
    go.replay()
    torch.cuda.synchronize()
    checks(torch.equal(o_old, o_eager), "stock (copied inputs) captured/replayed output != patched")
    print("  ok   stock: FULL capture + replay also still works and agrees")


def kernel_counts(checks: Checks, stock, patched) -> None:
    """One production-shaped call, counted by a TorchDispatchMode (no profiler: CUPTI both dropped the
    aten records and mis-attributed the copies in this image). stock must issue 4 MORE
    contiguous/copy_ ops (q/k/v/beta) than patched, whose only contiguous() is the kept `g`; and the
    captured graph replays exactly one CUDA kernel for patched."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class Counter(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.calls: dict[str, int] = {}

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.calls[func.__name__] = self.calls.get(func.__name__, 0) + 1
            return func(*args, **(kwargs or {}))

    inp = KC.make_decode_inputs(1, 8, H, K, DEV, seed=5)
    T = inp["q"].shape[1]

    def fresh() -> dict:
        """Strided views preserved (cloning a column slice falls back to contiguous and would hide
        the very copies we count); only the in-place-updated state gets its own buffer."""
        cur = dict(inp)
        cur["initial_state"] = inp["initial_state"].clone()
        return cur

    def tally(calls: dict[str, int], prefix: str) -> int:
        # dispatch keys carry the overload suffix ("contiguous.default")
        return sum(n for k, n in calls.items() if k.split(".")[0] == prefix)

    cs: dict[str, dict[str, int]] = {}
    for name, mod in (("stock", stock), ("patched", patched)):
        KC.run_kda(mod, fresh())  # JIT outside
        torch.cuda.synchronize()
        with Counter() as c:
            KC.run_kda(mod, fresh())
        torch.cuda.synchronize()
        cs[name] = c.calls
        print(f"  {name}: " + ", ".join(f"{k} x{n}" for k, n in sorted(c.calls.items())))
    ks, kp = cs["stock"], cs["patched"]
    # on this torch build Tensor.contiguous() on a strided tensor dispatches as aten::clone (no
    # aten::contiguous record), so the four copies show up as 4 extra clones in stock
    checks(tally(ks, "clone") - tally(kp, "clone") == 4,
           f"clone (the .contiguous() copies): stock {tally(ks, 'clone')} - patched {tally(kp, 'clone')} != 4")
    checks(tally(ks, "contiguous") == 0 and tally(kp, "contiguous") == 0,
           "unexpected aten::contiguous dispatch records")
    print(f"ok   .contiguous() copies per KDA layer call removed: clone dispatch {tally(ks, 'clone')}->"
          f"{tally(kp, 'clone')}")

    # and the captured graph replays exactly one CUDA kernel per call for the patched side
    from torch.profiler import ProfilerActivity, profile

    out = torch.empty(1, T, H, K, dtype=torch.bfloat16, device=DEV)
    sb = fresh()
    KC.run_kda(patched, sb, out=out)
    torch.cuda.synchronize()
    sb = fresh()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        KC.run_kda(patched, sb, out=out)
    g.replay()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        g.replay()
        torch.cuda.synchronize()
    evs = [
        e
        for e in prof.key_averages()
        if e.device_type == torch.autograd.DeviceType.CUDA and e.self_device_time_total > 0
    ]
    cnt = sum(e.count for e in evs)
    for e in evs:
        print(f"       {e.self_device_time_total:8.1f} us  x{e.count:<3} {e.key[:100]}")
    checks(cnt == 1, f"patched graph replay launches {cnt} CUDA kernels, expected 1")
    print("ok   patched replay: exactly 1 CUDA kernel per KDA layer call")


def main() -> None:
    checks = Checks()
    stock, patched, pk, tmp = KC.load_stock_and_patched()
    print("modules loaded: stock(fla_stock) patched(fla_patched)")
    for ns, ql, seed in ((1, 1, 0), (7, 1, 1), (1, 8, 2), (3, 3, 5)):
        print(f"- case {ns}x{ql} (H={H}, K={K})")
        run_case(checks, stock, patched, ns, ql, seed)
    print("- rejected layouts")
    reject_bad_layouts(checks, patched)
    print("- FULL CUDA-graph capture + replay")
    graph_capture(checks, stock, patched)
    print("- kernel counts per layer call")
    kernel_counts(checks, stock, patched)
    checks.summary()


if __name__ == "__main__":
    run_main(main)
