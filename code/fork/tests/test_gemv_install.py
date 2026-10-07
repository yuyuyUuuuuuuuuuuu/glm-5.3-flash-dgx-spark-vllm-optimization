"""GLM53_BF16_GEMV wiring (glm53_gemv_install.py, docs/BF16_GEMV.md) on the real vLLM classes of the image.

Modules: GateLinear, Indexer (the real class; its forward runs production's helpers _fused_indexer_k_norm,
fwht128_quant_fp8, _fused_indexer_weight_scale, with a capturing stub for the sparse indexer op), DFlashGroupedConv,
Glm5NextMoE + MoERunner (real classes, built without their heavy constructors, for the router dedup). Weights: the
real router / indexer weights of layers 10 / 11 (GLM53_CKPT, bf16 in the checkpoint) and a real DFlash2
kernel_projection (DRAFT_DIR); random ones when the files are absent.

Checks:
  G1  env off: plugin_install registers no op, hooks nothing, patches no class.
  G2  env on: ops registered, loader hooked; post_load wires exactly router/idx_wk/idx_head/idx_kpool/draft_conv/dedup,
      rejects nothing; the compile-cache tag is set in additional_config and changes vllm_config.compute_hash().
  G3  every M in 1..64 and prefill-sized M: served M -> within 1 bf16 ulp (bf16 / bf16-rounded outputs) or the fp32
      bound (head gate) of the exact value; unserved M -> bitwise equal to production's op; which path ran is what
      glm53_bf16_gemv.serve_plan / F32_MAX_M say.
  G4  Indexer: patched forward vs production's forward (same inputs): q_fp8 / q_scale bitwise, k / weights /
      gate_score within the rounding bound, at served and unserved M; the patched copy's AST differs from the
      production function's only in the two GEMM statements.
  G5  torch.compile(fullgraph=True, dynamic M): no graph break; the M-dependent choice is made per call (M = 5, 40, 64,
      100 through one compiled graph give the eager results bitwise); CUDA graph capture + replay == eager.
  G6  unknown handle -> production's op bitwise; a module whose self-test fails (mutated kernel) is not wired and keeps
      production's path; dedup off by env keeps the runner's gate.
Run: GPU_RUN_RO=$TF_EXL3_MODELS/GLM-5.3-Flash-Uncensored-NVFP4:$TF_EXL3_MODELS/GLM-5.3-Flash-DFlash2-dc77ff1c \
     tests/gpu_run.sh python3 tests/test_gemv_install.py
"""
from __future__ import annotations

import ast
import glob
import inspect
import json
import os
import socket
import struct
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from harness import Checks, run_main  # noqa: E402

CKPT = os.environ.get("GLM53_CKPT", os.path.join(os.environ.get("TF_EXL3_MODELS") or os.path.expanduser("~/models"), "GLM-5.3-Flash-Uncensored-NVFP4"))
DRAFT = os.environ.get("DRAFT_DIR", os.path.join(os.environ.get("TF_EXL3_MODELS") or os.path.expanduser("~/models"), "GLM-5.3-Flash-DFlash2-dc77ff1c"))
DEV = torch.device("cuda")


def ckpt_tensors(names: list[str]) -> dict[str, torch.Tensor]:
    from safetensors import safe_open
    idx = {}
    for f in sorted(glob.glob(os.path.join(CKPT, "model-*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            for k in json.loads(fh.read(n)):
                idx[k] = f
    out = {}
    for nm in names:
        if nm in idx:
            with safe_open(idx[nm], framework="pt", device="cpu") as f:
                out[nm] = f.get_tensor(nm)
    return out


def bf16_close(new: torch.Tensor, ref64: torch.Tensor, mag: torch.Tensor) -> bool:
    return bool(((new.double() - ref64).abs() <= ref64.abs() * 2.0 ** -8 + 1e-5 * mag + 1e-30).all())


class CaptureOp(torch.nn.Module):
    """Stands in for the sparse indexer op: returns what the Indexer computed for it."""

    def forward(self, hidden_states, q_fp8, k, weights, gate_score=None, compress_ape=None, index_kpool=None,
                positions=None):
        return q_fp8.float(), k.float(), weights.float(), gate_score.float()


def main() -> None:
    ck = Checks()
    import glm53_gemv_install as I
    import glm53_bf16_gemv as G
    import vllm.model_executor.model_loader.base_loader as BL
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    from vllm.model_executor.layers.layernorm import LayerNorm
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear, ReplicatedLinear
    from vllm.model_executor.models.qwen3_dflash2 import DFlashGroupedConv
    import vllm.models.glm5next.nvidia.attention as AT
    from vllm.models.glm5next.nvidia.model import Glm5NextMoE

    vc = VllmConfig()
    with socket.socket() as s_:
        s_.bind(("127.0.0.1", 0))
        port = s_.getsockname()[1]
    with set_current_vllm_config(vc):
        init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                     distributed_init_method=f"tcp://127.0.0.1:{port}", backend="nccl")
        ensure_model_parallel_initialized(1, 1)

    # ---------------------------------------------------------------- G1: env off
    print("== G1 env off")
    os.environ.pop(I.ENV, None)
    orig_pwal, orig_gate_fwd, orig_idx_fwd = BL.process_weights_after_loading, GateLinear.forward, AT.Indexer.forward
    I.plugin_install()
    ck(not I._STATE["ops"] and not hasattr(torch.ops.glm53_gemv, "linear"), "env off: no custom op registered")
    ck(BL.process_weights_after_loading is orig_pwal, "env off: loader not hooked")
    ck(GateLinear.forward is orig_gate_fwd and AT.Indexer.forward is orig_idx_fwd, "env off: no class patched")
    for v, want in (("0", False), ("", False), ("off", False), ("1", True), (" on ", True), ("yes", True)):
        ck(I.env_enabled({I.ENV: v}) is want, f"env_enabled({v!r}) == {want}")
    ck(I.env_kinds({}) == I.ALL_KINDS and I.env_kinds({I.ENV_KINDS: "router, idx_head"}) == ("router", "idx_head"),
       "env_kinds default / list")
    try:
        I.env_kinds({I.ENV_KINDS: "router,bogus"})
        ck(False, "env_kinds rejects an unknown kind")
    except ValueError:
        ck(True, "")
    ck(I.env_dedup({}) and not I.env_dedup({I.ENV_DEDUP: "0"}), "dedup default on, 0 -> off")

    # ---------------------------------------------------------------- G2: env on, build real modules
    print("== G2 env on")
    os.environ[I.ENV] = "1"
    I.plugin_install()
    ck(I._STATE["ops"] and hasattr(torch.ops.glm53_gemv, "linear") and hasattr(torch.ops.glm53_gemv, "head_gate"),
       "env on: custom ops registered")
    ck(BL.process_weights_after_loading is not orig_pwal and
       BL.process_weights_after_loading._glm53_gemv_orig is orig_pwal, "env on: loader hooked (wraps the original)")

    real = ckpt_tensors([f"model.language_model.layers.10.mlp.gate.weight",
                         "model.language_model.layers.11.self_attn.indexer.wk.weight",
                         "model.language_model.layers.11.self_attn.indexer.weights_proj.weight",
                         "model.language_model.layers.11.self_attn.indexer.index_kpool_compress_gate",
                         "model.language_model.layers.11.self_attn.indexer.k_norm.weight",
                         "model.language_model.layers.11.self_attn.indexer.k_norm.bias",
                         "model.language_model.layers.11.self_attn.indexer.wq_b.weight"])
    print(f"   real checkpoint tensors found: {len(real)}/7 ({CKPT})")
    gen = torch.Generator(device=DEV).manual_seed(7)

    def w_or_rand(name, shape, scale=0.02):
        t = real.get(name)
        if t is not None:
            assert tuple(t.shape) == shape, (name, t.shape)
            return t.to(DEV, torch.bfloat16)
        return (torch.randn(*shape, device=DEV, generator=gen) * scale).to(torch.bfloat16)

    L = "model.language_model.layers."
    torch.set_default_dtype(torch.bfloat16)   # as vLLM's loader does (set_default_torch_dtype(model_config.dtype))
    with set_current_vllm_config(vc), torch.device(DEV):
        gate = GateLinear(4096, 288, out_dtype=torch.float32, prefix="model.layers.10.mlp.gate")
        gate.weight.data.copy_(w_or_rand(L + "10.mlp.gate.weight", (288, 4096)))
        # MoE + runner (real classes, bare construction) for the dedup
        moe = Glm5NextMoE.__new__(Glm5NextMoE)
        torch.nn.Module.__init__(moe)
        moe.gate = gate
        runner = MoERunner.__new__(MoERunner)   # production: Glm5NextMoE.experts IS the MoERunner (FusedMoEFactory)
        torch.nn.Module.__init__(runner)
        runner.gate, runner._fse_fuse_gate, runner.routed_input_transform = gate, False, None
        moe.experts = runner
        # Indexer (real class, bare construction, the attributes its forward reads)
        idx = AT.Indexer.__new__(AT.Indexer)
        torch.nn.Module.__init__(idx)
        idx.n_head, idx.head_dim, idx.rope_dim, idx.index_kpool = 32, 128, 0, 4
        idx.quant_block_size, idx.scale_fmt, idx.softmax_scale = 128, "ue8m0", 128 ** -0.5
        idx.wq_b = ReplicatedLinear(1536, 4096, bias=False, params_dtype=torch.bfloat16, prefix="x.wq_b")
        idx.wq_b.weight.data.copy_(w_or_rand(L + "11.self_attn.indexer.wq_b.weight", (4096, 1536)))
        idx.wk_weights_proj = MergedColumnParallelLinear(4096, [128, 32], bias=False, quant_config=None,
                                                         disable_tp=True, params_dtype=torch.bfloat16,
                                                         prefix="x.wk_weights_proj")
        idx.wk_weights_proj.weight.data.copy_(torch.cat([w_or_rand(L + "11.self_attn.indexer.wk.weight", (128, 4096)),
                                                         w_or_rand(L + "11.self_attn.indexer.weights_proj.weight",
                                                                   (32, 4096))]))
        idx.k_norm = LayerNorm(128, eps=1e-6)
        for nm in ("weight", "bias"):
            t = real.get(L + f"11.self_attn.indexer.k_norm.{nm}")
            if t is not None:
                getattr(idx.k_norm, nm).data.copy_(t.to(DEV, getattr(idx.k_norm, nm).dtype))
        idx.index_kpool_compress_gate = torch.nn.Parameter(
            w_or_rand(L + "11.self_attn.indexer.index_kpool_compress_gate", (128, 4096)), requires_grad=False)
        idx.index_kpool_compress_ape = torch.nn.Parameter(torch.zeros(4, 128), requires_grad=False)
        idx.indexer_op = CaptureOp()
        conv = DFlashGroupedConv(4096, 2, 16, 8, torch.bfloat16, prefix="model.layers.45.attention_conv")
        kp = None
        try:
            from safetensors import safe_open
            with safe_open(os.path.join(DRAFT, "model.safetensors"), framework="pt", device="cpu") as f:
                kp = f.get_tensor("layers.0.attention_conv.kernel_projection.weight")
        except Exception as exc:  # noqa: BLE001
            print("   drafter weights not available:", repr(exc))
        conv.kernel_projection.weight.data.copy_(kp.to(DEV) if kp is not None else
                                                 (torch.randn(1024, 4096, device=DEV, generator=gen) * 0.02).bfloat16())

    torch.set_default_dtype(torch.float32)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.moe, self.indexer, self.conv = moe, idx, conv

    model = Model()
    h0 = vc.compute_hash()
    # production outputs before wiring (the reference for unserved M and for the indexer)
    Ms = list(range(1, 65)) + [65, 100, 256]
    X = {M: (torch.randn(M, 4096, device=DEV, generator=gen)).to(torch.bfloat16) for M in Ms}
    QR = {M: (torch.randn(M, 1536, device=DEV, generator=gen)).to(torch.bfloat16) for M in Ms}
    with torch.no_grad():
        prod = {M: dict(gate=gate(X[M])[0], wk=idx.wk_weights_proj(X[M])[0], conv=conv.kernel_projection(X[M]),
                        idx=idx(X[M], QR[M], None, None)) for M in Ms}
    with set_current_vllm_config(vc):
        out = I.post_load(model)
    print("   post_load:", {k: v for k, v in out.items() if k != "rejected"}, "rejected:", out["rejected"])
    ck(all(out[k] == 1 for k in I.ALL_KINDS) and out["dedup"] == 1 and not out["rejected"],
       "post_load wires router, idx_wk, idx_head, idx_kpool, draft_conv and the dedup, rejects nothing")
    ck(runner.gate is None and getattr(moe, "_glm53_gemv_dedup", False), "dedup: runner.gate is None")
    tag = vc.additional_config.get("glm53_bf16_gemv")
    ck(isinstance(tag, str) and tag.startswith(f"v{I.VERSION}:") and vc.compute_hash() != h0,
       f"compile-cache tag {tag!r} set and the config hash changed ({h0[:10]} -> {vc.compute_hash()[:10]})")
    ck(GateLinear.forward._glm53_gemv_orig is orig_gate_fwd and AT.Indexer.forward._glm53_gemv_orig is orig_idx_fwd,
       "classes patched once, originals kept")
    ck(type(idx.wk_weights_proj.quant_method).__name__.startswith("Glm53Gemv") and
       type(conv.kernel_projection.quant_method).__name__.startswith("Glm53Gemv"), "quant methods wrapped")

    # ---------------------------------------------------------------- G3 / G4: every M
    print("== G3/G4 outputs, every M")
    wg, wk, wc, kg = gate.weight, idx.wk_weights_proj.weight, conv.kernel_projection.weight, idx.index_kpool_compress_gate
    bad = []
    orig_forward = AT.Indexer.forward._glm53_gemv_orig
    with torch.no_grad():
        for M in Ms:
            x = X[M]
            ref = {n: (x.double() @ w.double().t(), x.double().abs() @ w.double().abs().t())
                   for n, w in (("gate", wg), ("wk", wk), ("conv", wc), ("kg", kg), ("hg", wk[128:]))}
            for name, new, w, om in (("gate", gate(x)[0], wg, 2), ("wk", idx.wk_weights_proj(x)[0], wk, 0),
                                     ("conv", conv.kernel_projection(x), wc, 0)):
                served = G.serve_plan(w.shape[0], w.shape[1], M) is not None
                if served:
                    ok = bf16_close(new, *ref[name]) and new.dtype == prod[M][name].dtype
                else:
                    ok = torch.equal(new, prod[M][name])
                if not ok:
                    bad.append(f"{name} M={M} served={served}")
            # indexer: (q_fp8, k, weights, gate_score) handed to the sparse indexer op
            qn, kn, wn, gn = idx(x, QR[M], None, None)
            cap_prod = orig_forward(idx, x, QR[M], None, None)
            # (production's forward calls the wk_weights_proj module, which is now wired: k matches only where
            # idx_wk is not served)
            same = [0, 2, 3] if G.serve_plan(160, 4096, M) is not None else [0, 1, 2, 3]
            ck(all(torch.equal(cap_prod[i], prod[M]["idx"][i]) for i in same),
               f"indexer production forward reproducible M={M}: " +
               str([(n, float((u - v).abs().max())) for n, u, v in zip("qkwg", cap_prod, prod[M]["idx"])]))
            qp_, kp_, wp_, gp_ = cap_prod
            s_wk = G.serve_plan(160, 4096, M) is not None
            s_hg = 1 <= M <= G.F32_MAX_M
            s_kg = G.serve_plan(128, 4096, M) is not None
            okq = torch.equal(qn, qp_)   # wq_b path untouched
            # weights = head_gate(x) * (per token, head) scale from q (identical on both sides): fp32 GEMM rounding
            okw = (bool(((wn - wp_).abs() <= 1e-5 * wp_.abs().amax(-1, keepdim=True)).all()) if s_hg
                   else torch.equal(wn, wp_))
            okg = bf16_close(gn, *ref["kg"]) if s_kg else torch.equal(gn, gp_)
            # k = LayerNorm(wk(x)[:, :128]): inputs equal up to one bf16 ulp of the GEMM -> loose, same scale
            okk = torch.equal(kn, kp_) if not s_wk else bool(((kn - kp_).abs() <= 0.02 + 0.02 * kp_.abs()).all())
            if not (okq and okw and okg and okk):
                bad.append(f"indexer M={M} q {okq} weights {okw} gate_score {okg} k {okk} (served wk {s_wk} "
                           f"hg {s_hg} kg {s_kg})")
            # the head-gate op itself against float64 (strict fp32 bound)
            if s_hg:
                hg = torch.ops.glm53_gemv.head_gate(x, wk, idx._wp_fp32, idx._glm53_gemv_idx[0], 128)
                if not bool(((hg.double() - ref["hg"][0]).abs() <= 1e-6 * ref["hg"][1] + 1e-30).all()):
                    bad.append(f"head_gate op M={M} outside the fp32 bound")
    ck(not bad, f"all M: served within rounding bound, unserved bitwise production ({bad[:4]})")
    sm = I.summary()
    print("   summary:", {k: {kk: vv for kk, vv in v.items() if kk != "M"} for k, v in sm.items()})
    for kind, (N, K) in (("router", (288, 4096)), ("idx_wk", (160, 4096)), ("idx_kpool", (128, 4096)),
                         ("draft_conv", (1024, 4096))):
        want = {M: ("gemv" if G.serve_plan(N, K, M) is not None else "production") for M in Ms}
        got = {M: v.split()[0] for M, v in sm[kind]["M"].items()}
        ck(got == want, f"{kind}: the path taken per M is the plan table's ({[M for M in Ms if got.get(M) != want[M]]})")
    got = {M: v for M, v in sm["idx_head"]["M"].items()}
    ck(all((got[M] == "gemm_f32") == (M <= G.F32_MAX_M) for M in Ms), "idx_head: gemm_f32 exactly for M <= F32_MAX_M")

    # AST: the patched copy differs from production's Indexer.forward only in the two GEMM statements
    def stmts(fn):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        return [ast.dump(s) for s in tree.body[0].body]
    a, b = stmts(orig_forward), stmts(I._indexer_forward_gemv)
    a_only = [s for s in a if s not in b]
    b_only = [s for s in b if s not in a]
    ck(len(a_only) == 2 and "torch.mm" not in str(b_only) and "mm'" in a_only[0] and "linear'" in a_only[1],
       f"Indexer copy: production-only statements = the head-gate torch.mm and the kpool F.linear ({len(a_only)})")
    ck(len(b_only) == 3, f"Indexer copy: 3 new statements (handle unpack + 2 branches), got {len(b_only)}")

    # ---------------------------------------------------------------- G5: torch.compile + CUDA graph
    print("== G5 torch.compile(fullgraph) + CUDA graph")

    def step(x, qr):
        return gate(x)[0], idx(x, qr, None, None), conv.kernel_projection(x), idx.wk_weights_proj(x)[0]

    def flat(t):   # k (index 2) excluded: production's _fused_indexer_k_norm is not bitwise stable between eager and
        return [t[0], t[1][0], t[1][2], t[1][3], t[2], t[3]]   # inductor (1 bf16 ulp at unserved M too)

    def k_close(a, b):
        return bool(((a[1][1] - b[1][1]).abs() <= 0.02 + 2 ** -6 * a[1][1].abs()).all())

    torch._dynamo.reset()
    cstep = torch.compile(step, fullgraph=True, dynamic=True)
    with torch.no_grad():
        for M in (5, 40, 64, 100, 8):
            e = step(X[M], QR[M])
            c = cstep(X[M], QR[M])
            ck(all(torch.equal(u, v) for u, v in zip(flat(e), flat(c))) and k_close(e, c),
               f"compiled == eager at M={M} (choice made per call): " +
               str([(n, float((u.float() - v.float()).abs().max()))
                    for n, u, v in zip(("gate", "q", "w", "g", "conv", "wk"), flat(e), flat(c))]))
        M = 8
        xs, qs = X[M].clone(), QR[M].clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            cstep(xs, qs)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            yg = cstep(xs, qs)
        for _ in range(3):
            xn = torch.randn(M, 4096, device=DEV, generator=gen).to(torch.bfloat16)
            qn = torch.randn(M, 1536, device=DEV, generator=gen).to(torch.bfloat16)
            xs.copy_(xn)
            qs.copy_(qn)
            g.replay()
            torch.cuda.synchronize()
            e = step(xn, qn)
            ck(all(torch.equal(u, v) for u, v in zip(flat(e), flat(yg))) and k_close(e, yg),
               "CUDA graph replay == eager")

    # ---------------------------------------------------------------- G6: fallbacks
    print("== G6 fallbacks")
    x = X[5]
    ck(torch.equal(torch.ops.glm53_gemv.linear(x, wg, 12345, 2), F.linear(x, wg).to(torch.float32)),
       "unknown handle -> production op bitwise")
    real_gemm = G.gemm
    try:
        G.gemm = lambda *a, **k: real_gemm(*a, **k) * 1.01   # mutated kernel: the self-test must reject it
        torch.set_default_dtype(torch.bfloat16)
        with set_current_vllm_config(vc), torch.device(DEV):
            gate2 = GateLinear(4096, 288, out_dtype=torch.float32, prefix="model.layers.12.mlp.gate")
            gate2.weight.data.copy_(wg)
        torch.set_default_dtype(torch.float32)

        class M2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.g = gate2
        with set_current_vllm_config(vc):
            out2 = I.post_load(M2())
    finally:
        G.gemm = real_gemm
    ck(out2["router"] == 0 and len(out2["rejected"]) == 1 and "self-test" in out2["rejected"][0] and
       getattr(gate2, "_glm53_gemv_handle", None) is None, f"mutated kernel -> rejected ({out2['rejected']})")
    ck(torch.equal(gate2(x)[0], F.linear(x, wg).to(torch.float32)), "rejected module keeps production's path")
    runner2 = MoERunner.__new__(MoERunner)
    torch.nn.Module.__init__(runner2)
    moe2 = Glm5NextMoE.__new__(Glm5NextMoE)
    torch.nn.Module.__init__(moe2)
    runner2.gate, runner2._fse_fuse_gate, runner2.routed_input_transform = gate2, False, None
    moe2.gate, moe2.experts = gate2, runner2

    class M3(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = moe2
    with set_current_vllm_config(vc):
        out3 = I.post_load(M3(), environ={I.ENV: "1", I.ENV_DEDUP: "0", I.ENV_KINDS: "router"})
    ck(runner2.gate is gate2 and out3["dedup"] == 0, "dedup off by env -> runner keeps its gate")
    ck.summary()


if __name__ == "__main__":
    run_main(main)
