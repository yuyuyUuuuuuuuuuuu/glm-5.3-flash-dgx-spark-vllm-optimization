"""GLM53_BF16_GEMV through vLLM's real plugin entry and real model loader (docs/BF16_GEMV.md).

integrate.plugin_register() (the vllm.general_plugins entry point, TF_EXL3_MOE unset) with GLM53_BF16_GEMV=1, then the
real DFlash2 drafter (incoai/GLM-5.3-Flash-DFlash2 @ dc77ff1c) is built by load_dflash_model -> get_model ->
DefaultModelLoader -> base_loader.process_weights_after_loading (wrapped by the plugin) with the production
speculative config (as tests/drafter/test_drafter_fp8_build.py). Checks: the hook fired inside vLLM's loader and wired
all 10 kernel_projection linears (5 layers x attention_conv / mlp_conv) after their real weights were loaded; the
compile-cache tag is in the live VllmConfig; each wired module's output against F.linear on its weight at the
drafter's decode M (8 B, B = 1..8): served M within 1 bf16 ulp of the exact value, unserved M bitwise production.
Run: GPU_RUN_RO=$TF_EXL3_MODELS/GLM-5.3-Flash-DFlash2-dc77ff1c:$TF_EXL3_MODELS/GLM-5.3-Flash-EXL3-TR3-4bpw-partial \
     tests/gpu_run.sh python3 tests/test_gemv_drafter_load.py
"""
from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from harness import Checks, run_main  # noqa: E402

DRAFT = os.environ.get("DRAFT_DIR", os.path.join(os.environ.get("TF_EXL3_MODELS") or os.path.expanduser("~/models"), "GLM-5.3-Flash-DFlash2-dc77ff1c"))
TGT = os.environ.get("TARGET_DIR", os.path.join(os.environ.get("TF_EXL3_MODELS") or os.path.expanduser("~/models"), "GLM-5.3-Flash-EXL3-TR3-4bpw-partial"))


def main() -> None:
    ck = Checks()
    torch.cuda.set_per_process_memory_fraction(min(1.0, 8 * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    os.environ.pop("TF_EXL3_MOE", None)
    os.environ["GLM53_BF16_GEMV"] = "1"
    import integrate
    integrate.plugin_register()   # the real plugin entry point
    import glm53_gemv_install as I
    import glm53_bf16_gemv as G
    import vllm.model_executor.model_loader.base_loader as BL
    ck(I._STATE["ops"] and I._STATE["loader"] and hasattr(BL.process_weights_after_loading, "_glm53_gemv_orig"),
       "plugin_register -> GLM53_BF16_GEMV installed (ops + loader hook)")

    from vllm.config import set_current_vllm_config
    from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
    from vllm.engine.arg_utils import EngineArgs
    import vllm.v1.worker.gpu.spec_decode.dflash.utils as DU

    vc = EngineArgs(model=TGT, skip_tokenizer_init=True, tensor_parallel_size=1, max_model_len=4096, enforce_eager=True,
                    gpu_memory_utilization=0.1, max_num_seqs=8, language_model_only=True, load_format="safetensors",
                    speculative_config={"method": "dflash", "model": DRAFT, "num_speculative_tokens": 7,
                                        "draft_sample_method": "probabilistic", "rejection_sample_method": "standard"}
                    ).create_engine_config()
    with socket.socket() as s_:
        s_.bind(("127.0.0.1", 0))
        port = s_.getsockname()[1]
    with set_current_vllm_config(vc):
        init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                     distributed_init_method=f"tcp://127.0.0.1:{port}", backend="nccl")
        ensure_model_parallel_initialized(1, 1)

    class StubTarget(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()

    h0 = vc.compute_hash()
    with set_current_vllm_config(vc):
        model = DU.load_dflash_model(StubTarget(), vc)
    torch.cuda.synchronize()
    tag = vc.additional_config.get("glm53_bf16_gemv") if isinstance(vc.additional_config, dict) else None
    ck(isinstance(tag, str) and vc.compute_hash() != h0, f"compile-cache tag in the live VllmConfig: {tag}")
    kps = [(n, m) for n, m in model.named_modules() if n.endswith("kernel_projection")]
    wired = [(n, m) for n, m in kps if getattr(m, "_glm53_gemv_handle", None) is not None]
    print(f"   kernel_projection modules: {len(kps)}, wired: {len(wired)}; summary {I.summary()}")
    ck(len(kps) == 10 and len(wired) == 10, "all 10 DFlash2 kernel_projection linears wired by the loader hook")
    ck(all(type(m.quant_method).__name__ == "Glm53GemvUnquantizedLinearMethod" for _, m in wired),
       "their quant_method is the wrapping subclass")
    gen = torch.Generator(device="cuda").manual_seed(3)
    with torch.no_grad():
        for n, m in wired:
            w = m.weight
            ck(w.dtype == torch.bfloat16 and tuple(w.shape) == (1024, 4096) and float(w.float().abs().sum()) > 0,
               f"{n}: real bf16 weight [1024, 4096]")
            for B in range(1, 9):
                M = 8 * B
                x = torch.randn(M, 4096, device="cuda", generator=gen).to(torch.bfloat16)
                y = m(x)
                y = y[0] if isinstance(y, tuple) else y
                ref = F.linear(x, w)
                if G.serve_plan(1024, 4096, M) is not None:
                    ex = x.double() @ w.double().t()
                    mag = x.double().abs() @ w.double().abs().t()
                    ok = bool(((y.double() - ex).abs() <= ex.abs() * 2.0 ** -8 + 1e-5 * mag).all())
                else:
                    ok = torch.equal(y, ref)
                ck(ok, f"{n} M={M}: {'kernel within 1 ulp of exact' if G.serve_plan(1024, 4096, M) else 'production op bitwise'}")
    print("   paths:", {k: v["M"] for k, v in I.summary().items()})
    ck.summary()


if __name__ == "__main__":
    run_main(main)
