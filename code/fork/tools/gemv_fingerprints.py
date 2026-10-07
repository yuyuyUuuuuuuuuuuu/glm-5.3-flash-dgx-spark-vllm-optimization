import sys, hashlib, inspect
sys.path.insert(0, "/w")
from integrate import source_fingerprint as fp
import vllm.model_executor.layers.fused_moe.router.gate_linear as GL
import vllm.model_executor.layers.fused_moe.runner.moe_runner as MR
import vllm.models.glm5next.nvidia.attention as AT
import vllm.models.glm5next.nvidia.model as MD
import vllm.model_executor.models.qwen3_dflash2 as QD2
import vllm.model_executor.layers.linear as LN
for label, fn in [("GateLinear.forward", GL.GateLinear.forward), ("MoERunner._forward_impl", MR.MoERunner._forward_impl),
                  ("Glm5NextMoE.forward", MD.Glm5NextMoE.forward), ("Indexer.forward", AT.Indexer.forward),
                  ("DFlashGroupedConv.prepare", QD2.DFlashGroupedConv.prepare),
                  ("UnquantizedLinearMethod.apply", LN.UnquantizedLinearMethod.apply)]:
    print(label, fp(fn), inspect.getsourcefile(fn))
for m in (GL, MR, AT, MD, QD2):
    print(hashlib.sha256(open(m.__file__, "rb").read()).hexdigest()[:16], m.__file__)
