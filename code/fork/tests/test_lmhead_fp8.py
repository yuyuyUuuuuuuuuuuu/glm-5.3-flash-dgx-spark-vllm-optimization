"""glm53_runtime.convert_lm_head_fp8 on vLLM's real ParallelLMHead + LogitsProcessor (TP=1, per-rank shard shape 77440 x 4096),
with production's overlay exl3.py bound over the image's (GPU_RUN_BIND=docs/ref/prod_live/overlay_exl3.py=<vllm>/.../exl3.py).
Checks: logits error vs BF16, top-1 agreement, BF16 weight freed (meta), memory delta, CUDA-graph capture/replay, timing."""
import os, sys, types, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29533")
import torch, torch.nn as nn
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import init_distributed_environment, initialize_model_parallel
import glm53_runtime as R
ok = True
with set_current_vllm_config(VllmConfig()):
    init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="tcp://127.0.0.1:29533", backend="nccl")
    initialize_model_parallel(1, 1)
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    V, K = 77440, 4096
    model = nn.Module(); model.language_model = nn.Module()          # production nests it (Glm5Next: language_model.lm_head)
    model.language_model.lm_head = ParallelLMHead(V, K).cuda().to(torch.bfloat16)
    g = torch.Generator(device="cuda").manual_seed(1)
    with torch.no_grad():
        model.language_model.lm_head.weight.copy_((torch.randn(V, K, device="cuda", generator=g) * 0.02).to(torch.bfloat16))
        model.language_model.lm_head.weight[:64] *= 8                      # a few heavy rows
    lp = LogitsProcessor(V)
    worker = types.SimpleNamespace(model_runner=types.SimpleNamespace(get_model=lambda: model),
                                   vllm_config=types.SimpleNamespace(model_config=types.SimpleNamespace(head_dtype=None)))
    xs = {m: (torch.randn(m, K, device="cuda", generator=g) * 1.5).to(torch.bfloat16) for m in (1, 5, 8, 64)}
    ref = {m: lp(model.language_model.lm_head, x).float() for m, x in xs.items()}
    def tms(fn, it=30):
        fn(); torch.cuda.synchronize(); ts = []
        for _ in range(it):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
        return statistics.median(ts)
    t_bf16 = {m: tms(lambda x=x: lp(model.language_model.lm_head, x)) for m, x in xs.items()}
    free0 = torch.cuda.mem_get_info()[0]
    R.convert_lm_head_fp8(worker)
    free1 = torch.cuda.mem_get_info()[0]
    print("weight device after:", model.language_model.lm_head.weight.device, "| quant_method:", type(model.language_model.lm_head.quant_method).__name__,
          f"| free +{(free1-free0)/2**20:.0f} MiB")
    ok &= model.language_model.lm_head.weight.device.type == "meta" and type(model.language_model.lm_head.quant_method).__name__ == "_Fp8HeadApply"
    for m, x in xs.items():
        out = lp(model.language_model.lm_head, x).float()
        rel = ((out - ref[m]).norm() / ref[m].norm()).item()
        top1 = (out.argmax(-1) == ref[m].argmax(-1)).float().mean().item()
        t = tms(lambda x=x: lp(model.language_model.lm_head, x))
        print(f"M={m:2d} rel_l2 {rel:.3e} top1 {top1:.3f}  time bf16 {t_bf16[m]*1e3:7.1f} us -> fp8 {t*1e3:7.1f} us")
        ok &= rel < 5e-2 and out.shape == ref[m].shape
    # CUDA graph capture/replay
    x = xs[8].clone(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2): y = lp(model.language_model.lm_head, x)
    torch.cuda.current_stream().wait_stream(s)
    gph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gph): y = lp(model.language_model.lm_head, x)
    x.copy_(xs[8]); gph.replay(); torch.cuda.synchronize()
    eq = torch.equal(y, lp(model.language_model.lm_head, xs[8]))
    print("graph replay equals eager:", eq); ok &= eq
print("ALL PASSED" if ok else "FAILED"); sys.exit(0 if ok else 1)
