"""B3 + C3: the drafter-FP8 overlays, verified with vLLM's own classes on one GB10 (TP=1).

Runs inside ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor (tests/gpu_run.sh, <= 8 GiB) with
  GPU_RUN_RO=$TF_EXL3_MODELS/GLM-5.3-Flash-DFlash2-dc77ff1c:$TF_EXL3_MODELS/GLM-5.3-Flash-EXL3-TR3-4bpw-partial
The image is never modified: overlay/patch_drafter_fp8.py and overlay/patch_drafter_lmhead_fp8.py are applied to
COPIES of qwen3_dflash.py / qwen3_dflash2.py / spec_decode/dflash/utils.py, which are then imported under their
canonical module names before anything else imports them.

VllmConfig comes from EngineArgs with the production speculative config (method dflash, DFlash2 @ dc77ff1c,
7 tokens, probabilistic/standard) and the real GLM-5.3-Flash EXL3 config.json (no target weights). The drafter is
built by the real load_dflash_model -> get_model -> DefaultModelLoader (safetensors) -> process_weights_after_loading,
with a stub target that shares nothing (builds 1, 2) or shares a real BF16 ParallelLMHead holding the target's
lm_head.weight (build 3).
  build 1: GLM53_DRAFT_FP8=0                                   (stock reference)
  build 2: GLM53_DRAFT_FP8=1                                   (decoder-layer projections FP8)
  build 3: GLM53_DRAFT_FP8=layers,fc + GLM53_DRAFT_LMHEAD_FP8=1 (+ fc FP8, + candidate-head FP8 copy)
Asserts (exit non-zero on failure): patch scripts idempotent / fail closed; B writes its mode into qwen3_dflash.py
(four modes -> four distinct texts; a changed mode re-patches only that line) and refuses to build a drafter whose
GLM53_DRAFT_FP8 differs from the written mode; vLLM's AOT compile-cache key is the same for every GLM53_DRAFT_FP8
value, and vLLM's own loader check (_verify_source_unchanged, on torch's SourceInfo of the drafter's traced code)
rejects an artifact recorded under another mode; which linears get which method;
Marlin outputs vs the BF16 build within FP8 error; context-K/V fusion stays bit-exact BF16; memory falls;
candidate head uses the copy while the target lm_head object and bytes are untouched and compute_logits is
bit-identical to F.linear; the copy's packed bytes equal the production Glm53DenseFp8Method's.
Not coverable here (TP=1, eager, no target): TP=2 sharding through the real classes, drafter CUDA-graph capture
/ torch.compile with Marlin inside, acceptance rate, real step time.
"""
import gc
import importlib
import importlib.util
import json
import os
import shutil
import socket
import sys
import tempfile
from pathlib import Path

import torch

avail = int(next(l for l in open("/proc/meminfo") if l.startswith("MemAvailable")).split()[1]) * 1024
assert avail > 40 * 2**30, f"host MemAvailable {avail / 2**30:.1f} GiB < 40"
torch.cuda.set_per_process_memory_fraction(min(1.0, 8 * 2**30 / torch.cuda.get_device_properties(0).total_memory))

import vllm  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DRAFT = os.environ.get("DRAFT_DIR", os.path.join(os.environ.get("TF_EXL3_MODELS") or os.path.expanduser("~/models"), "GLM-5.3-Flash-DFlash2-dc77ff1c"))
TGT = os.environ.get("TARGET_DIR", os.path.join(os.environ.get("TF_EXL3_MODELS") or os.path.expanduser("~/models"), "GLM-5.3-Flash-EXL3-TR3-4bpw-partial"))
LMH = os.path.join(TGT, "lm_head")
FAIL: list[str] = []


def check(cond, msg):
    print(("  PASS " if cond else "  FAIL ") + msg, flush=True)
    if not cond:
        FAIL.append(msg)


def raises(fn, exc, text=""):
    try:
        fn()
    except exc as e:
        return text in str(e)
    except SystemExit as e:
        return exc is SystemExit and text in str(e)
    return False


REL = {
    "qwen3_dflash": "model_executor/models/qwen3_dflash.py",
    "qwen3_dflash2": "model_executor/models/qwen3_dflash2.py",
    "dflash_utils": "v1/worker/gpu/spec_decode/dflash/utils.py",
}
TMP = Path(tempfile.mkdtemp(prefix="glm53_draftfp8_"))
site = TMP / "site"
for rel in REL.values():
    (site / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(vllm.__file__).parent / rel, site / rel)
stock = {k: (site / v).read_text() for k, v in REL.items()}
os.environ["GLM53_SITE"] = str(site)


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P_B = load_file("glm53_patch_draft_fp8", REPO / "overlay/patch_drafter_fp8.py")
P_C = load_file("glm53_patch_draft_lmhead_fp8", REPO / "overlay/patch_drafter_lmhead_fp8.py")

print("== patch scripts on copies of the image files")
for label, P, keys in (("B patch_drafter_fp8", P_B, ("qwen3_dflash",)), ("C patch_drafter_lmhead_fp8", P_C, ("qwen3_dflash2", "dflash_utils"))):
    check(P.main(["x", "--preflight"]) == 0 and all((site / REL[k]).read_text() == stock[k] for k in keys), f"{label}: --preflight writes nothing")
    check(P.main(["x"]) == 0 and all((site / REL[k]).read_text() != stock[k] for k in keys), f"{label}: patches")
    once = {k: (site / REL[k]).read_text() for k in keys}
    check(P.main(["x"]) == 0 and all((site / REL[k]).read_text() == once[k] for k in keys), f"{label}: second run is a no-op")
# drift: B on a file whose anchor moved; C with one of its two files drifted (neither file may be written)
d = TMP / "driftB" / REL["qwen3_dflash"]
d.parent.mkdir(parents=True)
d.write_text(stock["qwen3_dflash"].replace("get_draft_quant_config(vllm_config)\n", "get_draft_quant_config(vllm_config)  # moved\n"))
P_B.TARGET = d
check(raises(lambda: P_B.main(["x"]), SystemExit, "preflight failed") and "glm53" not in d.read_text(), "B: drifted anchor fails closed, file untouched")
P_B.TARGET = site / REL["qwen3_dflash"]
dc = TMP / "driftC"
for k in ("qwen3_dflash2", "dflash_utils"):
    (dc / REL[k]).parent.mkdir(parents=True, exist_ok=True)
    (dc / REL[k]).write_text(stock[k])
(dc / REL["qwen3_dflash2"]).write_text(stock["qwen3_dflash2"].replace("self.candidate_logits_processor(self.lm_head, hidden_states)", "self.candidate_logits_processor(self.lm_head, hidden_states * 1)"))
P_C.UTILS, P_C.MODEL = dc / REL["dflash_utils"], dc / REL["qwen3_dflash2"]
P_C.FILES = ((P_C.UTILS, "utils.py", P_C.FILES[0][2]), (P_C.MODEL, "qwen3_dflash2.py", P_C.FILES[1][2]))
check(raises(lambda: P_C.main(["x"]), SystemExit, "preflight failed") and (dc / REL["dflash_utils"]).read_text() == stock["dflash_utils"],
      "C: one drifted file fails closed and the other file is not written either")
for env, P, bad in (("GLM53_DRAFT_FP8", P_B, "fp4"), ("GLM53_DRAFT_LMHEAD_FP8", P_C, "yes")):
    os.environ[env] = bad
    check(raises(lambda: P.main(["x", "--preflight"]), SystemExit, "must be"), f"{env}={bad} rejected at install time")
    del os.environ[env]

# ------------------------------------------------------------------ patched copies under the canonical names
print("\n== import the patched copies under their canonical module names")
MODS = {}
for key, canonical in (("qwen3_dflash", "vllm.model_executor.models.qwen3_dflash"),
                       ("qwen3_dflash2", "vllm.model_executor.models.qwen3_dflash2"),
                       ("dflash_utils", "vllm.v1.worker.gpu.spec_decode.dflash.utils")):
    parent_name, leaf = canonical.rsplit(".", 1)
    parent = importlib.import_module(parent_name)
    assert canonical not in sys.modules, f"{canonical} already imported"
    spec = importlib.util.spec_from_file_location(canonical, site / REL[key])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[canonical] = mod
    spec.loader.exec_module(mod)
    setattr(parent, leaf, mod)
    MODS[key] = mod
QD, QD2, DU = MODS["qwen3_dflash"], MODS["qwen3_dflash2"], MODS["dflash_utils"]
check(hasattr(QD, "Glm53DraftFp8Config") and hasattr(DU, "_Glm53Fp8CandidateHead")
      and "glm53_candidate_head" in open(QD2.__file__).read() and QD2.DFlashQwen3Model is QD.DFlashQwen3Model,
      "patched modules active (qwen3_dflash2 subclasses the patched DFlashQwen3Model)")

print("\n== B selection logic (pure functions)")
G = QD._glm53_draft_fp8_group
table = [("model.layers.45.self_attn.qkv_proj", {"layers"}, "layers"), ("model.layers.49.self_attn.o_proj", {"layers"}, "layers"),
         ("model.layers.47.mlp.gate_up_proj", {"layers"}, "layers"), ("model.layers.46.mlp.down_proj", {"layers"}, "layers"),
         ("model.fc", {"layers"}, None), ("model.fc", {"layers", "fc"}, "fc"),
         ("model.layers.45.attention_conv.kernel_projection", {"layers", "fc"}, None),
         ("model.layers.45.mlp_conv.kernel_projection", {"layers", "fc"}, None),
         ("model.candidate_selector.hidden_projection", {"layers", "fc"}, None), ("lm_head", {"layers", "fc"}, None),
         ("model.layers.45.self_attn.attn", {"layers", "fc"}, None), ("model.layers.45.self_attn.qkv_proj", {"fc"}, None)]
bad = [(p, g, e, G(p, g)) for p, g, e in table if G(p, g) != e]
check(not bad, f"prefix -> group table ({len(table)} cases) {bad or ''}")
envs = [("", set()), ("off", set()), ("0", set()), ("1", {"layers"}), ("on", {"layers"}), ("layers", {"layers"}),
        ("layers,fc", {"layers", "fc"}), ("fc", {"fc"})]
okenv = all(set(QD._glm53_draft_fp8_parse(raw)) == exp for raw, exp in envs)
okenv &= raises(lambda: QD._glm53_draft_fp8_parse("layers,fp4"), ValueError, "GLM53_DRAFT_FP8")
check(okenv, "GLM53_DRAFT_FP8 parsing (off/0/1/on/layers/layers,fc/fc; unknown group raises)")
check(QD._GLM53_DRAFT_FP8_INSTALLED == "off", f"the imported copy was patched with GLM53_DRAFT_FP8 unset: written mode 'off' ({QD._GLM53_DRAFT_FP8_INSTALLED!r})")


def set_mode(env):
    """Simulate a container whose patch script ran with GLM53_DRAFT_FP8=env (writes the mode into the file)."""
    os.environ["GLM53_DRAFT_FP8"] = env
    QD._GLM53_DRAFT_FP8_INSTALLED = P_B.mode_of(P_B.parse_env(env))


okg = True
for env, installed, exp in (("1", "layers", {"layers"}), ("layers,fc", "layers,fc", {"layers", "fc"}), ("0", "off", set())):
    os.environ["GLM53_DRAFT_FP8"] = env
    QD._GLM53_DRAFT_FP8_INSTALLED = installed
    okg &= set(QD._glm53_draft_fp8_groups()) == exp
for env, installed in (("1", "off"), ("0", "layers"), ("layers,fc", "layers"), ("layers", "layers,fc")):
    os.environ["GLM53_DRAFT_FP8"] = env
    QD._GLM53_DRAFT_FP8_INSTALLED = installed
    okg &= raises(QD._glm53_draft_fp8_groups, RuntimeError, "was patched for")
check(okg, "runtime GLM53_DRAFT_FP8 must equal the mode written into qwen3_dflash.py; any mismatch fails closed at drafter build")


class _Q:
    def get_name(self):
        return "fp8"


set_mode("0")
base = _Q()
check(QD._glm53_draft_fp8_quant_config(None) is None and QD._glm53_draft_fp8_quant_config(base) is base, "off: the draft quant config is passed through unchanged")
set_mode("1")
check(isinstance(QD._glm53_draft_fp8_quant_config(None), QD.Glm53DraftFp8Config), "on + BF16 drafter: Glm53DraftFp8Config")
check(raises(lambda: QD._glm53_draft_fp8_quant_config(base), RuntimeError, "needs a BF16 drafter"), "on + already-quantized drafter: fails closed")
check(QD.Glm53DraftFp8Config({"layers"}).get_quant_method(torch.nn.Linear(4, 4), "model.layers.45.mlp.down_proj") is None, "non-LinearBase layers get no method")
os.environ["GLM53_DRAFT_LMHEAD_FP8"] = "2"
check(raises(DU._glm53_draft_lmhead_fp8_enabled, ValueError, "must be 0 or 1"), "GLM53_DRAFT_LMHEAD_FP8=2 raises at load")
os.environ["GLM53_DRAFT_LMHEAD_FP8"] = "1"
check(raises(lambda: DU._glm53_attach_candidate_head(torch.nn.Module(), None), RuntimeError, "needs a DFlash2 drafter"), "C on a drafter without compute_candidates fails closed")
os.environ["GLM53_DRAFT_LMHEAD_FP8"] = "0"
check(DU._glm53_attach_candidate_head(torch.nn.Module(), None) is None, "C off: attach is a no-op")

# ------------------------------------------------------------------ real vLLM config + single-rank distributed
from vllm.config import set_current_vllm_config  # noqa: E402
from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment  # noqa: E402
from vllm.engine.arg_utils import EngineArgs  # noqa: E402
from vllm.model_executor.layers.linear import LinearBase, ReplicatedLinear, UnquantizedLinearMethod  # noqa: E402
from vllm.model_executor.layers.quantization import exl3 as EXL3  # noqa: E402
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead  # noqa: E402


def make_config():
    return EngineArgs(model=TGT, skip_tokenizer_init=True, tensor_parallel_size=1, max_model_len=4096, enforce_eager=True,
                      gpu_memory_utilization=0.1, max_num_seqs=8, language_model_only=True,
                      # production uses LOAD_FORMAT=instanttensor: same DefaultModelLoader.load_model flow (load_weights ->
                      # process_weights_after_loading), other weights iterator; it stages the file in non-PyTorch GPU
                      # buffers (process reached 12.3 GiB here), which breaks the 8 GiB test cap, so safetensors is used.
                      load_format=os.environ.get("LOAD_FORMAT", "safetensors"),
                      speculative_config={"method": "dflash", "model": DRAFT, "num_speculative_tokens": 7,
                                          "draft_sample_method": "probabilistic", "rejection_sample_method": "standard"}
                      ).create_engine_config()


vc0 = make_config()
with socket.socket() as s_:
    s_.bind(("127.0.0.1", 0))
    port = s_.getsockname()[1]
with set_current_vllm_config(vc0):
    init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method=f"tcp://127.0.0.1:{port}", backend="nccl")
    ensure_model_parallel_initialized(1, 1)
from vllm.model_executor.models.utils import get_draft_quant_config  # noqa: E402

print("\n== how the drafter gets its quant method in the image (production speculative config)")
print(f"  target: quantization={vc0.model_config.quantization} quant_config={type(vc0.quant_config).__name__}; "
      f"draft: quantization={vc0.speculative_config.draft_model_config.quantization}; get_draft_quant_config -> {get_draft_quant_config(vc0)}")
check(vc0.speculative_config.draft_model_config.quantization is None and get_draft_quant_config(vc0) is None,
      "stock drafter quant config is None -> every drafter linear is UnquantizedLinearMethod (never reaches Exl3Config)")

# signature compatibility: the image's exl3.py has Glm53DenseFp8Method(group), the launcher overlay's (group, prefix).
# The real class of the module under test is checked directly; the other constructor shape through a stub.
import inspect  # noqa: E402

TAKES_PREFIX = "prefix" in inspect.signature(EXL3.Glm53DenseFp8Method.__init__).parameters
print(f"  production module: {'launcher overlay' if TAKES_PREFIX else 'image'} signature "
      f"Glm53DenseFp8Method{inspect.signature(EXL3.Glm53DenseFp8Method.__init__)}")


def fp8_method(group, prefix):
    return EXL3.Glm53DenseFp8Method(group, prefix) if TAKES_PREFIX else EXL3.Glm53DenseFp8Method(group)


with set_current_vllm_config(vc0):
    cfg = QD.Glm53DraftFp8Config({"layers"})
    lin = ReplicatedLinear(64, 64, bias=False, quant_config=cfg, prefix="model.layers.45.mlp.down_proj", params_dtype=torch.bfloat16)
    ok1 = (type(lin.quant_method) is EXL3.Glm53DenseFp8Method and lin.quant_method.group == "draft"
           and (not TAKES_PREFIX or lin.quant_method.prefix == "model.layers.45.mlp.down_proj"))
    lin2 = ReplicatedLinear(64, 64, bias=False, quant_config=cfg, prefix="model.fc", params_dtype=torch.bfloat16)
    ok2 = type(lin2.quant_method) is UnquantizedLinearMethod
    orig = EXL3.Glm53DenseFp8Method

    if TAKES_PREFIX:
        class OtherShape(orig):  # the image's constructor shape
            def __init__(self, group):
                UnquantizedLinearMethod.__init__(self)
                self.group, self.ready = group, False
    else:
        class OtherShape(orig):  # the launcher overlay's constructor shape
            def __init__(self, group, prefix):
                super().__init__(group)
                self.prefix = prefix

    EXL3.Glm53DenseFp8Method = OtherShape
    try:
        lin3 = ReplicatedLinear(64, 64, bias=False, quant_config=cfg, prefix="model.layers.45.mlp.down_proj", params_dtype=torch.bfloat16)
        ok3 = (type(lin3.quant_method) is OtherShape and lin3.quant_method.group == "draft"
               and (getattr(lin3.quant_method, "prefix", None) == "model.layers.45.mlp.down_proj") != TAKES_PREFIX)
    finally:
        EXL3.Glm53DenseFp8Method = orig
    del lin, lin2, lin3
check(ok1 and ok2, "real ReplicatedLinear construction: allow-listed prefix -> Glm53DenseFp8Method('draft'[, prefix]), other -> UnquantizedLinearMethod")
check(ok3, f"works with the other Glm53DenseFp8Method constructor shape too ({'(group)' if TAKES_PREFIX else '(group, prefix)'})")

print("\n== B vs vLLM's torch.compile caches (production: ENFORCE_EAGER=0, torch 2.13 -> VLLM_USE_AOT_COMPILE=1)")
import hashlib  # noqa: E402

from torch._dynamo.package import SourceInfo  # noqa: E402

from vllm.compilation.caching import aot_compile_hash_factors  # noqa: E402
from vllm.compilation.decorators import _model_hash_key, _verify_source_unchanged  # noqa: E402

keys = {}
for env in ("off", "1", "layers,fc"):
    os.environ["GLM53_DRAFT_FP8"] = env
    keys[env] = hashlib.sha256(str(aot_compile_hash_factors(vc0) + [_model_hash_key(QD.DFlashQwen3Model.forward)]).encode()).hexdigest()
print(f"  AOT cache key (decorators.py: aot_compile_hash_factors + _model_hash_key(forward)): " + ", ".join(f"{k}={v[:12]}" for k, v in keys.items()))
check(len(set(keys.values())) == 1, "the AOT key is identical for every GLM53_DRAFT_FP8 value (the variable never enters it)")
scratch = TMP / "modes" / REL["qwen3_dflash"]
scratch.parent.mkdir(parents=True)
texts = {}
P_B.TARGET = scratch
for env in ("off", "1", "fc", "layers,fc"):
    scratch.write_text(stock["qwen3_dflash"])
    os.environ["GLM53_DRAFT_FP8"] = env
    P_B.main(["x"])
    texts[env] = scratch.read_text()
os.environ["GLM53_DRAFT_FP8"] = "off"
P_B.main(["x"])
back = scratch.read_text()
P_B.TARGET = site / REL["qwen3_dflash"]
check(len(set(texts.values())) == 4, "the patched qwen3_dflash.py text differs for each of the 4 modes (so the traced-source check sees a mode change)")
check(back == texts["off"] and back.count('_GLM53_DRAFT_FP8_INSTALLED = "') == 1,
      "a file patched for another mode is re-patched: only the mode line changes, the result equals a fresh patch")
# torch records the root frame and every inlined frame (output_graph.py / aot_compile.py SourceInfo.add_code) with the
# module's full text; vLLM's loader recomputes the hash of those files and raises on a difference, which makes
# _try_load_aot_compiled_fn fall back to compiling again (decorators.py).
si = SourceInfo(inlined_sources=set())
for code in (QD.DFlashQwen3Model.forward.__code__, QD2.DFlash2Qwen3DecoderLayer.forward.__code__):
    si.add_code(code)
mods = sorted(x.module for x in si.inlined_sources)
print(f"  traced modules recorded by torch for the drafter's forward: {mods}")
check("vllm.model_executor.models.qwen3_dflash" in mods, "qwen3_dflash (DFlashQwen3Model.forward, the compiled root) is a recorded source")
check(not raises(lambda: _verify_source_unchanged(si, vc0), RuntimeError), "same mode: vLLM's loader check accepts the recorded artifact")
os.environ["GLM53_DRAFT_FP8"] = "layers"
P_B.main(["x"])   # the imported copy on disk now says "layers" (as after a restart with GLM53_DRAFT_FP8=1)
check(raises(lambda: _verify_source_unchanged(si, vc0), RuntimeError, "Source code has changed"),
      "mode changed (off -> layers): vLLM's loader check rejects the artifact recorded under 'off' -> recompile")
os.environ["GLM53_DRAFT_FP8"] = "off"
P_B.main(["x"])
check(not raises(lambda: _verify_source_unchanged(si, vc0), RuntimeError), "mode back to off: the file text is identical again and the check accepts")

# ------------------------------------------------------------------ builds
from safetensors import safe_open  # noqa: E402

ck = {}
with safe_open(os.path.join(DRAFT, "model.safetensors"), framework="pt", device="cpu") as f:
    for l in range(5):
        for s in ("k_proj", "v_proj"):
            ck[(l, s)] = f.get_tensor(f"layers.{l}.self_attn.{s}.weight")
ck_kv = torch.cat([torch.cat([ck[(l, "k_proj")], ck[(l, "v_proj")]]) for l in range(5)])
gen = torch.Generator(device="cuda").manual_seed(1)
X = {k: (torch.randn(8, n, device="cuda", generator=gen)).to(torch.bfloat16) for k, n in (("h", 4096), ("attn", 4096), ("fc", 20480))}
HC = (torch.randn(7 * 8, 4096, device="cuda", generator=gen) * 0.02).to(torch.bfloat16)


class StubTarget(torch.nn.Module):
    def __init__(self, lm_head=None):
        super().__init__()
        self.model = torch.nn.Module()  # no embed_tokens -> no embedding share
        if lm_head is not None:
            self.lm_head = lm_head


def build(draft_env, lmhead_env, lm_head=None):
    set_mode(draft_env)   # one container per mode: the patch script wrote this mode into qwen3_dflash.py
    os.environ["GLM53_DRAFT_LMHEAD_FP8"] = lmhead_env
    vc = make_config()  # fresh config: attention layers register by name per config
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    with set_current_vllm_config(vc):
        model = DU.load_dflash_model(StubTarget(lm_head), vc)
        torch.cuda.synchronize()
        # the drafter's own (replaced) lm_head sits in a reference cycle (Parameter -> bound weight_loader ->
        # module) until the cyclic GC runs; collect before measuring. get_rope() caches the 1M-position
        # cos/sin table (0.25 GiB) across builds, so only build 1 pays it: compare linear weight bytes instead.
        gc.collect()
        torch.cuda.empty_cache()
        info = {"alloc": torch.cuda.memory_allocated() - before, "peak": torch.cuda.max_memory_allocated() - before}
        lb = 0
        for _n, m in model.named_modules():
            if isinstance(m, LinearBase):
                for t in list(m.parameters(recurse=False)) + [getattr(m, "workspace", None)]:
                    if t is not None:
                        lb += t.numel() * t.element_size()
        info["linear_bytes"] = lb
        info["methods"] = {n: type(m.quant_method).__name__ for n, m in model.named_modules() if isinstance(m, LinearBase)}
        info["prefix"] = {n: getattr(m, "prefix", None) for n, m in model.named_modules() if isinstance(m, LinearBase)}
        info["fp8_ready"] = {n for n, m in model.named_modules() if isinstance(m, LinearBase) and getattr(m.quant_method, "ready", False)}
        info["fused_kv"] = model.model._fused_kv_weight.detach().cpu()
        out = {}
        for i, layer in enumerate(model.model.layers):
            out[(i, "qkv_proj")] = layer.self_attn.qkv_proj(X["h"])[0].float()
            out[(i, "o_proj")] = layer.self_attn.o_proj(X["attn"][:, : layer.self_attn.o_proj.input_size_per_partition])[0].float()
            out[(i, "mlp")] = layer.mlp(X["h"]).float()
        out["fc"] = model.model.fc(X["fc"]).float()
        info["out"] = out
        if draft_env != "0":
            # production captures the drafter in a FULL CUDA graph and (ENFORCE_EAGER=0) compiles DFlashQwen3Model
            layer0 = model.model.layers[0]
            fns = {"mlp": layer0.mlp, "qkv_proj": lambda x: layer0.self_attn.qkv_proj(x)[0]}
            info["graph_equal"], info["compile_rel"] = {}, {}
            for name, fn in fns.items():
                eager = fn(X["h"])
                sx = X["h"].clone()
                st = torch.cuda.Stream()
                st.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(st):
                    fn(sx)
                torch.cuda.current_stream().wait_stream(st)
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    gy = fn(sx)
                gr.replay()
                torch.cuda.synchronize()
                info["graph_equal"][name] = torch.equal(gy, eager)
                del gr
                cy = torch.compile(fn, fullgraph=True, dynamic=False)(X["h"])
                info["compile_rel"][name] = rel(cy.float(), eager.float())
    return model, info, vc


def rel(a, b):
    return float((a - b).norm() / b.norm())


print("\n== build 1: GLM53_DRAFT_FP8=0 (stock)")
m1, i1, _ = build("0", "0")
kinds1 = set(i1["methods"].values())
print(f"  {len(i1['methods'])} LinearBase modules, methods {sorted(kinds1)}; allocated {i1['alloc'] / 2**30:.3f} GiB (peak {i1['peak'] / 2**30:.3f})")
check(kinds1 == {"UnquantizedLinearMethod"} and not i1["fp8_ready"], "stock build: every drafter linear is UnquantizedLinearMethod")
check(torch.equal(i1["fused_kv"], ck_kv), "stock build: fused context K/V weight == checkpoint K/V rows (bitwise)")
names_layers = sorted(n for n in i1["methods"] if QD._glm53_draft_fp8_group(n, {"layers"}))
print("  drafter linear prefixes: " + ", ".join(sorted(i1["methods"])[:6]) + ", ...")
ref_out = {k: v.clone() for k, v in i1["out"].items()}
del m1
gc.collect()
torch.cuda.empty_cache()

print("\n== build 2: GLM53_DRAFT_FP8=1 (decoder-layer projections)")
m2, i2, _ = build("1", "0")
fp8_2 = sorted(n for n, t in i2["methods"].items() if t == "Glm53DenseFp8Method")
print(f"  FP8 modules ({len(fp8_2)}): {fp8_2[0]} ... {fp8_2[-1]}; others: {sorted(set(t for n, t in i2['methods'].items() if n not in fp8_2))}")
pref = sorted(i2["prefix"][n] for n in fp8_2)
print(f"  prefixes the quant config saw for them: {pref[0]} ... {pref[-1]} (none contains 'draft': {not any('draft' in x for x in pref)})")
print(f"  BF16 prefixes: {sorted(i2['prefix'][n] for n in i2['methods'] if n not in fp8_2)}")
print(f"  allocated {i2['alloc'] / 2**30:.3f} GiB (stock {i1['alloc'] / 2**30:.3f}, which includes the 0.25 GiB rope cache), peak {i2['peak'] / 2**30:.3f} GiB")
saved_lin = i1["linear_bytes"] - i2["linear_bytes"]
print(f"  drafter linear weights: stock {i1['linear_bytes'] / 2**30:.3f} GiB -> FP8 {i2['linear_bytes'] / 2**30:.3f} GiB, saved {saved_lin / 2**30:.3f} GiB at TP=1 "
      f"(= {saved_lin / 2 / 1e9:.3f} GB per rank at TP=2)")
check(fp8_2 == names_layers and len(fp8_2) == 20 and set(fp8_2) == i2["fp8_ready"], "exactly the 20 decoder-layer projections are FP8 and processed (ready)")
check(i2["methods"]["model.fc"] == "UnquantizedLinearMethod", "fc stays BF16 with GLM53_DRAFT_FP8=1")
check(all(t == "UnquantizedLinearMethod" for n, t in i2["methods"].items() if n not in fp8_2), "every other drafter linear (conv kernel_projection, selector hidden_projection, fc) stays BF16")
check(torch.equal(i2["fused_kv"], ck_kv), "FP8 build: fused context K/V weight is still the BF16 checkpoint rows (built before the repack)")
errs = {k: rel(v, ref_out[k]) for k, v in i2["out"].items() if k != "fc"}
worst = max(errs.values())
print("  module outputs vs the stock build (rel): " + ", ".join(f"{kind} max {max(e for (i, k), e in errs.items() if k == kind):.3e}" for kind in ("qkv_proj", "o_proj", "mlp")))
check(1e-3 < worst < 5e-2, f"FP8 module outputs within FP8 error of the BF16 build (max rel {worst:.3e}) and actually changed")
check(torch.equal(i2["out"]["fc"], ref_out["fc"]), "fc output bit-identical to stock (still BF16)")
exp_saved = 5 * (6144 * 4096 + 4096 * 4096 + 24576 * 4096 + 4096 * 12288)   # 1 byte per weight (BF16 -> e4m3)
check(abs(saved_lin - exp_saved) < 0.02 * exp_saved, f"linear weight bytes fall by {saved_lin / 1e9:.3f} GB (expected {exp_saved / 1e9:.3f} GB = half the BF16 bytes)")
check(all(i2["graph_equal"].values()), f"FP8 layer-0 mlp / qkv_proj: CUDA-graph capture + replay == eager bitwise {i2['graph_equal']}")
check(all(v < 1e-2 for v in i2["compile_rel"].values()), "FP8 layer-0 mlp / qkv_proj: torch.compile(fullgraph=True) traces the Marlin op; rel vs eager "
      + ", ".join(f"{k} {v:.2e}" for k, v in i2["compile_rel"].items()))
del m2
gc.collect()
torch.cuda.empty_cache()

print("\n== build 3: GLM53_DRAFT_FP8=layers,fc + GLM53_DRAFT_LMHEAD_FP8=1, stub target shares a real BF16 lm_head")
man = json.load(open(os.path.join(LMH, "manifest.json")))
Vv, Hh = man["shape"]
with set_current_vllm_config(vc0):
    tlm = ParallelLMHead(Vv, Hh, params_dtype=torch.bfloat16).cuda()
raw = torch.from_file(os.path.join(LMH, "lm_head.weight.bin"), shared=False, size=Vv * Hh, dtype=torch.bfloat16).view(Vv, Hh)
tlm.weight.data.copy_(raw)   # raw (CPU, from the file) is kept to compare bytes afterwards
gc.collect()
m3, i3, vc3 = build("layers,fc", "1", lm_head=tlm)
fp8_3 = sorted(n for n, t in i3["methods"].items() if t == "Glm53DenseFp8Method")
head = getattr(m3, "glm53_candidate_head", None)
print(f"  FP8 modules: {len(fp8_3)} (layers + model.fc); allocated {i3['alloc'] / 2**30:.3f} GiB (own lm_head replaced by the target's; + FP8 copy), peak {i3['peak'] / 2**30:.3f} GiB")
check(len(fp8_3) == 21 and "model.fc" in fp8_3 and set(fp8_3) == i3["fp8_ready"], "layers,fc: 20 projections + fc are FP8")
check(rel(i3["out"]["fc"], ref_out["fc"]) < 5e-2 and not torch.equal(i3["out"]["fc"], ref_out["fc"]), f"fc output within FP8 error ({rel(i3['out']['fc'], ref_out['fc']):.3e})")
check(head is not None and m3.lm_head is tlm, "candidate head attached; drafter lm_head is still the target's object")
check("glm53_candidate_head" not in dict(m3.named_modules()) and not any("glm53_candidate_head" in k for k in m3.state_dict()), "copy is not a registered submodule (invisible to loader passes / state_dict)")
check(tlm.weight.dtype == torch.bfloat16 and torch.equal(tlm.weight.data.cpu(), raw), "target lm_head bytes untouched (BF16, bitwise equal to lm_head.weight from the checkpoint)")
nbytes = sum(t.numel() * t.element_size() for t in (head.weight, head.weight_scale, head.workspace))
print(f"  FP8 copy: {nbytes / 2**20:.1f} MiB at TP=1 (full vocab {Vv}) -> {nbytes / 2 / 2**20:.1f} MiB per rank at TP=2")
with set_current_vllm_config(vc3), torch.inference_mode():
    ids_c, un_c = m3.compute_candidates(HC)
    ref_logits = m3.candidate_logits_processor(m3.lm_head, HC)
    un_b, ids_b = torch.topk(ref_logits, 16, dim=-1)
    logits_v = m3.compute_logits(HC)
    direct = torch.nn.functional.linear(HC, tlm.weight)
fp8_logits = head.quant_method.apply(head, HC)
check(torch.equal(logits_v, direct), "verification-side compute_logits == F.linear(h, BF16 lm_head) bitwise (target head unaffected)")
check(torch.equal(torch.topk(fp8_logits, 16, dim=-1).indices, ids_c), "compute_candidates consumes the FP8 copy (its top-16 == top-16 of the copy's logits)")
ov = (ids_c[:, :, None] == ids_b[:, None, :]).any(-1).float().mean().item()
top1 = (ids_c[:, 0] == ids_b[:, 0]).float().mean().item()
print(f"  candidates (random hidden x0.02, 56 rows): top-1 agreement {top1:.3f}, top-16 overlap {ov:.3f} vs the BF16 head")
check(ov > 0.8, "FP8 candidate top-16 largely agrees with BF16 (indicative only)")
# packed bytes identical to the production class (row-chunked quantization == one-shot), on a 16384-row slice
sl = tlm.weight.data[:16384].clone()


class _Src(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.weight = torch.nn.Parameter(w, requires_grad=False)
        self.tp_size = 1


mine = DU._Glm53Fp8CandidateHead(_Src(sl))


class _L(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.weight = torch.nn.Parameter(w.clone(), requires_grad=False)
        self.output_size_per_partition, self.input_size_per_partition = w.shape


prod = _L(sl)
pm = fp8_method("draft_lm_head", "draft.glm53_candidate_head")
pm.process_weights_after_loading(prod)
check(torch.equal(mine.weight, prod.weight) and torch.equal(mine.weight_scale, prod.weight_scale),
      "copy's packed Marlin weight and scales == production Glm53DenseFp8Method on the same rows (bitwise)")
xx = HC[:8]
check(torch.equal(mine.quant_method.apply(mine, xx), pm.apply(prod, xx)), "copy's apply == production apply (bitwise)")

shutil.rmtree(TMP, ignore_errors=True)
torch.distributed.destroy_process_group()
print(f"\n{'ALL PASSED' if not FAIL else 'FAILED: ' + str(len(FAIL))}")
for f_ in FAIL:
    print("  -", f_)
sys.exit(1 if FAIL else 0)
