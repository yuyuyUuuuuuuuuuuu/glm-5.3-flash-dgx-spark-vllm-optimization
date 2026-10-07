"""Runtime helpers for the GLM-5.3 serving worker, loaded from the tf_exl3_moe plugin (integrate.plugin_register).

Both features are inert unless their env var is non-empty; they only wrap methods of
vllm.v1.worker.gpu_worker.Worker (patched when that module is imported, i.e. only in worker processes).

GLM53_MEM_HYGIENE=1
    After Worker.load_model and after Worker.compile_or_warm_up_model: gc.collect(), release PyTorch's cached
    pinned host blocks (torch._C._host_emptyCache: only blocks no tensor uses) and glibc malloc_trim(0) (returns
    free heap pages to the OS). Logs a host/GPU memory breakdown before and after ("[glm53-mem] ..."), including
    the number of NCCL connection buffers (9,633,792-byte pinned /dev/zero maps = Simple 4 MiB + LL 512 KiB +
    LL128 4.69 MiB). Nothing GPU-side is freed and no numerics change.

GLM53_LMHEAD_FP8=1
    After Worker.load_model: the target's BF16 lm_head shard (ParallelLMHead, per rank 77440 x 4096) is replaced by an
    FP8 e4m3 per-output-channel copy on production's Marlin path (exl3.Glm53DenseFp8Method, the same quantization as
    GLM53_DENSE_FP8 and the drafter's candidate head) and the BF16 weight is freed (meta tensor). The DFlash2 drafter
    shares this lm_head object, so it uses the FP8 head too (leave GLM53_DRAFT_LMHEAD_FP8 empty: its copy is then
    redundant). A self-test against the BF16 projection runs first; any failure keeps the BF16 head (WARNING).
    Changes the target's logits (quality must be checked like any FP8 group).

GLM53_TF_PROFILE=<dir>   (e.g. /tmp/glm53-prof)
    Installs a SIGUSR2 handler in the worker's main thread. `kill -USR2 <worker pid>` arms torch.profiler for the
    next GLM53_TF_PROFILE_STEPS (default 40) Worker.execute_model calls (everything between the first and the
    last call, i.e. forward + sampling + drafting); then writes <dir>/rank<r>-<pid>-<time>.json.gz (chrome trace)
    and a .txt table of GPU time per kernel. No profiler object exists until the signal arrives.
    GLM53_DEC_PROF_DIAG=1 (default off): boot preflight + per-session CUDA-activity check with kineto's stderr and
    one automatic retry, rank-numbered file names (see "profiler diagnostics" below, docs/DEC_HOSTLOOP.md).
"""
from __future__ import annotations

import ctypes
import gc
import importlib.abc
import importlib.util
import logging
import os
import re
import signal
import sys
import time

_log = logging.getLogger("vllm.glm53_runtime")
TARGET = "vllm.v1.worker.gpu_worker"
NCCL_CONN_BYTES = 4194304 + 524288 + 4915200          # NCCL Simple + LL + LL128 default buffer sizes
_STATE = {"patched": False, "arm": 0, "prof": None, "left": 0, "dir": None}


def _on(name: str) -> str:
    return os.environ.get(name, "").strip()


# ---------------------------------------------------------------- memory breakdown
def host_breakdown() -> dict:
    out = {}
    try:
        for line in open("/proc/self/status"):
            if line.startswith(("RssAnon", "RssFile", "RssShmem", "VmRSS")):
                k, v = line.split(":", 1)
                out[k] = int(v.split()[0]) // 1024                               # MiB
    except OSError:
        pass
    zero_mib = heap_mib = 0
    nccl = 0
    cur = None
    try:
        for line in open("/proc/self/smaps"):
            m = re.match(r"^([0-9a-f]+)-([0-9a-f]+) \S+ \S+ \S+ \S+\s*(.*)$", line)
            if m:
                name = m.group(3)
                size = int(m.group(2), 16) - int(m.group(1), 16)
                cur = "zero" if name.startswith("/dev/zero") else ("heap" if name == "[heap]" else None)
                if cur == "zero" and size == NCCL_CONN_BYTES:
                    nccl += 1
                continue
            if cur and line.startswith("Rss:"):
                kb = int(line.split()[1])
                if cur == "zero":
                    zero_mib += kb
                else:
                    heap_mib += kb
    except OSError:
        pass
    out["pinned_devzero"] = zero_mib // 1024
    out["heap"] = heap_mib // 1024
    out["nccl_conn_buffers"] = nccl
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable"):
                out["sys_MemAvailable"] = int(line.split()[1]) // 1024
    except OSError:
        pass
    return out


def gpu_breakdown() -> dict:
    try:
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_initialized():
            return {}
        free, total = torch.cuda.mem_get_info()
        return {"torch_alloc": torch.cuda.memory_allocated() >> 20, "torch_reserved": torch.cuda.memory_reserved() >> 20,
                "cuda_free": free >> 20}
    except Exception:  # noqa: BLE001
        return {}


def _fmt(d: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in d.items())


def mem_report(tag: str) -> dict:
    d = {**host_breakdown(), **gpu_breakdown()}
    _log.info("[glm53-mem] %s (MiB): %s", tag, _fmt(d))
    return d


def hygiene(tag: str) -> None:
    before = host_breakdown()
    gc.collect()
    freed_host = "n/a"
    try:
        import torch
        fn = getattr(torch._C, "_host_emptyCache", None)
        if fn is not None:
            fn()
            freed_host = "ok"
    except Exception as exc:  # noqa: BLE001
        freed_host = f"error {exc!r}"
    try:
        trimmed = ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception as exc:  # noqa: BLE001
        trimmed = f"error {exc!r}"
    after = host_breakdown()
    delta = {k: after.get(k, 0) - before.get(k, 0) for k in ("VmRSS", "RssAnon", "RssShmem", "pinned_devzero", "heap")}
    _log.info("[glm53-mem] hygiene %s: host_emptyCache=%s malloc_trim=%s; delta MiB %s; now %s",
              tag, freed_host, trimmed, _fmt(delta), _fmt(after))


# ---------------------------------------------------------------- profiler diagnostics (GLM53_DEC_PROF_DIAG=1)
# In production the rank-1 (nodeB worker) trace of 06:30 had only cpu_op events: no cuda_runtime, no kernel, and a
# 6.4 s stall inside its first traced step, while rank 0's trace of the same moment was complete and the same worker
# process type produced complete traces at 01:00. So CUPTI recorded nothing in that session; the reason was printed
# only to the worker's stderr. With GLM53_DEC_PROF_DIAG=1 (default off = the code below behaves exactly as before):
#   * a tiny profiler session right after warm-up initializes CUPTI at boot and logs, per rank, whether CUDA activity
#     is recorded ("[glm53-prof] rank R preflight: ..."), with kineto/CUPTI's own stderr lines and the context
#     (free device/host memory, CUPTI libraries mapped, CUPTI env knobs, driver);
#   * every SIGUSR2 session captures kineto's stderr around start/stop, counts the device events it recorded and, if
#     there are none, logs a WARNING with that context and re-arms ONE more session automatically;
#   * traces are named rank<torch.distributed rank>-<pid>-<time>-s<session> (RANK is not set in vLLM workers, so
#     both ranks were written as 'rankx', and two sessions in the same second overwrote each other).
_DIAG = {"sessions": 0, "retried": False, "stderr": ""}


def _diag_on() -> bool:
    # "0"/"off"/"false"/"no" are OFF like every other GLM53_DEC_* knob (docs/DEC_HOSTLOOP.md: revert = unset or =0)
    return _on("GLM53_DEC_PROF_DIAG").lower() not in ("", "0", "off", "false", "no")


def _rank_label() -> str:
    if _diag_on():
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                return str(dist.get_rank())
        except Exception:  # noqa: BLE001
            pass
    return os.environ.get("RANK", os.environ.get("LOCAL_RANK", "x"))


def _capture_fd2(fn):
    """fn() with fd 2 redirected to a temporary file (kineto / CUPTI print their warnings there); the captured text
    is written back to the real stderr afterwards and returned."""
    import tempfile
    try:
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass
    tf = tempfile.TemporaryFile()
    saved = os.dup(2)
    try:
        os.dup2(tf.fileno(), 2)
        try:
            r = fn()
        finally:
            try:
                sys.stderr.flush()
            except Exception:  # noqa: BLE001
                pass
            os.dup2(saved, 2)
            os.close(saved)
        tf.seek(0)
        txt = tf.read().decode(errors="replace")
    finally:
        tf.close()
    if txt:
        try:
            os.write(2, txt.encode())
        except OSError:
            pass
    return r, txt


def _cupti_context() -> dict:
    import torch
    d: dict = {}
    try:
        d["kineto"] = bool(torch.autograd.kineto_available())
        d["activities"] = sorted(str(a).split(".")[-1] for a in torch.profiler.supported_activities())
    except Exception as exc:  # noqa: BLE001
        d["kineto"] = repr(exc)
    for k in ("TEARDOWN_CUPTI", "DISABLE_CUPTI_LAZY_REINIT", "CUDA_MODULE_LOADING", "KINETO_LOG_LEVEL",
              "KINETO_USE_DAEMON", "KINETO_CONFIG"):
        if k in os.environ:
            d[k] = os.environ[k]
    d["CUDA_INJECTION64_PATH_set"] = "CUDA_INJECTION64_PATH" in os.environ
    libs = set()
    try:
        for line in open("/proc/self/maps"):
            f = line.split()[-1]
            if "cupti" in f.lower() or "nvperf" in f.lower():
                libs.add(f)
    except OSError:
        pass
    d["cupti_libs"] = sorted(libs)
    try:
        d["cuda_free_mib"] = torch.cuda.mem_get_info()[0] >> 20
    except Exception:  # noqa: BLE001
        pass
    d["MemAvailable_mib"] = host_breakdown().get("sys_MemAvailable")
    try:
        d["driver"] = open("/proc/driver/nvidia/version").readline().strip()[:100]
    except OSError:
        pass
    d["sessions_before"] = _DIAG["sessions"]
    return d


def _device_events(prof) -> int:
    import torch
    try:
        return sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)
    except Exception:  # noqa: BLE001
        return -1


def _kineto_lines(txt: str) -> str:
    keep = [ln for ln in txt.splitlines() if re.search(r"cupti|kineto|profil|warn|error|fail", ln, re.I)
            and "SyncActivityProfilerHandler" not in ln]            # the USDT start/stop markers are not news
    return " | ".join(keep)[-1500:]


def prof_preflight() -> int:
    """GLM53_DEC_PROF_DIAG: one tiny session at boot (after warm-up). Returns the number of device events seen."""
    import torch
    from torch.profiler import ProfilerActivity, profile
    x = torch.ones(4096, device="cuda")
    torch.cuda.synchronize()
    t0 = time.time()
    p = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
    _, t1 = _capture_fd2(p.__enter__)
    y = (x * 2.0).sum()
    torch.cuda.synchronize()
    _, t2 = _capture_fd2(lambda: p.__exit__(None, None, None))
    del y
    n = _device_events(p)
    _DIAG["sessions"] += 1
    ctx = _cupti_context()
    msg = _kineto_lines(t1 + t2)
    if n > 0:
        _log.info("[glm53-prof] rank %s preflight: CUDA activity recorded (%d device events, %.2f s); kineto: %s; %s",
                  _rank_label(), n, time.time() - t0, msg or "-", ctx)
    else:
        _log.warning("[glm53-prof] rank %s preflight: NO CUDA activity recorded (%.2f s) -- SIGUSR2 traces of this "
                     "rank will be CPU-only; kineto said: %s; context %s", _rank_label(), time.time() - t0,
                     msg or "(nothing on stderr)", ctx)
    return n


# ---------------------------------------------------------------- profiler
def _arm(signum, frame) -> None:                                  # signal handler: only sets a counter
    if _STATE["prof"] is None:
        _STATE["arm"] = int(_on("GLM53_TF_PROFILE_STEPS") or 40)


def _prof_step() -> None:
    st = _STATE
    if st["prof"] is None and st["arm"] > 0:
        import torch
        from torch.profiler import ProfilerActivity, profile
        st["left"], st["arm"] = st["arm"], 0
        st["t0"] = time.time()
        st["prof"] = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False,
                             with_stack=False, profile_memory=False)
        torch.cuda.synchronize()
        if _diag_on():
            _DIAG["sessions"] += 1
            st["ctx"] = _cupti_context()
            st["steps"] = st["left"]
            _, _DIAG["stderr"] = _capture_fd2(st["prof"].__enter__)
        else:
            st["prof"].__enter__()
        _log.info("[glm53-prof] armed for %d execute_model calls", st["left"])
        return
    if st["prof"] is not None:
        st["left"] -= 1
        if st["left"] <= 0:
            import torch
            torch.cuda.synchronize()
            prof, st["prof"] = st["prof"], None
            wall = time.time() - st["t0"]
            diag = _diag_on()
            if diag:
                _, txt = _capture_fd2(lambda: prof.__exit__(None, None, None))
                _DIAG["stderr"] += txt
            else:
                prof.__exit__(None, None, None)
            d = st["dir"]
            os.makedirs(d, exist_ok=True)
            rank = _rank_label()
            base = os.path.join(d, f"rank{rank}-{os.getpid()}-{time.strftime('%Y%m%d-%H%M%S')}")
            if diag:
                base += f"-s{_DIAG['sessions']}"
            try:
                prof.export_chrome_trace(base + ".json")
                import gzip
                import shutil
                with open(base + ".json", "rb") as fi, gzip.open(base + ".json.gz", "wb") as fo:
                    shutil.copyfileobj(fi, fo)
                os.remove(base + ".json")
            except Exception as exc:  # noqa: BLE001
                _log.warning("[glm53-prof] chrome trace export failed: %r", exc)
            try:
                tab = prof.key_averages().table(sort_by="self_device_time_total", row_limit=80, max_name_column_width=90)
                with open(base + ".txt", "w") as f:
                    f.write(f"wall {wall:.3f} s\n{tab}\n")
            except Exception as exc:  # noqa: BLE001
                _log.warning("[glm53-prof] table export failed: %r", exc)
            _log.info("[glm53-prof] wrote %s.{json.gz,txt} (wall %.2f s)", base, wall)
            if diag:
                n = _device_events(prof)
                msg = _kineto_lines(_DIAG["stderr"])
                if n > 0:
                    _log.info("[glm53-prof] rank %s session %d: %d device events recorded; kineto: %s", rank,
                              _DIAG["sessions"], n, msg or "-")
                else:
                    again = not _DIAG["retried"]
                    _DIAG["retried"] = True
                    _log.warning("[glm53-prof] rank %s session %d recorded NO CUDA activity (CPU-only trace); kineto "
                                 "said: %s; context at start %s%s", rank, _DIAG["sessions"],
                                 msg or "(nothing on stderr)", st.get("ctx"),
                                 "; re-arming one more session now" if again else "")
                    if again:
                        st["arm"] = st.get("steps") or int(_on("GLM53_TF_PROFILE_STEPS") or 40)



# ---------------------------------------------------------------- target lm_head -> FP8 (Marlin)
class _Fp8HeadApply:
    """quant_method stand-in for the converted lm_head: LogitsProcessor calls lm_head.quant_method.apply(lm_head, x, bias)."""

    def __init__(self, method, holder) -> None:
        self.method, self.holder = method, holder

    def apply(self, layer, x, bias=None):
        return self.method.apply(self.holder, x, bias)


def _find_lm_head(worker):
    mr = getattr(worker, "model_runner", None)
    model = None
    for get in (lambda: mr.get_model(), lambda: mr.model):
        try:
            model = get()
            if model is not None:
                break
        except Exception:  # noqa: BLE001
            continue
    if model is None:
        return None, None
    lm = getattr(model, "lm_head", None)
    if type(lm).__name__ == "ParallelLMHead":
        return model, lm
    # multimodal wrappers keep it deeper (Glm5Next: language_model.lm_head): the target's only ParallelLMHead
    heads = [(n, m) for n, m in model.named_modules() if type(m).__name__ == "ParallelLMHead"]
    named = [(n, m) for n, m in heads if n == "lm_head" or n.endswith(".lm_head")]
    pick = named if len(named) == 1 else heads
    if len(pick) == 1:
        _log.info("[glm53-lmhead-fp8] target lm_head found at %s", pick[0][0])
        return model, pick[0][1]
    _log.warning("[glm53-lmhead-fp8] %d ParallelLMHead modules (%s)", len(heads), [n for n, _ in heads][:4])
    return model, None


def convert_lm_head_fp8(worker) -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    model, lm = _find_lm_head(worker)
    if lm is None or type(lm).__name__ != "ParallelLMHead":
        _log.warning("[glm53-lmhead-fp8] no ParallelLMHead found (%s); lm_head stays BF16", type(lm).__name__)
        return
    if type(lm.quant_method).__name__ != "UnquantizedEmbeddingMethod":
        _log.warning("[glm53-lmhead-fp8] lm_head quant_method is %s; unchanged", type(lm.quant_method).__name__)
        return
    hd = getattr(getattr(getattr(worker, "vllm_config", None), "model_config", None), "head_dtype", None)
    w = lm.weight.data
    if hd is not None and hd != w.dtype:
        _log.warning("[glm53-lmhead-fp8] head_dtype %s != weight dtype %s; unchanged", hd, w.dtype)
        return
    if w.dtype not in (torch.bfloat16, torch.float16) or w.dim() != 2 or w.device.type != "cuda":
        _log.warning("[glm53-lmhead-fp8] unexpected lm_head weight %s %s %s; unchanged", w.dtype, tuple(w.shape), w.device)
        return
    import inspect
    from vllm.model_executor.layers.quantization.exl3 import Glm53DenseFp8Method
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_layer_for_marlin
    n, k = w.shape
    holder = nn.Module()
    fp8 = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=w.device)
    scales = torch.empty(n, dtype=torch.float32, device=w.device)
    for a in range(0, n, 8192):                           # amax/448 per output channel, e4m3 (Glm53DenseFp8Method numerics)
        wf = w[a:a + 8192].float()
        sc = wf.abs().amax(dim=1).clamp(min=1e-12) / 448.0
        fp8[a:a + 8192] = (wf / sc[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        scales[a:a + 8192] = sc
        del wf
    holder.output_size_per_partition, holder.input_size_per_partition = n, k
    holder.orig_dtype = w.dtype
    holder.weight = nn.Parameter(fp8, requires_grad=False)
    holder.weight_scale = nn.Parameter(scales.to(w.dtype), requires_grad=False)
    holder.weight_block_size = None
    prepare_fp8_layer_for_marlin(holder, size_k_first=False)
    holder.glm53_fp8_n, holder.glm53_fp8_k = n, k
    if "prefix" in inspect.signature(Glm53DenseFp8Method.__init__).parameters:
        method = Glm53DenseFp8Method("lm_head", "lm_head")
    else:
        method = Glm53DenseFp8Method("lm_head")
    method.ready = True
    # self-test against the BF16 projection before switching (rows of the real weight, realistic hidden scale)
    g = torch.Generator(device=w.device).manual_seed(0)
    worst = 0.0
    for m in (1, 5, 8, 64):
        x = (torch.randn((m, k), generator=g, device=w.device, dtype=torch.float32) * 2.0).to(w.dtype)
        ref = F.linear(x.float(), w.float())
        out = method.apply(holder, x, None).float()
        rel = ((out - ref).norm() / ref.norm().clamp(min=1e-12)).item()
        top1 = (out.argmax(-1) == ref.argmax(-1)).float().mean().item()
        worst = max(worst, rel)
        if not (rel < 6e-2 and out.shape == ref.shape):
            _log.warning("[glm53-lmhead-fp8] self-test failed at M=%d (rel_l2 %.3e, shape %s); lm_head stays BF16",
                         m, rel, tuple(out.shape))
            return
    lm.quant_method = _Fp8HeadApply(method, holder)
    object.__setattr__(lm, "glm53_fp8_head", holder)
    freed = w.numel() * w.element_size()
    lm._parameters["weight"] = nn.Parameter(torch.empty((n, k), dtype=w.dtype, device="meta"), requires_grad=False)
    del w
    torch.cuda.empty_cache()
    nb = sum(t.numel() * t.element_size() for t in (holder.weight, holder.weight_scale, holder.workspace))
    _log.info("[glm53-lmhead-fp8] target lm_head %s -> FP8 Marlin (self-test rel_l2 max %.2e); freed %.1f MiB BF16, "
              "holds %.1f MiB", (n, k), worst, freed / 2**20, nb / 2**20)

# ---------------------------------------------------------------- patching
def _patch(mod) -> None:
    if _STATE["patched"]:
        return
    W = getattr(mod, "Worker", None)
    if W is None:
        _log.warning("[glm53-runtime] %s has no Worker; nothing patched", TARGET)
        return
    hyg, prof_dir, lmfp8 = _on("GLM53_MEM_HYGIENE"), _on("GLM53_TF_PROFILE"), _on("GLM53_LMHEAD_FP8")
    orig_load, orig_warm, orig_exec = W.load_model, W.compile_or_warm_up_model, W.execute_model

    def load_model(self, *a, **k):
        r = orig_load(self, *a, **k)
        if lmfp8:
            try:
                convert_lm_head_fp8(self)
            except Exception as exc:  # noqa: BLE001
                _log.warning("[glm53-lmhead-fp8] conversion failed (lm_head stays as loaded): %r", exc)
        if hyg:
            mem_report("after load_model")
            hygiene("after load_model")
        return r

    def compile_or_warm_up_model(self, *a, **k):
        if hyg:
            mem_report("before compile_or_warm_up_model (KV allocated)")
        r = orig_warm(self, *a, **k)
        if hyg:
            mem_report("after compile_or_warm_up_model")
            hygiene("after compile_or_warm_up_model")
        if prof_dir and _diag_on():
            try:
                prof_preflight()
            except Exception as exc:  # noqa: BLE001
                _log.warning("[glm53-prof] preflight failed: %r", exc)
        if prof_dir:
            try:
                signal.signal(signal.SIGUSR2, _arm)
                _log.info("[glm53-prof] ready: kill -USR2 %d profiles the next %s execute_model calls into %s",
                          os.getpid(), _on("GLM53_TF_PROFILE_STEPS") or 40, prof_dir)
            except Exception as exc:  # noqa: BLE001
                _log.warning("[glm53-prof] could not install SIGUSR2 handler: %r", exc)
        return r

    def execute_model(self, *a, **k):
        if _STATE["arm"] or _STATE["prof"] is not None:
            try:
                _prof_step()
            except Exception as exc:  # noqa: BLE001
                _STATE["prof"], _STATE["arm"] = None, 0
                _log.warning("[glm53-prof] profiler failed, disarmed: %r", exc)
        return orig_exec(self, *a, **k)

    W.load_model = load_model
    W.compile_or_warm_up_model = compile_or_warm_up_model
    if prof_dir:
        _STATE["dir"] = prof_dir
        W.execute_model = execute_model
    _STATE["patched"] = True
    _log.info("[glm53-runtime] patched Worker in pid %d: mem_hygiene=%s profile=%s lmhead_fp8=%s", os.getpid(), bool(hyg),
              prof_dir or "off", bool(lmfp8))


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return spec
        orig_exec = spec.loader.exec_module

        def exec_module(module):
            orig_exec(module)
            try:
                _patch(module)
            except Exception as exc:  # noqa: BLE001
                _log.warning("[glm53-runtime] patch failed (worker unchanged): %r", exc)
        spec.loader.exec_module = exec_module
        return spec


def install() -> None:
    """Called from integrate.plugin_register in every vLLM process. Never raises."""
    try:
        if not (_on("GLM53_MEM_HYGIENE") or _on("GLM53_TF_PROFILE") or _on("GLM53_LMHEAD_FP8")):
            return
        if TARGET in sys.modules:
            _patch(sys.modules[TARGET])
        elif not any(isinstance(f, _Finder) for f in sys.meta_path):
            sys.meta_path.insert(0, _Finder())
    except Exception as exc:  # noqa: BLE001
        _log.warning("[glm53-runtime] install failed (worker unchanged): %r", exc)
